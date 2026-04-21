import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from model import CV_FOLDS, RANDOM_SEED, build_model
from preprocess import Preprocessor, load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)



def _parse_args():
    p = argparse.ArgumentParser(
        description="Train an ATI classification model."
    )
    p.add_argument("--data",       required=True,  help="Path to train CSV")
    p.add_argument("--model-name", default="LightGBM",
                   help="Model to use (LightGBM, Ensemble, NeuralNet, …)")
    p.add_argument("--output-dir", default="weights")
    p.add_argument("--n-features", type=int, default=300,
                   help="Number of features after SelectKBest")
    p.add_argument("--n-svd",      type=int, default=150,
                   help="Number of TruncatedSVD components for proteins")
    p.add_argument("--calibrate",  action="store_true",
                   help="Wrap final model in isotonic calibration")
    p.add_argument("--seed",       type=int, default=RANDOM_SEED)
    return p.parse_args()



def run_cv(
    df:          "pd.DataFrame",
    model_name:  str,
    n_features:  int,
    n_svd:       int,
    seed:        int,
):
    
    import pandas as pd

    if "ati" not in df.columns:
        raise KeyError("'ati' column not found")

    df    = df.dropna(subset=["ati"]).reset_index(drop=True)
    y     = df["ati"].astype(int).values
    skf   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)

    oof_probs = np.zeros(len(y), dtype=float)
    fold_aucs = []

    logging.info(
        f"[{model_name}] Running {CV_FOLDS}-fold CV "
        f"(Preprocessor fit inside each fold — no leakage)..."
    )

    for fold, (tr_idx, val_idx) in enumerate(skf.split(df, y)):
        df_tr  = df.iloc[tr_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        y_tr   = y[tr_idx]
        y_val  = y[val_idx]

        
        prep  = Preprocessor(
            n_protein_components = n_svd,
            n_top_features       = n_features,
            random_state         = seed,
        )
        X_tr  = prep.fit_transform(df_tr, y_tr)
        X_val = prep.transform(df_val)


        clf = build_model(model_name)
        clf.fit(X_tr, y_tr)

        fold_probs              = clf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx]      = fold_probs
        fold_auc                = roc_auc_score(y_val, fold_probs)
        fold_aucs.append(fold_auc)
        logging.info(
            f"  Fold {fold+1}/{CV_FOLDS}  "
            f"n_train={len(y_tr)}  n_val={len(y_val)}  "
            f"AUC={fold_auc:.4f}"
        )

    return oof_probs, y, fold_aucs



# ─────────────────────────── main ────────────────────────────────────────────

def main():
    args    = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Model        : {args.model_name}")
    logging.info(f"Output dir   : {out_dir}")
    logging.info(f"Random seed  : {args.seed}")
    logging.info(f"n_features   : {args.n_features}")
    logging.info(f"n_svd        : {args.n_svd}")

    # ── load data ─────────────────────────────────────────────────────────────
    df = load_data(args.data)
    df = df.dropna(subset=["ati"]).reset_index(drop=True)
    y  = df["ati"].astype(int).values

    logging.info(
        f"Data shape   : {df.shape}  "
        f"(ATI={y.sum()}, No ATI={(y == 0).sum()})"
    )

    # ── CV evaluation ──────────────────────────────────────────────────────────
    oof_probs, y_true, fold_aucs = run_cv(
        df, args.model_name, args.n_features, args.n_svd, args.seed
    )

    oof_preds = (oof_probs >= 0.5).astype(int)
    cv_auc    = roc_auc_score(y_true, oof_probs)
    cv_loss   = log_loss(y_true, oof_probs)

    print("\n" + "=" * 60)
    print(f"  {args.model_name} — {CV_FOLDS}-Fold CV")
    print("=" * 60)
    print(
        classification_report(
            y_true, oof_preds, target_names=["No ATI", "ATI"]
        )
    )
    print(f"  AUC (ROC)    : {cv_auc:.4f}")
    print(f"  Log Loss     : {cv_loss:.4f}")
    print(f"  Per-fold AUC : {[f'{a:.4f}' for a in fold_aucs]}")
    print("=" * 60)

    # ── final model on all data ────────────────────────────────────────────────
    logging.info(
        f"\nTraining final {args.model_name} on all {len(df)} samples..."
    )

    prep_final = Preprocessor(
        n_protein_components = args.n_svd,
        n_top_features       = args.n_features,
        random_state         = args.seed,
    )
    X_all = prep_final.fit_transform(df, y)

    clf_final = build_model(args.model_name)

    if args.calibrate:
        logging.info("  Applying isotonic probability calibration (cv=3)…")
        clf_final = CalibratedClassifierCV(clf_final, method="isotonic", cv=3)

    clf_final.fit(X_all, y)

    # Sanity: training-set AUC (optimistic by definition)
    train_probs = clf_final.predict_proba(X_all)[:, 1]
    train_auc   = roc_auc_score(y, train_probs)
    train_loss  = log_loss(y, train_probs)

    print("\n" + "=" * 55)
    print(f"  Final model  : {args.model_name}")
    print(f"  Samples      : {len(df)}  (ATI={y.sum()}, No ATI={(y==0).sum()})")
    print(f"  Features     : {X_all.shape[1]}")
    print(f"  Calibrated   : {args.calibrate}")
    print(f"  CV AUC       : {cv_auc:.4f}   ← primary metric")
    print(f"  Train AUC    : {train_auc:.4f}  (optimistic — sanity only)")
    print(f"  Train Loss   : {train_loss:.4f}")
    print("=" * 55)

    # ── persist artefacts ─────────────────────────────────────────────────────
    model_path = out_dir / "model.pkl"
    prep_path  = out_dir / "preprocessor.pkl"   # replaces feature_selector.pkl
    names_path = out_dir / "feature_cols.json"
    summ_path  = out_dir / "training_summary.json"

    joblib.dump(clf_final,  model_path)
    joblib.dump(prep_final, prep_path)

    with open(names_path, "w") as f:
        json.dump(prep_final.feature_names_out_, f, indent=2)

    summary = {
        "model"       : args.model_name,
        "n_samples"   : len(df),
        "n_features"  : int(X_all.shape[1]),
        "n_svd"       : args.n_svd,
        "calibrated"  : args.calibrate,
        "cv_auc"      : round(cv_auc, 4),
        "cv_logloss"  : round(cv_loss, 4),
        "fold_aucs"   : [round(a, 4) for a in fold_aucs],
        "train_auc"   : round(train_auc, 4),
        "train_loss"  : round(train_loss, 4),
    }
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Saved model        : {model_path}")
    logging.info(f"Saved preprocessor : {prep_path}")
    logging.info(f"Saved features     : {names_path}")
    logging.info(f"Saved summary      : {summ_path}")

    print(f"\nAll weights saved to : {out_dir}/")
    print(f"Next step            : bash predict.sh /path/to/new_data.csv")


if __name__ == "__main__":
    main()