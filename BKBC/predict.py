import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess import load_data

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
)

_SCRIPT_DIR  = Path(__file__).resolve().parent          # …/hilton/BKBC/
_WEIGHTS_DIR = _SCRIPT_DIR / "weights"                  # …/hilton/BKBC/weights/
_MODEL_PATH  = _WEIGHTS_DIR / "model.pkl"
_PREP_PATH   = _WEIGHTS_DIR / "preprocessor.pkl"
_FEATS_PATH  = _WEIGHTS_DIR / "feature_cols.json"
_SUMM_PATH   = _WEIGHTS_DIR / "training_summary.json"

# Models that need float32 input (PyTorch-backed)
_FLOAT32_MODELS = {"NeuralNetClassifier", "_SoftVotingEnsemble"}



def _check_file(path: Path, label: str) -> Path:
    """
    Raise a clear FileNotFoundError that lists the directory contents
    so you immediately see what IS in the weights folder.
    """
    if path.is_file():
        return path

    if path.parent.is_dir():
        contents = "\n    ".join(
            sorted(f.name for f in path.parent.iterdir())
        ) or "(empty)"
    else:
        contents = "(directory does not exist)"

    raise FileNotFoundError(
        f"\n[{label}] not found: {path}\n"
        f"Contents of {path.parent}:\n    {contents}\n\n"
        f"Run train.py first, e.g.:\n"
        f"  python train.py --data <train.csv> --model-name LightGBM\n"
        f"  python train.py --data <train.csv> --model-name Ensemble"
    )



def _parse_args():
    p = argparse.ArgumentParser(
        description="Run a trained ATI model on new proteomics data."
    )
    p.add_argument(
        "--data",
        required = True,
        help     = "Path to input CSV (same feature schema as training data)",
    )
    p.add_argument(
        "--out",
        default = None,
        help    = (
            "Output CSV path. "
            "Default: predictions.csv inside the weights directory"
        ),
    )
    p.add_argument(
        "--model-dir",
        default = None,
        help    = (
            "Override the weights directory. "
            f"Default: {_WEIGHTS_DIR}"
        ),
    )
    p.add_argument(
        "--threshold",
        type    = float,
        default = 0.5,
        help    = "Probability threshold for ATI=1 (default: 0.5)",
    )
    return p.parse_args()



def _load_artifacts(weights_dir: Path):
    """
    Load model.pkl, preprocessor.pkl, and (optionally) feature_cols.json
    from the given weights directory.

    Returns
    -------
    model        : sklearn-compatible classifier
    prep         : fitted Preprocessor
    feature_cols : list[str] or None
    model_type   : str  class name of the loaded model
    """
    model_path = _check_file(weights_dir / "model.pkl",         "model.pkl")
    prep_path  = _check_file(weights_dir / "preprocessor.pkl",  "preprocessor.pkl")

    logging.info(f"  model.pkl        : {model_path}")
    logging.info(f"  preprocessor.pkl : {prep_path}")

    model = joblib.load(model_path)
    prep  = joblib.load(prep_path)

    model_type = type(model).__name__
    logging.info(f"  Model type       : {model_type}")

    # feature_cols.json is informational only — not required for inference
    feature_cols = None
    feat_path = weights_dir / "feature_cols.json"
    if feat_path.is_file():
        with open(feat_path) as f:
            feature_cols = json.load(f)
        logging.info(f"  feature_cols     : {len(feature_cols)} features listed")
    else:
        logging.info("  feature_cols.json not found — skipping (not needed for inference)")

    # Print training summary if available
    summ_path = weights_dir / "training_summary.json"
    if summ_path.is_file():
        with open(summ_path) as f:
            summ = json.load(f)
        logging.info(
            f"  Training summary : CV AUC={summ.get('cv_auc', 'N/A')}  "
            f"model={summ.get('model', 'N/A')}  "
            f"n_features={summ.get('n_features', 'N/A')}"
        )

    return model, prep, feature_cols, model_type



def _run_inference(
    model:      object,
    prep:       object,
    df:         pd.DataFrame,
    model_type: str,
    threshold:  float,
):
    """
    Full inference pipeline:
      1. Extract sample IDs and ground-truth labels (if present)
      2. Preprocess via prep.transform()
      3. Cast dtype to match model requirement
      4. Predict probabilities + hard labels
      5. Return results DataFrame

    Returns
    -------
    results    : pd.DataFrame  (sample_id, prob_ati, pred_label [, true_label])
    has_labels : bool
    """

    if "sample_id" in df.columns:
        sample_ids = df["sample_id"].values
    else:
        logging.warning("'sample_id' column not found — using row indices")
        sample_ids = np.arange(len(df))

    has_labels = "ati" in df.columns
    y_true     = df["ati"].astype(int).values if has_labels else None
    if not has_labels:
        logging.info("No 'ati' column — evaluation metrics will not be computed")

    logging.info("Preprocessing data...")
    try:
        X = prep.transform(df)
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise
    logging.info(f"  Feature matrix shape : {X.shape}")

    if model_type in _FLOAT32_MODELS:
        X = X.astype(np.float32)
        logging.info("  Cast to float32 (PyTorch model)")
    else:
        X = X.astype(np.float64)

    logging.info(f"Running inference on {X.shape[0]} samples...")
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise

    preds = (probs >= threshold).astype(int)
    logging.info(
        f"  Predicted ATI=1  : {preds.sum():,} / {len(preds):,} "
        f"({preds.mean():.1%})  threshold={threshold}"
    )

    results = pd.DataFrame({
        "sample_id"  : sample_ids,
        "prob_ati"   : np.round(probs, 6),
        "pred_label" : preds,
    })

    if has_labels:
        results["true_label"] = y_true

    return results, has_labels


def _evaluate(results: pd.DataFrame):
    """Print classification metrics when ground-truth labels are present."""
    from sklearn.metrics import classification_report, log_loss, roc_auc_score

    y_true = results["true_label"].values
    y_pred = results["pred_label"].values
    y_prob = results["prob_ati"].values

    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  Evaluation")
    print(f"{sep}")
    print(
        f"  Samples  : {len(results):,}"
        f"   |  No ATI : {(y_true == 0).sum():,}"
        f"   |  ATI : {(y_true == 1).sum():,}"
    )
    print(sep)

    if len(np.unique(y_true)) > 1:
        print(classification_report(y_true, y_pred, target_names=["No ATI", "ATI"]))
        print(f"  AUC (ROC)  : {roc_auc_score(y_true, y_prob):.4f}")
        print(f"  Log Loss   : {log_loss(y_true, y_prob):.4f}")
    else:
        print("  (Only one class present — AUC not defined)")

    print(f"{sep}\n")



def main():
    args = _parse_args()

    if args.model_dir:
        weights_dir = Path(args.model_dir).expanduser().resolve()
        if not weights_dir.is_dir():
            alt = _SCRIPT_DIR / args.model_dir
            if alt.is_dir():
                weights_dir = alt
                logging.info(
                    f"  --model-dir '{args.model_dir}' resolved relative to "
                    f"script dir: {weights_dir}"
                )
            else:
                raise FileNotFoundError(
                    f"Model directory not found: {weights_dir}\n"
                    f"Also tried: {alt}"
                )
    else:
        weights_dir = _WEIGHTS_DIR

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else weights_dir / "predictions.csv"
    )

    sep = "─" * 58
    print(f"\n{sep}")
    print("  ATI Prediction Pipeline")
    print(f"{sep}")
    logging.info(f"Script dir   : {_SCRIPT_DIR}")
    logging.info(f"Weights dir  : {weights_dir}")
    logging.info(f"Input data   : {args.data}")
    logging.info(f"Output path  : {out_path}")
    logging.info(f"Threshold    : {args.threshold}")

    logging.info("\nLoading model artifacts...")
    try:
        model, prep, feature_cols, model_type = _load_artifacts(weights_dir)
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    logging.info(f"\nLoading data from: {args.data}")
    try:
        df = load_data(args.data)
    except FileNotFoundError:
        logging.error(f"Data file not found: {args.data}")
        sys.exit(1)

    try:
        results, has_labels = _run_inference(
            model, prep, df, model_type, args.threshold
        )
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        sys.exit(1)

    if has_labels:
        _evaluate(results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logging.info(f"Predictions saved : {out_path}")
    print(f"Predictions saved : {out_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()