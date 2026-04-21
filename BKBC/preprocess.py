import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.preprocessing import PolynomialFeatures, RobustScaler

from model import CLINICAL_FEATURES

_VARIANCE_THRESHOLD = 1e-4
_N_SVD_COMPONENTS   = 150  
_N_TOP_FEATURES     = 300  
_WINSOR_LOWER       = 1.0
_WINSOR_UPPER       = 99.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)


def load_data(data_path: str) -> pd.DataFrame:
    """Load a CSV and return a DataFrame."""
    logging.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False, na_values=[".", ""])
    logging.info(f"  {len(df)} samples, {len(df.columns)} columns")
    return df



class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_protein_components: int   = _N_SVD_COMPONENTS,
        n_top_features:       int   = _N_TOP_FEATURES,
        variance_threshold:   float = _VARIANCE_THRESHOLD,
        winsor_lower:         float = _WINSOR_LOWER,
        winsor_upper:         float = _WINSOR_UPPER,
        random_state:         int   = 42,
    ):
        self.n_protein_components = n_protein_components
        self.n_top_features       = n_top_features
        self.variance_threshold   = variance_threshold
        self.winsor_lower         = winsor_lower
        self.winsor_upper         = winsor_upper
        self.random_state         = random_state


    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "Preprocessor":
        self._protein_cols_raw = sorted(
            c for c in df.columns if c.startswith("feature_")
        )
        self._clinical_cols = [c for c in CLINICAL_FEATURES if c in df.columns]

        missing_clin = [c for c in CLINICAL_FEATURES if c not in df.columns]
        if missing_clin:
            logging.warning(f"  Missing clinical columns: {missing_clin}")

        # compute imputation medians on train
        self._protein_medians = df[self._protein_cols_raw].median()
        self._clinical_medians = (
            df[self._clinical_cols].median()
            if self._clinical_cols
            else pd.Series(dtype=float)
        )

        prot_mat, miss_feat, df_w = self._impute_and_flags(df)

        # winsorisation bounds
        self._winsor_lo = np.percentile(prot_mat, self.winsor_lower, axis=0)
        self._winsor_hi = np.percentile(prot_mat, self.winsor_upper, axis=0)
        prot_mat = np.clip(prot_mat, self._winsor_lo, self._winsor_hi)
        logging.info(
            f"  Winsorisation     : clipped proteins to "
            f"{self.winsor_lower}%–{self.winsor_upper}%"
        )

        # variance filter
        self._var_sel = VarianceThreshold(self.variance_threshold)
        self._var_sel.fit(prot_mat)
        keep_mask = self._var_sel.get_support()
        self._protein_cols = [
            c for c, k in zip(self._protein_cols_raw, keep_mask) if k
        ]
        n_rem = len(self._protein_cols_raw) - len(self._protein_cols)
        prot_mat_filt = prot_mat[:, keep_mask]
        logging.info(
            f"  Variance filter   : removed {n_rem:,} → "
            f"{len(self._protein_cols):,} remain"
        )

        # RobustScaler
        self._scaler = RobustScaler()
        scaled = self._scaler.fit_transform(prot_mat_filt)

        # TruncatedSVD 
        n_comp = min(
            self.n_protein_components,
            scaled.shape[1],
            scaled.shape[0] - 1,
        )
        self._svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        svd_mat = self._svd.fit_transform(scaled)
        ev = self._svd.explained_variance_ratio_.sum()
        logging.info(
            f"  TruncatedSVD      : {n_comp} components "
            f"({ev:.1%} protein variance explained)"
        )

        # clinical features (fit polynomial)
        self._setup_poly(df_w)
        clin_mat, clin_names = self._build_clinical_mat(df_w)

        # assemble + SelectKBest
        X_full, all_names = self._assemble(svd_mat, miss_feat, clin_mat, clin_names)
        n_sel = min(self.n_top_features, X_full.shape[1])
        self._selector = SelectKBest(mutual_info_classif, k=n_sel)
        self._selector.fit(X_full, y)
        sel_mask = self._selector.get_support()
        self._feature_names = [n for n, m in zip(all_names, sel_mask) if m]
        logging.info(
            f"  Feature selection : {n_sel} / {X_full.shape[1]} features kept"
        )

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        prot_mat, miss_feat, df_w = self._impute_and_flags(df)
        prot_mat  = np.clip(prot_mat, self._winsor_lo, self._winsor_hi)
        prot_filt = prot_mat[:, self._var_sel.get_support()]
        scaled    = self._scaler.transform(prot_filt)
        svd_mat   = self._svd.transform(scaled)

        clin_mat, clin_names = self._build_clinical_mat(df_w)
        X_full, _ = self._assemble(svd_mat, miss_feat, clin_mat, clin_names)
        return self._selector.transform(X_full)

    @property
    def feature_names_out_(self) -> list:
        return getattr(self, "_feature_names", [])

    # private helpers

    def _impute_and_flags(self, df: pd.DataFrame):
        """
        Returns (protein_matrix, missingness_features, df_imputed).

        Missingness flags are captured BEFORE imputation so they remain
        informative.  Clinical flags are attached as __flag_<col> columns
        so _build_clinical_mat() can access them without extra state.
        """
        df = df.copy()

        # protein missingness (before imputing)
        prot_raw    = df[self._protein_cols_raw]
        miss_count  = prot_raw.isna().sum(axis=1).values.astype(float)
        miss_frac   = miss_count / max(len(self._protein_cols_raw), 1)
        miss_feat   = np.column_stack([miss_count, miss_frac])

        # clinical missingness flags (before imputing)
        # Build as a dict first, then concat once to avoid PerformanceWarning
        flag_dict = {
            f"__flag_{c}": df[c].isna().astype(float)
            for c in self._clinical_cols
        }
        if flag_dict:
            flags_df = pd.DataFrame(flag_dict, index=df.index)
            df = pd.concat([df, flags_df], axis=1)

        # impute proteins
        prot_imp = prot_raw.copy()
        for c in self._protein_cols_raw:
            prot_imp[c] = prot_imp[c].fillna(self._protein_medians[c])
        prot_mat = prot_imp.values.astype(float)

        # impute clinical
        for c in self._clinical_cols:
            df[c] = df[c].fillna(self._clinical_medians[c])

        return prot_mat, miss_feat, df

    def _setup_poly(self, df_w: pd.DataFrame) -> None:
        """Fit PolynomialFeatures on numeric clinical columns."""
        numeric_clin = [
            c for c in self._clinical_cols
            if pd.api.types.is_numeric_dtype(df_w[c].dtype)
        ]
        self._poly_numeric_clin = numeric_clin
        self._poly              = None
        self._poly_keep_idx     = []
        self._poly_names        = []

        if len(numeric_clin) >= 2:
            self._poly = PolynomialFeatures(
                degree=2, interaction_only=False, include_bias=False
            )
            poly_mat  = self._poly.fit_transform(
                df_w[numeric_clin].values.astype(float)
            )
            raw_names = list(self._poly.get_feature_names_out(numeric_clin))
            self._poly_keep_idx = [
                i for i, n in enumerate(raw_names) if n not in numeric_clin
            ]
            self._poly_names = [
                f"cpoly_{raw_names[i].replace(' ', '_').replace('^', '')}"
                for i in self._poly_keep_idx
            ]
            logging.info(
                f"  Clinical poly     : {len(self._poly_names)} polynomial features"
            )

    def _build_clinical_mat(self, df_w: pd.DataFrame):
        """Return (ndarray, names) for all clinical-derived features."""
        cols  = self._clinical_cols
        parts, names = [], []

        # raw clinical values
        for c in cols:
            parts.append(df_w[c].values.astype(float))
            names.append(c)

        # age bins
        if "age" in cols:
            parts.append(np.digitize(df_w["age"].values, [40, 60]).astype(float))
            names.append("age_bin")

        # eGFR-derived
        if "baseline_egfr_23" in cols:
            egfr     = df_w["baseline_egfr_23"].values.clip(0)
            log_egfr = np.log1p(egfr)
            ckd      = np.digitize(egfr, [15, 30, 60]).astype(float)  # G5→G1
            parts   += [log_egfr, ckd]
            names   += ["log_egfr", "egfr_ckd_stage"]

        # interactions
        if "age" in cols and "baseline_egfr_23" in cols:
            age      = df_w["age"].values.astype(float)
            egfr     = df_w["baseline_egfr_23"].values.astype(float)
            log_egfr = np.log1p(egfr.clip(0))
            parts   += [age * egfr, age * log_egfr]
            names   += ["age_x_egfr", "age_x_log_egfr"]

        # imputation flags
        for c in cols:
            fc = f"__flag_{c}"
            if fc in df_w.columns:
                parts.append(df_w[fc].values.astype(float))
                names.append(f"{c}_was_imputed")

        # polynomial clinical
        if self._poly is not None:
            pm = self._poly.transform(
                df_w[self._poly_numeric_clin].values.astype(float)
            )
            if self._poly_keep_idx:
                parts.append(pm[:, self._poly_keep_idx])
                names += self._poly_names

        if not parts:
            return np.zeros((len(df_w), 0)), []
        return np.column_stack(parts), names

    def _assemble(
        self,
        svd_mat:    np.ndarray,
        miss_feat:  np.ndarray,
        clin_mat:   np.ndarray,
        clin_names: list,
    ):
        """Combine all feature blocks into one matrix + name list."""
        parts, names = [], []

        # SVD components
        parts.append(svd_mat)
        names += [f"svd_{i + 1}" for i in range(svd_mat.shape[1])]

        # per-sample SVD statistics
        p25 = np.percentile(svd_mat, 25, axis=1)
        p75 = np.percentile(svd_mat, 75, axis=1)
        svd_stats = np.column_stack([
            svd_mat.mean(axis=1),
            svd_mat.std(axis=1),
            np.median(svd_mat, axis=1),
            p25, p75, p75 - p25,
            svd_mat.max(axis=1),
            svd_mat.min(axis=1),
        ])
        parts.append(svd_stats)
        names += [
            "svd_mean", "svd_std", "svd_median",
            "svd_p25",  "svd_p75", "svd_iqr",
            "svd_max",  "svd_min",
        ]

        # protein missingness
        parts.append(miss_feat)
        names += ["missing_protein_count", "missing_protein_frac"]

        # clinical block 
        if clin_mat.shape[1] > 0:
            parts.append(clin_mat)
            names += clin_names

        return np.column_stack(parts), names



def build_features_and_labels(df: pd.DataFrame):
    """
    Legacy one-shot wrapper: fit Preprocessor on ALL data → (X, y, names, prep).

    ⚠  For proper CV, fit Preprocessor INSIDE each fold (see train.py).
    """
    if "ati" not in df.columns:
        raise KeyError("Column 'ati' not found in dataframe")

    n_before = len(df)
    df = df.dropna(subset=["ati"]).reset_index(drop=True)
    if len(df) < n_before:
        logging.warning(
            f"Dropped {n_before - len(df)} rows with missing 'ati' label"
        )

    y    = df["ati"].astype(int).values
    prep = Preprocessor()
    X    = prep.fit_transform(df, y)

    logging.info(
        f"\nFinal dataset  : {len(df)} samples | "
        f"No ATI={(y == 0).sum()} | ATI={(y == 1).sum()}"
    )
    logging.info(f"  Total features : {X.shape[1]}")

    return X, y, prep.feature_names_out_, prep