import os
import logging
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

_NSLOTS = int(os.environ.get("NSLOTS", 1))
_NJOBS  = max(1, _NSLOTS - 1)

CLINICAL_FEATURES = ["age", "sex", "baseline_egfr_23"]
CV_FOLDS          = 5
RANDOM_SEED       = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)


class _ATINet:

    def __init__(self, input_dim: int, hidden_dims: tuple, dropout_rate: float):
        import torch
        import torch.nn as nn

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.SELU(),
                nn.AlphaDropout(dropout_rate),   # AlphaDropout preserves SeLU stats
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def __call__(self, x):            return self.net(x)
    def train(self):                  self.net.train()
    def eval(self):                   self.net.eval()
    def parameters(self):             return self.net.parameters()
    def state_dict(self):             return self.net.state_dict()
    def load_state_dict(self, state): self.net.load_state_dict(state)


class NeuralNetClassifier(BaseEstimator, ClassifierMixin):

    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_dims    : tuple = (128, 64),
        dropout_rate   : float = 0.6,
        learning_rate  : float = 1e-4,
        weight_decay   : float = 5e-3,
        n_epochs       : int   = 500,
        batch_size     : int   = 32,
        patience       : int   = 50,
        val_fraction   : float = 0.15,
        label_smoothing: float = 0.05,
        random_state   : int   = RANDOM_SEED,
    ):
        self.hidden_dims     = hidden_dims
        self.dropout_rate    = dropout_rate
        self.learning_rate   = learning_rate
        self.weight_decay    = weight_decay
        self.n_epochs        = n_epochs
        self.batch_size      = batch_size
        self.patience        = patience
        self.val_fraction    = val_fraction
        self.label_smoothing = label_smoothing
        self.random_state    = random_state

        self._net     = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size    = self.val_fraction,
            stratify     = y,
            random_state = self.random_state,
        )

        X_tr_t  = torch.from_numpy(X_tr)
        y_tr_t  = torch.from_numpy(y_tr).unsqueeze(1)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val).unsqueeze(1)

        self._net = _ATINet(X_tr_t.shape[1], self.hidden_dims, self.dropout_rate)

        optimiser = torch.optim.AdamW(         # AdamW: better weight decay
            self._net.parameters(),
            lr           = self.learning_rate,
            weight_decay = self.weight_decay,
        )

        def smooth_bce(logits, targets, eps=self.label_smoothing):
            targets_s = targets * (1.0 - eps) + 0.5 * eps
            return nn.functional.binary_cross_entropy_with_logits(logits, targets_s)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max   = self.n_epochs,
            eta_min = self.learning_rate * 0.01,
        )

        best_val_loss  = float("inf")
        patience_count = 0
        best_state     = None
        n_tr           = X_tr_t.shape[0]

        for epoch in range(self.n_epochs):
            self._net.train()
            perm = torch.randperm(n_tr)
            X_ep = X_tr_t[perm]
            y_ep = y_tr_t[perm]

            for start in range(0, n_tr - self.batch_size + 1, self.batch_size):
                X_b = X_ep[start : start + self.batch_size]
                y_b = y_ep[start : start + self.batch_size]
                optimiser.zero_grad()
                loss = smooth_bce(self._net(X_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimiser.step()

            scheduler.step()

            self._net.eval()
            with torch.no_grad():
                val_loss = smooth_bce(self._net(X_val_t), y_val_t).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss  = val_loss
                patience_count = 0
                best_state     = {k: v.clone() for k, v in self._net.state_dict().items()}
            else:
                patience_count += 1

            if patience_count >= self.patience:
                logging.info(
                    f"  NeuralNet early stop epoch {epoch+1:4d}  "
                    f"val_loss={best_val_loss:.4f}"
                )
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        if self._net is None:
            raise RuntimeError("Call fit() before predict_proba()")
        X = np.asarray(X, dtype=np.float32)
        self._net.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self._net(torch.from_numpy(X))).numpy().flatten()
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)




class _SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, named_estimators: list, weights: list = None):
        self.named_estimators = named_estimators
        self.weights          = weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SoftVotingEnsemble":
        self.classes_        = np.array([0, 1])
        self._fitted_estims_ = []

        for name, clf in self.named_estimators:
            logging.info(f"    Ensemble: fitting [{name}]...")
            clf.fit(X, y)
            self._fitted_estims_.append((name, clf))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_fitted_estims_"):
            raise RuntimeError("Call fit() before predict_proba()")

        all_probs = [clf.predict_proba(X) for _, clf in self._fitted_estims_]

        if self.weights is not None:
            w   = np.array(self.weights, dtype=float)
            w  /= w.sum()                         # normalise to sum=1
            out = sum(p * wi for p, wi in zip(all_probs, w))
        else:
            out = np.mean(all_probs, axis=0)

        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)



def _build_neuralnet(params: dict) -> NeuralNetClassifier:
    defaults = dict(
        hidden_dims     = (128, 64),
        dropout_rate    = 0.6,
        learning_rate   = 1e-4,
        weight_decay    = 5e-3,
        n_epochs        = 500,
        batch_size      = 32,
        patience        = 50,
        val_fraction    = 0.15,
        label_smoothing = 0.05,
        random_state    = RANDOM_SEED,
    )
    defaults.update(params)
    return NeuralNetClassifier(**defaults)


def _build_mlp(params: dict):

    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    defaults = dict(
        hidden_layer_sizes  = (128, 64),
        activation          = "relu",
        solver              = "adam",
        alpha               = 1e-2,
        batch_size          = 32,
        learning_rate_init  = 1e-4,
        max_iter            = 500,
        early_stopping      = True,
        validation_fraction = 0.15,
        n_iter_no_change    = 50,
        random_state        = RANDOM_SEED,
    )
    defaults.update(params)
    return Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(**defaults))])


def _build_lgbm(params: dict):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("Install with: pip install lightgbm")

    defaults = dict(
        n_estimators      = 500,
        max_depth         = 5,
        learning_rate     = 0.03,       # low lr → more trees, less overfit
        num_leaves        = 20,         # << 2^max_depth to regularise
        min_child_samples = 15,         # min samples per leaf — key regulariser
        subsample         = 0.8,        # row subsampling
        subsample_freq    = 1,
        colsample_bytree  = 0.5,        # feature subsampling
        reg_alpha         = 0.1,        # L1
        reg_lambda        = 1.0,        # L2
        class_weight      = "balanced",
        random_state      = RANDOM_SEED,
        n_jobs            = _NJOBS,
        verbose           = -1,
    )
    defaults.update(params)
    return lgb.LGBMClassifier(**defaults)


def _build_xgboost(params: dict):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("Install with: pip install xgboost")

    defaults = dict(
        n_estimators     = 500,
        max_depth        = 4,
        learning_rate    = 0.03,
        subsample        = 0.8,
        colsample_bytree = 0.5,
        min_child_weight = 5,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        eval_metric      = "logloss",
        random_state     = RANDOM_SEED,
        n_jobs           = _NJOBS,
    )
    defaults.update(params)
    return XGBClassifier(**defaults)


def _build_lasso_lr(params: dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    defaults = dict(
        C            = 0.05,           # strong regularisation
        penalty      = "elasticnet",
        solver       = "saga",
        l1_ratio     = 0.5,
        max_iter     = 3000,
        class_weight = "balanced",
        random_state = RANDOM_SEED,
        n_jobs       = _NJOBS,
    )
    defaults.update(params)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**defaults)),
    ])


def _build_catboost(params: dict):
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError("Install with: pip install catboost")

    defaults = dict(
        iterations    = 500,
        depth         = 5,
        learning_rate = 0.03,
        l2_leaf_reg   = 3.0,
        random_seed   = RANDOM_SEED,
        verbose       = 0,
        auto_class_weights = "Balanced",
    )
    defaults.update(params)
    return CatBoostClassifier(**defaults)


def _build_ensemble(params: dict):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("Install with: pip install lightgbm")

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # pop() so weights don't get passed to sub-estimators
    weights = params.pop("weights", [8, 2, 1])

    lgbm_clf = lgb.LGBMClassifier(
        n_estimators      = 500,
        max_depth         = 5,
        learning_rate     = 0.03,
        num_leaves        = 20,
        min_child_samples = 15,
        subsample         = 0.8,
        subsample_freq    = 1,
        colsample_bytree  = 0.5,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        class_weight      = "balanced",
        random_state      = RANDOM_SEED,
        n_jobs            = _NJOBS,
        verbose           = -1,
    )

    lr_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C            = 0.05,
            penalty      = "elasticnet",
            solver       = "saga",
            l1_ratio     = 0.5,
            max_iter     = 3000,
            class_weight = "balanced",
            random_state = RANDOM_SEED,
        )),
    ])

    nn_clf = NeuralNetClassifier(
        hidden_dims     = (128, 64),
        dropout_rate    = 0.6,
        learning_rate   = 1e-4,
        weight_decay    = 5e-3,
        n_epochs        = 500,
        batch_size      = 32,
        patience        = 50,
        val_fraction    = 0.15,
        label_smoothing = 0.05,
        random_state    = RANDOM_SEED,
    )

    return _SoftVotingEnsemble(
        named_estimators = [("lgbm", lgbm_clf), ("lr", lr_clf), ("nn", nn_clf)],
        weights          = weights,
    )

_BUILDERS = {
    "NeuralNet" : _build_neuralnet,
    "MLP"       : _build_mlp,
    "LightGBM"  : _build_lgbm,       
    "XGBoost"   : _build_xgboost,
    "CatBoost"  : _build_catboost,
    "Lasso LR"  : _build_lasso_lr,
    "Ensemble"  : _build_ensemble,
}


def build_model(model_name: str = "LightGBM", params: dict = None):
    """
    Return a fresh (unfitted) sklearn-compatible classifier.

    Parameters
    ----------
    model_name : str   One of: NeuralNet, MLP, LightGBM, XGBoost,
                               CatBoost, Lasso LR, Ensemble
    params     : dict  Override any model defaults.

    Examples
    --------
    >>> clf = build_model("LightGBM")
    >>> clf = build_model("LightGBM", {"num_leaves": 31, "n_estimators": 300})
    >>> clf = build_model("Ensemble", {"weights": [4, 2, 1]})
    """
    if model_name not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(_BUILDERS)}"
        )
    return _BUILDERS[model_name](params or {})


#  Smoke test

if __name__ == "__main__":
    print(f"Available models  : {list(_BUILDERS)}")
    print(f"Clinical features : {CLINICAL_FEATURES}")
    print(f"CV folds          : {CV_FOLDS}")
    print(f"Random seed       : {RANDOM_SEED}")
    print(f"n_jobs            : {_NJOBS}  (NSLOTS={_NSLOTS})")
    print()

    rng     = np.random.default_rng(0)
    X_dummy = rng.standard_normal((80, 20)).astype(np.float32)
    y_dummy = rng.integers(0, 2, 80)

    skip = {"CatBoost", "XGBoost"}   # skip if not installed
    for name in _BUILDERS:
        if name in skip:
            continue
        try:
            m = build_model(name)
            m.fit(X_dummy, y_dummy)
            proba = m.predict_proba(X_dummy)
            assert proba.shape == (80, 2), f"Bad shape: {proba.shape}"
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
            print(f"  [{name}] passed")
        except ImportError as e:
            print(f"  [{name}] skipped — {e}")

    print("\nAll checks passed.")