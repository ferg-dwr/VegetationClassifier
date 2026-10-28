"""
vegclassifier/models.py
=======================
The four classifiers from 03_B behind one interface, plus the persistence
layer 03_B lacked. Every model exposes:

    .fit(X, y)
    .predict(X)        -> (N,) int labels in scope-label space
    .predict_proba(X)  -> (N, n_classes) columns aligned to .classes_

`save_model` / `load_model` write a single joblib bundle carrying the fitted
model AND the metadata needed to reproduce inference: scope, class list,
feature names, normalization, and feature params. Predict/eval read that
bundle, so they never depend on notebook globals.

Faithful ports of 03_B cells 18-25 (GM, GMM, RF, SVM).
"""


import joblib
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_LOG_FLOOR = 1e-30


class _BaseModel:
    kind = "base"

    def __init__(self, classes: list[int]):
        self.classes_ = list(classes)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.array(self.classes_)[np.argmax(proba, axis=1)].astype(np.int32)

    # subclasses implement predict_proba, _state, _load_state


class GaussianModel(_BaseModel):
    """Per-class multivariate Gaussian (mean, cov, prior). 03_B cells 18/22."""
    kind = "gm"

    def fit(self, X, y):
        self.params_ = {}
        total = len(X)
        for cls in self.classes_:
            data = X[y == cls]
            if len(data) == 0:
                continue
            self.params_[cls] = {
                "mean":  data.mean(axis=0),
                "cov":   np.cov(data.T),
                "prior": len(data) / total,
            }
        return self

    def _logp(self, X):
        log_p = np.full((len(X), len(self.classes_)), -np.inf)
        for ci, cls in enumerate(self.classes_):
            if cls not in self.params_:
                continue
            p  = self.params_[cls]
            rv = multivariate_normal(mean=p["mean"], cov=p["cov"], allow_singular=True)
            log_p[:, ci] = rv.logpdf(X) + np.log(p["prior"] + _LOG_FLOOR)
        return log_p

    def predict_proba(self, X):
        log_p = self._logp(X)
        log_p = log_p - log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)

    def _state(self):
        return self.params_

    def _load_state(self, state):
        self.params_ = state
        return self


class GMMModel(_BaseModel):
    """Per-class Gaussian Mixture; BG gets bg_components, others 1. Cells 19/23."""
    kind = "gmm"

    def __init__(self, classes, bg_components=2, seed=42):
        super().__init__(classes)
        self.bg_components = bg_components
        self.seed = seed

    def fit(self, X, y):
        self.params_ = {}
        total = len(X)
        for cls in self.classes_:
            data = X[y == cls]
            if len(data) == 0:
                continue
            n_comp = self.bg_components if cls == 0 else 1
            gmm = GaussianMixture(n_components=n_comp, covariance_type="full",
                                  random_state=self.seed)
            gmm.fit(data)
            self.params_[cls] = {"gmm": gmm, "prior": len(data) / total}
        return self

    def predict_proba(self, X):
        log_p = np.full((len(X), len(self.classes_)), -np.inf)
        for ci, cls in enumerate(self.classes_):
            if cls not in self.params_:
                continue
            m = self.params_[cls]
            log_p[:, ci] = m["gmm"].score_samples(X) + np.log(m["prior"] + _LOG_FLOOR)
        log_p = log_p - log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)

    def _state(self):
        return self.params_

    def _load_state(self, state):
        self.params_ = state
        return self


class _SklearnModel(_BaseModel):
    """Shared predict_proba with explicit column alignment to scope classes."""

    def predict_proba(self, X):
        raw = self.est_.predict_proba(X)            # columns in est_.classes_ order
        out = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
        col = {int(c): j for j, c in enumerate(self.est_.classes_)}
        for ci, cls in enumerate(self.classes_):
            if cls in col:
                out[:, ci] = raw[:, col[cls]]
        return out

    def _state(self):
        return self.est_

    def _load_state(self, state):
        self.est_ = state
        return self


class RFModel(_SklearnModel):
    """RandomForest. 03_B cells 20/24."""
    kind = "rf"

    def __init__(self, classes, n_estimators=300, max_depth=None,
                 min_samples_leaf=5, max_features="sqrt",
                 class_weight="balanced", seed=42):
        super().__init__(classes)
        self.est_ = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            class_weight=class_weight, n_jobs=-1, random_state=seed,
        )

    def fit(self, X, y):
        self.est_.fit(X, y)
        return self

    @property
    def feature_importances_(self):
        return self.est_.feature_importances_


class SVMModel(_SklearnModel):
    """RBF-SVM with StandardScaler; subsamples large training sets. Cells 21/25."""
    kind = "svm"

    def __init__(self, classes, C=1.0, gamma="scale", max_train=40000,
                 class_weight="balanced", seed=42):
        super().__init__(classes)
        self.max_train = max_train
        self.seed = seed
        self.est_ = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=C, gamma=gamma, probability=True,
                        class_weight=class_weight, random_state=seed)),
        ])

    def fit(self, X, y):
        if len(X) > self.max_train:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(X), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        self.est_.fit(X, y)
        return self


_REGISTRY = {"gm": GaussianModel, "gmm": GMMModel, "rf": RFModel, "svm": SVMModel}


def make_model(model_cfg, scope_classes: list[int], seed: int) -> _BaseModel:
    """Build an unfitted model wrapper from a config.Model and scope classes."""
    algo = model_cfg.algo
    if algo == "gm":
        return GaussianModel(scope_classes)
    if algo == "gmm":
        return GMMModel(scope_classes, bg_components=model_cfg.gmm_bg_components, seed=seed)
    if algo == "rf":
        return RFModel(scope_classes, n_estimators=model_cfg.rf_n_estimators,
                       max_depth=model_cfg.rf_max_depth,
                       min_samples_leaf=model_cfg.rf_min_samples_leaf,
                       max_features=model_cfg.rf_max_features,
                       class_weight=model_cfg.rf_class_weight, seed=seed)
    if algo == "svm":
        return SVMModel(scope_classes, C=model_cfg.svm_C, gamma=model_cfg.svm_gamma,
                        max_train=model_cfg.svm_max_train,
                        class_weight=model_cfg.svm_class_weight, seed=seed)
    raise ValueError(f"unknown algo '{algo}'")


def save_model(model: _BaseModel, path, *, scope, feat_names, norm, feature_params):
    """Persist a fitted model plus the metadata predict/eval need."""
    bundle = {
        "kind":           model.kind,
        "classes":        model.classes_,
        "scope":          scope,
        "feat_names":     feat_names,
        "norm":           norm,             # {"mode","lo","hi"}
        "feature_params": feature_params,   # dict for build_feature_stack
        "state":          model._state(),
    }
    joblib.dump(bundle, path)
    return path


def load_model(path):
    """Load a bundle -> (model, metadata_dict)."""
    bundle = joblib.load(path)
    wrapper = _REGISTRY[bundle["kind"]].__new__(_REGISTRY[bundle["kind"]])
    _BaseModel.__init__(wrapper, bundle["classes"])
    wrapper._load_state(bundle["state"])
    meta = {k: bundle[k] for k in ("kind", "scope", "feat_names", "norm", "feature_params", "classes")}
    return wrapper, meta
