"""Random-forest SOH / good-bad model on physics features from the circuit fit.

Features are derived from fitted circuit parameters rather than raw spectra, so
the model needs far fewer labelled cells, and every Monte Carlo refit can be
pushed through the forest to get a prediction distribution.
"""
import os

import numpy as np

from . import io, model

SPOT_FREQS = (1000.0, 100.0, 10.0, 1.0, 0.1)
FEATURE_NAMES = (
    ["log_R0", "log_R1", "log_R2", "log_Rp", "log_Rtot", "a1", "a2", "log_Aw", "log_tau1", "log_tau2"]
    + [f"log|Z|@{f:g}Hz" for f in SPOT_FREQS]
    + [f"phase@{f:g}Hz" for f in SPOT_FREQS]
)
BASELINE_FEATURE = "log_Rtot"


def features(params, f_range):
    """Feature vector from one parameter set. Spot frequencies outside the
    measured range are NaN (the forest handles missing values)."""
    L, R0, R1, Q1, a1, R2, Q2, a2, Aw = params
    tau1 = (R1 * Q1) ** (1 / a1)
    tau2 = (R2 * Q2) ** (1 / a2)
    fmin, fmax = f_range
    spot = np.array(SPOT_FREQS)
    inside = (spot >= fmin * 0.99) & (spot <= fmax * 1.01)
    z = model.impedance(params, 2 * np.pi * spot)
    mag = np.where(inside, np.log(np.abs(z)), np.nan)
    phase = np.where(inside, np.degrees(np.angle(z)), np.nan)
    base = [np.log(R0), np.log(R1), np.log(R2), np.log(R1 + R2), np.log(R0 + R1 + R2),
            a1, a2, np.log(Aw), np.log(tau1), np.log(tau2)]
    return np.concatenate([base, mag, phase])


def spectrum_features(spec):
    res = model.fit(spec)
    return features(res.params, (spec.freq.min(), spec.freq.max())), res


def load_labels(path, data_dir=None):
    """CSV with a `file` column and either `soh` (numeric) or `label` (text)."""
    import csv

    data_dir = data_dir or os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8-sig") as fh:
        rows = [{k.strip().lower(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]
    if not rows or "file" not in rows[0]:
        raise ValueError("標籤檔需要 file 欄位，以及 soh 或 label 欄位")
    target = "soh" if "soh" in rows[0] else "label" if "label" in rows[0] else None
    if target is None:
        raise ValueError("標籤檔需要 soh（數值）或 label（類別）欄位")
    files = [r["file"] if os.path.isabs(r["file"]) else os.path.join(data_dir, r["file"]) for r in rows]
    y = [r[target] for r in rows]
    if target == "soh":
        return files, np.array([float(v) for v in y]), "regression"
    return files, np.array(y), "classification"


def _models(task, seed):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    if task == "regression":
        rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed)
        simple = LinearRegression()
    else:
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                    random_state=seed)
        simple = LogisticRegression(class_weight="balanced")
    return rf, simple


def cross_validate(X, y, task, repeats=10, seed=0):
    """Repeated k-fold CV of the forest vs. a one-feature (total resistance) model."""
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_validate as cv

    if task == "regression":
        splitter = RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
        scoring = {"MAE": "neg_mean_absolute_error", "R2": "r2"}
    else:
        k = int(min(5, np.unique(y, return_counts=True)[1].min()))
        if k < 2:
            raise ValueError("每個類別至少需要 2 顆電池才能做交叉驗證")
        splitter = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed)
        scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"}

    rf, simple = _models(task, seed)
    col = FEATURE_NAMES.index(BASELINE_FEATURE)
    out = {}
    for name, est, data in (("random_forest", rf, X), ("total_resistance_only", simple, X[:, [col]])):
        s = cv(est, data, y, cv=splitter, scoring=scoring)
        out[name] = {m: (float(abs(s[f"test_{m}"]).mean()) if m == "MAE" else float(s[f"test_{m}"].mean()),
                         float(s[f"test_{m}"].std()))
                     for m in scoring}
    return out


def train(files, y, task, seed=0, repeats=10, log=print):
    X = []
    for f in files:
        log(f"  特徵擷取 {os.path.basename(f)}")
        X.append(spectrum_features(io.load(f))[0])
    X = np.array(X)
    cv = cross_validate(X, y, task, repeats=repeats, seed=seed)
    rf, _ = _models(task, seed)
    rf.fit(X, y)
    importance = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda t: -t[1])
    return {"model": rf, "task": task, "features": list(FEATURE_NAMES), "cv": cv,
            "importance": [(n, float(v)) for n, v in importance], "n_train": len(y)}


def predict_mc(bundle, mc_params, f_range):
    """Push every Monte Carlo refit through the forest."""
    X = np.array([features(p, f_range) for p in mc_params])
    rf = bundle["model"]
    if bundle["task"] == "regression":
        pred = rf.predict(X)
        lo, med, hi = np.percentile(pred, [2.5, 50, 97.5])
        return {"soh": float(med), "soh_95": [float(lo), float(hi)],
                "cv_mae": bundle["cv"]["random_forest"]["MAE"][0]}
    proba = rf.predict_proba(X).mean(axis=0)
    classes = [str(c) for c in rf.classes_]
    best = int(np.argmax(proba))
    return {"label": classes[best], "confidence": float(proba[best]),
            "probabilities": dict(zip(classes, map(float, proba))),
            "cv_balanced_accuracy": bundle["cv"]["random_forest"]["balanced_accuracy"][0]}
