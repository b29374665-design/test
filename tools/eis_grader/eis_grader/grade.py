"""Turn fitted resistances into a health grade.

Industry practice treats a cell as end-of-life when its internal resistance has
doubled relative to new (resistance-based SOH). Growth is therefore measured
against a baseline: a fresh reference cell, datasheet values, or (for batch
sorting) the median of the batch.
"""
import numpy as np

# Upper bound of resistance growth (R / R_baseline) for each grade.
DEFAULT_GRADES = [
    ("A", "良好", 1.25),
    ("B", "尚可", 1.50),
    ("C", "老化", 2.00),
    ("D", "劣化 / 建議汰換", float("inf")),
]
EOL_FACTOR = 2.0
METRICS = ("R0", "Rp")


def grade_of(ratio, grades=DEFAULT_GRADES):
    for code, _, upper in grades:
        if ratio <= upper:
            return code
    return grades[-1][0]


def label(code, grades=DEFAULT_GRADES):
    return next(lbl for c, lbl, _ in grades if c == code)


def soh_r(r, r_new, eol=EOL_FACTOR):
    """Resistance-based SOH in %: 100 at baseline, 0 at eol * baseline."""
    return float(np.clip((eol * r_new - r) / ((eol - 1) * r_new), 0, 1) * 100)


def evaluate(point, samples, baseline, grades=DEFAULT_GRADES):
    """point: {metric: value}; samples: {metric: array of MC values}.

    The overall grade of a cell is the worse of its R0 and Rp grades, so a cell
    with a healthy ohmic resistance but a ballooning charge-transfer arc (or vice
    versa) is still caught.
    """
    codes = [g[0] for g in grades]
    per_metric = {}
    for m in METRICS:
        ratio = point[m] / baseline[m]
        per_metric[m] = {
            "value": point[m],
            "baseline": baseline[m],
            "ratio": ratio,
            "grade": grade_of(ratio, grades),
            "soh": soh_r(point[m], baseline[m]),
        }
    worst = max(codes.index(v["grade"]) for v in per_metric.values())

    n = len(samples[METRICS[0]])
    mc_codes = []
    for i in range(n):
        idx = max(codes.index(grade_of(samples[m][i] / baseline[m], grades)) for m in METRICS)
        mc_codes.append(codes[idx])
    prob = {c: mc_codes.count(c) / n for c in codes}
    soh_total = soh_r(point["R0"] + point["Rp"], baseline["R0"] + baseline["Rp"])
    return {
        "grade": codes[worst],
        "label": label(codes[worst], grades),
        "confidence": prob[codes[worst]],
        "probabilities": prob,
        "soh_r": soh_total,
        "metrics": per_metric,
    }
