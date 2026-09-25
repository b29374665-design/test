"""Command-line entry point: python -m eis_grader cell.csv [...] [--ref fresh.csv]"""
import argparse
import json
import os
import sys

import numpy as np

from . import grade, io, kk, model

KK_LIMIT = 0.02   # max Lin-KK residual (relative to |Z|) for trustworthy data
FIT_LIMIT = 0.02  # max RMS fit error


def analyse(spec, n_mc, seed):
    check = kk.lin_kk(spec)
    res = model.fit(spec)
    samples, sigma = model.monte_carlo(res, n=n_mc, seed=seed)
    mc = {
        "R0": samples[:, 1],
        "Rp": samples[:, 2] + samples[:, 5],
        "R1": samples[:, 2],
        "R2": samples[:, 5],
    }
    return {"spec": spec, "kk": check, "fit": res, "mc": mc, "sigma": sigma}


def _ci(a):
    lo, med, hi = np.percentile(a, [2.5, 50, 97.5])
    return float(lo), float(med), float(hi)


def _mohm(x):
    return f"{x * 1e3:.2f} mΩ"


def quality_notes(a):
    notes = []
    if a["kk"]["max_residual"] > KK_LIMIT:
        notes.append(f"Kramers-Kronig 殘差 {a['kk']['max_residual']:.1%} > {KK_LIMIT:.0%}：量測可能不穩定（漂移/非線性），判定僅供參考")
    if a["fit"].rms_error > FIT_LIMIT:
        notes.append(f"等效電路擬合誤差 {a['fit'].rms_error:.1%} 偏高：頻譜形狀與模型不符，判定僅供參考")
    if a["spec"].freq.min() > 1.0:
        notes.append("最低頻率 > 1 Hz：可能未量到完整的電荷轉移弧，Rp 可能被低估")
    return notes


def plot(a, path, verdict=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec, res = a["spec"], a["fit"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    f_dense = np.logspace(np.log10(spec.freq.max()), np.log10(spec.freq.min()), 300)
    z_fit = model.impedance(res.params, 2 * np.pi * f_dense)
    ax1.plot(spec.z.real * 1e3, -spec.z.imag * 1e3, "o", ms=4, mfc="none", label="measured")
    ax1.plot(z_fit.real * 1e3, -z_fit.imag * 1e3, "-", lw=1.5, label="fit")
    ax1.axvline(res.value("R0") * 1e3, ls=":", c="gray")
    ax1.axvline(res.value("Rtot") * 1e3, ls=":", c="gray")
    ax1.set_xlabel("Z' (mΩ)")
    ax1.set_ylabel("-Z'' (mΩ)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend()
    title = os.path.basename(spec.name)
    if verdict:
        title += f"  —  Grade {verdict['grade']}  (SOH_R {verdict['soh_r']:.0f}%)"
    ax1.set_title(title)

    ax2.hist(a["mc"]["R0"] * 1e3, bins=30, alpha=0.6, label="R0")
    ax2.hist(a["mc"]["Rp"] * 1e3, bins=30, alpha=0.6, label="Rp = R1 + R2")
    ax2.set_xlabel("resistance (mΩ)")
    ax2.set_ylabel("Monte Carlo count")
    ax2.set_title(f"Monte Carlo ({len(a['mc']['R0'])} refits)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eis_grader", description="由電池 EIS 數據自動判定電池優劣")
    ap.add_argument("files", nargs="+", help="EIS 檔案（CSV/TXT：頻率、Z'、Z''）")
    ap.add_argument("--ref", help="全新（基準）電池的 EIS 檔")
    ap.add_argument("--r0-new", type=float, help="全新電池 R0（Ω），例如規格書內阻")
    ap.add_argument("--rp-new", type=float, help="全新電池 Rp（Ω）")
    ap.add_argument("--mc", type=int, default=200, help="蒙地卡羅次數（預設 200）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eis_report", help="輸出資料夾（圖與 JSON）")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    results = []
    for path in args.files:
        print(f"分析 {path} ...", file=sys.stderr)
        results.append(analyse(io.load(path), args.mc, args.seed))

    baseline, mode = None, None
    if args.ref:
        ref = model.fit(io.load(args.ref))
        baseline = {"R0": ref.value("R0"), "Rp": ref.value("Rp")}
        mode = f"基準電池 {args.ref}"
    elif args.r0_new and args.rp_new:
        baseline = {"R0": args.r0_new, "Rp": args.rp_new}
        mode = "指定的全新電池數值"
    elif len(results) >= 3:
        baseline = {m: float(np.median([r["fit"].value(m) for r in results])) for m in grade.METRICS}
        mode = "本批電池中位數（相對排序，非絕對健康度）"

    report = []
    for a in results:
        res = a["fit"]
        point = {"R0": res.value("R0"), "Rp": res.value("Rp")}
        verdict = grade.evaluate(point, a["mc"], baseline) if baseline else None
        notes = quality_notes(a)
        name = os.path.basename(a["spec"].name)
        png = os.path.join(args.out, os.path.splitext(name)[0] + ".png")
        plot(a, png, verdict)

        print(f"\n=== {name} ===")
        print(f"數據品質：Lin-KK 最大殘差 {a['kk']['max_residual']:.2%}，擬合誤差 {res.rms_error:.2%}")
        for key, desc in (("R0", "歐姆內阻"), ("R1", "SEI/表面膜"), ("R2", "電荷轉移"), ("Rp", "極化電阻 R1+R2")):
            lo, _, hi = _ci(a["mc"][key])
            print(f"  {key:<3}{desc:<14} {_mohm(res.value(key)):>12}   95% 區間 [{_mohm(lo)}, {_mohm(hi)}]")
        if verdict:
            probs = "  ".join(f"{c}:{p:.0%}" for c, p in verdict["probabilities"].items())
            print(f"判定：{verdict['grade']}（{verdict['label']}）  可信度 {verdict['confidence']:.0%}")
            print(f"  內阻成長  R0 ×{verdict['metrics']['R0']['ratio']:.2f}   Rp ×{verdict['metrics']['Rp']['ratio']:.2f}"
                  f"   SOH_R ≈ {verdict['soh_r']:.0f}%")
            print(f"  蒙地卡羅等級機率  {probs}")
        else:
            print("判定：未提供基準（--ref 或 --r0-new/--rp-new，或一次給 3 顆以上電池），只輸出參數")
        for n in notes:
            print(f"  ⚠ {n}")
        print(f"  圖：{png}")

        report.append({
            "file": a["spec"].name,
            "fit": res.as_dict(),
            "kk": {k: v for k, v in a["kk"].items() if k != "residuals"},
            "monte_carlo": {k: dict(zip(("p2.5", "median", "p97.5"), _ci(v))) for k, v in a["mc"].items()},
            "mc_noise": a["sigma"],
            "verdict": verdict,
            "warnings": notes,
            "plot": png,
        })

    if baseline:
        print(f"\n基準：{mode}  R0={_mohm(baseline['R0'])}  Rp={_mohm(baseline['Rp'])}")
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump({"baseline": baseline, "baseline_source": mode, "cells": report}, fh,
                  ensure_ascii=False, indent=2, default=float)
    return 0
