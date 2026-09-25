"""Train the random-forest model: python -m eis_grader.train labels.csv --out model.joblib"""
import argparse
import sys

import joblib

from . import ml


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eis_grader.train", description="用已標記的電池 EIS 訓練隨機森林")
    ap.add_argument("labels", help="標籤 CSV：file 欄位 + soh（數值）或 label（類別）欄位")
    ap.add_argument("--data", help="EIS 檔所在資料夾（預設為標籤檔所在資料夾）")
    ap.add_argument("--out", default="eis_model.joblib")
    ap.add_argument("--repeats", type=int, default=10, help="5-fold 交叉驗證重複次數")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    files, y, task = ml.load_labels(args.labels, args.data)
    print(f"{len(files)} 顆電池，任務：{'SOH 迴歸' if task == 'regression' else '分類'}")
    bundle = ml.train(files, y, task, seed=args.seed, repeats=args.repeats)

    print("\n交叉驗證（平均 ± 標準差）：")
    for name, desc in (("random_forest", "隨機森林"), ("total_resistance_only", "只用總內阻")):
        scores = "  ".join(f"{m} {v:.3f}±{s:.3f}" for m, (v, s) in bundle["cv"][name].items())
        print(f"  {desc:<10} {scores}")
    rf, simple = bundle["cv"]["random_forest"], bundle["cv"]["total_resistance_only"]
    key = "MAE" if task == "regression" else "balanced_accuracy"
    better = rf[key][0] < simple[key][0] if key == "MAE" else rf[key][0] > simple[key][0]
    print("  → 隨機森林" + ("優於" if better else "沒有優於") + "只用總內阻的簡單模型"
          + ("" if better else "，建議用簡單模型或補充數據"))

    print("\n特徵重要度（前 8）：")
    for n, v in bundle["importance"][:8]:
        print(f"  {n:<16} {v:.3f}")
    joblib.dump(bundle, args.out)
    print(f"\n模型已存到 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
