"""对多组训练曲线做严格配对统计。

为什么需要这个：我们的验证集只有 300 张，单轮 mAP 波动可达 ±3 分。
直接比较"各自的最佳值"会被噪声主导。正确做法是**在同一轮次上逐一配对**，
再看差值的均值与显著性。
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]


def load(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["epoch"])] = float(row["metrics/mAP50-95(B)"])
            except (KeyError, ValueError):
                continue
    return out


def paired(a: dict[int, float], b: dict[int, float], name_a: str, name_b: str) -> None:
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        print(f"{name_a} vs {name_b}: 共同轮次不足（{len(common)}）")
        return
    diffs = [(a[e] - b[e]) * 100 for e in common]      # 换算成"分"
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else 0.0
    print(f"\n{name_a} − {name_b}   共同轮次 n={n}（epoch {common[0]}–{common[-1]}）")
    print(f"  平均差值 {mean:+.2f} 分   标准差 {sd:.2f}   标准误 {se:.2f}   t={t:+.2f}")
    print(f"  差值范围 {min(diffs):+.2f} ~ {max(diffs):+.2f}")
    verdict = "无显著差异（|t|<2）" if abs(t) < 2 else ("显著为正" if t > 0 else "显著为负")
    print(f"  粗判: {verdict}")


def main() -> None:
    files = {
        "RGB@960": WS / "runs" / "rgb_y8s_960" / "results.csv",
        "RGB+IR拼接@960": WS / "runs" / "mm4_960_results.csv",
        "DualStem融合@960": WS / "runs" / "fusion_960_results.csv",
        "RGB@1280": WS / "runs" / "rgb_1280_results.csv",
    }
    curves = {k: load(v) for k, v in files.items()}
    for k, v in curves.items():
        if v:
            best = max(v.values())
            print(f"{k:<20} 轮次 {len(v):>3}   最佳 {best*100:.2f} 分（epoch {max(v, key=v.get)}）")
        else:
            print(f"{k:<20} （无数据）")

    paired(curves["DualStem融合@960"], curves["RGB@960"], "融合@960", "RGB@960")
    paired(curves["DualStem融合@960"], curves["RGB+IR拼接@960"], "融合@960", "拼接@960")
    paired(curves["RGB+IR拼接@960"], curves["RGB@960"], "拼接@960", "RGB@960")


if __name__ == "__main__":
    main()
