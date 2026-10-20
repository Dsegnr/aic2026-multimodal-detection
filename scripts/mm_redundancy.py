"""红外到底有没有独立信息？先算一个最便宜的指标：与可见光的逐图相关度。

如果一份数据以白天场景为主，热红外与可见光灰度会高度相关（哪怕反相），
那么"融合没有收益"就有了数据层面的解释，而不是工程问题。

对验证集每张图算：
  · r_rgb_ir  : 可见光灰度 与 红外 的 Pearson 相关系数（取绝对值，因为热红外常反相）
  · 夜间/低照度判据：可见光灰度均值 < 60 视为暗场景，看其占比

用法：& $py scripts\\mm_redundancy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import argparse
import cv2
import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imgio import imread  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="splits/lbproxy_val.txt")
    ap.add_argument("--depth-sample", type=int, default=60)
    args = ap.parse_args()

    lst = Path(args.list)
    if not lst.is_absolute():
        lst = WS / lst
    images = [Path(ln.strip()) for ln in lst.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"清单 {lst.name}：{len(images)} 张")
    rgb_dir = WS / "data" / "aic" / "images"
    ir_dir = WS / "data" / "aic_ir" / "images"
    dp_dir = WS / "data" / "aic_dp" / "images"

    rs, lumas, dark = [], [], 0
    for p in images:
        bgr = imread(rgb_dir / p.name, cv2.IMREAD_COLOR)
        ir = imread(ir_dir / p.name, cv2.IMREAD_GRAYSCALE)
        if bgr is None or ir is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if ir.shape != gray.shape:
            ir = cv2.resize(ir, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        a = gray.astype(np.float64).ravel()
        b = ir.astype(np.float64).ravel()
        if a.std() < 1e-6 or b.std() < 1e-6:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        rs.append(abs(r))
        m = float(a.mean())
        lumas.append(m)
        if m < 60:
            dark += 1

    rs = np.array(rs)
    lumas = np.array(lumas)
    print(f"样本 {len(rs)} 张")
    print(f"|corr(可见光灰度, 红外)|  均值 {rs.mean():.3f}   中位数 {np.median(rs):.3f}   "
          f"最小 {rs.min():.3f}   最大 {rs.max():.3f}")
    for thr in (0.5, 0.7, 0.8, 0.9):
        print(f"  |r| > {thr:.1f} 的占比: {(rs > thr).mean() * 100:5.1f}%")
    print(f"\n可见光平均亮度: 均值 {lumas.mean():.1f}  中位数 {np.median(lumas):.1f}  "
          f"最暗 {lumas.min():.1f}  最亮 {lumas.max():.1f}")
    print(f"暗场景（<60）占比: {dark / max(len(lumas), 1) * 100:.1f}%")

    # 深度图有效像素占比（顺带看一眼深度质量）
    vals = []
    for p in images[: args.depth_sample]:
        d = imread(dp_dir / p.name, cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.float32)
        vals.append(float(((d >= 300) & (d <= 20000)).mean()))
    if vals:
        print(f"\n深度有效像素占比（前 60 张抽样）: 均值 {np.mean(vals) * 100:.1f}%  "
              f"最低 {np.min(vals) * 100:.1f}%  最高 {np.max(vals) * 100:.1f}%")


if __name__ == "__main__":
    main()
