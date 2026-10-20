"""逐类 AP 探针 + 分辨率子集拆分（回答 Astra 第六次的 Q3）。

功能：
  1. 逐类输出 AP@0.50 / 0.75 / 0.90 / 0.95 与 Recall@IoU0.50（贪心匹配，maxDets=100）。
  2. 按**源文件分辨率类型**把验证集拆成两部分分别评测：
     - `jpg`：官方原图就是 640x360 的那 149 张（val300 里占 31 张）
     - `png`：官方原图是 1920x1080 的那 1851 张（val300 里占 269 张）
     动机：val300 是**混合分辨率**的，10.3% 的图只有 1/9 的像素。
     若这部分图的 AP 显著更低，那么所有"推理尺度/训练分辨率"的结论都被它污染了。

评测逻辑与 `src/eval/map_eval.py` 完全一致（同类贪心匹配、全局按置信度排序、101 点插值），
只是把 IoU 阈值与图像集合变成可配置的，并额外算 recall。

用法：
    python src/eval/perclass_probe.py --gt-dir data/valgt_lbproxy \
        --pred-dir submissions/_dfine_ep56_tta1280 --split all
    python src/eval/perclass_probe.py ... --split jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from map_eval import (  # noqa: E402
    DEFAULT_NAMES,
    IOU_THRS,
    ap_from_flags,
    iou_matrix,
    read_gt_file,
    read_pred_file,
)


def img_kind(stem: str) -> str:
    """官方原图是 .png(1920x1080) 还是 .jpg(640x360)。"""
    d = WS / "data" / "aic" / "images"
    if (d / f"{stem}.png").exists():
        return "png"
    if (d / f"{stem}.jpg").exists():
        return "jpg"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--split", default="all", choices=["all", "png", "jpg"])
    ap.add_argument("--max-boxes", type=int, default=100)
    args = ap.parse_args()

    gt_dir, pred_dir = Path(args.gt_dir), Path(args.pred_dir)
    names = list(DEFAULT_NAMES)
    n_cls = len(names)
    # 主指标要用官方口径的 50 个点 → 10 个阈值；另外单独取 4 个阈值看细节
    thrs = list(IOU_THRS)
    show = [0.50, 0.75, 0.90, 0.95]
    show_idx = [thrs.index(round(s, 2)) for s in show]

    files = sorted(gt_dir.glob("*.txt"))
    if args.split != "all":
        files = [f for f in files if img_kind(f.stem) == args.split]
    if not files:
        sys.exit(f"{gt_dir} 下 --split {args.split} 没有匹配的标签")

    recs = []
    n_gt = np.zeros(n_cls, dtype=np.int64)
    for gf in files:
        gt = read_gt_file(gf)
        preds = read_pred_file(pred_dir / f"{gf.stem}.txt", max_boxes=args.max_boxes)
        recs.append((gf.stem, gt, preds))
        for c, b in gt.items():
            if 0 <= c < n_cls:
                n_gt[c] += len(b)

    # per[thr_idx, cls]
    per = np.full((len(thrs), n_cls), np.nan)
    # recall@0.5：该类 GT 中被任一预测以 IoU>=0.5 命中的比例
    hit = np.zeros(n_cls, dtype=np.int64)
    for ti, thr in enumerate(thrs):
        rows: dict[int, list[tuple[float, str, int, bool]]] = {}
        for stem, gt, preds in recs:
            by_cls: dict[int, list[tuple[float, int, np.ndarray]]] = {}
            for j, (conf, c, box) in enumerate(preds):
                by_cls.setdefault(c, []).append((conf, j, box))
            for c, plist in by_cls.items():
                if not 0 <= c < n_cls:
                    continue
                gtb = gt.get(c, np.zeros((0, 4)))
                matched = np.zeros(len(gtb), dtype=bool)
                for conf, j, box in plist:
                    is_tp = False
                    if len(gtb):
                        ious = iou_matrix(box[None, :], gtb)[0]
                        ious = np.where(matched, -1.0, ious)
                        b = int(np.argmax(ious))
                        if ious[b] >= thr:
                            matched[b] = True
                            is_tp = True
                    rows.setdefault(c, []).append((conf, stem, j, is_tp))
                if thr == 0.50:
                    hit[c] += int(matched.sum())
        for c in range(n_cls):
            r = rows.get(c, [])
            r.sort(key=lambda z: (-z[0], z[1], z[2]))
            v = ap_from_flags(np.array([z[3] for z in r], dtype=bool), int(n_gt[c]))
            if v is not None:
                per[ti, c] = v

    def m(sel):
        s = np.asarray(sel, dtype=float)
        s = s[~np.isnan(s)]
        return float(s.mean()) if s.size else float("nan")

    print(f"\n=== split={args.split}  图像 {len(recs)} 张  GT {int(n_gt.sum())} 框  "
          f"pred_dir={pred_dir.name} ===")
    print(f"mAP@50-95（10 阈值宏平均）= {m(per) * 100:.4f}   "
          f"mAP@50 = {m(per[show_idx[0]]) * 100:.4f}   "
          f"mAP@75 = {m(per[show_idx[1]]) * 100:.4f}")
    print(f"{'class':<12}{'AP50':>8}{'AP75':>8}{'AP90':>8}{'AP95':>8}"
          f"{'AP50-95':>9}{'Recall@0.5':>12}{'#gt':>6}")
    for c in range(n_cls):
        rec = (hit[c] / n_gt[c] * 100) if n_gt[c] else float("nan")
        col = per[:, c]
        colmean = float(col[~np.isnan(col)].mean()) if (~np.isnan(col)).any() else float("nan")
        print(f"{names[c]:<12}{per[show_idx[0], c] * 100:>8.1f}"
              f"{per[show_idx[1], c] * 100:>8.1f}{per[show_idx[2], c] * 100:>8.1f}"
              f"{per[show_idx[3], c] * 100:>8.1f}"
              f"{colmean * 100:>9.1f}{rec:>12.1f}{int(n_gt[c]):>6}")
    print(f"{'MACRO':<12}{m(per[show_idx[0]]) * 100:>8.1f}{m(per[show_idx[1]]) * 100:>8.1f}"
          f"{m(per[show_idx[2]]) * 100:>8.1f}{m(per[show_idx[3]]) * 100:>8.1f}"
          f"{m(per) * 100:>9.1f}")


if __name__ == "__main__":
    main()
