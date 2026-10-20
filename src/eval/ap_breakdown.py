"""把 mAP@50-95 拆成「按 IoU 阈值」和「按目标面积」两个维度。

为什么需要（Astra 第 2 次回答的第 2 节）：总分 47.7 只说"框不准"，
但没说清是哪一段 IoU、哪一档尺寸在丢分。只有拆开，才能判断下一步该做
  ① 提高小目标的定位（→ 分辨率 / 切块 / 尺度增强）
  ② 还是提高大中目标的框精度（→ 回归损失 / 更高质量的正样本）

面积分档沿用 COCO 的口径（换算到原图 1920×1080 像素）：
  small < 32²、medium < 96²、large ≥ 96²。

用法：
  & $py src\\eval\\ap_breakdown.py --gt-dir data\\valgt_lbproxy --pred-dir submissions\\_res_sweep\\B_1280_TTA
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src" / "eval"))

from map_eval import (  # noqa: E402
    DEFAULT_NAMES,
    IOU_THRS,
    ap_from_flags,
    iou_matrix,
    read_gt_file,
    read_pred_file,
)

IMG_W, IMG_H = 1920.0, 1080.0
SMALL, MEDIUM = 32.0**2, 96.0**2


def area_px(box: np.ndarray) -> float:
    return float((box[2] - box[0]) * IMG_W * (box[3] - box[1]) * IMG_H)


def ap_table(gt_dir: Path, pred_dir: Path, names, iou_thrs, max_boxes: int, area_filter=None):
    """返回 (per_class_ap[thr][cls], n_gt[cls])，可选按面积过滤 GT。"""
    n_cls = len(names)
    ap = np.full((len(iou_thrs), n_cls), np.nan)
    n_gt = np.zeros(n_cls, dtype=int)

    records = []
    for gt_file in sorted(gt_dir.glob("*.txt")):
        stem = gt_file.stem
        gt = read_gt_file(gt_file)
        preds = read_pred_file(pred_dir / f"{stem}.txt", max_boxes=max_boxes)
        if area_filter is not None:
            kept = {}
            for c, boxes in gt.items():
                sel = np.array([area_filter(area_px(b)) for b in boxes], dtype=bool)
                if sel.any():
                    kept[c] = boxes[sel]
            gt = kept
        records.append((gt, preds))

    for c in range(n_cls):
        n_gt[c] = int(sum(len(g.get(c, [])) for g, _ in records))

    for c in range(n_cls):
        for ti, thr in enumerate(iou_thrs):
            rows: list[tuple[float, str, int, bool]] = []
            for ri, (gt, preds) in enumerate(records):
                gt_c = gt.get(c, np.zeros((0, 4)))
                plist = [(cf, j, b) for j, (cf, cc, b) in enumerate(preds) if cc == c]
                if not len(gt_c) and not plist:
                    continue
                matched = np.zeros(len(gt_c), dtype=bool)
                for cf, j, b in plist:
                    is_tp = False
                    if len(gt_c):
                        ious = iou_matrix(b[None, :], gt_c)[0]
                        ious = np.where(matched, -1.0, ious)
                        best = int(np.argmax(ious))
                        if ious[best] >= thr:
                            matched[best] = True
                            is_tp = True
                    rows.append((cf, f"{ri:05d}", j, is_tp))
            rows.sort(key=lambda r: (-r[0], r[1], r[2]))
            flags = np.array([r[3] for r in rows], dtype=bool)
            val = ap_from_flags(flags, int(n_gt[c]))
            if val is not None:
                ap[ti, c] = val
    return ap, n_gt


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--gt-dir", required=True)
    ap_.add_argument("--pred-dir", required=True)
    ap_.add_argument("--max-boxes", type=int, default=100)
    args = ap_.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    ap, n_gt = ap_table(gt_dir, pred_dir, DEFAULT_NAMES, IOU_THRS, args.max_boxes)
    idx = {t: i for i, t in enumerate(IOU_THRS)}

    print(f"预测目录 {pred_dir}   max_boxes={args.max_boxes}")
    print("\n【按 IoU 阈值】每类 AP（越往下越考验框精度）")
    hdr = f"{'class':<13}{'#gt':>6}" + "".join(f"{t:>8.2f}" for t in IOU_THRS)
    print(hdr)
    for c, nm in enumerate(DEFAULT_NAMES):
        vals = "".join(f"{ap[ti, c] * 100:>8.2f}" if not np.isnan(ap[ti, c]) else f"{'-':>8}" for ti in range(len(IOU_THRS)))
        print(f"{nm:<13}{n_gt[c]:>6}{vals}")
    mean_row = "".join(
        f"{np.nanmean(ap[ti]) * 100:>8.2f}" for ti in range(len(IOU_THRS))
    )
    print(f"{'宏平均':<13}{int(n_gt.sum()):>6}{mean_row}")
    print(f"\nmAP50={np.nanmean(ap[idx[0.50]]) * 100:.2f}  "
          f"mAP75={np.nanmean(ap[idx[0.75]]) * 100:.2f}  "
          f"mAP90={np.nanmean(ap[idx[0.90]]) * 100:.2f}  "
          f"mAP95={np.nanmean(ap[idx[0.95]]) * 100:.2f}  "
          f"mAP50-95={np.nanmean(ap) * 100:.2f}")

    print("\n【按目标面积】GT 分档，预测不限制（COCO 口径，换算到原图像素）")
    print(f"{'档位':<10}{'#gt':>7}{'mAP50':>9}{'mAP75':>9}{'mAP50-95':>11}{'占框数':>9}")
    total_gt = 0
    for label, filt in (
        ("small", lambda a: a < SMALL),
        ("medium", lambda a: SMALL <= a < MEDIUM),
        ("large", lambda a: a >= MEDIUM),
    ):
        a2, n2 = ap_table(gt_dir, pred_dir, DEFAULT_NAMES, IOU_THRS, args.max_boxes, area_filter=filt)
        cnt = int(n2.sum())
        total_gt += cnt
        print(f"{label:<10}{cnt:>7}{np.nanmean(a2[idx[0.50]]) * 100:>9.2f}"
              f"{np.nanmean(a2[idx[0.75]]) * 100:>9.2f}"
              f"{np.nanmean(a2) * 100:>11.2f}{cnt / max(sum(n_gt), 1) * 100:>8.1f}%")
    print(f"{'合计':<10}{total_gt:>7}")


if __name__ == "__main__":
    main()
