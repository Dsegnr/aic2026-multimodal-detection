"""AIC2026 官方指标 mAP@50-95 的本地复刻实现。

官方口径（赛题页「六、性能指标要求」）：

1. IoU 阈值 T = {0.50, 0.55, ..., 0.95}，共 10 个。
2. 对每个类别 c 和每个阈值 t：把该类所有测试图像的预测框按置信度从高到低排序，
   依次遍历；若与某个「尚未被匹配」的真实框 IoU >= t，记 TP 并消耗该真实框，
   否则记 FP；遍历结束后未被匹配的真实框记 FN。
3. AP 使用 101 点插值法（召回率 0.00, 0.01, ..., 1.00 共 101 个采样点）：
   AP = mean_r max{ precision(r') : r' >= r }。
4. 最终 mAP@50-95 = 对 10 个 IoU 阈值、12 个类别取平均。

实现在两个点上刻意与 COCO 官方一致：
- 全类别 AP 是「所有图像合在一起」的整体 P-R 曲线，不是逐图平均；
- 每个 IoU 阈值下重新独立匹配。

约定：
- 真实框与预测框都用归一化 cx,cy,w,h，内部转成 xyxy 计算 IoU，不做裁剪。
- 某类别在验证集里没有真实框时 AP 记 None，默认从均值里剔除（COCO 同处理），
  并在报告里单独列出，避免"没框反而满分"的假象。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

IOU_THRS: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))

DEFAULT_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle",
    "car", "ball", "light", "garbage_can", "uav", "tricycle",
]


# --------------------------------------------------------------------------- #
# 基础几何
# --------------------------------------------------------------------------- #
def xywhn_to_xyxy(box: Sequence[float]) -> np.ndarray:
    """归一化 (cx, cy, w, h) -> 归一化 (x1, y1, x2, y2)。"""
    cx, cy, w, h = (float(v) for v in box[:4])
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def voc_ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    """101 点插值 AP。recall / precision 必须按置信度降序的累积序列给出。"""
    mrec = np.concatenate(([0.0], np.asarray(recall, dtype=np.float64), [1.0]))
    mpre = np.concatenate(([0.0], np.asarray(precision, dtype=np.float64), [0.0]))
    # 精度单调包络：从右向左取最大值
    for i in range(len(mpre) - 1, 0, -1):
        if mpre[i - 1] < mpre[i]:
            mpre[i - 1] = mpre[i]
    q = np.linspace(0.0, 1.0, 101)
    idx = np.clip(np.searchsorted(mrec, q, side="left"), 0, len(mpre) - 1)
    return float(np.mean(mpre[idx]))


def ap_from_flags(tp_flags: np.ndarray, n_gt: int) -> float | None:
    """tp_flags: 已按置信度降序排列的布尔序列。n_gt 为该类真实框总数。"""
    if n_gt <= 0:
        return None
    if len(tp_flags) == 0:
        return 0.0
    tp_cum = np.cumsum(tp_flags.astype(np.float64))
    fp_cum = np.cumsum((~tp_flags).astype(np.float64))
    recall = tp_cum / float(n_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return voc_ap_101(recall, precision)


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #
@dataclass
class ImageRecord:
    stem: str
    gt: dict[int, np.ndarray] = field(default_factory=dict)          # cls -> (M,4) xyxy
    preds: list[tuple[float, int, np.ndarray]] = field(default_factory=list)  # (conf, cls, xyxy)


def read_gt_file(path: Path) -> dict[int, np.ndarray]:
    """YOLO 标签: class cx cy w h（归一化）。"""
    per_cls: dict[int, list[np.ndarray]] = {}
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        per_cls.setdefault(cls, []).append(xywhn_to_xyxy(parts[1:5]))
    return {c: np.vstack(v) for c, v in per_cls.items()}


def read_pred_file(path: Path, max_boxes: int = 100) -> list[tuple[float, int, np.ndarray]]:
    """预测: class cx cy w h conf。按官方约束每图最多保留 max_boxes 个（置信度最高）。"""
    rows: list[tuple[float, int, np.ndarray]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        cls = int(float(parts[0]))
        conf = float(parts[5])
        rows.append((conf, cls, xywhn_to_xyxy(parts[1:5])))
    rows.sort(key=lambda r: -r[0])
    return rows[:max_boxes]


# --------------------------------------------------------------------------- #
# 主评测
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    map_50_95: float
    map_50: float
    map_75: float
    ap_per_class_50_95: dict[str, float | None]
    ap_per_class_50: dict[str, float | None]
    n_gt_per_class: dict[str, int]
    n_pred_per_class: dict[str, int]
    n_images: int
    n_images_missing_pred: int

    def summary(self) -> str:
        lines = [
            f"images={self.n_images}  缺预测文件={self.n_images_missing_pred}",
            f"mAP@50-95 = {self.map_50_95 * 100:.4f}",
            f"mAP@50    = {self.map_50 * 100:.4f}",
            f"mAP@75    = {self.map_75 * 100:.4f}",
            "",
            f"{'class':<14}{'AP50-95':>10}{'AP50':>10}{'#gt':>8}{'#pred':>8}",
        ]
        for name, ap in self.ap_per_class_50_95.items():
            ap_s = "  n/a" if ap is None else f"{ap * 100:10.4f}"
            ap50 = self.ap_per_class_50[name]
            ap50_s = "  n/a" if ap50 is None else f"{ap50 * 100:10.4f}"
            lines.append(
                f"{name:<14}{ap_s:>10}{ap50_s:>10}"
                f"{self.n_gt_per_class[name]:>8}{self.n_pred_per_class[name]:>8}"
            )
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


def evaluate(
    gt_dir: Path | str,
    pred_dir: Path | str,
    names: Sequence[str] = tuple(DEFAULT_NAMES),
    iou_thrs: Iterable[float] = IOU_THRS,
    exclude_absent: bool = True,
    max_boxes: int = 100,
    subset: Sequence[str] | None = None,
) -> EvalResult:
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    iou_thrs = tuple(float(t) for t in iou_thrs)

    gt_files = sorted(gt_dir.glob("*.txt"))
    if not gt_files:
        raise FileNotFoundError(f"{gt_dir} 下没有找到任何 .txt 标签")
    # `subset`：按 stem 挑图，**允许重复**（用于 bootstrap 有放回重采样——
    # 重复的 stem 会被当成两张独立的图参与统计）。默认 None = 用全部。
    if subset is not None:
        by_stem = {f.stem: f for f in gt_files}
        missing = [s for s in set(subset) if s not in by_stem]
        if missing:
            raise FileNotFoundError(f"subset 里有 {len(missing)} 个 stem 在 {gt_dir} 里找不到，例如 {missing[:3]}")
        gt_files = [by_stem[s] for s in subset]
        if not gt_files:
            raise FileNotFoundError("subset 为空")

    records: list[ImageRecord] = []
    missing_pred = 0
    for gf in gt_files:
        rec = ImageRecord(stem=gf.stem, gt=read_gt_file(gf))
        pf = pred_dir / f"{gf.stem}.txt"
        if not pf.exists():
            missing_pred += 1
        rec.preds = read_pred_file(pf, max_boxes=max_boxes)
        records.append(rec)

    n_cls = len(names)
    n_gt_per_class = np.zeros(n_cls, dtype=np.int64)
    n_pred_per_class = np.zeros(n_cls, dtype=np.int64)
    for rec in records:
        for c, boxes in rec.gt.items():
            if 0 <= c < n_cls:
                n_gt_per_class[c] += len(boxes)
        for _, c, _ in rec.preds:
            if 0 <= c < n_cls:
                n_pred_per_class[c] += 1

    # ap[thr_idx, cls]
    ap = np.full((len(iou_thrs), n_cls), np.nan, dtype=np.float64)

    for ti, thr in enumerate(iou_thrs):
        # 每张图、每个类别做一次贪心匹配，得到该类预测的 TP 标记
        per_class_rows: dict[int, list[tuple[float, str, int, bool]]] = {}
        for rec in records:
            by_cls: dict[int, list[tuple[float, int, np.ndarray]]] = {}
            for j, (conf, c, box) in enumerate(rec.preds):
                by_cls.setdefault(c, []).append((conf, j, box))
            for c, plist in by_cls.items():
                if c < 0 or c >= n_cls:
                    continue
                gt_boxes = rec.gt.get(c, np.zeros((0, 4)))
                matched = np.zeros(len(gt_boxes), dtype=bool)
                for k, (conf, j, box) in enumerate(plist):
                    is_tp = False
                    if len(gt_boxes):
                        ious = iou_matrix(box[None, :], gt_boxes)[0]
                        ious = np.where(matched, -1.0, ious)
                        best = int(np.argmax(ious))
                        if ious[best] >= thr:
                            matched[best] = True
                            is_tp = True
                    per_class_rows.setdefault(c, []).append((conf, rec.stem, j, is_tp))

        for c in range(n_cls):
            rows = per_class_rows.get(c, [])
            # 全局按置信度降序；同分用 (图像名, 预测序号) 做确定性排序
            rows.sort(key=lambda r: (-r[0], r[1], r[2]))
            flags = np.array([r[3] for r in rows], dtype=bool)
            val = ap_from_flags(flags, int(n_gt_per_class[c]))
            if val is not None:
                ap[ti, c] = val

    # 汇总
    def mean_over(sel: np.ndarray, axis: int) -> float:
        vals = np.where(np.isnan(sel), 0.0, sel) if not exclude_absent else sel[~np.isnan(sel)]
        return float(vals.mean()) if vals.size else float("nan")

    map_50_95 = mean_over(ap, axis=1)
    idx50 = int(np.argmin(np.abs(np.array(iou_thrs) - 0.50)))
    idx75 = int(np.argmin(np.abs(np.array(iou_thrs) - 0.75)))
    map_50 = mean_over(ap[idx50], axis=0)
    map_75 = mean_over(ap[idx75], axis=0)

    ap_per_class_50_95 = {
        names[c]: (None if np.isnan(ap[:, c]).all() else float(np.nanmean(ap[:, c])))
        for c in range(n_cls)
    }
    ap_per_class_50 = {
        names[c]: (None if np.isnan(ap[idx50, c]) else float(ap[idx50, c])) for c in range(n_cls)
    }

    return EvalResult(
        map_50_95=map_50_95,
        map_50=map_50,
        map_75=map_75,
        ap_per_class_50_95=ap_per_class_50_95,
        ap_per_class_50=ap_per_class_50,
        n_gt_per_class={names[c]: int(n_gt_per_class[c]) for c in range(n_cls)},
        n_pred_per_class={names[c]: int(n_pred_per_class[c]) for c in range(n_cls)},
        n_images=len(records),
        n_images_missing_pred=missing_pred,
    )


def _main() -> None:
    ap = argparse.ArgumentParser(description="AIC2026 mAP@50-95 本地评测")
    ap.add_argument("--gt-dir", required=True, help="真实标签目录（*.txt）")
    ap.add_argument("--pred-dir", required=True, help="预测结果目录（同名 *.txt）")
    ap.add_argument("--json-out", default=None, help="可选：把完整结果写入 JSON")
    ap.add_argument("--max-boxes", type=int, default=100,
                    help="每图保留的最大框数（官方限制 100）。把它调大可以与 300 对比，"
                         "用来量化「100 框上限」到底让我们丢了多少分。")
    args = ap.parse_args()

    res = evaluate(args.gt_dir, args.pred_dir, max_boxes=args.max_boxes)
    print(res.summary())
    if args.json_out:
        Path(args.json_out).write_text(res.to_json(), encoding="utf-8")
        print(f"\n结果已写入 {args.json_out}")


if __name__ == "__main__":
    _main()
