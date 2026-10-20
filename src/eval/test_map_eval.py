"""mAP@50-95 评测器的单元测试。

运行：
    .venv\\Scripts\\python.exe src\\eval\\test_map_eval.py

评测器是把官方口径翻译成代码，一旦它错了，后面所有实验结论都是幻觉。
因此这里用可手算的合成样例做断言。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_eval import evaluate, iou_matrix, voc_ap_101, xywhn_to_xyxy  # noqa: E402

NAMES = ["person", "boat", "animal", "seat"]


def _write(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _case(gt: dict[str, list[str]], pred: dict[str, list[str]]):
    """构造临时 gt/pred 目录。键为图像名，值为该文件的行。"""
    d = Path(tempfile.mkdtemp(prefix="mapeval_"))
    gt_dir, pred_dir = d / "gt", d / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    for stem, rows in gt.items():
        _write(gt_dir / f"{stem}.txt", rows)
    for stem, rows in pred.items():
        _write(pred_dir / f"{stem}.txt", rows)
    return gt_dir, pred_dir


# --------------------------------------------------------------------------- #
def test_iou_basics() -> None:
    a = np.array([[0.0, 0.0, 1.0, 1.0]])
    b = np.array([[0.0, 0.0, 1.0, 1.0]])
    assert abs(iou_matrix(a, b)[0, 0] - 1.0) < 1e-12
    # 半重叠：IoU = 1/3
    b2 = np.array([[0.5, 0.0, 1.5, 1.0]])
    assert abs(iou_matrix(a, b2)[0, 0] - 1.0 / 3.0) < 1e-9
    # 不相交
    b3 = np.array([[2.0, 2.0, 3.0, 3.0]])
    assert iou_matrix(a, b3)[0, 0] == 0.0
    # 空集合
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 1)
    print("[ok] test_iou_basics")


def test_ap101_handcase() -> None:
    # 2 个真实框；预测顺序 FP, TP, TP -> precision 3/3 处为 2/3
    # 精度包络后 PR 平台高度 = 2/3，101 点插值 AP = 2/3
    recall = np.array([0.0, 0.5, 1.0])
    precision = np.array([0.0, 0.5, 2.0 / 3.0])
    ap = voc_ap_101(recall, precision)
    assert abs(ap - 2.0 / 3.0) < 1e-9, ap

    # 全对 -> AP = 1
    assert abs(voc_ap_101(np.array([0.5, 1.0]), np.array([1.0, 1.0])) - 1.0) < 1e-9
    # 全错（无 TP）-> AP = 0
    assert abs(voc_ap_101(np.array([0.0]), np.array([0.0]))) < 1e-12
    print("[ok] test_ap101_handcase")


def test_perfect_prediction() -> None:
    gt = {"img1": ["0 0.5 0.5 0.2 0.2", "0 0.2 0.2 0.1 0.1"]}
    pred = {"img1": ["0 0.5 0.5 0.2 0.2 0.9", "0 0.2 0.2 0.1 0.1 0.8"]}
    g, p = _case(gt, pred)
    res = evaluate(g, p, names=NAMES)
    assert abs(res.map_50_95 - 1.0) < 1e-9, res.map_50_95
    assert abs(res.map_50 - 1.0) < 1e-9
    assert abs(res.ap_per_class_50_95["person"] - 1.0) < 1e-9
    # 其余类别在 GT 中不存在 -> None 且被剔除
    assert res.ap_per_class_50_95["boat"] is None
    print("[ok] test_perfect_prediction")


def test_no_prediction_file() -> None:
    gt = {"img1": ["0 0.5 0.5 0.2 0.2"]}
    pred: dict[str, list[str]] = {"img1": []}
    g, p = _case(gt, pred)
    res = evaluate(g, p, names=NAMES)
    assert res.map_50_95 == 0.0, res.map_50_95
    assert res.n_images_missing_pred == 0
    print("[ok] test_no_prediction_file")


def test_missing_pred_file_counts_as_zero() -> None:
    gt = {"img1": ["0 0.5 0.5 0.2 0.2"]}
    pred: dict[str, list[str]] = {}  # 完全不交预测文件
    g, p = _case(gt, pred)
    res = evaluate(g, p, names=NAMES)
    assert res.map_50_95 == 0.0
    assert res.n_images_missing_pred == 1
    print("[ok] test_missing_pred_file_counts_as_zero")


def test_fp_before_tp_penalty() -> None:
    # 2 个 GT，预测顺序 FP, TP, TP -> AP = 2/3
    gt = {"img1": ["0 0.1 0.1 0.1 0.1", "0 0.9 0.9 0.1 0.1"]}
    pred = {
        "img1": [
            "0 0.5 0.5 0.1 0.1 0.99",   # 与两个 GT 都不重叠 -> FP
            "0 0.1 0.1 0.1 0.1 0.90",   # TP
            "0 0.9 0.9 0.1 0.1 0.80",   # TP
        ]
    }
    g, p = _case(gt, pred)
    res = evaluate(g, p, names=NAMES)
    assert abs(res.map_50 - 2.0 / 3.0) < 1e-9, res.map_50
    print("[ok] test_fp_before_tp_penalty")


def test_class_isolation() -> None:
    # 预测类别写错 -> 不匹配，AP = 0
    gt = {"img1": ["0 0.5 0.5 0.2 0.2"]}
    pred = {"img1": ["1 0.5 0.5 0.2 0.2 0.9"]}
    g, p = _case(gt, pred)
    res = evaluate(g, p, names=NAMES)
    assert res.ap_per_class_50_95["person"] == 0.0
    assert res.ap_per_class_50_95["boat"] is None
    print("[ok] test_class_isolation")


def test_iou_threshold_boundary() -> None:
    # 等尺寸框水平偏移 d 时 IoU = (s-d)/(s+d)。用它精确构造 IoU=0.51 / 0.49 的两组。
    s, cx, cy = 0.2, 0.25, 0.25

    def pred_center_for(iou: float) -> float:
        d = s * (1.0 - iou) / (1.0 + iou)
        return cx + d

    gt = {"img1": [f"0 {cx} {cy} {s} {s}"]}

    # 情形一：IoU 略高于 0.50 -> 在 0.50 阈值算 TP，在 0.55 阈值算 FP
    p_hi = pred_center_for(0.51)
    iou_hi = float(iou_matrix(
        xywhn_to_xyxy([cx, cy, s, s])[None, :], xywhn_to_xyxy([p_hi, cy, s, s])[None, :]
    )[0, 0])
    assert abs(iou_hi - 0.51) < 1e-9, iou_hi
    g, p = _case(gt, {"img1": [f"0 {p_hi} {cy} {s} {s} 0.9"]})
    assert abs(evaluate(g, p, names=NAMES).map_50 - 1.0) < 1e-9
    assert evaluate(g, p, names=NAMES, iou_thrs=(0.55,)).map_50_95 == 0.0

    # 情形二：IoU 略低于 0.50 -> 在 0.50 阈值算 FP
    p_lo = pred_center_for(0.49)
    iou_lo = float(iou_matrix(
        xywhn_to_xyxy([cx, cy, s, s])[None, :], xywhn_to_xyxy([p_lo, cy, s, s])[None, :]
    )[0, 0])
    assert abs(iou_lo - 0.49) < 1e-9, iou_lo
    g2, p2 = _case(gt, {"img1": [f"0 {p_lo} {cy} {s} {s} 0.9"]})
    assert evaluate(g2, p2, names=NAMES).map_50 == 0.0
    print("[ok] test_iou_threshold_boundary")


def test_duplicate_matches_single_gt() -> None:
    # 1 个 GT，3 个几乎相同的预测：第一个 TP，后两个 FP；
    # 召回已到 1，精度包络后 AP 仍为 1（与 COCO 行为一致）
    gt = {"img1": ["0 0.5 0.5 0.2 0.2"]}
    pred = {
        "img1": [
            "0 0.5 0.5 0.2 0.2 0.95",
            "0 0.5 0.5 0.2 0.2 0.90",
            "0 0.5 0.5 0.2 0.2 0.85",
        ]
    }
    g, p = _case(gt, pred)
    res = evaluate(g, p, names=NAMES)
    assert abs(res.map_50_95 - 1.0) < 1e-9, res.map_50_95
    print("[ok] test_duplicate_matches_single_gt")


def test_max_boxes_truncation() -> None:
    # 101 个预测，唯一 TP 排在第 101 位 -> 被截断，AP = 0
    gt = {"img1": ["0 0.5 0.5 0.1 0.1"]}
    rows = ["0 0.05 0.05 0.02 0.02 0.99"] * 100
    rows.append("0 0.5 0.5 0.1 0.1 0.01")
    g, p = _case(gt, {"img1": rows})
    res = evaluate(g, p, names=NAMES)
    assert res.map_50_95 == 0.0, res.map_50_95
    assert res.n_pred_per_class["person"] == 100
    print("[ok] test_max_boxes_truncation")


def test_two_classes_macro_average() -> None:
    gt = {"img1": ["0 0.2 0.2 0.1 0.1", "1 0.8 0.8 0.1 0.1"]}
    # 类别 0 完美，类别 1 完全没有预测 -> AP: person=1, boat=0 -> 宏平均 0.5
    pred = {"img1": ["0 0.2 0.2 0.1 0.1 0.9"]}
    g, p = _case(gt, pred)
    res = evaluate(g, p, names=NAMES)
    assert abs(res.map_50_95 - 0.5) < 1e-9, res.map_50_95
    assert res.ap_per_class_50_95["boat"] == 0.0
    print("[ok] test_two_classes_macro_average")


def main() -> None:
    tests = [
        test_iou_basics,
        test_ap101_handcase,
        test_perfect_prediction,
        test_no_prediction_file,
        test_missing_pred_file_counts_as_zero,
        test_fp_before_tp_penalty,
        test_class_isolation,
        test_iou_threshold_boundary,
        test_duplicate_matches_single_gt,
        test_max_boxes_truncation,
        test_two_classes_macro_average,
    ]
    for t in tests:
        t()
    print(f"\n全部 {len(tests)} 个测试通过")


if __name__ == "__main__":
    main()
