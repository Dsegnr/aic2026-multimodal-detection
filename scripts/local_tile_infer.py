"""切块（SAHI 式）推理：把 1920×1080 的图按子窗口放大后再送进模型。

为什么做（本轮按面积拆分的诊断结果）：
  目标按面积分档后，mAP50-95 是 **small 3.4 / medium 19.3 / large 52.4**，
  而 small+medium 占了 58% 的框。也就是说分数被中小目标的高 IoU 定位拖住了。
  朴素做法（整图缩到 1280）会把 1920 宽的图**缩小 0.67 倍**，目标跟着变小；
  而切块后每一块只有原图的 1/4，缩放到 1280 反而是**放大 1.2~1.3 倍**——
  目标的等效像素尺寸翻倍，正好打在短板上。

这是**单模型多视图推理**，不是多模型集成；所有视图来自同一个 checkpoint。

视图组合：full 整图 / 2x2 切块 / 3x3 切块，用 + 连接。
合并时按类别做 NMS，坐标统一映射回原图，最后按置信度截断到 100 框（官方限制）。

用法：
  & $py scripts\\local_tile_infer.py --weights weights\\aic_rgb1280_best_48.01.pt --views full+2x2 --out submissions\\_tile\\full_2x2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(WS / ".ultralytics"))
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
sys.path.insert(0, str(WS / "src" / "eval"))

from map_eval import evaluate  # noqa: E402
from imgio import imread as imread_safe  # noqa: E402

# 自定义类的 checkpoint 里 pickle 了 ElongModel / MMFusionModel 等类。
# 不 import 对应模块时，ultralytics 会误以为缺 pip 包（AutoInstall 'elong'）→ 加载失败 →
# **一路空预测、静默 0 分**。这里先把它们 import 进来，保证能正确反序列化。
sys.path.insert(0, str(WS / "src" / "mm"))
for _mod in ("elong", "mm_multistage", "moe_fusion", "fusion"):
    try:
        __import__(_mod)
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] 预加载 {_mod} 失败（若权重不含该类可忽略）: {_e}")


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """单类 NMS，返回保留下标。boxes 为 xyxy。"""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


def cut_tiles(w: int, h: int, n: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """把 w×h 划成 n×n 块（带 overlap），返回 (x0, y0, x1, y1) 列表。"""
    if n <= 1:
        return [(0, 0, w, h)]
    tw = w / (n - (n - 1) * overlap)
    th = h / (n - (n - 1) * overlap)
    step_x = tw * (1 - overlap)
    step_y = th * (1 - overlap)
    out: list[tuple[int, int, int, int]] = []
    for i in range(n):
        for j in range(n):
            x0 = int(round(i * step_x))
            y0 = int(round(j * step_y))
            x1 = min(w, int(round(x0 + tw)))
            y1 = min(h, int(round(y0 + th)))
            x0 = max(0, x1 - int(round(tw)))
            y0 = max(0, y1 - int(round(th)))
            out.append((x0, y0, x1, y1))
    return out


def parse_views(text: str) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    for v in text.split("+"):
        if v == "full":
            specs.append(("full", 1))
        elif v in ("2x2", "3x3"):
            specs.append((v, int(v[0])))
        else:
            sys.exit(f"未知视图 {v}")
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val-list", default="splits/lbproxy_val.txt")
    ap.add_argument("--gt-dir", default="data/valgt_lbproxy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="full+2x2", help="full / 2x2 / 3x3，用 + 连接")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--device", default="0")
    ap.add_argument("--augment", action="store_true", help="单模型 TTA（多尺度 + 翻转）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（0=全部）")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = WS / weights
    val_list = WS / args.val_list
    gt_dir = WS / args.gt_dir
    out_dir = WS / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.txt"):
        p.unlink()

    images = [Path(ln.strip()) for ln in val_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        images = images[: args.limit]
    view_specs = parse_views(args.views)

    model = YOLO(str(weights))
    print(f"权重 {weights}\n视图 {args.views}  imgsz={args.imgsz} overlap={args.overlap} "
          f"nms_iou={args.nms_iou}  图像 {len(images)} 张", flush=True)

    t0 = time.time()
    n_unreadable = 0
    for k, img_path in enumerate(images, 1):
        # 用路径安全版：cv2.imread 在中文路径下会偶发返回 None（见 scripts/imgio.py）
        img = imread_safe(img_path, cv2.IMREAD_COLOR)
        if img is None:
            n_unreadable += 1
            (out_dir / f"{img_path.stem}.txt").write_text("", encoding="utf-8")
            continue
        h, w = img.shape[:2]
        crops: list[np.ndarray] = []
        meta: list[tuple[int, int, str]] = []
        for name, n in view_specs:
            for (x0, y0, x1, y1) in cut_tiles(w, h, n, args.overlap):
                crops.append(img[y0:y1, x0:x1])
                meta.append((x0, y0, name))

        results = model.predict(
            source=crops, imgsz=args.imgsz, conf=args.conf, iou=0.7,
            max_det=300, batch=4, device=args.device, augment=args.augment,
            verbose=False, stream=True, save=False,
        )
        boxes_all, cls_all, conf_all = [], [], []
        for res, (dx, dy, _tag) in zip(results, meta):
            b = res.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy().copy()
            xyxy[:, [0, 2]] += dx
            xyxy[:, [1, 3]] += dy
            boxes_all.append(xyxy)
            cls_all.append(b.cls.cpu().numpy().astype(int))
            conf_all.append(b.conf.cpu().numpy())
        if not boxes_all:
            (out_dir / f"{img_path.stem}.txt").write_text("", encoding="utf-8")
            continue

        B = np.concatenate(boxes_all)
        C = np.concatenate(cls_all)
        S = np.concatenate(conf_all)
        keep: list[int] = []
        for c in np.unique(C):
            idx = np.where(C == c)[0]
            kk = nms(B[idx], S[idx], args.nms_iou)
            keep.extend(idx[kk].tolist())
        keep_arr = np.array(keep)
        order = keep_arr[np.argsort(-S[keep_arr])][: args.max_det]

        lines = []
        for i in order:
            x1, y1, x2, y2 = B[i]
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{int(C[i])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(S[i]):.6f}")
        (out_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if k % 25 == 0:
            print(f"  {k}/{len(images)}  {time.time() - t0:.0f}s", flush=True)

    # 教训：清单路径写错时 cv2.imread 全部返回 None，评测会"安静地"给出 0 分。
    # 这种失败必须吵出来。
    if n_unreadable:
        print(f"[警告] {n_unreadable}/{len(images)} 张图读不出来（路径错误？）")
    if n_unreadable > len(images) * 0.2:
        sys.exit(f"读图失败率过高（{n_unreadable}/{len(images)}），评测结果不可信，已中止")

    res = evaluate(gt_dir, out_dir, max_boxes=args.max_det)
    print(res.summary())
    print(f"用时 {(time.time() - t0) / 60:.1f} 分钟")
    (out_dir / "_summary.json").write_text(
        json.dumps(
            {
                "views": args.views,
                "imgsz": args.imgsz,
                "overlap": args.overlap,
                "nms_iou": args.nms_iou,
                "mAP50-95": res.map_50_95,
                "mAP50": res.map_50,
                "mAP75": res.map_75,
                "per_class": res.ap_per_class_50_95,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
