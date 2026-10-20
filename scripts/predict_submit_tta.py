"""生成提交包：与验证时**完全同一条推理路径**（这是关键）。

验证时测出 49.35 的那条路径是：
    读图(cv2) → predict(imgsz=1280, conf=0.001, iou=0.7, max_det=300, augment=True)
              → 按类 NMS@0.70 → 按置信度截断到 100 框 → 归一化 xywh
注意这**不是** `predict.py --augment`：后者不做我们那第二道按类 NMS，
少了那一步在验证集上要低 0.78 分（48.57 vs 49.35）。

所以提交包必须由这个脚本产生，否则"本地 49.35"和"交上去的东西"就不是一个东西了。

用法：
  # 先在验证集上自证能复现 49.35
  & $py scripts\\predict_submit_tta.py --weights weights\\aic_rgb1280_best_48.01.pt `
        --list splits\\lbproxy_val.txt --out submissions\\_verify_49.35 --gt-dir data\\valgt_lbproxy

  # 再对测试集生成提交
  & $py scripts\\predict_submit_tta.py --weights weights\\aic_rgb1280_best_48.01.pt `
        --list splits\\test_all.txt --out submissions\\tta_49.35
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(WS / ".ultralytics"))
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
sys.path.insert(0, str(WS / "src" / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_tile_infer import nms  # noqa: E402
from imgio import imread as imread_safe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", required=True, help="图片清单 txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-dir", default=None, help="给了就顺便评测（验证集用）")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-augment", action="store_true", help="关掉 TTA（用于对比）")
    ap.add_argument("--no-nms", action="store_true",
                    help="跳过第二道「按类 NMS」。RT-DETR 是 NMS-free 的，"
                         "我们的按类 NMS 对它可能有害，这一项就是用来对照的。")
    args = ap.parse_args()

    import cv2

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = WS / weights
    lst = Path(args.list)
    if not lst.is_absolute():
        lst = WS / lst
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = WS / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.txt"):
        p.unlink()

    images = [Path(ln.strip()) for ln in lst.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"权重 {weights}\n清单 {lst}（{len(images)} 张）\n输出 {out_dir}")
    print(f"imgsz={args.imgsz} conf={args.conf} iou={args.iou} 二道NMS={args.nms_iou} "
          f"max_det={args.max_det} TTA={not args.no_augment}\n", flush=True)

    # 按文件名自动识别模型族：rtdetr-* 必须用 RTDETR 类，
    # 用 YOLO 类去加载 RT-DETR 权重会按 YOLO 的 DetectionModel 去构建，结果不对。
    if "rtdetr" in weights.name.lower():
        from ultralytics import RTDETR

        model = RTDETR(str(weights))
        print("[arch] RT-DETR 模式（NMS-free 模型）")
    else:
        from ultralytics import YOLO

        model = YOLO(str(weights))
    t0 = time.time()
    n_fail, n_boxes, n_empty, n_over = 0, 0, 0, 0
    for k, img_path in enumerate(images, 1):
        # 必须用路径安全版：cv2.imread 在中文路径下会偶发返回 None，
        # 那会写出空预测文件、让评测安静地给出 0 分。
        img = imread_safe(img_path, cv2.IMREAD_COLOR)
        if img is None:
            n_fail += 1
            (out_dir / f"{img_path.stem}.txt").write_text("", encoding="utf-8")
            continue
        h, w = img.shape[:2]
        res = model.predict(
            source=[img], imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            max_det=300, augment=not args.no_augment, batch=4,
            device=args.device, verbose=False,
        )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            (out_dir / f"{img_path.stem}.txt").write_text("", encoding="utf-8")
            n_empty += 1
            continue
        B = boxes.xyxy.cpu().numpy()
        C = boxes.cls.cpu().numpy().astype(int)
        S = boxes.conf.cpu().numpy()
        if args.no_nms:
            keep = list(range(len(B)))          # 对照：完全不做第二道 NMS
        else:
            keep = []
            for c in np.unique(C):
                idx = np.where(C == c)[0]
                keep.extend(idx[nms(B[idx], S[idx], args.nms_iou)].tolist())
        ka = np.array(keep)
        order = ka[np.argsort(-S[ka])]
        if len(order) > args.max_det:
            n_over += 1
        order = order[: args.max_det]
        lines = []
        for i in order:
            x1, y1, x2, y2 = B[i]
            lines.append(
                f"{int(C[i])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} {float(S[i]):.6f}"
            )
        (out_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_boxes += len(lines)
        if k % 100 == 0:
            print(f"  {k}/{len(images)}  {time.time() - t0:.0f}s", flush=True)

    print(f"\n完成：{len(images)} 张，共 {n_boxes} 框，空文件 {n_empty}，"
          f"被 100 框截断 {n_over} 张，读图失败 {n_fail}，用时 {(time.time() - t0) / 60:.1f} 分钟")
    if n_fail > len(images) * 0.2:
        sys.exit("读图失败率过高，结果不可信")

    if args.gt_dir:
        from map_eval import evaluate

        gt = Path(args.gt_dir)
        if not gt.is_absolute():
            gt = WS / gt
        r = evaluate(gt, out_dir, max_boxes=args.max_det)
        print(r.summary())


if __name__ == "__main__":
    main()
