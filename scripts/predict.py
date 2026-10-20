"""在测试集上推理并生成官方格式的提交文件。

官方输出要求（赛题页「九、成绩评价」）：
- 每张测试图一个同名 TXT；
- 每行 [class_id, norm_cx, norm_cy, norm_w, norm_h, confidence]；
- 每图最多 100 框，超出按置信度截断；
- 没有目标也必须提交空 TXT，不允许缺失；
- 全部打包为 zip 提交。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="权重路径（相对工作区或绝对）")
    ap.add_argument(
        "--source",
        default="data/test/images",
        help="测试图目录，或一个列出图片绝对路径的 .txt（相对工作区）",
    )
    ap.add_argument("--out", default=None, help="输出目录，默认 submissions/<weights 名>")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="0 表示 GPU，cpu 表示纯 CPU")
    ap.add_argument("--augment", action="store_true", help="TTA（多尺度+翻转）")
    ap.add_argument("--zip", action="store_true", help="打包 zip")
    ap.add_argument(
        "--chunk",
        type=int,
        default=100,
        help="分块推理的块大小。必须分块：一次性喂 1000 张会让 ultralytics 流式预测"
        "在某个张量上尝试分配 ~13 GB（CPU/GPU 同样复现），实测按 100 张分块可稳定通过。",
    )
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = WS / weights
    if not weights.exists():
        sys.exit(f"权重不存在: {weights}")

    source = Path(args.source)
    if not source.is_absolute():
        source = WS / source
    is_list = source.is_file() and source.suffix.lower() == ".txt"
    if not source.is_dir() and not is_list:
        sys.exit(f"测试目录/清单不存在: {source}")

    out_dir = Path(args.out) if args.out else WS / "submissions" / weights.stem
    if not out_dir.is_absolute():
        out_dir = WS / out_dir
    if out_dir.exists():
        # 只清理本脚本自己的输出目录，绝不触碰原始数据
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import os

    os.environ.setdefault("YOLO_CONFIG_DIR", str(WS / ".ultralytics"))
    os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

    from ultralytics import YOLO

    if is_list:
        images = [Path(ln.strip()) for ln in source.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    print(f"待推理图片数量: {len(images)}")
    print(f"权重: {weights}")
    print(f"输出: {out_dir}")
    print(f"imgsz={args.imgsz} conf={args.conf} iou={args.iou} max_det={args.max_det} augment={args.augment}")

    model = YOLO(str(weights))
    # 注意 1：不能把 .txt 清单路径直接交给 ultralytics —— 它内部用系统默认编码
    #   （中文 Windows 上是 GBK）读清单，遇到中文路径的 UTF-8 字节会抛 UnicodeDecodeError。
    #   因此统一传图片路径列表。
    # 注意 2：必须分块。一次性传入 1000 张时，ultralytics 流式预测会在某一刻尝试
    #   分配 13,107,200,000 字节（12.21 GiB），CPU 与 GPU 报同样的字节数，
    #   与 batch(=4/8)、置信度阈值无关；按 100 张分块则 100% 通过。
    n_boxes = 0
    n_empty = 0
    n_done = 0
    for start in range(0, len(images), args.chunk):
        chunk = images[start : start + args.chunk]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            augment=args.augment,
            batch=args.batch,
            device=args.device,
            stream=True,
            verbose=False,
            save=False,
        )
        for res in results:
            img_path = Path(res.path)
            stem = img_path.stem
            h, w = res.orig_shape[:2]
            boxes = res.boxes
            lines: list[str] = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()
                order = conf.argsort()[::-1][: args.max_det]
                for i in order:
                    x1, y1, x2, y2 = xyxy[i]
                    cw, ch = x2 - x1, y2 - y1
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    lines.append(
                        f"{int(cls[i])} {cx / w:.6f} {cy / h:.6f} {cw / w:.6f} {ch / h:.6f} {float(conf[i]):.6f}"
                    )
            if not lines:
                n_empty += 1
            n_boxes += len(lines)
            (out_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            n_done += 1
        print(f"  已完成 {n_done}/{len(images)}")

    print(f"完成：{len(images)} 张图，共 {n_boxes} 个预测框，其中 {n_empty} 张为空")

    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.glob("*.txt")):
                zf.write(p, arcname=p.name)
        print(f"已打包: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
