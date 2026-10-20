"""AIC2026 训练入口。

用脚本而不是一行式命令，避免中文路径在 PowerShell 转义中损坏。

示例：
    .venv\\Scripts\\python.exe scripts\\train.py --model weights\\yolov8s.pt \\
        --data configs\\aic_lbproxy.yaml --epochs 100 --imgsz 640 --batch 16 --name rgb_y8s_640
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/yolov8s.pt", help="本地权重路径或模型名")
    ap.add_argument("--data", default="configs/aic_lbproxy.yaml", help="数据集 yaml（相对工作区）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="run")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--cache", default=None, help="ram / disk / None")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="其余透传给 ultralytics 的 key=value")
    args = ap.parse_args()

    # ultralytics 会把设置和字体缓存写到用户目录；统一收到工作区内，避免沙箱写权限问题
    os.environ.setdefault("YOLO_CONFIG_DIR", str(WS / ".ultralytics"))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # polars 的 CPU 特性自检在本机误报 'unknown feature flag: sse3'，
    # 会在训练结束保存模型时抛 RuntimeError，直接跳过该检查。
    os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = WS / model_path
    if model_path.suffix and not model_path.exists():
        sys.exit(f"权重不存在: {model_path}（请先运行 scripts/download_weights.py）")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = WS / data_path

    overrides = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(WS / "runs"),
        "name": args.name,
        "seed": args.seed,
        "patience": args.patience,
        "exist_ok": True,
        "resume": args.resume,
    }
    if args.cache:
        overrides["cache"] = args.cache

    for kv in args.extra or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    v = {"true": True, "false": False, "none": None}.get(v.lower(), v)
            overrides[k] = v

    print("训练配置:")
    for k, v in overrides.items():
        print(f"  {k} = {v}")

    model = YOLO(str(model_path))
    model.train(**overrides)

    metrics = model.val()
    print("\n验证结果:")
    print(f"  mAP@50-95 = {metrics.box.map:.4f}")
    print(f"  mAP@50    = {metrics.box.map50:.4f}")
    print(f"  mAP@75    = {metrics.box.map75:.4f}")


if __name__ == "__main__":
    main()
