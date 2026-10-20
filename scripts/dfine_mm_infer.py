"""6 通道（可见光 + 红外/深度/掩码）D-FINE / MMDFINE 推理 + 3 视图 TTA → 提交格式。

## 为什么要单独写

训练侧（`D-FINE/src/mm_dfine_2026.py::MMCocoDetection.load_item`）喂给模型的 6 通道是：

    0-2  可见光 **RGB**（PIL 读，来自 ./data/images 的 BMP 缓存）
    3    红外灰度        （aux PNG 的 R 通道）
    4    深度归一化 8bit  （aux PNG 的 G 通道，无效像素 0）
    5    深度有效掩码     （aux PNG 的 B 通道，0/255）

而现成的 `dfine_infer.py` 只吃 RGB（`T.Resize` + `ToTensor`），喂给 MMDFINE 会在骨干第一层
直接 shape 报错；即便侥幸跑通，通道顺序错了也会**静默掉分**。
所以这里把训练侧的通道构造**逐行复刻**（同一个 `mm_build.build_mm_array`，
同一份深度解码口径 dmin=300 / dmax=20000 / linear），只把「读缓存 PNG」换成
「读缓存 PNG **或** 现算（原始红外/深度目录）」两种模式。

## 与训练侧严格对齐的三点

1. **拼接顺序**：先拼 6 通道 uint8，**再** Resize，**再** /255（= MMCocoDetection → Resize → ScaleMMImage）；
2. **aux 尺寸**：与可见光不一致时按 BILINEAR 缩放到可见光尺寸（训练侧同一逻辑）；
3. **翻转**：TTA 的水平翻转作用于**全部 6 通道**（训练侧 RandomHorizontalFlip 同样作用于整张 6 通道张量）。

## 用法

    # ① 在 val300 上自证（与训练日志里自己报的 val mAP 对齐，量级一致才说明通道构造正确）
    python scripts/dfine_mm_infer.py --config configs/dfine/custom/dfine_hgnetv2_s_aic_mm.yml \
        --weights output/dfine_hgnetv2_s_aic_mm/best_stg1.pth \
        --list splits/lbproxy_val.txt --out submissions/_mm_eval --aux-dir /dev/shm/aic_aux \
        --imgsz 1280 --scales 1.0,1.0,0.83 --flips none,lr,none

    # ② 复赛测试集打包（aux 现算，用赛题给的 phase2 红外/深度）
    python scripts/dfine_mm_infer.py ... --list splits/phase2_test_all_srv.txt \
        --out submissions/_mm_phase2 --ir ./data/phase2_infrared \
        --dp ./data/phase2_depth
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src"))
sys.path.insert(0, str(WS / "scripts"))
sys.path.insert(0, str(WS / "src" / "mm"))

from local_tile_infer import nms  # noqa: E402

D_IR_DIR = "./data/phase2_infrared"
D_DP_DIR = "./data/phase2_depth"


def build_aux(rgb_path: Path, aux_dir: str | None, ir_dir: str | None, dp_dir: str | None) -> np.ndarray:
    """返回 HxWx3 uint8 的 [IR, depth8, mask]（与训练侧 aux 缓存同源）。"""
    if aux_dir:
        ap = Path(aux_dir) / rgb_path.name
        if not ap.exists():
            raise FileNotFoundError(f"aux 缓存缺图：{ap}")
        return np.asarray(PILImage.open(ap).convert("RGB"), dtype=np.uint8)
    # 现算：必须用与缓存**同一实现**（./src/mm/mm_build.py）
    from mm_build import build_mm_array

    arr = build_mm_array(
        rgb_path,
        (ir_dir or D_IR_DIR, dp_dir or D_DP_DIR),
        ("ir", "depth"),
        300.0,
        20000.0,
        "linear",
        tag="mm-infer",
    )
    return np.ascontiguousarray(arr[:, :, 3:6])


def load_array6(rgb_path: Path, aux_dir: str | None, ir_dir: str | None, dp_dir: str | None) -> np.ndarray:
    """构造 6 通道 uint8 `HxWx6`：RGB(可见光) + [IR, depth8, mask]。"""
    vis = np.asarray(PILImage.open(rgb_path).convert("RGB"), dtype=np.uint8)
    aux = build_aux(rgb_path, aux_dir, ir_dir, dp_dir)
    if aux.shape[:2] != vis.shape[:2]:
        aux = np.asarray(
            PILImage.fromarray(aux).resize((vis.shape[1], vis.shape[0]), PILImage.BILINEAR),
            dtype=np.uint8,
        )
    return np.ascontiguousarray(np.concatenate([vis, aux], axis=2))


def load_model(dfine_root: Path, cfg_path: str, weights: str, device: str):
    sys.path.insert(0, str(dfine_root))
    import torch.nn as nn

    from src.core import YAMLConfig

    # ★ MMDFINE 的注册由 coco_dataset 末尾的一行 import 触发（训练侧也是这么挂的），
    #   这里显式再导一次，避免只看模型配置时注册不到。
    try:
        import importlib

        importlib.import_module("src.mm_dfine_2026")
    except Exception as exc:  # noqa: BLE001
        print(f"[mm] 显式导入 mm 模块失败（可能已由 coco_dataset 注册）：{type(exc).__name__}: {exc}")

    cfg = YAMLConfig(str(cfg_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    if Path(str(weights)).name.lower() in ("none", "", "null"):
        print("[mm] 未加载权重（冒烟模式）")
    else:
        ckpt = torch.load(weights, map_location="cpu")
        if "ema" in ckpt and ckpt["ema"]:
            state = ckpt["ema"]["module"] if "module" in ckpt["ema"] else ckpt["ema"]
            print("[mm] 用 EMA 权重")
        else:
            state = ckpt["model"]
        cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    return Model().to(device).eval()


@torch.no_grad()
def infer_one(model, arr6: np.ndarray, size: int, flip: bool, device: str):
    """单视图：arr6 为 uint8 HxWx6；返回原图坐标的框/类别/分数。"""
    h, w = arr6.shape[:2]
    a = arr6[:, ::-1] if flip else arr6
    x = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0).float()  # 1,6,H,W
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    x = (x / 255.0).to(device)
    orig = torch.tensor([[w, h]], device=device)
    labels, boxes, scores = model(x, orig)
    return (
        boxes[0].cpu().numpy().astype(np.float32),
        labels[0].cpu().numpy().astype(int),
        scores[0].cpu().numpy().astype(np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--scales", default="1.0,1.0,0.83")
    ap.add_argument("--flips", default="none,lr,none")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--aux-dir", default="")
    ap.add_argument("--ir", default="")
    ap.add_argument("--dp", default="")
    args = ap.parse_args()

    dfine_root = WS / "D-FINE" if (WS / "D-FINE" / "train.py").exists() else WS / "reference" / "D-FINE"
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        # 容错：允许传 "reference\D-FINE\configs\..."（自动剥离前缀），
        # 也允许传相对 dfine_root 的 "configs\..."（历史用法）。
        _s = str(cfg_path).replace("\\", "/")
        _pfx = "reference/D-FINE/"
        if _s.lower().startswith(_pfx.lower()):
            _s = _s[len(_pfx):]
        cfg_path = dfine_root / _s
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

    if not args.aux_dir and not (args.ir and args.dp):
        raise SystemExit("必须给 --aux-dir（缓存）或 --ir+--dp（现算）之一")

    scales = [float(x) for x in args.scales.split(",")]
    flips = args.flips.split(",")
    images = [Path(x.strip()) for x in lst.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        images = images[: args.limit]
    print(f"[mm] 配置 {cfg_path}\n[mm] 权重 {weights}")
    print(f"[mm] {len(images)} 张  imgsz={args.imgsz}  视图={list(zip(scales, flips))}  "
          f"aux={'缓存 ' + args.aux_dir if args.aux_dir else '现算'}")

    os.chdir(dfine_root)
    model = load_model(dfine_root, str(cfg_path), str(weights), args.device)
    t0 = time.time()
    n_boxes = n_empty = 0
    for k, p in enumerate(images, 1):
        arr6 = load_array6(p, args.aux_dir or None, args.ir or None, args.dp or None)
        w = arr6.shape[1]
        allB, allL, allS = [], [], []
        for s, fl in zip(scales, flips):
            raw = args.imgsz * s
            size = max(64, int(round(raw / 64.0)) * 64)
            B, L, S = infer_one(model, arr6, size, fl == "lr", args.device)
            if len(B) == 0:
                continue
            if fl == "lr":
                x1 = w - B[:, 2]
                x2 = w - B[:, 0]
                B = np.stack([x1, B[:, 1], x2, B[:, 3]], axis=1)
            allB.append(B)
            allL.append(L)
            allS.append(S)
        if not allB:
            (out_dir / f"{p.stem}.txt").write_text("", encoding="utf-8")
            n_empty += 1
            continue
        B = np.concatenate(allB)
        L = np.concatenate(allL)
        S = np.concatenate(allS)
        m = S >= args.conf
        B, L, S = B[m], L[m], S[m]
        h = arr6.shape[0]
        if len(B):
            B[:, 0] = np.clip(B[:, 0], 0.0, w)
            B[:, 1] = np.clip(B[:, 1], 0.0, h)
            B[:, 2] = np.clip(B[:, 2], 0.0, w)
            B[:, 3] = np.clip(B[:, 3], 0.0, h)
            ok = (B[:, 2] > B[:, 0]) & (B[:, 3] > B[:, 1])
            B, L, S = B[ok], L[ok], S[ok]
        keep = []
        for c in np.unique(L):
            idx = np.where(L == c)[0]
            keep.extend(idx[nms(B[idx], S[idx], args.nms_iou)].tolist())
        keep = np.array(keep, dtype=int)
        order = keep[np.argsort(-S[keep])][: args.max_det]
        lines = []
        for i in order:
            x1, y1, x2, y2 = B[i]
            lines.append(f"{int(L[i])} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                         f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f} {float(S[i]):.6f}")
        (out_dir / f"{p.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_boxes += len(lines)
        if k % 200 == 0:
            print(f"[mm]   {k}/{len(images)}  {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"[mm] 完成 {len(images)} 张，共 {n_boxes} 框，空文件 {n_empty}，"
          f"用时 {(time.time()-t0)/60:.1f} 分钟 → {out_dir}")


if __name__ == "__main__":
    main()
