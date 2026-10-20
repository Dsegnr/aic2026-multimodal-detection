"""D-FINE 推理 + 多视图 TTA → 我们的提交格式。

## 为什么必须单独写

RT-DETR 那套（`detr_tta_infer.py`）走的是 **ultralytics 的推理接口**；
而 D-FINE 是独立仓库，用自己的 `src.core.YAMLConfig` + `model(im_data, orig_size)`。
两者**不通用**。而且 D-FINE 的 img 变换是 **Resize 到正方形（会改变长宽比）**，
不是 letterbox —— 这一点也必须跟训练配置对齐。

## TTA 与 RT-DETR 那套保持一致（同一类操作，非多模型集成）

    视图 V1 = 原图 @ size S
    视图 V2 = 水平翻转 @ size S（框翻回来）
    视图 V3 = 原图 @ size 0.83·S
    合并   = 按类 NMS

## 输出

与官方提交同格式的 TXT：`class cx cy w h conf`（归一化）。

用法：
    python scripts/dfine_infer.py --config <cfg.yml> --weights <ckpt.pth> \
        --list splits/lbproxy_val.txt --out submissions/_dfine_tta --imgsz 1280
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src" / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_tile_infer import nms  # noqa: E402


def load_model(dfine_root: Path, cfg_path: str, weights: str, device: str):
    sys.path.insert(0, str(dfine_root))
    import torch.nn as nn

    from src.core import YAMLConfig

    cfg = YAMLConfig(str(cfg_path))
    # 官方推理脚本的做法：关掉 HGNetv2 的预训练下载。
    # （我们后面会用 checkpoint 覆盖全部权重，所以主干初始值无所谓；
    #   关掉它能避免在无网环境里卡在下载上。）
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    # --weights none：只验证"管线通不通"（配置→建模→推理→TTA 合并→输出格式），
    # 不加载权重。用来做冒烟测试——因为 COCO 的 80 类权重装不进 12 类模型，
    # 分类头形状不匹配会直接抛异常。
    if Path(str(weights)).name.lower() in ("none", "", "null"):
        print("[dfine] 未加载权重（冒烟模式），输出为随机结果，仅用于验证管线")
    else:
        ckpt = torch.load(weights, map_location="cpu")
        if "ema" in ckpt and ckpt["ema"]:
            state = ckpt["ema"]["module"] if "module" in ckpt["ema"] else ckpt["ema"]
            print("[dfine] 用 EMA 权重")
        else:
            state = ckpt["model"]
        cfg.model.load_state_dict(state)

    # ★ 必须走官方推理范式：deploy() + postprocessor.deploy()
    #   直接 model(images, orig_size) 会返回 (outputs, aux) 两元组，解包成三个就报错。
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    return Model().to(device).eval()


@torch.no_grad()
def infer_one(model, pil: Image.Image, size_h: int, size_w: int, device: str):
    """单视图推理，返回 boxes(原图坐标) / labels / scores。"""
    w, h = pil.size
    orig = torch.tensor([[w, h]], device=device)
    tf = T.Compose([T.Resize((size_h, size_w)), T.ToTensor()])
    x = tf(pil).unsqueeze(0).to(device)
    labels, boxes, scores = model(x, orig)
    labels = labels[0].cpu().numpy().astype(int)
    boxes = boxes[0].cpu().numpy().astype(np.float32)
    scores = scores[0].cpu().numpy().astype(np.float32)
    return boxes, labels, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--rect-h", type=int, default=0,
                    help="矩形推理高度（32 的倍数）；>0 时与 --rect-w 一起启用矩形口径")
    ap.add_argument("--rect-w", type=int, default=0,
                    help="矩形推理宽度（32 的倍数）；>0 时与 --rect-h 一起启用矩形口径")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--scales", default="1.0,1.0,0.83")
    ap.add_argument("--flips", default="none,lr,none")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（冒烟测试用）")
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

    scales = [float(x) for x in args.scales.split(",")]
    flips = args.flips.split(",")
    images = [Path(x.strip()) for x in lst.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        images = images[: args.limit]
    print(f"[dfine] 仓库 {dfine_root}")
    print(f"[dfine] 配置 {cfg_path}\n[dfine] 权重 {weights}")
    print(f"[dfine] {len(images)} 张  基准 imgsz={args.imgsz}  视图={list(zip(scales, flips))}\n")

    # ★ 必须切到 D-FINE 根目录：它按**相对路径**找 HGNetV2 主干预训练权重
    #   （weight/hgnetv2/...），相对的是 CWD 而不是包路径。
    #   服务器上的 dfine_run.py 里有 chdir 所以没事；这里漏了就会一直重试下载。
    os.chdir(dfine_root)
    model = load_model(dfine_root, str(cfg_path), str(weights), args.device)
    t0 = time.time()
    n_boxes = n_empty = 0
    for k, p in enumerate(images, 1):
        pil = Image.open(p).convert("RGB")
        w, h = pil.size
        allB, allL, allS = [], [], []
        for s, fl in zip(scales, flips):
            im = pil.transpose(Image.FLIP_LEFT_RIGHT) if fl == "lr" else pil
            # ★ 输入边长必须取整到 **64 的倍数**：
            #   D-FINE 的 FPN 是"深层上采样 2 倍再与浅层 concat"，
            #   一旦某个尺度出现奇数尺寸（例如 0.83×1280=1062），
            #   上采样后 68 与浅层 67 就对不上，直接
            #   RuntimeError: Expected size 68 but got size 67。
            base_h = args.rect_h if args.rect_h > 0 else args.imgsz
            base_w = args.rect_w if args.rect_w > 0 else args.imgsz
            raw_h, raw_w = base_h * s, base_w * s
            size_h = max(32, int(round(raw_h / 32.0)) * 32)
            size_w = max(32, int(round(raw_w / 32.0)) * 32)
            B, L, S = infer_one(model, im, size_h, size_w, args.device)
            if len(B) == 0:
                continue
            if fl == "lr":                      # 框翻回来
                x1 = w - B[:, 2]
                x2 = w - B[:, 0]
                B = np.stack([x1, B[:, 1], x2, B[:, 3]], axis=1)
            allB.append(B); allL.append(L); allS.append(S)
        if not allB:
            (out_dir / f"{p.stem}.txt").write_text("", encoding="utf-8")
            n_empty += 1
            continue
        B = np.concatenate(allB); L = np.concatenate(allL); S = np.concatenate(allS)
        m = S >= args.conf
        B, L, S = B[m], L[m], S[m]
        # ★ 必须先**裁到图像范围内**再归一化：
        #   D-FINE 的 postprocessor 会给出略微越界的框（实测最多到 w=1.0054、h=1.0070），
        #   而队长那边的自动提交没有人工复核，我们的 check_submission.py 会把越界判为 FAIL。
        #   裁框对 mAP 只可能有利（GT 一定在图像内），不会引入新风险。
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

    print(f"[dfine] 完成 {len(images)} 张，共 {n_boxes} 框，空文件 {n_empty}，"
          f"用时 {(time.time()-t0)/60:.1f} 分钟 → {out_dir}")


if __name__ == "__main__":
    main()
