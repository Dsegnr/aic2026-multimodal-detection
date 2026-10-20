"""三模态（6 通道）推理与提交包生成 —— 补上"训练有、推理没有"的那条路径。

【为什么必须补】
  训练侧 `src/mm/mm_patch.py` 的 `MultiModalYOLODataset.load_image` 会把
      BGR(3) + 红外灰度(1) + 深度归一化(1) + 深度有效掩码(1) = 6 通道
  拼好喂给模型；训练日志里能看到 stem 被扩展成 `[6, 48, 3, 2]`。
  但推理侧的脚本（local_tile_infer / predict_submit_tta）都是 `cv2.imread` 读 3 通道，
  而 ultralytics 的 `model.predict(path)` 会自己重读图、**把辅助通道丢掉**。
  ⇒ 后果有两个：① 本地无法给三模态模型打提交口径的分；
                ② 复赛要求代码能复现榜单结果（细则十三·4）+ 算法须读取三模态（细则十·一）。

【通道一致性怎么保证】
  * 通道构造**逐行复刻** `mm_patch._load_extra`：同样的读图 flag、同样的深度有效区间
    [300, 20000] mm、同样的线性归一化、无效置 0、并显式产出掩码通道。
  * BGR→RGB 的翻转：ultralytics 预测侧是 `if im.shape[1] == 3: im.flip(1)`，
    训练侧是 `img[::-1] if ... and img.shape[0] == 3` —— **两边都只对 3 通道生效**，
    所以 6 通道时两边都不翻转，天然一致（这一条我读过源码确认过）。
  * 手法：把 `ultralytics.data.loaders.imread` 换成我们的 6 通道构造器，
    其余（letterbox / TTA / NMS / 坐标还原）全部沿用 ultralytics 原路径。

【自证方式】在有 GT 的 val300 上跑一遍，与训练时自己报的 val mAP50-95 对比；
量级一致才说明通道构造正确。（深度通道只要编码不一致，分数会静默崩掉而且不报错。）

用法：
  & $py scripts\\mm_infer.py --weights weights\\current\\mm6_y8m_200.pt ^
        --list splits\\lbproxy_val.txt --out submissions\\_mm_val ^
        --gt-dir data\\valgt_lbproxy --imgsz 1280
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
sys.path.insert(0, str(WS / "src" / "mm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_tile_infer import nms  # noqa: E402
from imgio import imread as imread_safe  # noqa: E402

import cv2  # noqa: E402


def build_mm6(rgb_path: Path, dmin: float = 300.0, dmax: float = 20000.0,
              depth_mode: str = "linear", extra_dirs: tuple = (),
              kinds: tuple = ("ir", "depth")) -> np.ndarray:
    """6 通道构造 —— **转发到 `mm_build.build_mm_array`**（训练/增强/推理唯一真源）。

    2026-09-23 改：原先这里有一份"逐行复刻"的实现，与训练侧各写一遍。
    两份实现漂移直接导致了"深度通道恒为全 0"的事故（8bit 数据套毫米阈值）。
    现在只保留转发，杜绝再次分叉。
    """
    from mm_build import build_mm_array

    return build_mm_array(
        rgb_path, extra_dirs=extra_dirs, kinds=kinds,
        dmin=dmin, dmax=dmax, encoding=depth_mode, tag="infer",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", required=True, help="图片清单 txt（可见光）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-dir", default=None, help="给了就顺便评测")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-augment", action="store_true", help="关掉 TTA")
    ap.add_argument("--depth-min", type=float, default=300.0)
    ap.add_argument("--depth-max", type=float, default=20000.0)
    ap.add_argument("--depth-mode", default="linear", choices=["linear", "inverse", "log"])
    ap.add_argument("--data-yaml", default=None,
                    help="训练用的 data yaml。给了就以它为准（辅助模态目录/编码与训练逐字一致）；"
                         "强烈建议总是带上它，否则只能靠路径段猜模态目录")
    args = ap.parse_args()

    # 模态目录与深度编码：**优先从训练 yaml 读**，保证与训练完全同源。
    extra_dirs: tuple = ()
    kinds: tuple = ()
    dmin, dmax, dmode = args.depth_min, args.depth_max, args.depth_mode
    if args.data_yaml:
        from mm_build import load_mm_config

        cfg = load_mm_config(args.data_yaml)
        extra_dirs, kinds = cfg["extra_dirs"], cfg["kinds"]
        dmin, dmax, dmode = cfg["dmin"], cfg["dmax"], cfg["encoding"]
        print(f"[cfg] 来自 {args.data_yaml}: kinds={kinds} dirs={extra_dirs} "
              f"depth=({dmin:g},{dmax:g},{dmode}) channels={cfg['channels']}", flush=True)

    # 自定义类 checkpoint 需要先 import 对应模块，否则反序列化失败 → 一路空预测、静默 0 分
    for mod in ("elong", "mm_multistage", "mm_multistage_rtdetr", "moe_fusion", "fusion", "mm_patch"):
        try:
            __import__(mod)
        except Exception:
            pass

    weights = Path(args.weights)
    use_rtdetr = "rtdetr" in weights.name.lower()
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
    # ⚠️ 不能靠"替换 ultralytics 的读图函数"：它对一批图片会用 multiprocessing 池，
    #    子进程里没有我们的补丁（实测报错 "expected input[1,3,...] to have 6 channels"）。
    #    正确做法是**自己构造 6 通道数组后直接喂给 predict**——ultralytics 的
    #    LoadPilAndNumpy 接受 ndarray，letterbox / TTA / NMS / 坐标还原全部照旧。
    #    另：预测侧只在 `im.shape[1] == 3` 时才做 BGR→RGB 翻转，6 通道天然不翻转，
    #    与训练侧 `img.shape[0] == 3` 的条件一致。
    print(f"多模态通道：BGR(3)+{'红外(1)+深度(1)+掩码(1)' if kinds == ('ir', 'depth') else kinds}"
          f"｜深度口径 ({dmin:g},{dmax:g},{dmode})｜imgsz={args.imgsz} TTA={not args.no_augment}\n",
          flush=True)

    if use_rtdetr:
        from ultralytics import RTDETR

        print("[arch] RT-DETR 模式", flush=True)
        model = RTDETR(str(weights))
    else:
        from ultralytics import YOLO

        print("[arch] YOLO 模式", flush=True)
    model = YOLO(str(weights))
    t0 = time.time()
    n_fail, n_boxes, n_empty, n_over = 0, 0, 0, 0
    for k, img_path in enumerate(images, 1):
        try:
            arr = build_mm6(img_path, dmin, dmax, dmode, extra_dirs, kinds)
        except Exception as e:
            print(f"  !! 构造 6 通道失败 {img_path.name}: {type(e).__name__}: {e}", flush=True)
            n_fail += 1
            (out_dir / f"{img_path.stem}.txt").write_text("", encoding="utf-8")
            continue
        h, w = arr.shape[:2]
        # RT-DETR 的 `augment=True`（ultralytics 内置 TTA）在 8.4 上不被支持；
        # DETR 族的 TTA 一律走我们自己的多视图脚本 `detr_tta_infer.py`。
        res = model.predict(
            source=[arr], imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            max_det=300, augment=(not args.no_augment) and not use_rtdetr, batch=4,
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
        keep: list[int] = []
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

    if args.gt_dir:
        from map_eval import evaluate

        gt = Path(args.gt_dir)
        if not gt.is_absolute():
            gt = WS / gt
        print(evaluate(gt, out_dir, max_boxes=args.max_det).summary())


if __name__ == "__main__":
    main()
