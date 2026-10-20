"""服务器端多模态训练入口（ultralytics 8.4 + 自研多模态数据管线）。

与本地 train.py 的区别：
1. 支持 channels>3（RGB+IR[+Depth]），由 data yaml 的 `channels` 与
   `extra_modality_dirs` 驱动；
2. 用 `expand_stem_weights` 把 COCO 预训练的第一层扩展成多通道，
   避免 ultralytics 因形状不匹配而丢弃整个 stem；
3. 支持显存占用上限，便于与同卡上正在跑的科研实验共存。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "mm"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "weights" / "yolov8s.pt"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default=str(ROOT / "runs"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--gpu-frac", type=float, default=0.0,
                    help="显存占用上限（0~1），0 表示不限；用于与同卡其他任务共存")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="使用训练集的比例（冒烟测试用 0.03 这类小值）")
    ap.add_argument("--fusion", default="none", choices=["none", "dualstem", "moe", "multistage"],
                    help="dualstem = 非对称双分支+门控；moe = 按模态分专家+逐像素路由；"
                         "multistage = 多阶段融合（模态各自编码，P2/P3/P4 三处注入，见 src/mm/mm_multistage.py）")
    ap.add_argument("--ms-drop", type=float, default=0.15,
                    help="multistage 的辅助模态 dropout 概率")
    ap.add_argument("--ms-rgb-drop", type=float, default=0.0,
                    help="multistage 的 RGB dropout 概率")
    ap.add_argument("--ms-mode", default="diff", choices=["diff", "add"],
                    help="multistage 的融合方式：diff=门控差分（背景相消、目标凸显）；add=直接相加")
    ap.add_argument("--ms-no-bidir", action="store_true",
                    help="关掉双向更新（融合特征只回加到 RGB 分支，作为消融对照）")
    ap.add_argument("--ms-lr-mult", type=float, default=3.0,
                    help="多阶段融合里**新加模块**的学习率倍数（预训练主干保持 1×）；"
                         "1.0 表示不启用差分学习率")
    ap.add_argument("--elong", action="store_true",
                    help="注入面向细长目标（bicycle/sign/uav）的各向异性卷积模块，见 src/mm/elong.py")
    ap.add_argument("--elong-k", type=int, default=11, help="横竖一维大核的长度")
    ap.add_argument("--elong-points", default="4,6", help="注入位置（backbone 层号，逗号分隔）")
    ap.add_argument("--elong-lr-mult", type=float, default=3.0, help="新模块学习率倍数")
    ap.add_argument("--moe-mode", default="residual", choices=["residual", "softmax"],
                    help="residual = 残差式逐模态门控（起点严格等于 RGB 基线，保守）；"
                         "softmax = 竞争式 softmax 路由（更接近标准 MoE，但有路由塌缩风险）")
    ap.add_argument("--moe-drop", type=float, default=0.15,
                    help="训练时按此概率把某个辅助模态整体置零（防塌缩 + 满足细则鲁棒性要求）")
    ap.add_argument("--moe-rgb-drop", type=float, default=0.0,
                    help="训练时按此概率把 RGB 整体置零（更激进，逼网络用辅助模态）")
    ap.add_argument("--lam-init", type=float, default=0.0,
                    help="DualStem 的 λ 初始值；0 表示从纯 RGB 基线出发")
    ap.add_argument("--p2", action="store_true",
                    help="加 P2（stride 4）检测层级：按 yolov8-p2.yaml 构建并迁移预训练权重")
    ap.add_argument("--p2-scale", default="s", help="P2 模型的 scale（n/s/m/l/x）")
    ap.add_argument("--extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    import torch

    if args.gpu_frac and args.gpu_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_frac, device=0)
        print(f"[gpu] 显存上限设为 {args.gpu_frac:.0%}")

    import yaml

    data_cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    channels = int(data_cfg.get("channels", 3))
    print(f"[data] channels = {channels}, extra = {data_cfg.get('extra_modality_dirs')}")

    _rare = bool((data_cfg.get("rare_paste") or {}).get("enable"))
    # 按文件名自动识别模型族：rtdetr-* 走 RTDETR，其余走 YOLO。
    # （RT-DETR 用的是 RTDETRDataset，不是 YOLODataset，所以数据侧补丁要分别挂。）
    #
    # ★ 2026-09-24 修复：**续训/微调时模型路径是 `last.pt`/`best.pt`，文件名里没有 "rtdetr"**
    #   于是被误判成 YOLO ⇒ 用 v8DetectionLoss 装到 RT-DETR 上 ⇒
    #   `AttributeError: 'RTDETRDecoder' object has no attribute 'stride'`（实测踩到，
    #   整条续训臂白崩一次）。现在改成"文件名优先，其次读 checkpoint 里的模型类名"。
    def _looks_rtdetr(model_path: str) -> bool:
        name = Path(model_path).name.lower()
        if "rtdetr" in name:
            return True
        p = Path(model_path)
        if p.suffix not in (".pt", ".pth") or not p.is_file():
            return False
        try:
            import torch as _torch

            ck = _torch.load(str(p), map_location="cpu", weights_only=False)
            m = ck
            if isinstance(ck, dict):
                m = ck.get("ema") if ck.get("ema") is not None else ck.get("model")
            cls = type(m).__name__.upper() if m is not None else ""
            return "RTDETR" in cls
        except Exception as e:  # noqa: BLE001
            print(f"[arch] 读 checkpoint 判族失败（按文件名处理）: {type(e).__name__}: {e}")
            return False

    _is_rtdetr = _looks_rtdetr(args.model)
    print(f"[arch] 模型族判定: {'RT-DETR' if _is_rtdetr else 'YOLO'}（来源 {args.model}）")
    if channels > 3 or _rare:
        import mm_patch

        mm_patch.install(channels)
        if _is_rtdetr:
            mm_patch.install_rtdetr()
        print(f"[mm] 已安装自定义数据管线（channels={channels}, rare_paste={_rare}, "
              f"arch={'rtdetr' if _is_rtdetr else 'yolo'}）")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

    if _is_rtdetr:
        from ultralytics import RTDETR

        model = RTDETR(args.model)
        print(f"[arch] RT-DETR 模式：{args.model}")
    else:
        from ultralytics import YOLO

        model = YOLO(args.model)
    trainer_cls = None

    if args.elong:
        import elong as _elong

        pts = tuple(int(x) for x in args.elong_points.split(",") if x.strip())
        trainer_cls = _elong.make_trainer(points=pts, k=args.elong_k, lr_mult=args.elong_lr_mult)
        print(f"[Elong] 已启用各向异性卷积（位置 {pts}，核长 {args.elong_k}，"
              f"新模块学习率×{args.elong_lr_mult}）")

        def _log_elong(trainer):
            try:
                md = getattr(trainer.model, "elong", None)
                if md is None:
                    return
                vals = []
                for key, blk in md.items():
                    r = getattr(blk, "last_res_ratio", None)
                    vals.append(f"L{key}={r.item():.3f}" if r is not None else f"L{key}=n/a")
                print("    [Elong] 残差/主特征 量级比: " + "  ".join(vals), flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    [Elong] 统计读取失败: {e}", flush=True)

        model.add_callback("on_train_epoch_end", _log_elong)

    if args.p2:
        import p2_inject

        trainer_cls = p2_inject.make_trainer(args.model, scale=args.p2_scale)
        print(f"[P2] 已启用 P2 检测层级（scale={args.p2_scale}）")

    if channels > 3 and args.fusion == "dualstem":
        # 非对称融合：RGB 分支完整继承 COCO 权重，辅助模态走独立分支，门控 + λ=0 起步
        import fusion

        trainer_cls = fusion.make_trainer(lam_init=args.lam_init, ckpt_path=args.model)
        print(f"[fusion] DualStem 已启用（λ_init={args.lam_init}）")

        def _log_lambda(trainer):
            """每轮打印 tanh(λ) —— 判断"辅助模态到底有没有被用起来"的唯一仪表。

            ★ 2026-09-23 修正：不再假设 DualStem 挂在 `model.model[0]`
            （RT-DETR 挂在 `model.0.stem1`），改用 fusion.find_dual_stem() 全模型搜索；
            同时把 EMA 里的 λ 也打出来（EMA 是实际被评分的权重）。
            """
            try:
                ds = fusion.find_dual_stem(trainer.model)
                msg = ""
                if ds is not None:
                    msg = f"tanh(λ) = {torch.tanh(ds.lam).item():+.5f}"
                ema = getattr(trainer, "ema", None)
                if ema is not None:
                    ds_e = fusion.find_dual_stem(ema.ema)
                    if ds_e is not None:
                        msg += f" | EMA tanh(λ) = {torch.tanh(ds_e.lam).item():+.5f}"
                if msg:
                    print(f"    [DualStem] {msg}", flush=True)
                else:
                    print("    [DualStem][!!] 模型里找不到 DualStem —— 注入没生效！", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    [DualStem] λ 统计失败: {type(e).__name__}: {e}", flush=True)

        model.add_callback("on_train_epoch_end", _log_lambda)

    elif channels > 3 and args.fusion == "moe":
        import moe_fusion

        trainer_cls = moe_fusion.make_trainer(
            mode=args.moe_mode,
            drop_p=args.moe_drop,
            drop_rgb_p=args.moe_rgb_drop,
            ckpt_path=args.model,
        )
        print(f"[MoE] 已启用（mode={args.moe_mode}, aux_drop={args.moe_drop}, "
              f"rgb_drop={args.moe_rgb_drop}）")

        def _log_routing(trainer):
            """每轮打印路由权重——这是判断 MoE 是否真的在专化的唯一仪表。"""
            try:
                seq = trainer.model.model if hasattr(trainer.model, "model") else trainer.model
                stem = seq[0]
                m = getattr(stem, "last_g_mean", None)
                s = getattr(stem, "last_g_std", None)
                if m is None:
                    return
                names = ["RGB", "IR", "Depth+Mask"][: len(m)]
                means = "  ".join(f"{n}={v:.3f}" for n, v in zip(names, m.tolist()))
                stds = "  ".join(f"{n}±{v:.3f}" for n, v in zip(names, s.tolist())) if s is not None else ""
                print(f"    [MoE] mean(g): {means}   |  跨图 std: {stds}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    [MoE] 路由统计读取失败: {type(e).__name__}: {e}", flush=True)

        model.add_callback("on_train_epoch_end", _log_routing)

    elif channels > 3 and args.fusion == "multistage":
        # ★ 2026-09-23：按模型族选实现。mm_multistage 是 YOLO 专属（继承 DetectionModel、
        # 覆写 _predict_once）；RT-DETR 入口是 predict()、头是 RTDETRDecoder，
        # 必须走 mm_multistage_rtdetr（复用同一套 AuxStages/AGFFusion，融合点 3/7/9）。
        if _is_rtdetr:
            import mm_multistage_rtdetr as mm_ms

            print("[MM] 使用 RT-DETR 版多阶段融合（融合点 layer 3/7/9 = P3/P4/P5）")
        else:
            import mm_multistage as mm_ms

        trainer_cls = mm_ms.make_trainer(
            aux_in=channels - 3,
            drop_p=args.ms_drop,
            drop_rgb_p=args.ms_rgb_drop,
            fuse_mode=args.ms_mode,
            bidir=not args.ms_no_bidir,
            lr_mult=args.ms_lr_mult,
        )
        print(f"[MM] 多阶段三模态融合已启用（aux={channels - 3}ch, 辅助dropout={args.ms_drop}, "
              f"RGBdropout={args.ms_rgb_drop}, mode={args.ms_mode}, 双向={not args.ms_no_bidir}, "
              f"新模块学习率×{args.ms_lr_mult}）")

        def _log_mm(trainer):
            try:
                seq = trainer.model.model if hasattr(trainer.model, "model") else trainer.model
                fusions = getattr(trainer.model, "fusions", None)
                if fusions is None:
                    return
                vals = []
                for i, f in enumerate(fusions):
                    m = getattr(f, "last_gate_mean", None)
                    vals.append(f"P{2 + i}={m.item():.3f}" if m is not None else f"P{2 + i}=n/a")
                print(f"    [MM] 融合门均值: " + "  ".join(vals), flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    [MM] 门统计读取失败: {type(e).__name__}: {e}", flush=True)

        model.add_callback("on_train_epoch_end", _log_mm)

    elif channels > 3:
        # 朴素通道拼接：扩展预训练 stem 到多通道
        import mm_patch

        def _expand_stems(trainer):
            """把预训练 stem 扩展到多通道；**失败就中断**（不能带着随机 stem 跑 6 小时）。

            注意必须同时处理 `trainer.ema`：`on_pretrain_routine_end` 触发时
            EMA 已经创建（它是当时的权重快照），只改 model 会让 EMA 里的 stem
            停留在随机初始化上，早期验证分被系统性压低。
            """
            ok = mm_patch.expand_stem_weights(trainer.model, args.model)
            ema = getattr(trainer, "ema", None)
            if ema is not None:
                ok_ema = mm_patch.expand_stem_weights(ema.ema, args.model)
                if not ok_ema:
                    raise RuntimeError("EMA 的 stem 扩展失败")
            if not ok:
                raise RuntimeError(
                    "stem 权重扩展失败（见上方 [stem] 日志）。带着随机初始化的第一层训练 "
                    "等于把多模态实验变成假结论，因此直接中断。"
                )

        model.add_callback(
            "on_pretrain_routine_end",
            _expand_stems,
        )

    overrides = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "patience": args.patience,
        "fraction": args.fraction,
        "exist_ok": True,
    }
    for kv in args.extra or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    v = {"true": True, "false": False}.get(v.lower(), v)
            overrides[k] = v

    print("[train]", {k: overrides[k] for k in ("data", "epochs", "imgsz", "batch", "name")})
    if trainer_cls is not None:
        model.train(trainer=trainer_cls, **overrides)
    else:
        model.train(**overrides)

    metrics = model.val()
    print(f"\n最终 mAP@50-95 = {metrics.box.map:.4f}   mAP@50 = {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
