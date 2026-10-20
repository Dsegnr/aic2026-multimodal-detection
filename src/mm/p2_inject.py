"""给 COCO 预训练的 YOLOv8s 加一个 P2（stride 4）检测层级。

## 为什么做这个（依据）

EEG 项目 29 连负定律：**正增益只来自「去瓶颈 / 更好利用既有信息 / 训练校准」。**
他们最大的单项增益（Scout +3.23pp）来自「换掉限制 Scout 的前端」——是去瓶颈，不是加模块。

我们的检测器存在同构的瓶颈：
- 目标框归一化宽度中位数仅 **0.0375**（1280 输入下约 48 px，640 下仅 24 px）；
- 检测头只有 P3/P4/P5（stride 8/16/32），**P2（stride 4）整层缺失**；
- 48 px 的目标落在 stride 8 的 P3 上只剩 6 个格子，高 IoU 定位无从谈起。

## 技术难点与解法

ultralytics 没有 P2 版的 COCO 预训练权重，直接 `yolov8s-p2.yaml` 从零训在 2000 张图上会崩。
本模块做「外科手术式」迁移：把预训练权重按**语义对应**填进 P2 模型。

层索引映射（见 yolov8.yaml 与 yolov8-p2.yaml 对比）：

| p2 索引 | baseline 索引 | 说明 |
| --- | --- | --- |
| 0–15 | 0–15 | 主干与前三段 neck，结构完全相同 |
| 16–21 | — | 新增 P2 分支（Upsample/Concat/C2f/Conv/C2f） |
| 22–27 | 16–21 | 整体偏移 +6（因为前面插了 6 层） |
| 28 | 22 | Detect 头，3 分支 → 4 分支 |

新增层的初始化策略：
- 18 号 C2f(128) ← 主干 2 号 C2f(128)，形状相同，直接复制；
- 21 号 C2f(256) ← 原 head 15 号 C2f(256)，形状相同，直接复制；
- 19 号 Conv(128→128, s2) 无同形来源 → 保持默认初始化；
- Detect 新增的 P2 分支：第 2/3 层从 P3 分支复制（形状无关输入通道），
  第 1 层按输入通道切片复制（256 → 128，取前 128 个输入通道）。
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml

# 需要把 baseline 的哪一层搬到 p2 的哪一层（只列非恒等映射）
LAYER_MAP: dict[int, int] = {
    18: 2,    # 新增 P2 的 C2f(128) ← 主干 P2 的 C2f(128)
    21: 15,   # 新增 P3 的 C2f(256) ← 原 head 的 P3 C2f(256)
    22: 16,
    23: 17,
    24: 18,
    25: 19,
    26: 20,
    27: 21,
}


def resolve_yaml(name: str = "yolov8-p2.yaml") -> Path:
    """把配置名解析成 ultralytics 包内的绝对路径。"""
    p = Path(name)
    if p.is_file():
        return p
    import ultralytics

    cand = Path(ultralytics.__file__).parent / "cfg" / "models" / "v8" / name
    if cand.is_file():
        return cand
    # 兼容只给名字的情况：在整个 cfg/models 下搜
    for hit in (Path(ultralytics.__file__).parent / "cfg" / "models").rglob(name):
        return hit
    raise FileNotFoundError(f"找不到模型配置: {name}")


def build_p2_model(yaml_path: str | Path | None = None, nc: int = 12, scale: str = "s", ch: int = 3):
    """按 yolov8-p2.yaml 构建指定 scale 的检测模型（nc 可覆盖）。"""
    from ultralytics.nn.tasks import DetectionModel

    cfg_path = resolve_yaml(str(yaml_path or "yolov8-p2.yaml"))
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["nc"] = nc
    cfg["scale"] = scale
    model = DetectionModel(copy.deepcopy(cfg), ch=ch, nc=nc, verbose=True)
    model.yaml["scale"] = scale
    model.yaml["nc"] = nc
    return model


def _copy_matching(dst: torch.nn.Module, src: torch.nn.Module, tag: str) -> tuple[int, int]:
    """按 key+形状 严格匹配地复制参数，返回 (成功数, 跳过数)。"""
    sd_src = src.state_dict()
    sd_dst = dst.state_dict()
    ok = skip = 0
    for k, v in sd_dst.items():
        if k in sd_src and sd_src[k].shape == v.shape:
            v.copy_(sd_src[k])
            ok += 1
        else:
            skip += 1
    return ok, skip


def transfer_weights(p2_model, ckpt_path: str | Path, verbose: bool = True) -> dict:
    """把 yolov8s.pt（P3/P4/P5）的权重迁移到 P2 模型上。"""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    base = ckpt["model"]
    base_sd = base.state_dict()

    # 1) 先用 ultralytics 自带的 load 加载「同索引同形状」的层（含类别名重映射）
    p2_model.load(base)  # 内部走 intersect_dicts + cls_remap

    dst_seq = p2_model.model
    stats = {"direct": 0, "mapped": 0, "skipped": 0, "new_layers": []}

    # 2) 按语义映射搬非恒等的层
    for dst_i, src_i in LAYER_MAP.items():
        try:
            d, s = _copy_matching(dst_seq[dst_i], base.model[src_i], f"{dst_i}<-{src_i}")
            if d:
                stats["mapped"] += d
            if s:
                stats["skipped"] += s
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[P2] 映射 {dst_i}<-{src_i} 失败: {type(e).__name__}: {e}")

    # 3) Detect 头：把原 3 分支搬到后 3 个位置，新 P2 分支从 P3 分支派生
    d_det = dst_seq[-1]
    b_det = base.model[-1]
    for name in ("cv2", "cv3"):
        d_list = getattr(d_det, name)
        b_list = getattr(b_det, name)
        # 后 3 个分支 ← 原 3 个分支
        for offset, bi in enumerate(range(len(b_list))):
            di = len(d_list) - len(b_list) + offset
            d, s = _copy_matching(d_list[di], b_list[bi], f"detect.{name}[{di}]")
            stats["mapped"] += d
            stats["skipped"] += s
        # 新 P2 分支 ← 派生自 P3 分支（索引 0）
        src_branch = b_list[0]
        new_branch = d_list[0]
        with torch.no_grad():
            for k in new_branch.state_dict():
                dv = new_branch.state_dict()[k]
                sv = src_branch.state_dict()[k]
                if dv.shape == sv.shape:
                    dv.copy_(sv)
                elif dv.dim() == 4 and sv.dim() == 4 and dv.shape[0] == sv.shape[0]:
                    # 第一层卷积：输入通道数不同（P2 128 vs P3 256），取前若干输入通道
                    dv.copy_(sv[:, : dv.shape[1]])
                    stats["mapped"] += 1
        stats["new_layers"].append(f"Detect P2 分支 ← 派生自 P3 分支")

    stats["direct"] = len(base_sd)
    if verbose:
        print(
            f"[P2] 权重迁移完成：同索引加载 + 语义映射 {stats['mapped']} 项，"
            f"跳过 {stats['skipped']} 项；新增层：{stats['new_layers']}"
        )
    return stats


def verify_transfer(p2_model, ckpt_path: str | Path) -> dict:
    """独立校验：确认主干权重确实被搬进来了（而不是停留在随机初始化）。"""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    base = ckpt["model"]
    out = {}
    # 主干第 4 层（C2f）与映射层 24（← baseline 18）各查一项
    checks = [("model.4.cv2.conv.weight", 4, 4), ("model.24.cv2.conv.weight", 24, 18)]
    for name, di, si in checks:
        try:
            a = p2_model.model[di].state_dict()
            key = [k for k in a if k.endswith("cv2.conv.weight")][0]
            b = base.model[si].state_dict()[key]
            out[name] = float((a[key] - b).abs().max().item())
        except Exception as e:  # noqa: BLE001
            out[name] = f"校验失败: {e}"
    return out


def make_trainer(ckpt_path: str | Path, yaml_path: str = "yolov8-p2.yaml", scale: str = "s"):
    """返回一个在 get_model 里自动构建 P2 模型并迁移权重的 DetectionTrainer 子类。"""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class P2DetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            nc = self.data["nc"]
            model = build_p2_model(yaml_path, nc=nc, scale=scale, ch=self.data.get("channels", 3))
            transfer_weights(model, ckpt_path, verbose=verbose)
            # 训练器与损失函数依赖这两个属性
            model.nc = nc
            model.names = self.data["names"]
            diffs = verify_transfer(model, ckpt_path)
            print(f"[P2] 迁移校验（应与 0 接近）: {diffs}")
            return model

    return P2DetectionTrainer
