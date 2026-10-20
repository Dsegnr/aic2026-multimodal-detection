"""多阶段三模态融合 v2 —— 模态各自编码 + 逐级双向门控差分融合。

## 为什么是这个结构（三条独立证据汇合）

1. **我们自己的机械诊断**（第 9.2 节）：辅助通道换成随机噪声掉 24.6 分、
   换成全零只掉 0.31 分 → 通道被重度使用但内容没被提取（modality laziness）。
2. **我们自己的设计缺陷**（第 14 节）：此前融合全部发生在**输入层 stem**，
   而且辅助模态只有**一层卷积（约 900 参数）**——没有容量去表示任何东西。
   所以"融合≈0"是**欠功效实验**，不是"多模态无用"。
3. **公开工程方案**（第 16 节，YOLO 双 backbone + AdaptiveGatedFusion）：
   双独立骨干 + 逐级融合 + **门控差分** + **双向残差更新**；
   在 LLVIP 上其 `double_backbone_fuse` 内部对照显示该结构值约 +6 分。

## 结构

```
输入 (B, 6, H, W) = [RGB(3) | 红外(1) | 深度(1) | 有效掩码(1)]
  RGB ──► 原 backbone（ch=3，完整继承 COCO 权重，不被稀释）
            层层推进，每到一个融合点：
  辅助 ──► AuxStages（四级，逐级下采样，容量与 RGB 主干同级）
            两级在融合点处汇合：

  AGFFusion:  g = σ(W·[r, a])
              d = attn( silu( bn( W·((r − a)·g) ) ) )   # 门控差分：背景相消、目标凸显
              r ← r + W_r·d        a ← a + W_a·d        # 双向更新
```

**起点保证**：`W_r`、`W_a` 零初始化 → 第 0 步输出**严格等于纯 RGB 基线**。
代价是辅助编码器第一次反传梯度恰为 0（一步延迟，已自检确认无害）。

**模态 dropout**：训练时按概率把某个模态整体置零（分组：红外 / 深度+掩码）。
既约束门控塌缩，也直接对应《竞赛细则》七（二）的鲁棒性要求。

## 实现

覆写 `DetectionModel._predict_once`（训练与推理的共同入口，`loss()` 第 549 行调它）。
模型按 `ch=3` 构建，因此 stem 卷积能完整继承 COCO 权重——这是与"朴素拼接"最关键的区别。
已知限制：`augment=True` 会绕过本函数，需要 TTA 时另行实现。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel

# YOLOv8 系列 backbone 的 stage 末端：2=P2/4, 4=P3/8, 6=P4/16, 9=P5/32
DEFAULT_FUSION_POINTS = (2, 4, 6, 9)


class AuxStages(nn.ModuleList):
    """辅助模态自己的编码器：四级，逐级降分辨率到与 P2/P3/P4/P5 对齐。"""

    def __init__(self, c_in: int, widths: tuple[int, ...]) -> None:
        blocks: list[nn.Module] = []
        c = c_in
        for i, w in enumerate(widths):
            if i == 0:
                # 第一级一次降到 stride 4（对应 P2/4）
                layers = [
                    Conv(c, max(w // 4, 8), 3, 2),
                    Conv(max(w // 4, 8), max(w // 2, 8), 3, 1),
                    Conv(max(w // 2, 8), w, 3, 2),
                ]
            else:
                layers = [Conv(c, w, 3, 2), Conv(w, w, 3, 1)]
            blocks.append(nn.Sequential(*layers))
            c = w
        super().__init__(blocks)


class AGFFusion(nn.Module):
    """自适应门控差分融合 + 双向更新（起点严格等于 RGB 基线）。"""

    def __init__(self, c_rgb: int, c_aux: int, mode: str = "diff", bidir: bool = True) -> None:
        super().__init__()
        self.mode = mode
        self.bidir = bidir
        c = c_rgb
        # 辅助特征先投影到 RGB 的特征空间，才能做"差分"（两者必须在同一空间里相减）
        self.proj = nn.Conv2d(c_aux, c, 1)
        # ★ 修复 1：尺度对齐。`proj` 是随机初始化的 1×1 卷积，输出尺度和主干的 rgb
        #   特征未必可比。若两者量级差很多，`(rgb − aux)` 会一边倒，差分分支学到的是噪声，
        #   再通过 r_out 注回主干就是有害的。归一化后再相减，差分才有意义。
        self.aux_norm = nn.BatchNorm2d(c)
        self.gate = nn.Conv2d(2 * c, c, 1)                                    # 逐通道自适应门控
        self.diff = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU())
        hid = max(c // 4, 8)
        self.attn = nn.Sequential(nn.Conv2d(c, hid, 1), nn.SiLU(), nn.Conv2d(hid, c, 1))  # 双池化通道注意力
        self.r_out = nn.Conv2d(c, c, 1)
        self.a_out = nn.Conv2d(c, c_aux, 1) if bidir else None      # 回写到辅助分支要还原通道数
        for m in (self.gate, self.r_out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        if self.a_out is not None:
            nn.init.zeros_(self.a_out.weight)
            nn.init.zeros_(self.a_out.bias)
        self.last_gate_mean: torch.Tensor | None = None

    def forward(self, rgb: torch.Tensor, aux: torch.Tensor):
        if aux.shape[-2:] != rgb.shape[-2:]:
            aux = nn.functional.interpolate(aux, size=rgb.shape[-2:], mode="nearest")
        aux_p = self.proj(aux)        # 投影到 RGB 特征空间，才能做差分
        aux_p = self.aux_norm(aux_p)  # 尺度对齐（见上）
        g = self.gate(torch.cat((rgb, aux_p), dim=1)).sigmoid()
        # 差分：背景处两模态高度重合 → 响应低；目标处差异显著 → 响应高
        d = self.diff((rgb - aux_p) * g if self.mode == "diff" else aux_p * g)
        s = self.attn(d.mean(dim=(2, 3), keepdim=True) + d.amax(dim=(2, 3), keepdim=True))
        d = d * s.sigmoid()
        r2 = rgb + self.r_out(d)
        a2 = aux + self.a_out(d) if self.a_out is not None else aux   # a_out 输出还原为辅助通道数
        with torch.no_grad():
            self.last_gate_mean = g.mean().detach()
        return r2, a2


class MMFusionModel(DetectionModel):
    """YOLO 检测模型 + 多阶段双向三模态融合。"""

    # ⚠️ 必须是类属性：`DetectionModel.__init__` 会先跑一次前向算 stride，
    # 那时实例属性还没赋值，`_predict_once` 会被提前调用。
    _mm_ready: bool = False
    fusion_points: tuple[int, ...] = DEFAULT_FUSION_POINTS
    aux_in: int = 0
    aux_groups: tuple[int, ...] = ()
    drop_p: float = 0.0
    drop_rgb_p: float = 0.0
    fuse_mode: str = "diff"
    bidir: bool = True

    def __init__(
        self,
        cfg: str = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        fusion_points: tuple[int, ...] = DEFAULT_FUSION_POINTS,
        aux_in: int = 3,
        aux_groups: tuple[int, ...] = (1, 2),
        drop_p: float = 0.15,
        drop_rgb_p: float = 0.0,
        fuse_mode: str = "diff",
        bidir: bool = True,
        aux_width_scale: float = 0.5,
    ) -> None:
        # ch 是 **RGB 分支** 的通道数（3），不是数据管线的 6
        super().__init__(cfg, ch, nc, verbose)
        self.fusion_points = tuple(fusion_points)
        self.aux_in = int(aux_in)
        self.aux_groups = tuple(aux_groups)
        assert sum(self.aux_groups) == self.aux_in, f"分组 {self.aux_groups} 与 aux_in={self.aux_in} 不符"
        self.drop_p = float(drop_p)
        self.drop_rgb_p = float(drop_rgb_p)
        self.fuse_mode = fuse_mode
        self.bidir = bool(bidir)

        widths = self._probe_widths()
        aw = tuple(max(int(round(w * aux_width_scale / 8)) * 8, 8) for w in widths)
        self.aux_stages = AuxStages(self.aux_in, aw)
        self.fusions = nn.ModuleList([AGFFusion(w, a, mode=fuse_mode, bidir=bidir)
                                      for w, a in zip(widths, aw)])
        self._segments = self._build_segments()
        self._mm_ready = True

        if verbose:
            n_aux = sum(p.numel() for p in self.aux_stages.parameters())
            n_f = sum(p.numel() for p in self.fusions.parameters())
            print(f"[MM] 多阶段双向融合装配完成：融合点 {self.fusion_points}"
                  f"（RGB 宽度 {widths} / 辅助宽度 {aw}）；mode={fuse_mode}, 双向更新={bidir}")
            print(f"[MM] 辅助编码器 {n_aux / 1e6:.2f}M 参数 + 融合模块 {n_f / 1e6:.2f}M 参数")

    # ------------------------------------------------------------------ #
    def _probe_widths(self) -> tuple[int, ...]:
        """跑一次纯 RGB 前向，读出各融合点处的真实通道数（随模型 scale 变化）。"""
        chans: dict[int, int] = {}
        hooks = []
        for i in self.fusion_points:
            hooks.append(
                self.model[i].register_forward_hook(
                    (lambda idx: lambda _m, _i, out: chans.__setitem__(idx, int(out.shape[1])))(i)
                )
            )
        was_training = self.training
        self.eval()
        with torch.no_grad():
            DetectionModel._predict_once(self, torch.zeros(1, self.yaml.get("channels", 3), 256, 256))
        for h in hooks:
            h.remove()
        if was_training:
            self.train()
        missing = [i for i in self.fusion_points if i not in chans]
        if missing:
            raise RuntimeError(f"融合点 {missing} 没取到通道数，请检查 fusion_points")
        return tuple(chans[i] for i in self.fusion_points)

    def _build_segments(self) -> list[tuple[int, int, int]]:
        """把 backbone 切成若干 stage：每段以某个融合点结尾。"""
        segs, start = [], 0
        for idx in self.fusion_points:
            segs.append((start, idx, idx))
            start = idx + 1
        return segs

    # ------------------------------------------------------------------ #
    def _modality_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        drop_rgb = self.drop_rgb_p > 0 and torch.rand(1).item() < self.drop_rgb_p
        drop_aux = [i for i in range(len(self.aux_groups))
                    if self.drop_p > 0 and torch.rand(1).item() < self.drop_p]
        if not drop_rgb and not drop_aux:
            return x
        x = x.clone()
        if drop_rgb:
            x[:, :3] = 0
        off = 3
        for i, g in enumerate(self.aux_groups):
            if i in drop_aux:
                x[:, off : off + g] = 0
            off += g
        return x

    # ------------------------------------------------------------------ #
    def _predict_once(self, x, profile=False, embed=None):
        """ultralytics 训练与推理的共同入口：逐级交替跑双分支并做双向融合。"""
        if not self._mm_ready:
            return DetectionModel._predict_once(self, x[:, :3] if x.shape[1] > 3 else x, profile, embed)

        x = self._modality_dropout(x)
        y: list[torch.Tensor | None] = [None] * len(self.model)
        r = x[:, :3]      # RGB 分支
        a = x[:, 3:]      # 辅助分支（红外 + 深度 + 掩码）

        for i, (start, end, save_idx) in enumerate(self._segments):
            for j in range(start, end + 1):
                m = self.model[j]
                if m.f != -1:
                    r = y[m.f] if isinstance(m.f, int) else [r if t == -1 else y[t] for t in m.f]
                r = m(r)
            a = self.aux_stages[i](a)          # 辅助分支推进一级
            r, a = self.fusions[i](r, a)       # 双向融合：两个分支都被更新
            y[save_idx] = r if save_idx in self.save else None

        head_start = self._segments[-1][1] + 1
        xx = r
        for m in self.model[head_start:]:
            if m.f != -1:
                xx = y[m.f] if isinstance(m.f, int) else [xx if j == -1 else y[j] for j in m.f]
            xx = m(xx)
            y[m.i] = xx if m.i in self.save else None
        return xx


def make_trainer(
    fusion_points: tuple[int, ...] = DEFAULT_FUSION_POINTS,
    aux_in: int = 3,
    drop_p: float = 0.15,
    drop_rgb_p: float = 0.0,
    fuse_mode: str = "diff",
    bidir: bool = True,
    lr_mult: float = 3.0,
):
    """生成用 MMFusionModel 构建模型的 DetectionTrainer 子类。

    ★ 修复 2（差分学习率）：我们往一个 11.1M 的**预训练**模型里塞进了
    1.17M 辅助编码器 + 1.92M 融合模块的**随机初始化**参数，却让它们和预训练主干
    共用同一个学习率。这是"新架构直接加上去就掉分"的经典成因：
    新参数需要学得更快，预训练权重需要动得更慢。

    ultralytics 自己在 MuSGD 路径上就是这么做的——它把检测头的 cv3 学习率
    放大 3 倍（见 engine/trainer.py 的 `lr * 3`）。我们对新模块做同样的事。
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    NEW_PREFIXES = ("aux_stages.", "fusions.")

    class MMFusionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            model = MMFusionModel(
                cfg,
                ch=3,                       # RGB 分支 = 3 通道，完整继承 COCO 权重
                nc=getattr(self, "data", {}).get("nc"),
                verbose=verbose and self.args.verbose,
                fusion_points=fusion_points,
                aux_in=aux_in,
                drop_p=drop_p,
                drop_rgb_p=drop_rgb_p,
                fuse_mode=fuse_mode,
                bidir=bidir,
            )
            if weights:
                model.load(weights)
            return model

        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
            optim = super().build_optimizer(model, name, lr, momentum, decay, iterations)
            if lr_mult == 1.0:
                return optim
            target = model
            ours = {
                id(p)
                for n, p in target.named_parameters()
                if n.startswith(NEW_PREFIXES)
            }
            if not ours:
                return optim
            moved_groups = []
            for g in optim.param_groups:
                kept, moved = [], []
                for p in g["params"]:
                    (moved if id(p) in ours else kept).append(p)
                g["params"] = kept
                if moved:
                    ng = {k: v for k, v in g.items() if k != "params"}
                    ng["params"] = moved
                    ng["lr"] = g["lr"] * lr_mult
                    moved_groups.append(ng)
            for ng in moved_groups:
                optim.add_param_group(ng)
            # 学习率调度器读的是 initial_lr，必须给新组补上（已有的不覆盖）
            for g in optim.param_groups:
                g.setdefault("initial_lr", g["lr"])
            n_new = sum(len(g["params"]) for g in moved_groups)
            print(f"[MM] 差分学习率：{n_new} 个新参数张量 ×{lr_mult}（其余保持原学习率）")
            return optim

    return MMFusionTrainer
