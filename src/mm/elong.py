"""面向**细长目标**的各向异性多尺度卷积（Anisotropic Elongation Module）。

## 为什么单独做这个

逐类诊断（第 19.1 节）里，**bicycle 是全项目最差的一类：AP 27.2**，
而它**不是样本少**——验证集有 109 个框、训练集有 730 个框，样本充足。
所以它是**结构问题**：自行车是"细长 + 镂空"的目标，
标准 3×3 方形卷积核与方形先验的检测头对它天然不利
（同样问题也出现在 sign 的长条、uav 的细翼上）。

对照证据：那篇双模态工程方案的模块清单里有一条
「**多尺度非对称卷积单元：并行搭载 5×5 标准深度卷积、多组横竖分解一维大核卷积**，
轻量化捕获多尺度、**长条形目标**空间轮廓特征」——正是为这个问题设计的。

## 结构

```
分支 1: 5×5 深度卷积         —— 各向同性的局部上下文
分支 2: 1×k 横向深度卷积      —— 水平延伸的长条
分支 3: k×1 纵向深度卷积      —— 垂直延伸的长条
        ↓ 拼接 → 1×1 投影（零初始化）→ 残差叠加回原特征
```

**零初始化投影保证第 0 步输出严格等于原模型**，不破坏 COCO 预训练特征。
与多阶段融合模块一样，这条"起点等于基线"的性质是我们所有结构改动的硬要求。

## 注入位置

检测头读取 backbone 的 P3(stride 8, layer 4) 与 P4(stride 16, layer 6)；
bicycle / sign 这类目标主要落在 P3，所以默认注入 (4, 6) 两处。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel

DEFAULT_POINTS = (4, 6)


class ElongBlock(nn.Module):
    """各向异性多尺度卷积 + 零初始化残差。"""

    def __init__(self, c: int, k: int = 11, act: bool = True) -> None:
        super().__init__()
        self.dw = Conv(c, c, 5, 1, g=c, act=act)          # 5×5 深度卷积
        self.h = Conv(c, c, (1, k), 1, g=c, act=act)      # 横向 1×k
        self.v = Conv(c, c, (k, 1), 1, g=c, act=act)      # 纵向 k×1
        self.proj = nn.Conv2d(3 * c, c, 1)
        nn.init.zeros_(self.proj.weight)                  # 零初始化 → 起点等于基线
        nn.init.zeros_(self.proj.bias)
        self.last_res_ratio: torch.Tensor | None = None
        self.f = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.proj(torch.cat((self.dw(x), self.h(x), self.v(x)), dim=1))
        with torch.no_grad():
            # 残差相对于主特征的量级——判断这个模块到底有没有在起作用
            denom = x.detach().abs().mean().clamp_min(1e-6)
            self.last_res_ratio = (d.detach().abs().mean() / denom)
        return x + d


class ElongModel(DetectionModel):
    """在 backbone 的指定 stage 输出上注入各向异性卷积（纯 RGB，单变量实验用）。"""

    _elong_ready: bool = False
    points: tuple[int, ...] = DEFAULT_POINTS
    k: int = 11

    def __init__(
        self,
        cfg: str = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        points: tuple[int, ...] = DEFAULT_POINTS,
        k: int = 11,
    ) -> None:
        super().__init__(cfg, ch, nc, verbose)
        self.points = tuple(points)
        self.k = int(k)
        widths = self._probe_widths()
        self.elong = nn.ModuleDict({str(i): ElongBlock(w, k) for i, w in zip(self.points, widths)})
        self._elong_ready = True
        if verbose:
            n = sum(p.numel() for p in self.elong.parameters())
            print(f"[Elong] 已注入各向异性卷积：位置 {self.points}（宽度 {widths}），"
                  f"核长 {k}，共 {n / 1e3:.1f}K 参数（起点严格等于基线）")

    def _probe_widths(self) -> tuple[int, ...]:
        chans: dict[int, int] = {}
        hooks = []
        for i in self.points:
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
        return tuple(chans[i] for i in self.points)

    def _predict_once(self, x, profile=False, embed=None):
        if not self._elong_ready:
            return DetectionModel._predict_once(self, x, profile, embed)
        y: list[torch.Tensor | None] = [None] * len(self.model)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if m.i in self.points:
                x = self.elong[str(m.i)](x)
            y[m.i] = x if m.i in self.save else None
        return x


def make_trainer(
    points: tuple[int, ...] = DEFAULT_POINTS,
    k: int = 11,
    lr_mult: float = 3.0,
):
    """生成用 ElongModel 构建模型的 DetectionTrainer 子类（新模块用差分学习率）。"""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class ElongTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            model = ElongModel(
                cfg,
                ch=3,
                nc=getattr(self, "data", {}).get("nc"),
                verbose=verbose and self.args.verbose,
                points=points,
                k=k,
            )
            if weights:
                model.load(weights)
            return model

        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
            optim = super().build_optimizer(model, name, lr, momentum, decay, iterations)
            if lr_mult == 1.0:
                return optim
            ours = {id(p) for n, p in model.named_parameters() if n.startswith("elong.")}
            if not ours:
                return optim
            moved = []
            for g in optim.param_groups:
                keep, mv = [], []
                for p in g["params"]:
                    (mv if id(p) in ours else keep).append(p)
                g["params"] = keep
                if mv:
                    ng = {k2: v for k2, v in g.items() if k2 != "params"}
                    ng["params"] = mv
                    ng["lr"] = g["lr"] * lr_mult
                    moved.append(ng)
            for ng in moved:
                optim.add_param_group(ng)
            for g in optim.param_groups:
                g.setdefault("initial_lr", g["lr"])
            print(f"[Elong] 差分学习率：新模块 ×{lr_mult}")
            return optim

    return ElongTrainer
