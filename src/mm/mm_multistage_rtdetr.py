"""多阶段三模态融合 —— **RT-DETR 版**（复用 mm_multistage 的 AuxStages / AGFFusion）。

## 为什么需要单独一版

`mm_multistage.py` 是 **YOLO 专属**的：它继承 `DetectionModel`、覆写 `_predict_once`，
并把 `self.model[head_start:]` 当作"检测头"。RT-DETR 结构完全不同：

* 训练/推理入口是 **`RTDETRDetectionModel.predict(x, profile, batch, augment, embed)`**
  （`loss()` 内部就是调 `predict(img, batch=targets)`）；`_predict_once` 它**根本不用**；
* 头是 `self.model[-1]` = `RTDETRDecoder`，调用形式是 `head([y[j] for j in head.f], batch)`；
* 骨干里真正被下游引用的输出是 **layer 3（P3/512）、7（P4/1024）、9（P5/2048）**
  （见 `cfg/models/rt-detr/rtdetr-l.yaml`：layer 14 取 `[7]`、layer 19 取 `[3]`、
  layer 10 取 `[-1]`(=9)）。

## 与 YOLO 版完全一致的设计（依据见 mm_multistage.py 开头的三条证据）

```
输入 (B,6,H,W) = [RGB(3) | 红外(1) | 深度(1) | 掩码(1)]
  RGB ──► 原骨干（ch=3，完整继承 COCO 权重，不被稀释）
  辅助 ──► AuxStages（三级，逐级下采样到 P3/P4/P5 分辨率）
  每个融合点： AGFFusion：门控差分 + 双向残差更新，W_r/W_a **零初始化**
              ⇒ 第 0 步输出**严格等于纯 RGB 基线**
```

模型按 `ch=3` 构建（RGB 分支），数据管线仍喂 6 通道，由本文件在 `predict()` 里切分。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.tasks import RTDETRDetectionModel

from mm_multistage import AGFFusion, AuxStages

# RT-DETR-l 的骨干融合点：P3/8=layer3(512ch)、P4/16=layer7(1024ch)、P5/32=layer9(2048ch)
DEFAULT_FUSION_POINTS = (3, 7, 9)


class MMFusionRTDETRModel(RTDETRDetectionModel):
    """RT-DETR 检测模型 + 多阶段双向三模态融合。"""

    # ⚠️ 类属性：`__init__` 里父类会先跑一次 forward，那时实例属性还没赋值
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
        cfg: str = "rtdetr-l.yaml",
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
        # ch=3：RGB 分支走原骨干（**不是** 6），这样 stem 能完整继承 COCO 权重
        super().__init__(cfg, ch, nc, verbose)
        # ★ 防御性设置：本版 ultralytics 的 RTDETRDetectionModel **不设 self.nc**
        #   （只有 self.yaml["nc"] 与 self.model[-1].nc），而 `init_criterion()` 读的是
        #   `self.nc` —— 普通训练路径靠 trainer 兜住，但手工构造模型（自检、离线评估）
        #   会直接 AttributeError。显式设上，行为与 YOLO 模型一致。
        if nc is not None:
            self.nc = int(nc)
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
        self._stage_of_layer = {int(layer): i for i, layer in enumerate(self.fusion_points)}
        self._mm_ready = True

        if verbose:
            n_aux = sum(p.numel() for p in self.aux_stages.parameters())
            n_f = sum(p.numel() for p in self.fusions.parameters())
            print(f"[MM-RTDETR] 多阶段双向融合装配完成：融合点 {self.fusion_points}（层号）"
                  f" = P3/P4/P5；RGB 宽度 {widths} / 辅助宽度 {aw}；mode={fuse_mode}, 双向={bidir}")
            print(f"[MM-RTDETR] 辅助编码器 {n_aux / 1e6:.2f}M 参数 + 融合模块 {n_f / 1e6:.2f}M 参数")

    # ------------------------------------------------------------------ #
    def _probe_widths(self) -> tuple[int, ...]:
        """跑一次纯 RGB 前向，读出各融合点的真实通道数（挂 forward hook）。"""
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
            RTDETRDetectionModel.predict(self, torch.zeros(1, 3, 256, 256))
        for h in hooks:
            h.remove()
        if was_training:
            self.train()
        missing = [i for i in self.fusion_points if i not in chans]
        if missing:
            raise RuntimeError(f"融合点 {missing} 没取到通道数，请检查 fusion_points（RT-DETR-l 应为 3/7/9）")
        return tuple(chans[i] for i in self.fusion_points)

    # ------------------------------------------------------------------ #
    def _modality_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """训练时按概率整体置零某个模态（分组：红外 / 深度+掩码）。"""
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
    def predict(self, x, profile=False, batch=None, augment=False, embed=None):
        """覆写 RT-DETR 的推理/训练共同入口：双分支逐级推进 + 在 P3/P4/P5 融合。"""
        if not self._mm_ready:
            # __init__ 期间父类会先跑一次 3 通道前向
            return RTDETRDetectionModel.predict(self, x, profile, batch, augment, embed)

        x = self._modality_dropout(x)
        # ⚠️ y 必须**逐个 append**，不能预分配定长列表：
        #   ultralytics 的 f 允许负索引（如 RT-DETR 的 [-2,-1] 拼接层），
        #   负索引是"相对当前已产出的输出"取值；定长列表会让 y[-2] 取到 None，
        #   导致 `torch.cat` 报 "expected Tensor ... but got NoneType"。
        #   （这是自检逮到的真 bug，YOLO 版没暴露只因它的模型不用 -2。）
        y: list[torch.Tensor | None] = []
        r = x[:, :3]      # RGB 分支
        a = x[:, 3:]      # 辅助分支（红外 + 深度 + 掩码）

        for m in self.model[:-1]:                      # 骨干 + FPN/PAN（不含解码器）
            if m.f != -1:
                prev = y[m.f] if isinstance(m.f, int) else [r if j == -1 else y[j] for j in m.f]
            else:
                prev = r
            if profile:
                self._profile_one_layer(m, prev, [])
            r = m(prev)
            y.append(r if m.i in self.save else None)
            stage = self._stage_of_layer.get(int(m.i))
            if stage is not None:
                a = self.aux_stages[stage](a)
                r, a = self.fusions[stage](r, a)
                if m.i in self.save:
                    y[-1] = r                          # 让下游用**融合后**的特征

        head = self.model[-1]                          # RTDETRDecoder
        return head([y[j] for j in head.f], batch)


def make_trainer(
    fusion_points: tuple[int, ...] = DEFAULT_FUSION_POINTS,
    aux_in: int = 3,
    drop_p: float = 0.15,
    drop_rgb_p: float = 0.0,
    fuse_mode: str = "diff",
    bidir: bool = True,
    lr_mult: float = 3.0,
):
    """生成用 MMFusionRTDETRModel 构建模型的 RTDETRTrainer 子类（含差分学习率）。"""
    from ultralytics.models.rtdetr.train import RTDETRTrainer

    NEW_PREFIXES = ("aux_stages.", "fusions.")

    class MMFusionRTDETRTrainer(RTDETRTrainer):
        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            model = MMFusionRTDETRModel(
                cfg,
                ch=3,                       # RGB 分支 3 通道 ⇒ 完整继承 COCO 权重
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
            """新模块（辅助编码器 + 融合模块）用 ×lr_mult 的学习率，预训练主干保持 1×。

            理由同 YOLO 版：往 11M 预训练模型里塞进 ~1.2M+1.9M 随机初始化参数，
            若共用同一个学习率，就是"新架构直接加上去就掉分"的经典成因。
            """
            optim = super().build_optimizer(model, name, lr, momentum, decay, iterations)
            if lr_mult == 1.0:
                return optim
            ours = {id(p) for n, p in model.named_parameters() if n.startswith(NEW_PREFIXES)}
            if not ours:
                print("[MM-RTDETR][!!] 没找到新参数（命名前缀不匹配）—— 差分学习率未生效")
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
            for g in optim.param_groups:
                g.setdefault("initial_lr", g["lr"])
            n_new = sum(len(g["params"]) for g in moved_groups)
            print(f"[MM-RTDETR] 差分学习率：{n_new} 个新参数张量 ×{lr_mult}（其余保持原学习率）")
            return optim

    return MMFusionRTDETRTrainer
