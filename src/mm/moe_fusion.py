"""MoE 多模态融合：按模态分专家 + 路由 + 模态 dropout。

## 为什么是 MoE（不是又一次拍脑袋）

我们已有的证伪实验给出了病灶，而且指向性很强：

  把 4 通道模型的**红外换成随机噪声** → mAP50-95 掉 **24.6 分**；
  换成**全零** → 只掉 **0.31 分**。

这两条合起来说明：那个辅助通道**被网络重度使用**（所以噪声能把它打崩），
但网络**没有从红外内容里取信息**（所以全零几乎无损）。
教科书式的解释是"单模态主导 / 模态惰性"：RGB 已经能解掉大部分目标，
辅助通道的梯度信号被稀释，权重 drift 到一个近似常数就停了。

MoE 针对的正是这个机制：**给每个模态独立的参数（专家），让"用不用它"由一个显式路由决定**，
而不是让它在共享卷积核里跟 RGB 抢梯度。

## 两种路由模式（这是本模块最关键的设计点）

`mode="residual"`（默认，保守）：
    out = E_rgb(x_rgb) + Σ_{i≥1} sigmoid(r_i(x)) · E_i(x_i)
    · RGB 专家完整继承 COCO 预训练权重；
    · 辅助专家**零初始化** → 第 0 步输出**严格等于 RGB 基线**；
    · 每个辅助模态有**自己的**门（不像 DualStem 那样共用一个标量 λ）。
    优点：绝不可能一上来就破坏预训练特征，起点等于我们最强的基线。
    代价：没有"专家竞争"，属于 MoE 家族里最温和的一支。

`mode="softmax"`（激进，真正的竞争式路由）：
    out = Σ_i softmax(z(x)/T)_i · E_i(x_i)
    · 路由是逐像素的 softmax 竞争；
    · router bias 初始化为 [b,0,0]（b=6）→ 起点仍近似 RGB；
    · 风险：**路由塌缩**（所有像素都给 RGB 专家），在 1700 张图 + 主模态压倒性强时是高概率事件。

## 必看的诊断（这是本模块存在的一半理由）

每轮记录 `mean(g)`（各专家平均路由权重，逐模态）与 `std(g)`（跨图标准差）。

    mean(g_rgb) → 1.0              路由塌缩，等于没做
    各 mean 都接近先验、std ≈ 0     均匀路由，没有专化
    std 明显 > 0 且随图变化          真的在按图/按像素专化  ← 只有这种情况才可能涨分

没有这套仪表，"MoE 没用"和"MoE 没学起来"就分不开——那正是我们在 P2 / DualStem 上犯过的错。

## 模态 dropout

训练时按概率把某个模态置零。作用有两个：
1. 逼路由不能只靠 RGB（对上面的塌缩是一道直接约束）；
2. 直接对上《竞赛细则》七（二）鲁棒性要求：
   「在一个或者多个模态数据成像质量较差时，算法也不应出现大幅性能波动」。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


def infer_groups(c_in: int) -> tuple[int, ...]:
    """把输入通道按模态分组：RGB(3) / 红外(1) / 深度+掩码(2)。"""
    if c_in == 4:
        return (3, 1)
    if c_in == 6:
        return (3, 1, 2)
    return (3,) + tuple([1] * (c_in - 3))


class MoEStem(nn.Module):
    """替换主干第一层 conv1：按模态分专家 + 路由融合。"""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
        s: int = 2,
        groups: tuple[int, ...] | None = None,
        mode: str = "residual",
        drop_p: float = 0.15,
        drop_rgb_p: float = 0.0,
        temp: float = 1.0,
        rgb_bias: float = 6.0,
    ) -> None:
        super().__init__()
        self.c_in = c_in
        self.groups = tuple(groups or infer_groups(c_in))
        assert sum(self.groups) == c_in, f"分组 {self.groups} 与 c_in={c_in} 不符"
        self.mode = mode
        self.drop_p = drop_p
        self.drop_rgb_p = drop_rgb_p
        self.temp = temp

        self.experts = nn.ModuleList([Conv(g, c_out, k, s) for g in self.groups])
        # 路由：与专家同 stride，逐像素。输入用原始多模态图像（省算力，且不看专家输出避免自激）
        self.router = nn.Conv2d(c_in, len(self.groups), kernel_size=1, stride=s)

        self._init_weights(rgb_bias)

        # 诊断量（每轮由回调读取）
        self.last_g_mean: torch.Tensor | None = None
        self.last_g_std: torch.Tensor | None = None
        # ultralytics 的模块需要 .f 指明输入层
        self.f = -1

    # ------------------------------------------------------------------ #
    def _init_weights(self, rgb_bias: float) -> None:
        with torch.no_grad():
            if self.mode == "residual":
                # 辅助专家零初始化 → 第 0 步输出严格等于 RGB 专家
                for e in list(self.experts)[1:]:
                    nn.init.zeros_(e.conv.weight)
                    if e.conv.bias is not None:
                        nn.init.zeros_(e.conv.bias)
                # 门控 logits bias 置 0 → sigmoid=0.5（既不压倒也不忽略），权重置 0
                nn.init.zeros_(self.router.weight)
                nn.init.zeros_(self.router.bias)
            else:
                # 竞争式路由：初始强烈偏向 RGB 专家，保证起点≈基线
                nn.init.zeros_(self.router.weight)
                bias = torch.zeros(len(self.groups))
                bias[0] = rgb_bias
                self.router.bias.copy_(bias)
                for e in list(self.experts)[1:]:
                    nn.init.zeros_(e.conv.weight)
                    if e.conv.bias is not None:
                        nn.init.zeros_(e.conv.bias)

    # ------------------------------------------------------------------ #
    def _modality_dropout(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        """训练时随机把某个模态整体置零。返回新的列表（不原地改，避免影响别的分支）。"""
        if not self.training:
            return xs
        out = list(xs)
        if self.drop_rgb_p > 0 and torch.rand(1).item() < self.drop_rgb_p:
            out[0] = torch.zeros_like(out[0])
        for i in range(1, len(out)):
            if self.drop_p > 0 and torch.rand(1).item() < self.drop_p:
                out[i] = torch.zeros_like(out[i])
        return out

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs: list[torch.Tensor] = []
        off = 0
        for g in self.groups:
            xs.append(x[:, off : off + g])
            off += g
        xs = self._modality_dropout(xs)

        feats = [e(xi) for e, xi in zip(self.experts, xs)]
        z = self.router(x)                       # (B, N, h, w)

        if self.mode == "residual":
            g = z.sigmoid()                      # 每个辅助模态一个独立的逐像素门
            out = feats[0]
            for i in range(1, len(feats)):
                out = out + g[:, i : i + 1] * feats[i]
        else:
            g = torch.softmax(z / self.temp, dim=1)
            out = feats[0] * g[:, 0:1]
            for i in range(1, len(feats)):
                out = out + g[:, i : i + 1] * feats[i]

        with torch.no_grad():
            self.last_g_mean = g.mean(dim=(0, 2, 3)).detach()
            # 跨图标准差：衡量"路由是否按图变化"，这是专化的必要条件
            self.last_g_std = g.mean(dim=(2, 3)).std(dim=0).detach()
        return out


def inject_moe_stem(
    model: nn.Module,
    ckpt_path: str | Path,
    mode: str = "residual",
    drop_p: float = 0.15,
    drop_rgb_p: float = 0.0,
    verbose: bool = True,
) -> bool:
    """把 conv1 换成 MoEStem，并把 COCO 预训练权重装进 **RGB 专家**。"""
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
    except Exception as e:  # noqa: BLE001
        print(f"[MoE] 读取预训练权重失败: {e}")
        return False

    key = next((k for k in sd if k.endswith("0.conv.weight")), None)
    if key is None:
        print("[MoE] 未找到预训练 stem 权重")
        return False
    bias_key = f"{key[:-6]}bias"
    w = sd[key]

    seq = model.model if hasattr(model, "model") else model
    old = seq[0]
    old_conv = getattr(old, "conv", old)
    if not isinstance(old_conv, nn.Conv2d):
        print(f"[MoE] 第一层不是卷积: {type(old)}")
        return False

    c_out = old_conv.out_channels
    k = old_conv.kernel_size[0]
    s = old_conv.stride[0]
    c_in = old_conv.in_channels
    if c_in <= 3:
        print(f"[MoE] 输入通道为 {c_in}，没有辅助模态，跳过")
        return False

    new = MoEStem(c_in, c_out, k, s, mode=mode, drop_p=drop_p, drop_rgb_p=drop_rgb_p)
    with torch.no_grad():
        if w.shape[0] != c_out or w.shape[1] != 3:
            print(f"[MoE] 预训练 stem 形状 {tuple(w.shape)} 与 c_out={c_out} 不符")
            return False
        new.experts[0].conv.weight.copy_(w)
        if new.experts[0].conv.bias is not None and bias_key in sd:
            new.experts[0].conv.bias.copy_(sd[bias_key])
    seq[0] = new

    # parse_model 会给每个模块附 i/f/type/np，手工注入必须补齐
    new.i = 0
    new.f = -1
    new.type = "MoEStem"
    new.np = sum(p.numel() for p in new.parameters())

    if verbose:
        print(
            f"[MoE] 已注入: 输入 {c_in}ch 分组 {new.groups} -> {c_out}ch, "
            f"mode={mode}, 模态dropout={drop_p:.2f}, RGB专家=COCO权重, 辅助专家=零初始化"
        )
    return True


def make_trainer(
    mode: str = "residual",
    drop_p: float = 0.15,
    drop_rgb_p: float = 0.0,
    ckpt_path: str | Path | None = None,
):
    """生成在 get_model 里自动注入 MoEStem 的 DetectionTrainer 子类。"""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class MoEDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            if ckpt_path:
                ok = inject_moe_stem(
                    model, ckpt_path, mode=mode, drop_p=drop_p, drop_rgb_p=drop_rgb_p
                )
                self._moe_injected = ok
            return model

    return MoEDetectionTrainer
