"""非对称融合模块：DualStem。

## 为什么这么设计（有实验依据，不是拍脑袋）

E4 配对实验（960 分辨率、22 个配对轮次、RGB vs RGB+红外四通道拼接）得到的结论是：
**平均差值 −0.05 分，即"直接拼通道"没有可靠增益。**

机理：
- 收益：红外确实带来互补信息（第 7 轮曾领先 3.3 分）；
- 代价：`channels != 3` 时 ultralytics 丢弃预训练的第一层卷积，第 4 通道只能用
  RGB 权重均值"半随机"初始化，稀释了已经学好的 COCO 低层滤波器。
两者抵消，净值为零。

因此本模块针对性地解决"代价侧"：
1. **RGB 走自己的分支，权重从 COCO 完整继承**（不再被稀释）；
2. **辅助模态（红外/深度）走独立分支**，可以学自己的低层特征；
3. **门控 + λ 从 0 初始化**：`输出 = f_rgb + tanh(λ)·gate·f_aux`。
   λ=0 时输出**严格等于** f_rgb，也就是"训练第一步与 RGB 基线完全一致"，
   之后由网络自己决定要不要、以及在哪些位置上采纳辅助模态。

这与 CSSA（RGB-T 对称的通道切换）的区别在于：
它把 RGB 当作必须保住的语义主干，只让辅助模态做"增量修正"，
并且把"何时相信辅助模态"交给一个显式门控学习。
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


class DualStem(nn.Module):
    """替换主干第一层：RGB 与辅助模态分路，门控融合。

    入参 c_in 由数据 yaml 的 channels 决定（RGB=3 + 辅助模态数）。
    输出通道与原始 stem 保持一致，因此**下游网络结构完全不变**，
    COCO 预训练的后继层全部可以正常继承。
    """

    def __init__(self, old_module: nn.Module, c_in: int, lam_init: float = 0.0):
        """用**原 stem 模块**复制出两条分支（保证 BN/激活/卷积参数设置逐位一致）。

        ★ 2026-09-23 重写：旧版用 `Conv(3, c_out, k, s)` 新建分支，会丢掉原模块的
        激活函数（RT-DETR 的 HGStem.stem1 用 **ReLU**，而 ultralytics Conv 默认 **SiLU**），
        于是"RGB 分支等于原 stem"这个前提就不成立了。现在改成 deepcopy 原模块后
        只替换其内部 `conv` 的输入通道数（并按通道切片继承预训练权重）。
        """
        super().__init__()
        self.c_out = _inner_conv(old_module).out_channels
        self.c_in = c_in
        self.c_aux = max(c_in - 3, 1)
        self.rgb = _rebuild_branch(old_module, 3, take_weights=True)
        self.aux = _rebuild_branch(old_module, self.c_aux, take_weights=False)
        self.gate = nn.Conv2d(2 * self.c_out, self.c_out, 1)
        # 门控初始化为 0 -> sigmoid=0.5（既不压倒也不忽略辅助模态）
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        # λ 初始化为 lam_init（0 表示从纯 RGB 基线出发）
        self.lam = nn.Parameter(torch.tensor(float(lam_init)))
        # ultralytics 的 BaseModel._predict_once 依赖每个模块的 .f 属性
        # （表示输入来自哪一层）。-1 = 使用上一层输出。
        # parse_model 构建的模块会自动带上它，我们手工注入的必须自己设。
        self.f = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rgb = x[:, :3]
        x_aux = x[:, 3:]
        f_rgb = self.rgb(x_rgb)
        f_aux = self.aux(x_aux)
        g = self.gate(torch.cat((f_rgb, f_aux), dim=1)).sigmoid()
        return f_rgb + torch.tanh(self.lam) * g * f_aux


def _inner_conv(m: nn.Module) -> nn.Conv2d:
    """取出模块内部的 nn.Conv2d（ultralytics 的 Conv 包装器是 m.conv；否则可能是它本身）。"""
    inner = getattr(m, "conv", None)
    if isinstance(inner, nn.Conv2d):
        return inner
    if isinstance(m, nn.Conv2d):
        return m
    raise TypeError(f"模块里找不到 nn.Conv2d: {type(m).__name__}")


def _rebuild_branch(old_module: nn.Module, in_ch: int, take_weights: bool) -> nn.Module:
    """deepcopy 原模块，只把内部 conv 的输入通道改成 in_ch（可选：切片继承权重）。"""
    branch = copy.deepcopy(old_module)
    old_inner = _inner_conv(old_module)
    new_inner = nn.Conv2d(
        in_ch,
        old_inner.out_channels,
        old_inner.kernel_size,
        old_inner.stride,
        old_inner.padding,
        old_inner.dilation,
        old_inner.groups,
        old_inner.bias is not None,
    )
    with torch.no_grad():
        if take_weights:
            # 预训练 stem 一定是 3 通道（RGB），按输入通道切片继承
            src = old_inner.weight
            k = min(3, src.shape[1], in_ch)
            new_inner.weight[:, :k].copy_(src[:, :k])
        if old_inner.bias is not None and new_inner.bias is not None:
            new_inner.bias.copy_(old_inner.bias)
    if getattr(branch, "conv", None) is old_inner:
        branch.conv = new_inner
    else:
        branch = new_inner  # 原模块本身就是裸卷积
    return branch


def _find_stem_module(root: nn.Module) -> tuple[nn.Module | None, str, str]:
    """定位"吃输入的那一层"，返回 (父模块, 属性名, 用于日志的路径)。

    - YOLO：`model.0`（ultralytics Conv 包装器）—— 把 `model.model[0]` 换掉；
    - RT-DETR：`model.0` 是 HGStem，真正的第一层卷积是 `model.0.stem1.conv`，
      所以要把 **`stem1`（Conv 包装器）** 换掉（HGStem.forward 第一句就是 `x = self.stem1(x)`）。
    """
    first_conv_name = None
    for name, mod in root.named_modules():
        if isinstance(mod, nn.Conv2d):
            first_conv_name = name
            break
    if first_conv_name is None:
        return None, "", ""
    parts = first_conv_name.split(".")
    wrapper_path = ".".join(parts[:-1])
    wrapper = root.get_submodule(wrapper_path) if wrapper_path else root
    # 若父模块是"卷积包装器"（有 bn/act），就整体替换它，保留 BN 与激活
    if wrapper_path and (hasattr(wrapper, "bn") or hasattr(wrapper, "act")):
        target_path = wrapper_path
    else:
        target_path = first_conv_name
    parent_path, attr = target_path.rsplit(".", 1) if "." in target_path else ("", target_path)
    parent = root.get_submodule(parent_path) if parent_path else root
    return parent, attr, target_path


def inject_dual_stem(
    model: nn.Module,
    ckpt_path: str | Path,
    lam_init: float = 0.0,
    verbose: bool = True,
) -> bool:
    """把模型第一层卷积替换成 DualStem，并把预训练权重装进 RGB 分支。

    注意：必须在 `model.train()` 之前调用。ultralytics 的
    `BaseTrainer.setup_model()` 对已经构建好的 nn.Module 直接 return，
    不会重建模型，所以注入是安全的；且优化器在之后才创建，新参数会被正确纳入。

    ★ 2026-09-23 重写要点：
      1. 源权重不再用字符串后缀匹配（旧法在 RT-DETR 上会抓到错层）；
        改成"预训练模型里第一个 `in_channels==3` 的 Conv2d"；
      2. 目标层也按模块对象定位（支持 YOLO 的 `model.0` 与 RT-DETR 的 `model.0.stem1`）；
      3. 输出通道与预训练对不上时**返回 False**（调用方必须据此中断，不准带随机 stem 训练）。
    """
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # ★ 2026-09-24：训练检查点里 `["model"]` 可能是 None（ultralytics 有时只存 EMA），
        #   旧代码直接 `ckpt["model"].modules()` ⇒ `'NoneType' object has no attribute 'modules'`。
        #   这里按 model → ema → 自身 的顺序取，取不到就明确返回 False。
        pretrained = None
        if isinstance(ckpt, dict):
            for key in ("model", "ema"):
                cand = ckpt.get(key)
                if cand is not None:
                    pretrained = cand
                    break
        else:
            pretrained = ckpt
        if pretrained is None or not hasattr(pretrained, "modules"):
            print(f"[DualStem] 检查点里没有可用的模型权重（model/ema 均为空）: {ckpt_path}")
            return False
    except Exception as e:  # noqa: BLE001
        print(f"[DualStem] 读取预训练权重失败: {e}")
        return False

    src_mod = next((m for m in pretrained.modules()
                    if isinstance(m, nn.Conv2d) and m.in_channels == 3), None)
    if src_mod is None:
        print("[DualStem] 预训练权重里找不到 3 通道卷积")
        return False
    src_w = src_mod.weight.detach().clone()

    parent, attr, target_path = _find_stem_module(model)
    if parent is None:
        print("[DualStem] 模型里找不到任何 Conv2d")
        return False
    old = parent.get_submodule(attr) if hasattr(parent, "get_submodule") else getattr(parent, attr)
    old_conv = _inner_conv(old)
    c_out, c_in = old_conv.out_channels, old_conv.in_channels
    if c_in <= 3:
        print(f"[DualStem] 输入通道为 {c_in}，没有辅助模态，跳过")
        return False
    if src_w.shape[0] != c_out:
        print(f"[DualStem] 输出通道不匹配：目标 {c_out} vs 预训练 {src_w.shape[0]}，拒绝注入")
        return False

    new = DualStem(old, c_in, lam_init=lam_init)
    with torch.no_grad():
        # RGB 分支已经用 deepcopy 拿到了原模块的 BN/激活；这里把它的 conv 权重
        # 换成**预训练**的 3 通道权重（严谨继承 COCO 先验）
        rgb_inner = _inner_conv(new.rgb)
        rgb_inner.weight.copy_(src_w.to(rgb_inner.weight.dtype))
        if rgb_inner.bias is not None and src_mod.bias is not None:
            rgb_inner.bias.copy_(src_mod.bias.detach().to(rgb_inner.bias.dtype))
    setattr(parent, attr, new)

    # ultralytics 的 parse_model 会给每个模块附加 i/f/type/np 四个属性，
    # 手工注入的模块必须补齐，否则前向、模型摘要、保存逻辑都会报 AttributeError。
    new.i = 0
    new.f = -1
    new.type = "DualStem"
    new.np = sum(p.numel() for p in new.parameters())

    if verbose:
        print(
            f"[DualStem] 已注入到 {target_path}: 输入 {c_in}ch (RGB 3 + 辅助 {c_in - 3}) -> "
            f"{c_out}ch, λ={lam_init}（λ=0 时输出严格等于 RGB 分支 = 从基线出发）"
        )
    return True


def find_dual_stem(model: nn.Module) -> DualStem | None:
    """在模型里找出注入的 DualStem（供日志/统计用，兼容 YOLO 与 RT-DETR 两种挂点）。"""
    for m in model.modules():
        if isinstance(m, DualStem):
            return m
    return None


def make_trainer(lam_init: float = 0.0, ckpt_path: str | Path | None = None):
    """生成一个在 get_model 里自动注入 DualStem 的训练器子类。

    ★ 2026-09-23 修正：**按模型族选基类**。旧版无条件继承 `DetectionTrainer`（YOLO 专用），
    而 RT-DETR 必须走 `RTDETRTrainer`（数据类、验证器、get_model 都不一样），
    否则 `--fusion dualstem` 配 RT-DETR 会走错训练器。
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    base = DetectionTrainer
    if ckpt_path and "rtdetr" in Path(str(ckpt_path)).name.lower():
        from ultralytics.models.rtdetr.train import RTDETRTrainer

        base = RTDETRTrainer

    class FusionDetectionTrainer(base):  # type: ignore[misc, valid-type]
        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            # ★ 2026-09-24 更正：**续训时必须照常注入**。
            #   原因：ultralytics 续训会用 `get_model()` 重建一个"空壳"模型，再把检查点里的
            #   `ema` 权重 + **优化器状态**灌进来。若我们因为在 resume 就跳过注入，
            #   重建出来的模型没有 DualStem ⇒ 参数组数对不上 ⇒
            #   `ValueError: loaded state dict contains a parameter group that doesn't match
            #    the size of optimizer's group`（实测踩到）。
            #   注入是"把结构搭对"，权重与优化器状态随后由检查点恢复，所以不会重复包裹。
            if ckpt_path:
                ok = inject_dual_stem(model, ckpt_path, lam_init=lam_init)
                if not ok:
                    # 失败即中断：带着随机 stem 跑 8 小时等于把实验变成假结论
                    raise RuntimeError(
                        "DualStem 注入失败（见上方 [DualStem] 日志）—— 拒绝训练，请修好再跑"
                    )
            return model

    return FusionDetectionTrainer
