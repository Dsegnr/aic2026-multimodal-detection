"""多模态（RGB + 红外 + 深度）数据管线补丁，面向 ultralytics 8.4.x。

设计要点
--------
1. **零额外磁盘**：不预生成融合图片，而是在 `load_image` 里就地读取并拼接通道。
   服务器 /home 只剩 ~20GB，生成 4 通道 TIFF 需要 8-16GB，不可行。
2. **零侵入**：不修改 ultralytics 源码，全部通过子类 + monkeypatch 完成；
   升级 ultralytics 不会破坏我们的代码。
3. **通道数由数据 yaml 的 `channels` 键驱动**：ultralytics 8.4 的
   `DetectionTrainer.get_model()` 会读取 `data["channels"]` 来构建主干第一层，
   所以模型侧不需要改结构。

实测确认（ultralytics 8.4.150）：
- mosaic 画布用 `labels["img"].shape[2]` 取通道数 → 通道无关
- LetterBox 用 `np.full((..., c), ...)` → 通道无关
- RandomPerspective 的 borderValue 已经是 4 元组，注释明确说是为多光谱设计
- RandomHSV 有 `if img.shape[-1] != 3: return labels` 保护
所以只需替换图像加载环节。
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.patches import imread

from depth_codec import DEFAULT_DMAX_MM, DEFAULT_DMIN_MM, decode_depth
from mm_build import build_mm_array


class MultiModalYOLODataset(YOLODataset):
    """在标准 YOLO 数据集基础上，把若干辅助模态按通道拼到主图上。

    数据 yaml 需要两个额外键：
        channels: 4                      # 1 + 辅助模态数（RGB=3）
        extra_modality_dirs:             # 与主图同名的辅助模态目录
          - /path/to/infrared/images
          - /path/to/depth/images        # 可选
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_dirs = [Path(p) for p in (self.data.get("extra_modality_dirs") or [])]
        # 每个辅助模态的类型：'ir'（热红外灰度）或 'depth'（深度）
        self._kinds = self.data.get("extra_modality_types") or ["ir"] * len(self.extra_dirs)
        # 深度归一化参数：**仅对 >=16bit 的度量深度生效**；8bit 数据走 depth_codec 的
        # "0=无效、非零即有效"分支（本赛题实际就是 8bit，见 depth_codec 模块注释）。
        self._dmin = float(self.data.get("depth_min_mm", DEFAULT_DMIN_MM))
        self._dmax = float(self.data.get("depth_max_mm", DEFAULT_DMAX_MM))
        self._depth_mode = self.data.get("depth_encoding", "linear")  # linear / inverse / log
        self._rare_paste = None
        self._depth_logged = False

    # ------------------------------------------------------------------ #
    def _init_rare_paste(self):
        """惰性构造稀有类粘贴增强（只在训练时启用）。"""
        from rare_paste import RareClassPaste

        cfg = self.data.get("rare_paste") or {}
        if not cfg.get("enable"):
            return None
        # 深度目录：优先用 yaml 里声明的深度模态目录
        depth_dir = None
        for d, kind in zip(self.extra_dirs, self._kinds):
            if kind == "depth":
                depth_dir = d
        if depth_dir is None:
            cfg_depth = cfg.get("depth_dir")
            depth_dir = Path(cfg_depth) if cfg_depth else None
        return RareClassPaste(
            labels_dir=Path(self.label_files[0]).parent if hasattr(self, "label_files") else Path(cfg["labels_dir"]),
            image_dir=Path(self.im_files[0]).parent,
            depth_dir=depth_dir,
            rare_classes=tuple(cfg.get("classes", [1, 7, 11])),
            prob=float(cfg.get("prob", 0.5)),
            max_paste=int(cfg.get("max_paste", 2)),
            depth_tol=float(cfg.get("depth_tol", 0.35)),
            # ★ 2026-09-23 修复：donor 补丁必须与训练图**同通道构成**，否则 6 通道下
            #   `patch.shape[2] != img.shape[2]` 会让所有粘贴被静默跳过。
            extra_dirs=tuple(self.extra_dirs),
            extra_kinds=tuple(self._kinds),
            depth_encoding=self._depth_mode,
            depth_check=bool(cfg.get("depth_check", False)),
        )

    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        if self.augment and self.data.get("rare_paste", {}).get("enable"):
            if self._rare_paste is None:
                self._rare_paste = self._init_rare_paste()
            if self._rare_paste is not None:
                try:
                    label = self._rare_paste(label)
                except Exception as e:  # noqa: BLE001
                    # 增强失败不应中断训练
                    if not getattr(self, "_rare_paste_warned", False):
                        print(f"[rare_paste] 跳过一次粘贴: {type(e).__name__}: {e}")
                        self._rare_paste_warned = True
        return label

    # ------------------------------------------------------------------ #
    # 说明：辅助模态的读取/拼接**已全部搬到 `mm_build.build_mm_array()`**。
    # 这里不再保留第二份实现 —— 两份实现漂移正是 2026-09-23 那次
    # "深度通道恒为 0" 事故的根因。要改通道构造，请只改 mm_build。

    # ------------------------------------------------------------------ #
    def load_image(self, i: int, rect_mode: bool = True, resize_short: bool = False):
        """与 BaseDataset.load_image 逻辑一致，但在读取后拼接辅助模态通道。"""
        if not self.extra_dirs:
            return super().load_image(i, rect_mode=rect_mode, resize_short=resize_short)

        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is not None:  # 已在 RAM 缓存
            return self.ims[i], self.im_hw0[i], self.im_hw[i]

        if fn.exists():
            try:
                im = np.load(fn)
                n_ch = im.shape[-1] if im.ndim >= 3 else 1
                if n_ch != self.channels:
                    Path(fn).unlink(missing_ok=True)
                    im = None
            except Exception:
                Path(fn).unlink(missing_ok=True)
                im = None

        if im is None:
            # ★ 2026-09-23：通道构造统一走 mm_build（训练/增强/推理唯一真源）。
            im = build_mm_array(
                f,
                extra_dirs=tuple(self.extra_dirs),
                kinds=tuple(self._kinds),
                dmin=self._dmin,
                dmax=self._dmax,
                encoding=self._depth_mode,
                tag="train",
            )
            if im.shape[2] != self.channels:
                raise RuntimeError(
                    f"构造出的通道数 {im.shape[2]} != data yaml 的 channels={self.channels}（{f}）"
                )

        h0, w0 = im.shape[:2]
        if rect_mode:
            if resize_short:
                r = self.imgsz / min(h0, w0)
                if r != 1:
                    w, h = (math.ceil(w0 * r), self.imgsz) if h0 < w0 else (self.imgsz, math.ceil(h0 * r))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        if self.augment and self.cache != "ram":
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

        return im, (h0, w0), im.shape[:2]


def install(channels: int = 4) -> None:
    """让 ultralytics 的构建流程使用上面的多模态数据集。"""
    import ultralytics.data.build as build_mod

    # 注意：必须无条件挂载。即使 channels==3（纯 RGB），只要启用了
    # rare_paste 之类的数据侧增强，也需要走我们的数据集类。
    # 没有额外模态时 load_image 会自动回退到父类实现，无副作用。
    build_mod.YOLODataset = MultiModalYOLODataset


def install_rtdetr() -> None:
    """让 RT-DETR 走**同一套**多模态数据集（通道拼接 + 稀有类粘贴）。

    为什么需要单独一个：RT-DETR 用的是
    `ultralytics.models.rtdetr.val.RTDETRDataset`。它虽然**继承自** YOLODataset，
    但 `build_dataset` 是直接按类名引用的，**不会**经过 build_mod.YOLODataset 这个名字，
    所以 `install()` 里那次替换对它无效。

    ## 2026-09-23 修正（重要）

    旧实现让这个子类**只继承 `RTDETRDataset`**（=纯 YOLODataset），并显式把
    `extra_dirs` 清空、注释写着"只跑 RGB 单模态，不做通道拼接"。
    后果：`channels: 6` 的 RT-DETR 臂根本拼不上辅助通道（模型建 6 通道、
    数据给 3 通道），而且因为 RTDETRDataset 自己重写了 `load_image` 转调
    `super().load_image`，我们的 `MultiModalYOLODataset.load_image` 也**不会被调用**。

    现在改成**继承 `MultiModalYOLODataset`**，并保留 RT-DETR 的 `load_image(i, rect_mode=False)`
    签名（RT-DETR 训练/验证都用 rect=False）。

    另外必须同时改 `ultralytics.models.rtdetr.train.RTDETRDataset`：
    那一行 `from .val import RTDETRDataset` 在 import 期就把名字绑定了，
    只改 `val` 模块的全局名不影响它（这个坑会让"改了却没生效"）。
    """
    import ultralytics.models.rtdetr.val as rv

    base = rv.RTDETRDataset
    if getattr(base, "_aic_paste_patched", False):
        return

    class RTDETRDatasetWithPaste(MultiModalYOLODataset):
        """多模态 RT-DETR 数据集（通道拼接 + 稀有类粘贴）。"""

        _aic_paste_patched = True

        def load_image(self, i, rect_mode=False):
            # RT-DETR 侧固定 rect_mode=False；通道拼接逻辑完全复用父类。
            return MultiModalYOLODataset.load_image(self, i, rect_mode=rect_mode)

    rv.RTDETRDataset = RTDETRDatasetWithPaste
    patched = ["val.RTDETRDataset"]
    try:
        import ultralytics.models.rtdetr.train as rt

        rt.RTDETRDataset = RTDETRDatasetWithPaste
        patched.append("train.RTDETRDataset")
    except Exception as e:  # noqa: BLE001
        print(f"[mm] 警告：未能改写 rtdetr.train 的引用（{type(e).__name__}: {e}）")
    print(f"[mm] RT-DETR 数据集已换成多模态版（含通道拼接 + 稀有类粘贴）：{', '.join(patched)}")


def expand_stem_weights(model, ckpt_path: str | Path, verbose: bool = True) -> bool:
    """把 3 通道预训练主干的第一层扩展到 N 通道（**返回 False 表示没成功**）。

    ultralytics 在 `channels != 3` 时会因为形状不匹配而跳过第一层卷积，
    导致整个 stem 随机初始化，白丢 COCO 先验。这里手动把原 3 通道权重填进去，
    新增通道用原权重的均值初始化（等价于把 RGB 灰度化后接上去）。

    ## 2026-09-23 重写（原实现有静默 bug，会毁掉整条多模态实验）

    旧实现用**字符串匹配**找源权重：`next(k for k in sd if k.endswith("0.conv.weight"))`。
    * YOLO 族：`model.0.conv.weight` 排在最前，侥幸正确；
    * **RT-DETR 族**：stem 叫 `model.0.stem1.conv`，而 `...0.conv.weight` 会匹配到
      别的层（实测拿到 `(48,48,3,3)` 的张量）。于是输出通道对不上 ⇒ 返回 False ⇒
      **第一层卷积保持随机初始化**，而调用方过去没有检查返回值。
      这正好是"多模态明明该有增益却总是 0/负"的可疑病因之一。

    新实现**不依赖任何字符串**：
      1. 目标卷积 = 模型里**第一个 `in_channels == 目标通道数` 的 Conv2d**
         （目标通道数取自 `model.yaml["ch"]`，由 ultralytics 按 data yaml 的
         `channels` 写入）；
      2. 源权重 = 预训练 ckpt 里**第一个 `in_channels == 3` 的 Conv2d 的权重**；
      3. 两者 out_channels 必须一致，否则返回 False 并由调用方**中断训练**。
    """
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        pretrained = ckpt["model"]
    except Exception as e:  # noqa: BLE001
        print(f"[stem] 读取预训练权重失败: {e}")
        return False

    # 目标层 = 模块顺序上的第一个 Conv2d（YOLO 的 model.0.conv、RT-DETR 的
    # model.0.stem1.conv 都是它）；它的输入通道数就是本模型的实际输入通道数。
    # ⚠️ 不能用 model.yaml["ch"]：实测 RTDETRDetectionModel(ch=6) 建出来的模型
    #    第一层确实是 6 通道，但 yaml 字典里仍写着 3（ultralytics 没回写），
    #    用它会误判成"无需扩展"。
    stem = next((m for m in model.modules() if isinstance(m, torch.nn.Conv2d)), None)
    if stem is None:
        print("[stem] 模型里没有任何 Conv2d")
        return False
    want_in = int(stem.in_channels)
    if want_in <= 3:
        if verbose:
            print(f"[stem] 第一层输入通道 = {want_in}，无需扩展")
        return True
    src_mod = next((m for m in pretrained.modules()
                    if isinstance(m, torch.nn.Conv2d) and m.in_channels == 3), None)
    if src_mod is None:
        print("[stem] 预训练权重里找不到 3 通道卷积")
        return False

    src = src_mod.weight.detach().clone()
    if stem.weight.shape[0] != src.shape[0]:
        print(f"[stem] 输出通道不一致：目标 {tuple(stem.weight.shape)} vs 源 {tuple(src.shape)}，拒绝扩展")
        return False
    extra = want_in - 3
    # 均值在 float32 里算：预训练 ckpt 是 fp16，若直接在 fp16 上做 reduce，
    # 累积误差可达 1e-3 量级（会被后续的"是否真的扩展成功"自检抓出来）。
    mean_ch = src.float().mean(dim=1, keepdim=True)
    with torch.no_grad():
        stem.weight[:, :3].copy_(src.to(stem.weight.dtype))
        stem.weight[:, 3:].copy_(mean_ch.repeat(1, extra, 1, 1).to(stem.weight.dtype))
        if (stem.bias is not None and src_mod.bias is not None
                and stem.bias.shape == src_mod.bias.shape):
            stem.bias.copy_(src_mod.bias.detach().to(stem.bias.dtype))
    if verbose:
        print(f"[stem] 已扩展 {tuple(src.shape)} -> {tuple(stem.weight.shape)}；"
              f"新增 {extra} 个通道用 RGB 权重均值初始化；目标层={type(stem).__name__}")
    return True
