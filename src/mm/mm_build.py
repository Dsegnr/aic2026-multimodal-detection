"""多模态通道构造 —— **训练、增强、推理三处共用同一份实现**。

## 为什么必须共用

通道构造一旦在训练侧和推理侧不一致（位深、归一化、通道顺序、无效像素处理），
后果是**静默掉分**：模型能跑、日志正常、分数接近随机。本项目已有实例
（8bit 深度被当成毫米值 ⇒ 深度通道恒为全 0，见 `depth_codec` 的模块注释）。

所以这里规定：**任何要拼多模态通道的地方，都必须调 `build_mm_array()`。**

## 通道顺序（与 `configs/server_mm6_rare.yaml`、历史 6 次实验一致）

```
0-2  BGR（cv2 读入的原样，不翻转 —— ultralytics 只在 3 通道时才做 BGR→RGB）
3    红外灰度
4    深度归一化（无效像素 0）
5    深度有效掩码（0/255）
```

新增模态只需在 yaml 的 `extra_modality_dirs` / `extra_modality_types` 里追加。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from depth_codec import DEFAULT_DMAX_MM, DEFAULT_DMIN_MM, decode_depth

_LOGGED: set[str] = set()


def _read_exact(path: Path, flags: int, what: str) -> np.ndarray:
    """读图；失败**直接抛错**（调用方需要决定是跳过整个样本还是中断）。"""
    from ultralytics.utils.patches import imread

    im = imread(path, flags=flags)
    if im is None:
        raise FileNotFoundError(f"{what}读取失败: {path}")
    return im


def resolve_modality_path(rgb_path: Path, kind: str, extra_dirs: tuple[Path, ...] = ()) -> Path:
    """由可见光路径推出同 stem 的辅助模态路径。

    两种策略（按顺序尝试；都不成 ⇒ 抛错并列出试过的路径，绝不静默返回 None）：

    1. **显式目录**（推荐、也是 yaml 里的方式）：`extra_dirs` 里每个目录下找同名文件；
    2. **路径段替换**（历史兼容）：把路径里的 `aic` 段换成 `aic_ir` / `aic_dp` /
       `aic_ir_images` / `aic_dp_images`（5090 上的实际命名）。
    """
    tried: list[str] = []
    for d in extra_dirs:
        cand = Path(d) / rgb_path.name
        tried.append(str(cand))
        if cand.exists():
            return cand
    alias = {"ir": ("aic_ir", "aic_ir_images", "infrared"), "depth": ("aic_dp", "aic_dp_images", "depth")}[kind]
    parts = list(rgb_path.parts)
    for i, seg in enumerate(parts):
        if seg in ("aic", "images", "visible"):
            for a in alias:
                cand = Path(*parts[:i], a, *parts[i + 1:])
                tried.append(str(cand))
                if cand.exists():
                    return cand
    raise FileNotFoundError(
        f"找不到 {kind} 模态图（试过 {len(tried)} 个路径，前 4 个：{tried[:4]}）: {rgb_path}"
    )


def build_mm_array(
    rgb_path: str | Path,
    extra_dirs: tuple[str | Path, ...] = (),
    kinds: tuple[str, ...] = ("ir", "depth"),
    dmin: float = DEFAULT_DMIN_MM,
    dmax: float = DEFAULT_DMAX_MM,
    encoding: str = "linear",
    tag: str = "build",
) -> np.ndarray:
    """构造多模态数组 `HxWxC`（C = 3 + 每个辅助模态的通道数）。

    深度模态贡献 2 个通道（归一化深度 + 有效掩码），与历史实现一致。
    """
    rgb_path = Path(rgb_path)
    extra_dirs = tuple(Path(p) for p in extra_dirs)
    kinds = tuple(kinds) if kinds else ("ir",) * len(extra_dirs)
    if len(kinds) != len(extra_dirs):
        raise ValueError(f"kinds({len(kinds)}) 与 extra_dirs({len(extra_dirs)}) 长度不一致")

    parts = [_read_exact(rgb_path, cv2.IMREAD_COLOR, "主图")]
    for kind, d in zip(kinds, extra_dirs):
        p = resolve_modality_path(rgb_path, kind, (d,))
        if kind == "depth":
            raw = _read_exact(p, cv2.IMREAD_UNCHANGED, "深度图")
            d8, mask, info = decode_depth(raw, dmin, dmax, encoding, src=p)
            key = f"depth:{info['scale']}:{info['encoding']}"
            if key not in _LOGGED:
                _LOGGED.add(key)
                if info["valid_frac"] <= 0:
                    raise RuntimeError(
                        f"[mm_build] 深度有效率为 0，配置与数据不匹配：{p} dtype={info['dtype']}"
                    )
                print(
                    f"[mm_build:{tag}] 深度解码={info['scale']} dtype={info['dtype']} "
                    f"encoding={info['encoding']} 本样本有效率={info['valid_frac']:.3f}",
                    flush=True,
                )
            parts.append(np.stack([d8, mask], axis=2))
        else:
            im = _read_exact(p, cv2.IMREAD_GRAYSCALE, f"{kind} 图")
            if im.ndim == 3 and im.shape[2] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if im.ndim == 2:
                im = im[:, :, None]
            parts.append(im)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=2)


def channels_of(extra_kinds: tuple[str, ...]) -> int:
    """总通道数：RGB 3 + 每个 ir 1 + 每个 depth 2。"""
    return 3 + sum(2 if k == "depth" else 1 for k in extra_kinds)


def load_mm_config(yaml_path: str | Path) -> dict:
    """读训练用 data yaml，取出构造通道所需的最小配置集（保证与训练同源）。"""
    import yaml

    cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    kinds = tuple(cfg.get("extra_modality_types") or ["ir"] * len(cfg.get("extra_modality_dirs") or []))
    return {
        "extra_dirs": tuple(cfg.get("extra_modality_dirs") or []),
        "kinds": kinds,
        "dmin": float(cfg.get("depth_min_mm", DEFAULT_DMIN_MM)),
        "dmax": float(cfg.get("depth_max_mm", DEFAULT_DMAX_MM)),
        "encoding": cfg.get("depth_encoding", "linear"),
        "channels": int(cfg.get("channels", channels_of(kinds))),
    }
