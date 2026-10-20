"""深度图编解码 —— **全项目唯一真源**。

## 为什么必须单独成模块

2026-09-23 发现一个**静默失效**：本赛题深度图是 **8bit JPEG**（0=无效，非零即有效），
而训练侧 `mm_patch._load_extra`、增强侧 `rare_paste._median_depth`、推理侧
`mm_infer.build_mm6` 三处都写死了"16bit 毫米值，有效区间 [300, 20000]"。

后果（全都不报错）：

* 训练侧的深度通道 + 有效掩码**恒为全 0** ⇒ 模型看到的"深度模态"是一张黑图；
* 增强侧 `_median_depth` 恒返回 None ⇒ **深度一致性校验从未生效过**，
  我们榜分最好的那一臂（50.1170）实际跑的是**普通 copy-paste**；
* 推理侧同样全 0 ⇒ 就算训练对了，推理也会喂黑图。

⇒ 归结为一条工程纪律：**跨模块共享的数值约定必须只有一个实现**，
   而且第一个样本对不上时就要**报错**，不能 `continue`。

## 本数据集实测（2026-09-23，**全量 1000+2000 张逐张统计**）

⚠️ 深度是**两族混编**的，不是单一位深：

| 数据 | 族 A | 族 B |
|---|---|---|
| 训练集 depth（2000） | **1851 张 uint16** PNG 1920×1080（毫米值，直接用 `[300,20000]` mm 判有效） | **149 张 uint8** JPEG 360×640（R=G=B，0=无效，已是归一化值） |
| 复赛 depth（1000） | **883 张 uint16** PNG 1920×1080 | **117 张 uint8** JPEG 360×640（其中 38 张因 JPEG 色度子采样出现 \|R-G\|max=6） |
| 不可读/异常 | — | 0 张 |

**两族归一化后同尺度**（实测）：族 A 用 `(d_mm−300)/19700×255` 得到的非零中位为
53~138，族 B 的非零中位为 138~146 ⇒ 混编后仍可比较（这条支撑 rare_paste 的深度一致性校验）。

族 B 的非零值域恒铺满 1–255（多张图中位都是 146），说明是**全局归一化**而非逐图 min-max。

### 旧口径的真实后果（**不是"全网全 0"，而是按族分裂**）

旧代码写死 `valid = (300 <= d <= 20000)`：

* 族 A（uint16 毫米）：**口径恰好正确**，正常参与训练；
* 族 B（uint8 归一化）：`d ∈ [0,255]` ⇒ **恒不接受** ⇒ 该图深度通道与掩码**全为 0**，
  即 **训练集 149/2000 = 7.5%**、复赛 **117/1000 = 11.7%** 的样本深度是黑图，
  而且**不报错**。

（我第一次抽样时按文件名排序取前 12 张，恰好全是族 B，于是误判为"全数据集恒为全 0"——
教训：**抽样要分层/随机，不能取排序后的前 N 个**。）

## 两种位深各自怎么处理

* `uint8`（本赛题实际）：`valid = d > 0`，归一化 `n = d / 255`——数据已经是归一化好的，
  **不得**再套毫米阈值。
* `uint16`/更大整数位深（兼容官方可能换发的 16bit 版本）：`valid = dmin <= d <= dmax`，
  按 `encoding` 做 linear / inverse / log 归一化。

对不上就 `raise`，让日志里一眼能看到。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# 官方细则：深度像素值为 0 或过小代表无效深度区域。默认毫米区间（仅对 >=16bit 生效）。
DEFAULT_DMIN_MM = 300.0
DEFAULT_DMAX_MM = 20000.0

# JPEG 色度子采样（4:2:0）会让"灰度存成彩色"的图出现 R≠G 的微小偏差。
# 实测复赛深度集：117 张 8bit JPEG 中 38 张 |R-G|max == 6，其余为 0；
# 真正的彩色映射（colormap 渲染的深度）差异会是几十上百。故取 16 作判据。
GRAY_TOL = 16


class DepthFormatError(RuntimeError):
    """深度图格式与配置不符 —— 必须让人看到，不允许静默降级。"""


def _to_2d(raw: np.ndarray, src: str | Path | None = None) -> np.ndarray:
    if raw is None:
        raise DepthFormatError(f"深度图为空/读取失败: {src}")
    if raw.ndim == 3:
        c = raw.shape[2]
        if c == 1:
            # ★ 实测：ultralytics 的 imread 会把灰度图补成 (H, W, 1)，
            #   而 cv2.imread 给的是 (H, W)。两种都要能处理（否则读第二张就 IndexError）。
            raw = raw[..., 0]
        elif c >= 3:
            # JPEG 灰度会存成 3 通道（R=G=B）。**不能要求严格相等**：色度子采样会
            # 引入 ≤6 量级的偏差（实测），严格判定会把正常的 38/117 张图判成异常。
            # 用容差判据：差异在 GRAY_TOL 内 ⇒ 当灰度处理（取通道均值，抑制噪声）；
            # 超过 ⇒ 真的是彩色（例如 colormap 渲染），无法当标量深度，直接报错。
            diff = int(np.abs(raw[..., 0].astype(np.int16) - raw[..., 1].astype(np.int16)).max())
            if diff > GRAY_TOL:
                raise DepthFormatError(
                    f"深度图不是灰度（|R-G|max={diff} > {GRAY_TOL}），无法按标量深度解读: {src} "
                    f"dtype={raw.dtype} shape={raw.shape}"
                )
            raw = raw.mean(axis=2).round().astype(raw.dtype)
        else:
            raise DepthFormatError(f"深度图通道数异常 {c}: {src} shape={raw.shape}")
    if raw.ndim != 2:
        raise DepthFormatError(f"深度图维度异常: {src} shape={raw.shape}")
    return raw


def decode_depth(
    raw: np.ndarray,
    dmin: float = DEFAULT_DMIN_MM,
    dmax: float = DEFAULT_DMAX_MM,
    encoding: str = "linear",
    src: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """把原始深度数组解码成 `(depth_u8, valid_mask_u8, info)`。

    返回的两个数组都是 `HxW uint8`：

    * `depth_u8`：无效像素为 0；有效像素按 `encoding` 归一化到 1..255（线性时近似）。
    * `valid_mask_u8`：有效为 255，无效为 0。

    `info` 里带位深/刻度/有效占比，调用方**应当把它打印出来**（否则又是一次静默）。
    """
    raw = _to_2d(raw, src)
    dt = raw.dtype

    if dt == np.uint8:
        d = raw.astype(np.float32)
        valid = d > 0.0
        scale = "uint8_normalized"
        if encoding == "inverse":
            n = 1.0 - d / 255.0
        elif encoding == "log":
            # 已归一化的 0..255 梯度上做 log 展宽（把近处细节拉开）
            dd = np.maximum(d, 1.0)
            n = np.log(dd) / np.log(255.0)
        else:
            n = d / 255.0
    elif dt.itemsize >= 2 and np.issubdtype(dt, np.integer):
        if dmax <= dmin:
            raise DepthFormatError(f"深度区间非法: dmin={dmin} dmax={dmax}")
        d = raw.astype(np.float32)
        valid = (d >= dmin) & (d <= dmax)
        scale = f"metric_mm[{dmin:g},{dmax:g}]"
        if encoding == "inverse":
            z = 1.0 / np.maximum(d, 1e-6)
            lo, hi = 1.0 / dmax, 1.0 / dmin
            n = (z - lo) / (hi - lo)
        elif encoding == "log":
            dd = np.clip(d, dmin, dmax)
            n = (np.log(dd) - np.log(dmin)) / (np.log(dmax) - np.log(dmin))
        else:
            n = (d - dmin) / (dmax - dmin)
    else:
        raise DepthFormatError(f"不支持的深度位深 {dt}（{src}）")

    d8 = np.clip(n * 255.0, 0, 255).astype(np.uint8)
    d8[~valid] = 0
    mask = valid.astype(np.uint8) * 255
    info = {
        "scale": scale,
        "dtype": str(dt),
        "valid_frac": float(valid.mean()) if valid.size else 0.0,
        "encoding": encoding,
    }
    return d8, mask, info


def assert_depth_usable(
    samples: list[tuple[str, np.ndarray]],
    dmin: float = DEFAULT_DMIN_MM,
    dmax: float = DEFAULT_DMAX_MM,
    encoding: str = "linear",
) -> dict:
    """**fail-fast 探针**：给定若干 `(名字, 原始深度数组)`，确认深度通道不是全 0。

    全部样本的 `valid_frac == 0` ⇒ 直接抛错。理由：深度通道全 0 时训练**不会报错**，
    只会安静地把"多模态"退化成"单模态 + 两张黑图"，把一次 6 小时的实验变成假结论。
    """
    infos = []
    for name, raw in samples:
        _d8, mask, info = decode_depth(raw, dmin, dmax, encoding, src=name)
        info["name"] = str(name)
        infos.append(info)
    if not infos:
        raise DepthFormatError("深度探针没有拿到任何样本")
    best = max(i["valid_frac"] for i in infos)
    if best <= 0.0:
        detail = "; ".join(f"{i['name']}:{i['dtype']} valid={i['valid_frac']:.3f}" for i in infos[:5])
        raise DepthFormatError(
            "深度通道全为无效（valid_frac=0）—— 位深/阈值不匹配，拒绝继续。样本：" + detail
        )
    summary = {
        "n": len(infos),
        "scale": infos[0]["scale"],
        "dtype": infos[0]["dtype"],
        "valid_frac_mean": sum(i["valid_frac"] for i in infos) / len(infos),
        "valid_frac_min": min(i["valid_frac"] for i in infos),
        "valid_frac_max": best,
    }
    return summary
