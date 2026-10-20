"""深度一致的稀有类粘贴增强（Depth-Consistent Rare-Class Copy-Paste）。

## 为什么需要它

本赛题是 **12 类宏平均**，每一类独占 8.33 分。而训练集极端长尾：

| 类别 | 训练框数 |
| --- | --- |
| tricycle | 25 |
| ball | 89 |
| boat | 135 |

这三类合计占 25% 的分数权重，却只有不到 250 个样本。**把它们从 AP 0.34 提到 0.6，
宏平均就是 +6.5 分** —— 这是目前识别出的最大单项杠杆。

## 为什么不能直接用常规 copy-paste

常规 copy-paste 会破坏几何一致性：把一个远处的物体贴到近处，它的像素尺度、
透视关系、以及（如果我们用深度）深度值全都对不上。这样教出来的模型会学到
错误的尺度-上下文关联。

## 我们的做法

利用本赛题独有的两个属性：
1. **三模态严格空间对齐** → RGB/红外/深度补丁可以一起搬；
2. **度量深度（毫米级绝对距离）** → 可以判断"这个目标贴在哪儿在几何上是合理的"。

粘贴时做两道几何校验：
- **深度一致性**：目标框的中位深度与目标落点的局部中位深度必须在容许范围内；
- **遮挡一致性**：若落点与某个更近的真实目标重叠，则该次粘贴作废（不能把东西
  贴在别人前面还指望模型学对）。

属于「更好利用既有数据」这一类——EEG 项目 29 连负定律里少数被验证有效的类别之一。
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ultralytics.utils.patches import imread

from depth_codec import DEFAULT_DMAX_MM, DEFAULT_DMIN_MM, decode_depth


class RareClassPaste:
    """按类别稀缺度采样 donor，并做深度一致性校验后粘贴。"""

    def __init__(
        self,
        labels_dir: str | Path,
        image_dir: str | Path,
        depth_dir: str | Path | None = None,
        rare_classes: tuple[int, ...] = (1, 7, 11),   # boat / ball / tricycle
        prob: float = 0.5,
        max_paste: int = 2,
        depth_tol: float = 0.35,
        dmin: float = DEFAULT_DMIN_MM,
        dmax: float = DEFAULT_DMAX_MM,
        depth_check: bool = False,
        extra_dirs: tuple[str | Path, ...] = (),
        extra_kinds: tuple[str, ...] = (),
        depth_encoding: str = "linear",
        tries: int = 40,
        feather: int = 3,
    ) -> None:
        self.labels_dir = Path(labels_dir)
        self.image_dir = Path(image_dir)
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.rare = tuple(rare_classes)
        self.prob = prob
        self.max_paste = max_paste
        self.depth_tol = depth_tol
        self.dmin, self.dmax = dmin, dmax
        # ★ 2026-09-23：深度一致性校验此前**从未生效**（阈值用了毫米、数据是 8bit，
        #   每次都在 `valid.size < 10` 处 continue）。修复后它才真的会拒绝，
        #   这是一个**行为变化**，所以用开关显式控制，默认保持"不过滤"以与
        #   已经打过榜的 rare_paste 臂（榜分 50.1170）可比。
        self.depth_check = bool(depth_check)
        self.depth_encoding = depth_encoding
        self.extra_dirs = tuple(Path(p) for p in extra_dirs)
        self.extra_kinds = tuple(extra_kinds) if extra_kinds else ("ir",) * len(self.extra_dirs)
        self.tries = tries
        self.feather = feather
        # 统计（用于确认增强真的在生效）
        self.n_calls = 0
        self.n_trigger = 0
        self.n_pasted = 0
        self.n_reject_depth = 0
        self.n_reject_iou = 0
        self.n_skip_shape = 0
        self._warned_dead = False

        # donor 池：每个稀有类 -> [(图像名, 框(归一化), 该框的中位深度)]
        self.pool: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)
        for lf in sorted(self.labels_dir.glob("*.txt")):
            rows = []
            for ln in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = ln.split()
                if len(p) >= 5:
                    c = int(float(p[0]))
                    if c in self.rare:
                        rows.append((c, np.array([float(x) for x in p[1:5]])))
            for c, b in rows:
                self.pool[c].append((lf.stem, b))
        self.classes = [c for c in self.rare if self.pool.get(c)]

    # ------------------------------------------------------------------ #
    def _depth_of(self, stem: str) -> np.ndarray | None:
        """返回**解码后**的深度图（float32，0 表示无效），与训练通道构造同源。"""
        if not self.depth_dir:
            return None
        for ext in (".png", ".jpg", ".jpeg"):
            f = self.depth_dir / f"{stem}{ext}"
            if f.exists():
                d = imread(f, flags=cv2.IMREAD_UNCHANGED)
                if d is None:
                    return None
                d8, _mask, _info = decode_depth(
                    d, self.dmin, self.dmax, self.depth_encoding, src=f
                )
                return d8.astype(np.float32)
        return None

    def _donor_image(self, stem: str) -> np.ndarray | None:
        """按**训练侧同样的通道构成**构造 donor 全图（BGR + 各辅助模态）。

        为什么必须这样：`__call__` 里有 `patch.shape[2] != img.shape[2] -> continue`，
        而 donor 原本只读 RGB（3 通道）。在 6 通道训练下这会让 100% 的粘贴被跳过，
        且**不报错**——稀有类过采样会静默消失。
        """
        base = None
        for ext in (".png", ".jpg", ".jpeg"):
            f = self.image_dir / f"{stem}{ext}"
            if f.exists():
                base = imread(f, flags=cv2.IMREAD_COLOR)
                break
        if base is None:
            return None
        h, w = base.shape[:2]
        parts = [base]
        for d, kind in zip(self.extra_dirs, self.extra_kinds):
            p = None
            for ext in (".png", ".jpg", ".jpeg"):
                cand = d / f"{stem}{ext}"
                if cand.exists():
                    p = cand
                    break
            if p is None:
                return None
            if kind == "depth":
                raw = imread(p, flags=cv2.IMREAD_UNCHANGED)
                if raw is None:
                    return None
                d8, mask, _info = decode_depth(raw, self.dmin, self.dmax, self.depth_encoding, src=p)
                if d8.shape[:2] != (h, w):
                    d8 = cv2.resize(d8, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                parts.append(np.stack([d8, mask], axis=2))
            else:
                im = imread(p, flags=cv2.IMREAD_GRAYSCALE)
                if im is None:
                    return None
                if im.ndim == 2:
                    im = im[:, :, None]
                if im.shape[:2] != (h, w):
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                    if im.ndim == 2:
                        im = im[:, :, None]
                parts.append(im)
        return np.concatenate(parts, axis=2)

    @staticmethod
    def _box_px(box: np.ndarray, h: int, w: int) -> tuple[int, int, int, int]:
        cx, cy, bw, bh = box
        x1 = int(max(0, (cx - bw / 2) * w))
        y1 = int(max(0, (cy - bh / 2) * h))
        x2 = int(min(w, (cx + bw / 2) * w))
        y2 = int(min(h, (cy + bh / 2) * h))
        return x1, y1, x2, y2

    @staticmethod
    def _median_depth(depth: np.ndarray | None, box_px) -> float | None:
        if depth is None:
            return None
        x1, y1, x2, y2 = box_px
        if x2 <= x1 or y2 <= y1:
            return None
        patch = depth[y1:y2, x1:x2]
        # 解码后的深度：0 = 无效
        valid = patch[patch > 0]
        return float(np.median(valid)) if valid.size >= 10 else None

    # ------------------------------------------------------------------ #
    def __call__(self, label: dict) -> dict:
        """在 ultralytics 的 label 字典上就地做粘贴。label 来自 get_image_and_label。"""
        self.n_calls += 1
        # 前 10 次每次都报（便于冒烟测试确认生效），之后每 500 次报一次
        if self.n_calls <= 10 or self.n_calls % 500 == 0:
            print(
                f"[rare_paste] 调用 {self.n_calls} 次 | 触发 {self.n_trigger} | "
                f"成功粘贴 {self.n_pasted} 个 | 因深度不符拒绝 {self.n_reject_depth} | "
                f"因重叠拒绝 {self.n_reject_iou} | 因通道不符跳过 {self.n_skip_shape} | "
                f"深度一致性={'开' if self.depth_check else '关'}",
                flush=True,
            )
        # ★ fail-fast：粘贴机制"看起来开着、实际一个都没贴上"是本项目已经踩过的坑
        #   （通道不符 -> continue -> 静默失效）。这里让它在日志里喊出来。
        if (not self._warned_dead) and self.n_calls >= 200:
            self._warned_dead = True
            if self.n_pasted == 0:
                print(
                    f"[rare_paste][!! 失效 !!] 已调用 {self.n_calls} 次、触发 {self.n_trigger} 次，"
                    f"但一次都没贴成功：通道不符跳过 {self.n_skip_shape} 次。"
                    f"请检查 extra_dirs / extra_kinds 与训练通道是否一致。",
                    flush=True,
                )
            elif self.depth_check and self.n_reject_depth == 0 and self.n_calls >= 500:
                print(
                    "[rare_paste][!! 可疑 !!] 深度一致性已开，但 500 次里一次都没拒绝过——"
                    "深度通道可能仍是空的（回顾 2026-09-23 的 8bit/毫米口径 bug）。",
                    flush=True,
                )
        if not self.classes or random.random() > self.prob:
            return label
        self.n_trigger += 1
        img = label.get("img")
        inst = label.get("instances")
        if img is None or inst is None or len(inst) == 0:
            return label

        im_file = Path(label["im_file"])
        hi, wi = img.shape[:2]
        ori_h, ori_w = label["ori_shape"][:2]
        sx, sy = wi / ori_w, hi / ori_h           # 原图 -> 当前 img 的缩放

        # 当前图的原始框（归一化 xywh）。注意：ultralytics 的 Instances 只存 bboxes，
        # 类别单独存在 label["cls"]（形状 n×1）里。
        cur = inst.bboxes.clone().numpy() if hasattr(inst.bboxes, "numpy") else np.asarray(inst.bboxes)
        cur_cls = np.asarray(label["cls"]).reshape(-1).astype(int) if label.get("cls") is not None \
            else np.zeros(len(cur), dtype=int)

        depth_cur = self._depth_of(im_file.stem)
        if depth_cur is not None and depth_cur.shape[:2] != (ori_h, ori_w):
            depth_cur = cv2.resize(depth_cur, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

        n_paste = 0
        for _ in range(self.max_paste):
            cls = random.choice(self.classes)      # 每类等概率 -> 稀有类被过采样
            stem, box = random.choice(self.pool[cls])
            donor_img = self._donor_image(stem)
            if donor_img is None:
                continue
            dh, dw = donor_img.shape[:2]
            x1, y1, x2, y2 = self._box_px(box, dh, dw)
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            patch = donor_img[y1:y2, x1:x2]
            if patch.ndim == 2:
                patch = patch[:, :, None]
            if patch.shape[2] != img.shape[2]:
                self.n_skip_shape += 1
                continue

            # donor 目标的中位深度
            donor_depth_map = self._depth_of(stem)
            donor_z = self._median_depth(
                donor_depth_map, (x1, y1, x2, y2)
            ) if donor_depth_map is not None else None

            ph, pw = patch.shape[:2]
            if ph >= hi or pw >= wi:
                continue

            placed = False
            for _t in range(self.tries):
                ty = random.randint(0, hi - ph)
                tx = random.randint(0, wi - pw)
                bx1, by1 = tx / sx, ty / sy                    # 转回原图坐标
                bw_, bh_ = pw / sx, ph / sy
                nb = np.array([(bx1 + bw_ / 2) / ori_w, (by1 + bh_ / 2) / ori_h,
                               bw_ / ori_w, bh_ / ori_h])

                # ① 不与现有目标重叠（IoU < 0.15）
                if len(cur) and self._max_iou(nb, cur) > 0.15:
                    self.n_reject_iou += 1
                    continue

                # ② 深度一致性 + 遮挡一致性（仅当 depth_check 打开）
                if self.depth_check and depth_cur is not None and donor_z is not None:
                    px1, py1 = int(bx1), int(by1)
                    px2, py2 = int(bx1 + bw_), int(by1 + bh_)
                    patch_d = depth_cur[max(0, py1):py2, max(0, px1):px2]
                    valid = patch_d[patch_d > 0]
                    if valid.size < 10:
                        continue
                    dest_z = float(np.median(valid))
                    if abs(dest_z - donor_z) / donor_z > self.depth_tol:
                        self.n_reject_depth += 1
                        continue                                  # 深度不符：贴过去尺度会错
                    # 落点若被更近的真实目标压住 -> 拒绝
                    blocked = False
                    for b_, c_ in zip(cur, cur_cls):
                        if self._iou(nb, b_) > 0.1:
                            ez = self._median_depth(depth_cur, self._box_px(b_, ori_h, ori_w))
                            if ez is not None and ez < donor_z * 0.8:
                                blocked = True
                                break
                    if blocked:
                        continue

                self._blend(img, patch, tx, ty)
                cur = np.vstack([cur, nb]) if len(cur) else nb[None, :]
                cur_cls = np.append(cur_cls, cls)
                n_paste += 1
                self.n_pasted += 1
                placed = True
                # 立刻同步标签，保证"图像改动"与"标签改动"原子发生
                self._sync_labels(label, cur, cur_cls)
                break
            if placed and n_paste >= self.max_paste:
                break

        return label

    # ------------------------------------------------------------------ #
    def _blend(self, img: np.ndarray, patch: np.ndarray, tx: int, ty: int) -> None:
        h, w = patch.shape[:2]
        if img.shape[2] != patch.shape[2]:
            return
        a = np.ones((h, w), dtype=np.float32)
        f = max(1, self.feather)
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        a[:f, :] *= ramp[:, None]
        a[-f:, :] *= ramp[::-1, None]
        a[:, :f] *= ramp[None, :]
        a[:, -f:] *= ramp[::-1][None, :]
        a = a[:, :, None]
        dst = img[ty:ty + h, tx:tx + w].astype(np.float32)
        img[ty:ty + h, tx:tx + w] = (a * patch + (1 - a) * dst).astype(img.dtype)

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
        ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
        bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
        bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = a[2] * a[3] + b[2] * b[3] - inter
        return float(inter / ua) if ua > 0 else 0.0

    @classmethod
    def _max_iou(cls, nb: np.ndarray, boxes: np.ndarray) -> float:
        return max((cls._iou(nb, b) for b in boxes), default=0.0)

    @staticmethod
    def _sync_labels(label: dict, boxes: np.ndarray, cls: np.ndarray) -> None:
        """把新框与类别写回 label。

        关键：ultralytics 的 `Instances` 只承载 bboxes/segments/keypoints，
        类别单独存在 `label["cls"]`（形状 n×1）。两者必须同步更新，
        否则会出现"图上有目标、标签里没有"的假阴性，反而伤害训练。
        `Instances(bboxes, segments, keypoints, bbox_format, normalized)` 不接受 cls 参数。
        """
        from ultralytics.utils.instance import Instances

        # ⚠️ 必须传 numpy —— ultralytics 的 Instances 全程按 numpy 约定工作。
        # 传 torch.Tensor 会让 bboxes.dtype == torch.float32，下游
        # `np.ones(..., dtype=bboxes.dtype)` 会抛
        # "TypeError: Cannot interpret 'torch.float32' as a data type"。
        # 这个 bug 曾在第 71 轮把训练打断。
        label["instances"] = Instances(
            bboxes=np.asarray(boxes, dtype=np.float32),
            bbox_format="xywh",
            normalized=True,
        )
        label["cls"] = np.asarray(cls, dtype=np.float32).reshape(-1, 1)
