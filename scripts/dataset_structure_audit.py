"""数据本体的结构性审计（回答"我们是不是根本没看到根本原因"）。

查四件事，全部是**我们从未测过**的：
  1. 训练集内部的近重复度 —— 2000 张里到底有多少个"不同场景"？
     如果大量是同一段视频的相邻帧，那 ①有效数据量远小于 2000；
     ②我们的 val300 可能与训练集近重复（本地分被高估）；
     ③但更关键：**测试集里也可能有训练集的近重复**。
  2. 训练集 ↔ 测试集的近重复度 —— 若测试帧与训练帧大量近重复，
     那就解释了"为什么 60+ 的队伍这么多"（他们可能在用这个结构）。
  3. 文件名来源前缀的分布（train vs test）——数据是拼来的，
     若 test 的来源构成与 train 不同，就是**域偏移**。
  4. 分辨率 / 深度位深的构成（train vs test）——若 test 的异构子集比例不同，
     同样构成域偏移。

近重复用 32×32 灰度图 + 余弦相似度判定（对 JPEG 压缩、轻微位移鲁棒）。

用法：
    & $py scripts\\dataset_structure_audit.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"./data")
TRAIN_VIS = ROOT / "训练集" / "AIC2026_Train_2000" / "visible"
TRAIN_DP = ROOT / "训练集" / "AIC2026_Train_2000" / "depth"
TEST_VIS = ROOT / "测试集" / "AIC2026_PHASE_1_1000" / "visible"
TEST_DP = ROOT / "测试集" / "AIC2026_PHASE_1_1000" / "depth"
VAL_LIST = ROOT / "工作区" / "splits" / "lbproxy_val.txt"

N = 32


def thumb(p: Path) -> np.ndarray | None:
    try:
        im = Image.open(p)
        im.draft("L", (N * 4, N * 4))  # 先让 JPEG 解码器降采样，快很多
        g = np.asarray(im.convert("L").resize((N, N), Image.BILINEAR), dtype=np.float32)
    except Exception:
        return None
    v = g.reshape(-1)
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def build(files: list[Path]) -> tuple[np.ndarray, list[str]]:
    vecs, names = [], []
    for i, p in enumerate(files):
        v = thumb(p)
        if v is not None:
            vecs.append(v)
            names.append(p.stem)
        if (i + 1) % 500 == 0:
            print(f"    已处理 {i + 1}/{len(files)}", flush=True)
    return np.stack(vecs), names


def prefix(name: str) -> str:
    if re.fullmatch(r"\d+", name):
        return "纯数字(原始)"
    m = re.match(r"([A-Za-z]+)", name)
    return m.group(1) + "_" if m else "其它"


def main() -> None:
    tr_files = sorted(TRAIN_VIS.glob("*.jpg")) + sorted(TRAIN_VIS.glob("*.png"))
    te_files = sorted(TEST_VIS.glob("*.jpg")) + sorted(TEST_VIS.glob("*.png"))
    print(f"训练 visible {len(tr_files)} 张，测试 visible {len(te_files)} 张\n")

    print("[1/4] 提取训练集缩略图向量")
    A, an = build(tr_files)
    print("[2/4] 提取测试集缩略图向量")
    B, bn = build(te_files)
    print()

    # ---- 训练集内部：每张图与"除自己外最像的一张"的相似度
    S = A @ A.T
    np.fill_diagonal(S, -1.0)
    tr_best = S.max(axis=1)

    # ---- 测试集 → 训练集：每张测试图与训练集最像的一张
    S2 = B @ A.T
    te_best = S2.max(axis=1)

    # ---- 测试集内部：场景聚簇程度
    S3 = B @ B.T
    np.fill_diagonal(S3, -1.0)
    te_self = S3.max(axis=1)

    def hdr(t: str) -> None:
        print(f"\n=== {t} ===")
        print(f"{'相似度阈值':<14}{'训练集内部':>12}{'测试→训练':>12}{'测试内部':>12}")
        for th in (0.99, 0.97, 0.95, 0.90, 0.80):
            print(f"  >= {th:<10.2f}"
                  f"{(tr_best >= th).mean() * 100:>10.1f}%"
                  f"{(te_best >= th).mean() * 100:>11.1f}%"
                  f"{(te_self >= th).mean() * 100:>11.1f}%")
        print(f"  {'中位数':<11}{(np.median(tr_best)):>11.4f}"
              f"{(np.median(te_best)):>12.4f}{(np.median(te_self)):>12.4f}")

    hdr("近重复统计（占比 = 有多少张图能找到这么像的另一张）")

    # ---- val300 是否与训练集近重复（本地分是否被高估）
    val_stems = [Path(x.strip()).stem for x in VAL_LIST.read_text(encoding="utf-8").splitlines() if x.strip()]
    idx = {n: i for i, n in enumerate(an)}
    vi = [idx[s] for s in val_stems if s in idx]
    print(f"\nval300 命中训练集索引 {len(vi)} 张")
    if vi:
        Sv = A[vi] @ A.T
        Sv[np.arange(len(vi)), np.array(vi)] = -1.0
        vb = Sv.max(axis=1)
        for th in (0.99, 0.97, 0.95, 0.90):
            print(f"  val 能?? 到 >= {th:.2f} 的训练图：{(vb >= th).sum():>4} 张 "
                  f"({(vb >= th).mean() * 100:.1f}%)")
        print(f"  val 的最近邻相似度中位数 = {np.median(vb):.4f}")

    # ---- 最像的若干对（测试→训练），打印出来供人肉核对
    print("\n=== 测试→训练 最相似的 12 对（相似度 / 测试图 / 训练图）===")
    flat = np.argsort(S2.ravel())[::-1][:12]
    for k in flat:
        ti, ai = divmod(int(k), S2.shape[1])
        print(f"  {S2[ti, ai]:.4f}   {bn[ti]:<34} <- {an[ai]}")

    # ---- 文件名来源前缀
    print("\n=== 文件名来源前缀构成 ===")
    pt, pe = Counter(prefix(n) for n in an), Counter(prefix(n) for n in bn)
    keys = sorted(set(pt) | set(pe), key=lambda k: -(pt[k] + pe[k]))
    print(f"{'前缀':<18}{'训练':>8}{'占比':>9}{'测试':>8}{'占比':>9}")
    for k in keys:
        print(f"{k:<18}{pt[k]:>8}{pt[k] / len(an) * 100:>8.1f}%{pe[k]:>8}{pe[k] / len(bn) * 100:>8.1f}%")

    # ---- 分辨率 / 深度位深
    print("\n=== 分辨率与深度位深构成 ===")

    def stats(files: list[Path], label: str) -> None:
        sizes = Counter()
        for p in files:
            try:
                sizes[Image.open(p).size] += 1
            except Exception:
                pass
        print(f"  [{label}] 分辨率分布（前 5）：")
        for s, c in sizes.most_common(5):
            print(f"      {s[0]}x{s[1]:<6} {c:>5} 张  ({c / len(files) * 100:.1f}%)")

    stats(tr_files, "训练")
    stats(te_files, "测试")

    for dp_dir, label in ((TRAIN_DP, "训练"), (TEST_DP, "测试")):
        files = sorted(dp_dir.glob("*.png"))
        if not files:
            files = sorted(dp_dir.glob("*.jpg"))
        modes = Counter()
        sub = files[:: max(1, len(files) // 400)]
        for p in sub:
            try:
                im = Image.open(p)
                modes[(im.mode, str(im.size))] += 1
            except Exception:
                pass
        print(f"  [{label}] 深度图（抽样 {len(sub)} 张）mode/尺寸 分布：")
        for m, c in modes.most_common(5):
            print(f"      {m[0]:<8} {m[1]:<12} {c:>4}  ({c / max(1, len(sub)) * 100:.1f}%)")


if __name__ == "__main__":
    main()
