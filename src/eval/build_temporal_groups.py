"""P0：把 2000 张训练图按**内容相似度**归成"时间/场景组"，为分组验证做准备。

背景（Astra 第 5 次回答）：
    我们的随机 300 张 val 来自训练池，连续视频的同源帧会在 train/val 之间
    产生"近邻泄漏"，导致 val 对 medium/large 目标系统性偏心（实测
    见过 37.92 vs 没见过 19.34）。因此**不允许按单张图片随机分层**，
    必须先把帧分组，再做 group-level 划分。

为什么用内容相似度而不是文件名：
    - 命名有四五个互不相同的家族（纯数字 6/8 位、shuming_、hehe_、
      NNNNNN_NNN_、NNNNNN_NNN_NNNNNNNN），编号空间并不统一；
    - 实测过"按编号找最近帧"搬标注只有 3.86 分，说明编号≠时间；
    - 相似度是**可直接测量**的，分组结果可以写进报告供复核。

本脚本**只读官方训练数据**，不读测试集、不读任何测试标签；
输出只有分组文件与统计报告，不产生任何提交结果。

用法：
    python src/eval/build_temporal_groups.py --stage embed
    python src/eval/build_temporal_groups.py --stage group
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path(__file__).resolve().parents[2]
IMG_DIR = WS / "data" / "aic" / "images"
CACHE = WS / "runs" / "_temporal_groups"
THUMB = 48


def embed_one(path: Path) -> np.ndarray | None:
    """灰度缩略图 → 零均值单位范数向量（余弦相似度 = 点积）。"""
    try:
        with Image.open(path) as im:
            im.draft("L", (THUMB * 4, THUMB * 4))  # JPEG 快速降采样
            g = np.asarray(im.convert("L").resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! 读取失败 {path.name}: {exc}")
        return None
    v = g.ravel() - g.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else v


def stage_embed() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    files = sorted(IMG_DIR.iterdir(), key=lambda p: p.name)
    files = [p for p in files if p.suffix.lower() in {".jpg", ".png"}]
    print(f"训练图 {len(files)} 张，缩略图 {THUMB}x{THUMB}，嵌入中…")

    names: list[str] = []
    vecs: list[np.ndarray] = []
    for i, p in enumerate(files):
        v = embed_one(p)
        if v is not None:
            names.append(p.stem)
            vecs.append(v)
        if (i + 1) % 400 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)

    E = np.stack(vecs)
    np.save(CACHE / "emb.npy", E)
    (CACHE / "names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    print(f"嵌入完成：{E.shape} → {CACHE / 'emb.npy'}")


def stage_similarity() -> None:
    E = np.load(CACHE / "emb.npy")
    names = json.loads((CACHE / "names.json").read_text(encoding="utf-8"))
    print(f"相似度矩阵 {E.shape[0]}x{E.shape[0]}…")
    S = E @ E.T
    np.fill_diagonal(S, -1.0)
    nn = S.max(axis=1)

    print("\n【最近邻相似度分布】")
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"  P{q:<3} = {np.percentile(nn, q):.4f}")
    print(f"  均值 = {nn.mean():.4f}   最大 = {nn.max():.4f}")

    print("\n【不同阈值下的近邻对数量】")
    iu = np.triu_indices(len(names), k=1)
    for th in (0.99, 0.98, 0.97, 0.95, 0.93, 0.90, 0.85, 0.80, 0.75):
        k = int((S[iu] >= th).sum())
        print(f"  sim >= {th:.2f} : {k:>7} 对   （占全部 {k / len(iu[0]) * 100:.3f}%）")

    # 保存 NN 结果，供人工抽查
    order = np.argsort(-nn)
    lines = []
    for i in order[:200]:
        j = int(np.argmax(S[i]))
        lines.append(f"{nn[i]:.4f}\t{names[i]}\t{names[j]}")
    (CACHE / "nn_top200.tsv").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n最近邻最高的 200 条 → {CACHE / 'nn_top200.tsv'}")


def stage_group() -> None:
    E = np.load(CACHE / "emb.npy")
    names = json.loads((CACHE / "names.json").read_text(encoding="utf-8"))
    n = len(names)
    S = E @ E.T
    np.fill_diagonal(S, -1.0)

    print(f"{'阈值':>6} {'组数':>7} {'最大组':>7} {'中位组':>7} {'单张组占比':>10} {'前5大组':>30}")
    report: dict[str, list[list[str]]] = {}
    for th in (0.99, 0.97, 0.95, 0.93, 0.90):
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        iu = np.triu_indices(n, k=1)
        hits = S[iu] >= th
        for a, b in zip(iu[0][hits].tolist(), iu[1][hits].tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        buckets: dict[int, list[str]] = {}
        for i in range(n):
            buckets.setdefault(find(i), []).append(names[i])
        comps = sorted(buckets.values(), key=len, reverse=True)
        sizes = np.array([len(c) for c in comps])
        singles = int((sizes == 1).sum())
        print(
            f"{th:>6.2f} {len(comps):>7} {sizes.max():>7} {int(np.median(sizes)):>7} "
            f"{singles / len(comps) * 100:>9.1f}% {str([int(x) for x in sizes[:5]]):>30}"
        )
        report[str(th)] = comps

    out = CACHE / "groups.json"
    out.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(f"\n全部分组结果 → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["embed", "similarity", "group"], required=True)
    args = ap.parse_args()
    {"embed": stage_embed, "similarity": stage_similarity, "group": stage_group}[args.stage]()


if __name__ == "__main__":
    main()
