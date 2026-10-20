"""P0：按"内容分组"做 group-level 划分，替换掉随机 val300。

依据（Astra 第 5 次回答 §Q1.1）：
    "建议把 2000 张图按连续视频块、文件名邻近关系、来源序列先分成若干 group，
     再做 group-level assignment，**不允许相邻帧跨 train/val**。"

分组来源：`build_temporal_groups.py` 的相似度连通分量（近重复帧 = 同一时间块）。
划分原则：
    1. 同一个 group 只出现在同一侧（train 或 val），绝不劈开；
    2. 5 折，每折 val ≈ 400 张 / train ≈ 1600 张；
    3. 用 LPT（最长优先）贪心平衡每折的**总图数**与**12 类目标数**；
    4. 不强行对齐低分辨率子集比例——按 Astra 的说法，低分辨率子集
       **单独报告指标**，而不是把它揉进同一个数里。

只读官方训练数据与标签，不碰测试集。

用法：
    python src/eval/make_group_folds.py --tau 0.93 --folds 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
IMG_DIR = WS / "data" / "aic" / "images"
LBL_DIR = WS / "data" / "aic" / "labels"
CACHE = WS / "runs" / "_temporal_groups"
SPLITS = WS / "splits"
NCLS = 12
SERVER_IMG = "./data/aic/images"


def load_groups(tau: float) -> list[list[str]]:
    data = json.loads((CACHE / "groups.json").read_text(encoding="utf-8"))
    return data[str(tau)]


def class_vector(stem: str) -> np.ndarray:
    v = np.zeros(NCLS, dtype=np.float64)
    f = LBL_DIR / f"{stem}.txt"
    if f.exists():
        for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = ln.split()
            if len(p) >= 5:
                try:
                    c = int(float(p[0]))
                except ValueError:
                    continue
                if 0 <= c < NCLS:
                    v[c] += 1
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.93)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="gfold")
    args = ap.parse_args()

    groups = load_groups(args.tau)
    sizes = np.array([len(g) for g in groups])
    print(f"τ={args.tau}: {len(groups)} 组，共 {sizes.sum()} 张，最大组 {sizes.max()}，单张组 {(sizes == 1).sum()}")

    cv = np.stack([class_vector(s) for g in groups for s in g])
    off = np.cumsum([0] + [len(g) for g in groups])
    gv = np.stack([cv[off[i]:off[i + 1]].sum(axis=0) for i in range(len(groups))])
    gs = np.array([len(g) for g in groups], dtype=np.float64)

    # 全局类别权重（用于平衡）：按类别总数的倒数归一化
    tot = gv.sum(axis=0)
    w = np.where(tot > 0, tot.sum() / np.maximum(tot, 1), 0.0)
    w = w / (w.sum() or 1.0)

    K = args.folds
    size_load = np.zeros(K)
    cls_load = np.zeros((K, NCLS))
    assign = np.full(len(groups), -1, dtype=int)
    order = np.argsort(-gs)  # LPT：先放大组
    glob_prop = tot / tot.sum()
    for gi in order:
        # 两级排序：**先**比"该折当前总图数"（保证规模均衡），
        # 规模相同时才比"类别分布偏离全局的程度"（保证类别均衡）。
        # 单张组极多、规模天然大量持平，因此第二个键实际承担了主要的类别平衡工作。
        keys = []
        for k in range(K):
            dc = cls_load[k] + gv[gi]
            prop = dc / max(dc.sum(), 1.0)
            dev = float(np.sum(np.abs(prop - glob_prop) * w))
            keys.append((size_load[k], dev, k))
        best = min(keys)[2]
        assign[gi] = best
        size_load[best] += gs[gi]
        cls_load[best] += gv[gi]

    print("\n各折规模：", [int(x) for x in size_load])

    # 输出
    all_stems = set()
    report = {"tau": args.tau, "folds": K, "n_groups": len(groups), "sizes": [int(x) for x in size_load]}
    for k in range(K):
        val_stems = [s for gi in range(len(groups)) if assign[gi] == k for s in groups[gi]]
        tr_stems = [s for gi in range(len(groups)) if assign[gi] != k for s in groups[gi]]
        all_stems.update(val_stems)

        def real_path(s: str) -> Path:
            p = IMG_DIR / f"{s}.png"
            return p if p.exists() else IMG_DIR / f"{s}.jpg"

        def to_lines(stems: list[str], server: bool) -> str:
            # 扩展名必须按真实文件给：低分辨率那 149 张是 .jpg，
            # 写死 .png 会让服务器端 149 张图直接找不到（静默丢样本）。
            if server:
                body = "\n".join(f"{SERVER_IMG}/{real_path(s).name}" for s in stems)
            else:
                body = "\n".join(str(real_path(s)) for s in stems)
            return body + "\n"          # 尾换行：否则 wc -l 少数一行

        def write_list(path: Path, text: str) -> None:
            """**必须写 LF**。Windows 默认会把 \\n 翻成 \\r\\n，
            上传到 Linux 后每行末尾多一个 \\r，路径全部失效——
            这会变成"跑完 200 轮但一张图都没读到"的静默失败。"""
            with path.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)

        write_list(SPLITS / f"{args.tag}{k}_val.txt", to_lines(val_stems, False))
        write_list(SPLITS / f"{args.tag}{k}_train.txt", to_lines(tr_stems, False))
        write_list(SPLITS / f"{args.tag}{k}_val_srv.txt", to_lines(val_stems, True))
        write_list(SPLITS / f"{args.tag}{k}_train_srv.txt", to_lines(tr_stems, True))

        n_low = sum(1 for s in val_stems if (IMG_DIR / f"{s}.jpg").exists())
        report[f"fold{k}"] = {
            "n_val": len(val_stems),
            "n_train": len(tr_stems),
            "lowres_val": n_low,
            "lowres_pct": round(100 * n_low / max(len(val_stems), 1), 2),
            "per_class_val": {str(i): int(x) for i, x in enumerate(cls_load[k].astype(int))},
        }
        print(
            f"  fold{k}: val {len(val_stems):>4}  train {len(tr_stems):>4}  "
            f"低分辨率 val {n_low:>3} ({100 * n_low / max(len(val_stems), 1):.1f}%)"
        )

    assert len(all_stems) == len(cv), f"覆盖不全：{len(all_stems)} vs {len(cv)}"
    print(f"\n5 折 val 并集 = {len(all_stems)} 张（= 训练集总数，每张恰好属于一折）")

    # 诊断：现有随机 val300 里有多少张在 train1700 里有"孪生帧"
    diag = {}
    for name in ("lbproxy_val", "train300_probe"):
        f = SPLITS / f"{name}.txt"
        if not f.exists():
            continue
        stems = {Path(x.strip()).stem for x in f.read_text(encoding="utf-8").splitlines() if x.strip()}
        group_of = {s: gi for gi, g in enumerate(groups) for s in g}
        twin = sum(1 for s in stems if s in group_of and len(groups[group_of[s]]) > 1)
        diag[name] = {"n": len(stems), "with_twin_in_pool": twin,
                      "pct": round(100 * twin / max(len(stems), 1), 1)}
        print(f"  诊断 {name}: {len(stems)} 张中 {twin} 张（{diag[name]['pct']}%）在图内有孪生帧")
    report["existing_split_diag"] = diag

    (CACHE / f"folds_report_{args.tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n报告 → {CACHE / f'folds_report_{args.tag}.json'}")


if __name__ == "__main__":
    main()
