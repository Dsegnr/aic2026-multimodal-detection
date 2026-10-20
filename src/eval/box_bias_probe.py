"""诊断：标注框的"系统性偏差"是不是**尺寸**型的？（不是中心偏移型的）

动机（09-19 逐类探针的结果）：
    recall@IoU0.5 全类 83~100%，但 AP90 只有 19.7、AP95 只有 7.7。
    ⇒ 分数**完全死在定位精度上，而不是漏检上**。
    对 1536 下短边中位 44.8 px 的目标，要拿到 IoU≥0.90 需要框边误差 ≈1.2 px。
    所以问题变成："模型和标注到底差在**哪里**？"

    如果是**中心偏移**（有方向性），那说明估计器有偏；
    如果是**尺寸比**（pred/gt 的 w、h 之比稳定偏离 1），那说明标注/模型有**框约定差异**——
    这种偏差**可以用一个纯后处理的、按类的尺度校正补回来**，而且只用官方训练数据即可拟合。

输出：
  * 每类：匹配对的 w 比 / h 比（中位与四分位）、面积比、中心偏移（按目标尺寸归一化）
  * 每类的"最优单参数尺度 α"：把预测框按 α 缩放后能最大化该类的 AP50-95（在 val 上扫）

用法：
    python src/eval/box_bias_probe.py --gt-dir data/valgt_lbproxy \
        --pred-dir submissions/_dfine_ep56_tta1280
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from map_eval import DEFAULT_NAMES, iou_matrix, read_gt_file, read_pred_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--max-boxes", type=int, default=100)
    args = ap.parse_args()

    gt_dir, pred_dir = Path(args.gt_dir), Path(args.pred_dir)
    names = list(DEFAULT_NAMES)

    wr: dict[int, list[float]] = {c: [] for c in range(len(names))}
    hr: dict[int, list[float]] = {c: [] for c in range(len(names))}
    dx: dict[int, list[float]] = {c: [] for c in range(len(names))}
    dy: dict[int, list[float]] = {c: [] for c in range(len(names))}
    giou: dict[int, list[float]] = {c: [] for c in range(len(names))}

    for gf in sorted(gt_dir.glob("*.txt")):
        gt = read_gt_file(gf)
        preds = read_pred_file(pred_dir / f"{gf.stem}.txt", max_boxes=args.max_boxes)
        for c, g in gt.items():
            if not 0 <= c < len(names):
                continue
            p = [(conf, b) for conf, cc, b in preds if cc == c and conf >= args.conf]
            if len(p) == 0 or len(g) == 0:
                continue
            pb = np.stack([b for _, b in p])
            ious = iou_matrix(pb, g)
            used_p, used_g = set(), set()
            order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
            for pi, gj in order:
                if ious[pi, gj] < 0.3:
                    break
                if pi in used_p or gj in used_g:
                    continue
                used_p.add(int(pi)); used_g.add(int(gj))
                pw, ph = pb[pi, 2] - pb[pi, 0], pb[pi, 3] - pb[pi, 1]
                gw, gh = g[gj, 2] - g[gj, 0], g[gj, 3] - g[gj, 1]
                if gw <= 0 or gh <= 0:
                    continue
                wr[c].append(pw / gw); hr[c].append(ph / gh)
                pcx, pcy = (pb[pi, 0] + pb[pi, 2]) / 2, (pb[pi, 1] + pb[pi, 3]) / 2
                gcx, gcy = (g[gj, 0] + g[gj, 2]) / 2, (g[gj, 1] + g[gj, 3]) / 2
                dx[c].append((pcx - gcx) / gw); dy[c].append((pcy - gcy) / gh)
                giou[c].append(ious[pi, gj])

    print(f"\n=== 尺寸/中心偏差（同类最高 IoU 贪心配对, conf>={args.conf}, IoU>=0.3）===")
    print(f"{'class':<12}{'n':>5}{'w比中位':>9}{'h比中位':>9}{'面积比':>9}"
          f"{'dcx中位':>9}{'dcy中位':>9}{'|dx|>0.1':>9}{'匹配IoU中位':>12}")
    for c in range(len(names)):
        if len(wr[c]) < 5:
            continue
        w = np.array(wr[c]); h = np.array(hr[c])
        a = w * h
        ddx = np.array(dx[c]); ddy = np.array(dy[c])
        big = (np.abs(ddx) > 0.1).mean() * 100
        print(f"{names[c]:<12}{len(w):>5}{np.median(w):>9.3f}{np.median(h):>9.3f}"
              f"{np.median(a):>9.3f}{np.median(ddx):>9.3f}{np.median(ddy):>9.3f}"
              f"{big:>9.1f}{np.median(giou[c]):>12.3f}")
    allw = np.concatenate([np.array(v) for v in wr.values() if v])
    allh = np.concatenate([np.array(v) for v in hr.values() if v])
    print(f"{'ALL':<12}{len(allw):>5}{np.median(allw):>9.3f}{np.median(allh):>9.3f}"
          f"{np.median(allw * allh):>9.3f}")
    print(f"  w比 处在 [0.95,1.05] 的比例 = {((allw > 0.95) & (allw < 1.05)).mean() * 100:.1f}%")
    print(f"  h比 处在 [0.95,1.05] 的比例 = {((allh > 0.95) & (allh < 1.05)).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
