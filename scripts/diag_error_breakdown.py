"""错误类型拆解：把每个 GT 框归类为「定位不准 / 分类错 / 漏检」。

为什么要做：第 182 节发现弱类是"小目标高频类"，但**没区分**是
  (a) 模型检到了、只是框不够准（定位问题 ⇒ 高分辨率/推理放大对症）
  (b) 模型根本没检到该目标（召回问题 ⇒ 容量/数据/增广对症）
两者的解法完全不同，而现在我们正要把算力押到其中一边。

做法（不依赖 COCO 评测内部状态，直接算）：
  对每个 GT 框，在同图预测里找**任意类别**的最大 IoU 匹配：
    * IoU ≥ 0.75 且类别正确            → 已检好（TP-high）
    * 0.50 ≤ IoU < 0.75 且类别正确     → 检到但框偏（TP-mid，mAP@50-95 的主要失分）
    * 0.10 ≤ IoU < 0.50 且类别正确     → 检到但定位差（TP-low）
    * IoU ≥ 0.50 但类别错              → 分类错（cls-wrong）
    * 其余（无任何 IoU≥0.1 的预测）     → 漏检（miss）

用法：.venv/Scripts/python.exe scripts/diag_error_breakdown.py preds/ep9 [preds/...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]


def iou_matrix(gt, dt):
    """gt (n,4) xyxy, dt (m,4) xyxy → (n,m) IoU"""
    if len(gt) == 0 or len(dt) == 0:
        return np.zeros((len(gt), len(dt)), np.float32)
    g = gt[:, None, :]
    d = dt[None, :, :]
    xx1 = np.maximum(g[..., 0], d[..., 0])
    yy1 = np.maximum(g[..., 1], d[..., 1])
    xx2 = np.minimum(g[..., 2], d[..., 2])
    yy2 = np.minimum(g[..., 3], d[..., 3])
    w = np.clip(xx2 - xx1, 0, None)
    h = np.clip(yy2 - yy1, 0, None)
    inter = w * h
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    ad = (dt[:, 2] - dt[:, 0]) * (dt[:, 3] - dt[:, 1])
    union = ag[:, None] + ad[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def main() -> None:
    gt_json = json.loads((WS / "data" / "aic_coco" / "val.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in gt_json["categories"]}
    by_img = defaultdict(list)
    for a in gt_json["annotations"]:
        by_img[a["image_id"]].append(a)
    stem2id = {Path(im["file_name"]).stem: im["id"] for im in gt_json["images"]}
    size = {im["id"]: (im["width"], im["height"]) for im in gt_json["images"]}

    for pred_dir in sys.argv[1:]:
        pdir = Path(pred_dir) if Path(pred_dir).is_absolute() else WS / pred_dir
        if not pdir.is_dir():
            print(f"[跳过] {pred_dir}")
            continue
        # 预测：每图 (boxes, labels)
        preds = {}
        for txt in pdir.glob("*.txt"):
            iid = stem2id.get(txt.stem)
            if iid is None:
                continue
            W, H = size[iid]
            b, l = [], []
            for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = line.split()
                if len(p) < 6:
                    continue
                c, cx, cy, w, h = (float(x) for x in p[:5])
                b.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
                l.append(int(c))
            preds[iid] = (np.array(b, np.float32).reshape(-1, 4), np.array(l, np.int64))

        stat = defaultdict(lambda: defaultdict(int))
        tot = defaultdict(int)
        conf = defaultdict(int)          # (gt_cls, pred_cls) 计数：IoU≥0.5 但类别错
        for iid, anns in by_img.items():
            if iid not in preds or not anns:
                continue
            dt, dl = preds[iid]
            for a in anns:
                cid = a["category_id"]
                g = np.array([a["bbox"][0], a["bbox"][1],
                              a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]], np.float32)[None, :]
                if len(dt) == 0:
                    stat[cid]["miss"] += 1
                    tot[cid] += 1
                    continue
                ious = iou_matrix(g, dt)[0]
                j = int(np.argmax(ious))
                iou = float(ious[j])
                cls_ok = int(dl[j]) == cid
                if iou >= 0.75 and cls_ok:
                    stat[cid]["TP-high"] += 1
                elif iou >= 0.50 and cls_ok:
                    stat[cid]["TP-mid"] += 1
                elif iou >= 0.10 and cls_ok:
                    stat[cid]["TP-low"] += 1
                elif iou >= 0.50 and not cls_ok:
                    stat[cid]["cls-wrong"] += 1
                    conf[(cid, int(dl[j]))] += 1
                else:
                    stat[cid]["miss"] += 1
                tot[cid] += 1

        print(f"\n===== {pdir.name} =====（每个 GT 框归一类；占比按该类样本数）")
        print(f"{'class':14s}{'GT数':>6s}{'检好':>8s}{'框偏(0.5-0.75)':>16s}{'定位差(0.1-0.5)':>17s}{'分类错':>8s}{'漏检':>8s}")
        for cid in sorted(names, key=lambda c: -tot.get(c, 0)):
            n = tot[cid]
            if n == 0:
                continue
            s = stat[cid]
            f = lambda k: f"{s[k]/n*100:7.1f}%"  # noqa: E731
            print(f"{names[cid]:14s}{n:6d}{f('TP-high'):>8s}{f('TP-mid'):>16s}"
                  f"{f('TP-low'):>17s}{f('cls-wrong'):>8s}{f('miss'):>8s}")
        allc = ["TP-high", "TP-mid", "TP-low", "cls-wrong", "miss"]
        N = sum(sum(stat[c].values()) for c in stat)
        print("总体占比：" + "  ".join(f"{k} {sum(stat[c][k] for c in stat)/N*100:.1f}%" for k in allc))
        print("判读：若 '框偏+定位差' 占大头 ⇒ 定位问题（高分辨率/推理放大对症）；"
              "若 '漏检' 占大头 ⇒ 召回问题（容量/数据/增广对症）。")

        # 分类混淆对（IoU≥0.5 但类别错）——直接告出"谁被认成了谁"
        if conf:
            print("\n分类混淆 Top（GT 类 → 被预测成）：")
            for (a, b), n in sorted(conf.items(), key=lambda kv: -kv[1])[:10]:
                gtn = sum(1 for (x, _), v in conf.items() if x == a for _ in [0]) or 1
                print(f"  {names[a]:12s} → {names[b]:12s} {n:4d} 次"
                      f"（占 {names[a]} 的 {n/max(tot[a],1)*100:.1f}%）")


if __name__ == "__main__":
    main()
