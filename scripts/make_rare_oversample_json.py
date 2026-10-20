"""为 D-FINE 造一个"稀类过采样"的训练标注（纯数据侧改动，不动任何代码）。

## 为什么要这样做

官方指标是 **12 类宏平均**，而 val300 的逐类 AP 里最弱的四类是
bicycle 39.9 / ball 32.5 / sign 45.9 / garbage_can 51.6，
**它们各自占宏平均的 1/12**。算术上只要把这 4 类各抬到 55，宏平均就 **+4.2**。

`rare_paste`（稀有类贴图增强）只实现在我们 ultralytics 那条路径上，
**D-FINE 直接读 COCO json，用不到它**。所以对 D-FINE 这条线，
最干净的等价手段是：**把"含弱类目标的图"在训练集里重复若干次** ——
这等价于提高这些类的采样权重，且完全不需要改训练代码。

## 做法
对每张训练图，数它含有的"弱类"目标数：
  弱类 = {boat(1), sign(4), bicycle(5), ball(7), garbage_can(9), tricycle(11)}
按"含有弱类的个数"给重复次数 k（最多 k_max），把该图与其标注整体复制 k 份
（image_id / annotation id 全部重新编号，保证 COCO 合法）。

用法：
    python scripts/make_rare_oversample_json.py --in data/aic_coco/train.json \
        --out data/aic_coco/train_rare.json --k-max 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WS = Path(__file__).resolve().parents[1]

# 名称顺序来自官方 12 类
NAMES = ['person', 'boat', 'animal', 'seat', 'sign', 'bicycle',
         'car', 'ball', 'light', 'garbage_can', 'uav', 'tricycle']
DEFAULT_WEAK = [1, 4, 5, 7, 9, 11]     # boat, sign, bicycle, ball, garbage_can, tricycle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", default="data/aic_coco/train.json")
    ap.add_argument("--out", default="data/aic_coco/train_rare.json")
    ap.add_argument("--weak", default=",".join(str(c) for c in DEFAULT_WEAK))
    ap.add_argument("--k-max", type=int, default=3)
    args = ap.parse_args()

    src = json.loads((WS / args.in_json).read_text(encoding="utf-8"))
    weak = {int(x) for x in args.weak.split(",")}
    by_img: dict[int, list[dict]] = {}
    for a in src["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    out = {"images": [], "annotations": [], "categories": src["categories"]}
    next_iid, next_aid = 1, 1
    n_rep_total = 0
    rep_hist: dict[int, int] = {}
    for im in src["images"]:
        anns = by_img.get(im["id"], [])
        k_weak = sum(1 for a in anns if a["category_id"] in weak)
        # 含弱类越多，重复越多；上限 k_max
        k = 1 + min(k_weak, args.k_max - 1)
        rep_hist[k] = rep_hist.get(k, 0) + 1
        for _ in range(k):
            ni = dict(im); ni["id"] = next_iid
            out["images"].append(ni)
            for a in anns:
                na = dict(a); na["id"] = next_aid; na["image_id"] = next_iid
                out["annotations"].append(na); next_aid += 1
            next_iid += 1; n_rep_total += 1

    (WS / args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{args.in_json} → {args.out}")
    print(f"  原图 {len(src['images'])}  原标注 {len(src['annotations'])}")
    print(f"  弱类集合 {[NAMES[c] for c in sorted(weak)]}")
    print(f"  复制后图 {len(out['images'])}（{n_rep_total} 条）  标注 {len(out['annotations'])}")
    print(f"  重复次数分布 {dict(sorted(rep_hist.items()))}")


if __name__ == "__main__":
    main()
