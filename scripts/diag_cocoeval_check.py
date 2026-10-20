"""用**官方 pycocotools** 复算我们的 val300 预测，与我们自己的 map_eval 对比。

为什么需要（2026-09-30）：
  我们一直用自研 `src/eval/map_eval.py` 当本地尺子，但它和榜方（官方 COCO 评测）
  是否**逐位一致**从未验证过。如果两边系统性差 5~6 分，那么"本地 58 → 榜 51.7"
  这个 6.6 分的差距里，有多少是"尺子差异"、有多少是"数据更难"就说不清。

用法：
    python scripts/diag_cocoeval_check.py --gt data/aic_coco/val.json \
        --pred submissions/_tta_L_six
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

WS = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/aic_coco/val.json")
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()

    gt_path = WS / a.gt if not Path(a.gt).is_absolute() else Path(a.gt)
    pred_dir = WS / a.pred if not Path(a.pred).is_absolute() else Path(a.pred)

    coco = COCO(str(gt_path))
    # file_name 词干 -> image_id
    stem2id = {}
    for img in coco.dataset["images"]:
        stem2id[Path(img["file_name"]).stem] = img["id"]

    dets = []
    n_files = 0
    for txt in sorted(pred_dir.glob("*.txt")):
        iid = stem2id.get(txt.stem)
        if iid is None:
            continue
        n_files += 1
        img = coco.loadImgs(iid)[0]
        w, h = img["width"], img["height"]
        for line in txt.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 6:
                continue
            c, cx, cy, bw, bh, s = (float(x) for x in p[:6])
            dets.append({
                "image_id": iid,
                "category_id": int(c),
                "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                "score": s,
            })
    print(f"[diag] GT 图 {len(coco.dataset['images'])} 张；匹配上的预测文件 {n_files}；检测框 {len(dets)}")

    dt = coco.loadRes(dets)
    ev = COCOeval(coco, dt, "bbox")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    print(f"[diag] 官方 pycocotools mAP@50-95 = {ev.stats[0]*100:.4f}")
    print(f"[diag] 官方 mAP@50 = {ev.stats[1]*100:.4f} / mAP@75 = {ev.stats[2]*100:.4f}")
    print("[diag] 把上面这三个数与自研 map_eval 的输出对照（同一份预测）")


if __name__ == "__main__":
    main()
