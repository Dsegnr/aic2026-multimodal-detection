"""把我们的 YOLO 格式数据转成 D-FINE 需要的 COCO 格式。

**关键约定（照抄自 D-FINE 的源码，不是猜的）**：
    `reference/D-FINE/src/data/dataset/coco_dataset.py` 里

        if self.remap_mscoco_category:
            ... category2label=mscoco_category2label
        else:
            labels = [obj["category_id"] for obj in anno]   # ← 直接当 label 用

    配合 custom 配置里的 `remap_mscoco_category: False` + `num_classes: 12`，
    ⇒ **category_id 必须 0 基（0~11）**，否则 12 类的分类头会越界。
    （COCO 官方惯例是 1 基，这里必须反着来。）

输出：COCO json（images / annotations / categories 三段），
      img_folder 仍指向我们原来的图片目录，所以不需要复制图片。

用法：
    python src/eval/yolo2coco.py --list splits/lbproxy_train.txt --out data/aic_coco/train.json
    python src/eval/yolo2coco.py --list splits/lbproxy_val.txt   --out data/aic_coco/val.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

WS = Path(__file__).resolve().parents[2]
IMG_DIR = WS / "data" / "aic" / "images"
LBL_DIR = WS / "data" / "aic" / "labels"
NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle",
         "car", "ball", "light", "garbage_can", "uav", "tricycle"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = [x.strip() for x in (WS / args.list).read_text(encoding="utf-8").splitlines() if x.strip()]
    stems = [Path(x).stem for x in lines]
    print(f"清单 {args.list}：{len(stems)} 张")

    images, anns = [], []
    n_box = n_missing_img = n_missing_lbl = 0
    bad_cls = 0
    for i, s in enumerate(stems, start=1):
        img_path = IMG_DIR / f"{s}.png"
        if not img_path.exists():
            img_path = IMG_DIR / f"{s}.jpg"
        if not img_path.exists():
            n_missing_img += 1
            continue
        with Image.open(img_path) as im:
            W, H = im.size
        images.append({"id": i, "file_name": img_path.name, "width": W, "height": H})

        lbl = LBL_DIR / f"{s}.txt"
        if not lbl.exists():
            n_missing_lbl += 1
            continue
        for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            q = ln.split()
            if len(q) < 5:
                continue
            c = int(float(q[0]))
            if not (0 <= c < len(NAMES)):      # 0 基，越界直接报出来
                bad_cls += 1
                continue
            cx, cy, w, h = (float(v) for v in q[1:5])
            x = (cx - w / 2) * W
            y = (cy - h / 2) * H
            bw = w * W
            bh = h * H
            if bw <= 0 or bh <= 0:
                continue
            n_box += 1
            anns.append({
                "id": len(anns) + 1,
                "image_id": i,
                "category_id": c,              # ★ 0 基
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            })

    coco = {
        "images": images,
        "annotations": anns,
        # ★ 0 基 category_id，与 D-FINE 的 remap_mscoco_category=False 路径一致
        "categories": [{"id": i, "name": n} for i, n in enumerate(NAMES)],
    }
    out = WS / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    print(f"写出 {out}")
    print(f"  images      = {len(images)}   缺失图片 {n_missing_img}")
    print(f"  annotations = {n_box}   缺标注文件 {n_missing_lbl}   类别越界 {bad_cls}")
    print(f"  categories  = {len(NAMES)}（id 0~{len(NAMES) - 1}，0 基）")
    print(f"  文件大小     = {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
