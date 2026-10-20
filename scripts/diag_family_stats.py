"""逐家族 GT 统计：框密度、框面积分布、类别混合——找 num6（测试集占 61%）为何最弱。

用法：.venv/Scripts/python.exe scripts/diag_family_stats.py
"""

from __future__ import annotations

import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
FAMILIES = [
    ("num6", re.compile(r"^\d{6}$")),
    ("num6_3_8", re.compile(r"^\d{6}_\d{3}_\d{8}$")),
    ("shuming", re.compile(r"^shuming_")),
    ("num8", re.compile(r"^\d{8}$")),
    ("hehe", re.compile(r"^hehe_")),
]


def fam(stem: str) -> str:
    for n, p in FAMILIES:
        if p.match(stem):
            return n
    return "other"


def main() -> None:
    d = json.load(open(WS / "data/aic_coco/val.json", encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    img_meta = {im["id"]: im for im in d["images"]}
    img2fam = {im["id"]: fam(Path(im["file_name"]).stem) for im in d["images"]}

    boxes = defaultdict(list)
    cls = defaultdict(Counter)
    nimg = Counter(img2fam.values())
    nbox = Counter()
    for a in d["annotations"]:
        f = img2fam[a["image_id"]]
        im = img_meta[a["image_id"]]
        boxes[f].append(a["bbox"][2] * a["bbox"][3] / (im["width"] * im["height"]))
        cls[f][cats[a["category_id"]]] += 1
        nbox[f] += 1

    print(f"{'family':10s}{'图数':>6s}{'框/图':>8s}{'面积中位':>10s}{'p25':>9s}{'p75':>9s}")
    for f in ["num6", "num6_3_8", "shuming", "num8", "hehe", "other"]:
        if not boxes[f]:
            continue
        b = sorted(boxes[f])
        q = lambda p: b[int((len(b) - 1) * p)]  # noqa: E731
        print(f"{f:10s}{nimg[f]:6d}{nbox[f]/nimg[f]:8.1f}{st.median(b):10.5f}{q(0.25):9.5f}{q(0.75):9.5f}")
    print()
    for f in ["num6", "num6_3_8", "shuming", "num8", "hehe"]:
        c = cls[f]
        tot = sum(c.values())
        top = ", ".join(f"{k}:{v/tot*100:.0f}%" for k, v in c.most_common(6))
        print(f"{f:10s} {top}")


if __name__ == "__main__":
    main()
