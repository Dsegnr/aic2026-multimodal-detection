"""按"命名家族"切分 val300，看**每个家族的类别构成**——用来判断"家族弱"到底是
"类别弱"还是"场景弱"。

用法： python scripts/diag_family_class.py
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
FAM = [("num6", r"^\d{6}$"), ("num6_3_8", r"^\d{6}_\d{3}_\d{8}$"), ("shuming", r"^shuming_"),
       ("num8", r"^\d{8}$"), ("hehe", r"^hehe_")]
NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball",
         "light", "garbage_can", "uav", "tricycle"]


def fam(s: str) -> str:
    for n, p in FAM:
        if re.match(p, s):
            return n
    return "other"


def main() -> None:
    d = json.loads((WS / "data/aic_coco/val.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    iid2stem = {i["id"]: Path(i["file_name"]).stem for i in d["images"]}
    per = collections.defaultdict(collections.Counter)
    imgs = collections.Counter()
    for i in d["images"]:
        imgs[fam(Path(i["file_name"]).stem)] += 1
    for a in d["annotations"]:
        per[fam(iid2stem[a["image_id"]])][cats[a["category_id"]]] += 1

    header = f"{'family':10s}{'图数':>5s}{'框数':>6s}" + "".join(f"{n[:6]:>8s}" for n in NAMES)
    print(header)
    for f, c in sorted(per.items(), key=lambda x: -sum(x[1].values())):
        tot = sum(c.values()) or 1
        row = f"{f:10s}{imgs[f]:5d}{tot:6d}" + "".join(f"{100*c[n]/tot:7.1f}%" for n in NAMES)
        print(row)
    print("\n（列是该家族内各类框占比；对照测试集家族占比：num6 61.3% / num6_3_8 18.0% / num8 11.7% / shuming 5.4%）")


if __name__ == "__main__":
    main()
