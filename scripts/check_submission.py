"""提交格式自检。

按官方约束逐条检查，任何一条不通过都可能让整份结果被判无效：
- 每张测试图必须有同名 TXT，不能缺失；
- 每行 6 个字段 [class_id, cx, cy, w, h, conf]；
- class_id 为 0..11 的整数；
- 坐标与置信度都在 [0,1]；
- 每图不超过 100 行；
- 不出现 NaN / inf；
- 三类违规单独计数，并给出前若干条样例。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
NC = 12
MAX_BOXES = 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--test-dir", default="data/test/images")
    ap.add_argument("--max-report", type=int, default=12)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_absolute():
        pred_dir = WS / pred_dir
    test_dir = Path(args.test_dir)
    if not test_dir.is_absolute():
        test_dir = WS / test_dir

    images = sorted(p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    stems = [p.stem for p in images]

    missing = [s for s in stems if not (pred_dir / f"{s}.txt").exists()]
    extra = [p.name for p in pred_dir.glob("*.txt") if p.stem not in set(stems)]

    problems: list[str] = []
    n_lines_total = 0
    n_empty = 0
    n_over100 = 0
    n_bad_field: list[str] = []
    n_bad_class: list[str] = []
    n_bad_range: list[str] = []
    n_bad_num: list[str] = []

    for stem in stems:
        f = pred_dir / f"{stem}.txt"
        if not f.exists():
            continue
        lines = [ln for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        if not lines:
            n_empty += 1
        if len(lines) > MAX_BOXES:
            n_over100 += 1
            problems.append(f"{f.name}: {len(lines)} 行 > {MAX_BOXES}")
        for ln in lines:
            n_lines_total += 1
            parts = ln.split()
            if len(parts) != 6:
                n_bad_field.append(f"{f.name}: {ln[:60]}")
                continue
            try:
                cls = float(parts[0])
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                n_bad_num.append(f"{f.name}: {ln[:60]}")
                continue
            if any(math.isnan(v) or math.isinf(v) for v in vals):
                n_bad_num.append(f"{f.name}: {ln[:60]}")
                continue
            if not (cls.is_integer() and 0 <= cls < NC):
                n_bad_class.append(f"{f.name}: {ln[:60]}")
            if any(v < 0.0 or v > 1.0 for v in vals):
                n_bad_range.append(f"{f.name}: {ln[:60]}")

    print(f"测试图: {len(stems)}   预测文件: {len(list(pred_dir.glob('*.txt')))}")
    print(f"预测框总数: {n_lines_total}   空文件: {n_empty}   超过100行的文件: {n_over100}")

    def report(name: str, items: list[str]) -> None:
        flag = "OK " if not items else "FAIL"
        print(f"[{flag}] {name}: {len(items)}")
        for s in items[: args.max_report]:
            print(f"        {s}")

    report("缺失预测文件", missing)
    report("多余预测文件", extra)
    report("字段数不为6", n_bad_field)
    report("非法类别", n_bad_class)
    report("坐标/置信度越界", n_bad_range)
    report("NaN/Inf/解析失败", n_bad_num)
    report("每图框数超限", problems)

    bad = bool(missing or extra or n_bad_field or n_bad_class or n_bad_range or n_bad_num or problems)
    print("\n结论:", "存在违规，提交前必须修复" if bad else "格式自检通过")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
