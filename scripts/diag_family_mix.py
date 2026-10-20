"""诊断"本地分不转移"的一个具体可能：**val300 与复赛测试集的"命名家族"构成不同**。

背景（2026-09-30）：
  官方 2000 张训练图有 6 个互不相同的命名家族（`000002`、`000013_040_suppl_00000167`、
  `shuming_486_00000001`、`00000004`、`hehe_101_000001_049_00000022`、`000002_080_00000048`）。
  如果我们的 val300 里"好判的家族"占比高、而复赛测试集里"难判的家族"占比高，
  那么本地分（58.4）与榜分（51.7）之间那 6.6 分就有一大块可以用**构成差异**解释，
  而不是"模型不行"——这也能告诉我们：**该往哪个家族上使劲**。

做法：
  1) 按正则把 val300 与复赛测试集各自分到 6 个家族；
  2) 对 val300 的每个家族，单独跑一次官方口径评测（同 gt/pred 目录，只取该家族）；
  3) 用**测试集的家族占比**对 val 的家族 AP 加权，得到"按测试集构成校正后的本地分"。

用法：
    python scripts/diag_family_mix.py --pred submissions/_tta_fine_fineC \
        --gt-dir data/valgt_lbproxy --val-list splits/lbproxy_val.txt \
        --test-list splits/phase2_test_all_local.txt
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
PY = str(WS / ".venv" / "Scripts" / "python.exe")

FAMILIES = [
    ("num6", re.compile(r"^\d{6}$")),
    ("num6_3_8", re.compile(r"^\d{6}_\d{3}_\d{8}$")),
    ("shuming", re.compile(r"^shuming_")),
    ("num8", re.compile(r"^\d{8}$")),
    ("hehe", re.compile(r"^hehe_")),
    ("num6_3", re.compile(r"^\d{6}_\d{3}_$")),
]


def family_of(stem: str) -> str:
    for name, pat in FAMILIES:
        if pat.match(stem):
            return name
    return "other"


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=WS, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return (p.stdout or "") + (p.stderr or "")


def stems(path: Path) -> list[str]:
    return [Path(x.strip()).stem for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt-dir", default="data/valgt_lbproxy")
    ap.add_argument("--val-list", default="splits/lbproxy_val.txt")
    ap.add_argument("--test-list", default="splits/phase2_test_all_local.txt")
    a = ap.parse_args()

    val = stems(WS / a.val_list)
    test = stems(WS / a.test_list)
    fv = Counter(family_of(s) for s in val)
    ft = Counter(family_of(s) for s in test)

    print(f"[family] val300 家族分布 : {dict(fv)}")
    print(f"[family] 测试集家族分布 : {dict(ft)}   （共 {len(test)} 张）")
    print(f"{'family':12s}{'val张数':>8s}{'val占比':>10s}{'test张数':>10s}{'test占比':>10s}")
    for name, _ in FAMILIES:
        print(f"{name:12s}{fv[name]:8d}{fv[name]/len(val):10.3f}{ft[name]:10d}{ft[name]/len(test):10.3f}")

    # 逐家族评测（把该家族的预测与 GT 各自软链到临时目录）
    pred_root = (WS / a.pred) if not Path(a.pred).is_absolute() else Path(a.pred)
    gt_root = WS / a.gt_dir
    per_family: dict[str, float] = {}
    tmp = Path(tempfile.mkdtemp(prefix="fam_"))
    try:
        for name, _ in FAMILIES:
            sel = [s for s in val if family_of(s) == name]
            if len(sel) < 10:
                print(f"  {name:12s} 样本太少({len(sel)}) 跳过")
                continue
            pd, gd = tmp / f"pred_{name}", tmp / f"gt_{name}"
            pd.mkdir(exist_ok=True)
            gd.mkdir(exist_ok=True)
            for s in sel:
                for root, dst in ((pred_root, pd), (gt_root, gd)):
                    f = root / f"{s}.txt"
                    if f.exists():
                        shutil.copy2(f, dst / f.name)
            ev = run([PY, "src/eval/map_eval.py", "--gt-dir", str(gd), "--pred-dir", str(pd)])
            m = re.search(r"mAP@50-95\s*=\s*([0-9.]+)", ev)
            val_ap = float(m.group(1)) if m else float("nan")
            per_family[name] = val_ap
            print(f"  {name:12s} n={len(sel):3d}  val mAP={val_ap:.4f}  "
                  f"（测试集该家族占比 {ft[name]/len(test):.3f}）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 按测试集构成加权
    num = sum(per_family[n] * ft[n] for n in per_family)
    den = sum(ft[n] for n in per_family)
    print(f"\n[family] 按测试集构成加权的本地分 ≈ {num/den:.4f}"
          f"（未加权平均 {sum(per_family.values())/len(per_family):.4f}）")
    print("⇒ 若加权分明显低于未加权分，说明『测试集里我们弱的家族占比更高』，这就是本地虚高的来源之一。")


if __name__ == "__main__":
    main()
