"""带**显存上限**的 D-FINE 启动器。

为什么需要它：D-FINE 是第三方仓库，**没有我们 `remote_launch.sh` 里的 `--gpu-frac` 保护**。
而这张 4090 是和 SEED 科研实验共用的——不设上限它可能把 24 GB 吃满，
把别人的实验挤掉（这是我们绝对不能做的事）。

做法：在**导入 torch 之后、任何 CUDA 分配之前**设
`torch.cuda.set_per_process_memory_fraction(frac, 0)`，等价于我们的 `--gpu-frac`。

用法（其余参数原样透传给 D-FINE 的 train.py）：
    AIC_GPU_FRAC=0.65 CUDA_VISIBLE_DEVICES=0 python scripts/dfine_run.py \
        -c configs/dfine/custom/dfine_hgnetv2_s_aic.yml \
        -t ./weights/dfine_s_obj2coco.pth \
        -d cuda:0 --use-amp
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    frac = float(os.environ.get("AIC_GPU_FRAC", "0.65"))
    root = Path(os.environ.get("DFINE_ROOT", str(Path.home() / "aic2026" / "D-FINE")))
    if not (root / "train.py").exists():
        sys.exit(f"[dfine_run] 找不到 {root}/train.py")

    sys.path.insert(0, str(root))
    os.chdir(root)          # D-FINE 的 output_dir / 相对配置路径都依赖 cwd

    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            free, _ = torch.cuda.mem_get_info(0)
            reserve = int(os.environ.get("AIC_GPU_RESERVE_GB", "2")) * 2**30
            # ★ 两个上限取小：
            #   a) 绝对上限 AIC_GPU_FRAC × 总显存（硬顶）
            #   b) **"现在空闲的显存 − 留给别人的余量"**
            # 只设 (a) 是会翻车的：`set_per_process_memory_fraction` 是**占总量**的比例，
            # 不是占空量的比例。y26m 那次就是 0.65×23.65 = 15.4 GB 的额度，
            # 可当时 SEED 实验已经占了 8.4 GB，8.4+15.4 > 23.65 ⇒ 直接 OOM。
            budget = min(int(frac * total), max(0, int(free - reserve)))
            budget = max(budget, 2 * 2**30)      # 兜底，别把自己卡死
            torch.cuda.set_per_process_memory_fraction(budget / total, 0)
            print(f"[dfine_run] 总显存 {total / 2**30:.1f} GB，当前空闲 {free / 2**30:.1f} GB "
                  f"⇒ 本进程上限 {budget / 2**30:.1f} GB（预留 {reserve / 2**30:.0f} GB 给别人）",
                  flush=True)
        else:
            print("[dfine_run] 警告：CUDA 不可用", flush=True)
    except Exception as exc:  # noqa: BLE001  设不上不该中断训练，但要说出来
        print(f"[dfine_run] 设显存上限失败（继续运行）：{type(exc).__name__}: {exc}", flush=True)

    print(f"[dfine_run] 工作目录 {root}；参数 {sys.argv[1:]}", flush=True)
    sys.argv = [str(root / "train.py"), *sys.argv[1:]]
    runpy.run_path(str(root / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
