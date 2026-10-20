"""路径安全的图像读取。

踩坑记录（2026-09-14）：`cv2.imread("含中文的路径.jpg")` 会在 Windows 上**偶发**返回 None，
但同一个文件用 `open()` 能读、用 `cv2.imdecode(np.fromfile(...))` 也能读。
原因是 OpenCV 的 `imread` 走窄字符路径接口，中文路径的编解码不稳定；
而 `np.fromfile` 走 Python 的文件 API（宽字符），再用 `imdecode` 解码，全程不依赖编码。

这个坑很危险：读图失败时脚本会写出**空预测文件**，评测就"安静地"给出 0 分。
所以全项目统一用这里的 `imread`，不要再直接调 `cv2.imread`。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR):
    """与 cv2.imread 同签名，但能读中文路径。失败返回 None。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)
