"""低复制读取 DataFrame 列数组的内部工具。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np


def column_values_without_attrs(frame: pd.DataFrame, column: str) -> Any:
    """读取只读列数组，避免构造 Series 时深拷贝整份 DataFrame.attrs。"""
    position = frame.columns.get_loc(column)
    if not isinstance(position, (int, np.integer)):
        raise ValueError(f"列名必须唯一：{column}")
    position = int(position)
    manager = getattr(frame, "_mgr", None)
    getter = getattr(manager, "iget_values", None)
    if callable(getter):
        return getter(position)
    # 兼容没有列管理器读取接口的 pandas 版本；调用方评估表应保持窄列。
    return frame.to_numpy(copy=False)[:, position]
