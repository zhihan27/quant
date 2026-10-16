"""Parquet 列式面板存储与 DuckDB 向量化截面查询。

大规模面板数据不再依赖 SQLite 行式 JSON 缓存：逐股日线、清洗后的因子面板与市值行业暴露
统一保存为 Parquet，由 DuckDB 完成每日截面的读取与 groupby 计算。SQLite 仍保留接口请求
事件与历史同步状态，两者职责不重叠。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import settings

LOGGER = logging.getLogger("quant.panel_store")
ADJUSTMENT_LABELS = {"qfq": "qfq", "raw": "raw"}
# 按 code=*.parquet 分区的衍生数据目录：PIT 财务面板、筹码分布因子、高频日内因子，
# 以及用于累积历史深度的原始分钟线。
PARTITION_DIRECTORIES = ("fundamentals", "chip", "intraday", "intraday_bars")


def _now_text() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _atomic_write(frame: pd.DataFrame, path: Path) -> Path:
    """先写临时文件再替换，避免中断留下半个面板。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression=settings.PARQUET_COMPRESSION)
    os.replace(temporary, path)
    return path


class PanelStore:
    """Parquet 面板读写与 DuckDB 查询入口。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or settings.PANEL_DIR)
        self.history_dir = self.root / "history"
        self.factor_dir = self.root / "factors"
        self.exposure_dir = self.root / "exposures"
        self.universe_dir = self.root / "universe"
        self.partition_dirs = {name: self.root / name for name in PARTITION_DIRECTORIES}

    # ---------------------------------------------------------------- 历史行情
    def history_path(self, code: str, adjustment: str = "qfq") -> Path:
        adjustment = ADJUSTMENT_LABELS.get(str(adjustment), "raw")
        return self.history_dir / f"adjustment={adjustment}" / f"code={str(code).zfill(6)}.parquet"

    def write_history(self, code: str, frame: pd.DataFrame, adjustment: str = "qfq", meta: dict[str, Any] | None = None) -> Path:
        """保存单只股票的清洗后日线；同一代码重复写入整体覆盖，不做增量拼接。"""
        stored = frame.copy()
        stored["code"] = str(code).zfill(6)
        stored["stored_at"] = _now_text()
        stored["history_source"] = str((meta or {}).get("source", ""))
        return _atomic_write(stored, self.history_path(code, adjustment))

    def read_history(self, code: str, adjustment: str = "qfq") -> pd.DataFrame | None:
        path = self.history_path(code, adjustment)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            LOGGER.warning("读取 Parquet 历史失败 code=%s detail=%s", code, exc)
            return None

    def history_codes(self, adjustment: str = "qfq") -> list[str]:
        directory = self.history_dir / f"adjustment={ADJUSTMENT_LABELS.get(adjustment, 'raw')}"
        if not directory.exists():
            return []
        return sorted(path.stem.split("=", 1)[-1] for path in directory.glob("code=*.parquet"))

    # ---------------------------------------------------------------- 因子面板
    def panel_path(self, version: str | None = None) -> Path:
        return self.factor_dir / f"panel_v{version or settings.MODEL_VERSION}.parquet"

    def save_panel(self, frame: pd.DataFrame, version: str | None = None, context_id: str | None = None) -> Path:
        """保存当日截面因子面板；context_id 与模型上下文绑定，防止错配复用。"""
        stored = frame.copy()
        if context_id is not None:
            stored.attrs["context_id"] = context_id
        return _atomic_write(stored, self.panel_path(version))

    def load_panel(self, version: str | None = None) -> pd.DataFrame | None:
        path = self.panel_path(version)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            LOGGER.warning("读取 Parquet 因子面板失败 detail=%s", exc)
            return None

    # ---------------------------------------------------------------- 暴露数据
    def save_exposures(self, frame: pd.DataFrame) -> Path:
        return _atomic_write(frame, self.exposure_dir / "exposures.parquet")

    def load_exposures(self) -> pd.DataFrame | None:
        path = self.exposure_dir / "exposures.parquet"
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            LOGGER.warning("读取 Parquet 暴露数据失败 detail=%s", exc)
            return None

    def exposures_meta(self) -> dict[str, Any]:
        path = self.exposure_dir / "exposures.meta.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_exposures_meta(self, meta: dict[str, Any]) -> None:
        path = self.exposure_dir / "exposures.meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    # ------------------------------------------------------- 按代码分区的衍生数据
    def partition_dir_for(self, name: str) -> Path:
        """返回已登记的衍生数据分区目录；未登记的名称直接报错，避免写到别处。"""
        if name not in self.partition_dirs:
            raise ValueError(f"未登记的分区名称：{name}；可用 {'、'.join(PARTITION_DIRECTORIES)}。")
        return self.partition_dirs[name]

    def partition_path(self, name: str, code: str) -> Path:
        return self.partition_dir_for(name) / f"code={str(code).zfill(6)}.parquet"

    def write_partition(self, name: str, code: str, frame: pd.DataFrame) -> Path:
        """整体覆盖写入单只股票的分区文件；重复写入不做增量拼接，避免新旧口径混存。"""
        stored = frame.copy()
        stored["code"] = str(code).zfill(6)
        stored["stored_at"] = _now_text()
        return _atomic_write(stored, self.partition_path(name, code))

    def read_partition(self, name: str, code: str) -> pd.DataFrame | None:
        path = self.partition_path(name, code)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            LOGGER.warning("读取 Parquet 分区失败 name=%s code=%s detail=%s", name, code, exc)
            return None

    def partition_codes(self, name: str) -> list[str]:
        directory = self.partition_dir_for(name)
        if not directory.exists():
            return []
        return sorted(path.stem.split("=", 1)[-1] for path in directory.glob("code=*.parquet"))

    def load_partition(self, name: str, columns: list[str] | None = None) -> pd.DataFrame | None:
        """拼接某个分区的全部股票文件；目录不存在时返回 None，由调用方决定是否报错。"""
        directory = self.partition_dir_for(name)
        if not directory.exists() or not any(directory.glob("code=*.parquet")):
            return None
        frames = []
        for path in sorted(directory.glob("code=*.parquet")):
            try:
                frames.append(pd.read_parquet(path, columns=columns))
            except (OSError, ValueError) as exc:
                LOGGER.warning("读取 Parquet 分区文件失败 path=%s detail=%s", path.name, exc)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def load_universe_status(self) -> pd.DataFrame | None:
        """读取股票池状态快照（含退市与特别处理名单）。"""
        path = self.universe_dir / "status.parquet"
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            LOGGER.warning("读取股票池状态失败 detail=%s", exc)
            return None

    def save_universe_status(self, frame: pd.DataFrame) -> Path:
        return _atomic_write(frame, self.universe_dir / "status.parquet")

    # ---------------------------------------------------------------- DuckDB
    @staticmethod
    def duckdb_available() -> bool:
        try:
            import duckdb  # noqa: F401
        except ImportError:
            return False
        return True

    def connect(self, extra_views: dict[str, pd.DataFrame] | None = None) -> Any:
        """建立 DuckDB 连接并把本地 Parquet 目录映射为视图；缺失目录不注册空视图。"""
        import duckdb

        connection = duckdb.connect(database=":memory:")
        history_glob = str(self.history_dir / "adjustment=*" / "code=*.parquet")
        if any(self.history_dir.glob("adjustment=*/code=*.parquet")):
            connection.execute(
                f"CREATE VIEW history AS SELECT * FROM read_parquet('{history_glob}', hive_partitioning=true)"
            )
        panel_glob = str(self.factor_dir / "panel_v*.parquet")
        if any(self.factor_dir.glob("panel_v*.parquet")):
            connection.execute(f"CREATE VIEW factor_panel AS SELECT * FROM read_parquet('{panel_glob}')")
        # 财务、筹码与高频衍生因子按 code=*.parquet 分区，同样映射为只读视图，
        # 便于与 history 做 ASOF JOIN 等高性能关联查询。
        for name in PARTITION_DIRECTORIES:
            directory = self.partition_dirs[name]
            if any(directory.glob("code=*.parquet")):
                glob = str(directory / "code=*.parquet")
                connection.execute(
                    f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
                )
        for name, frame in (extra_views or {}).items():
            connection.register(name, frame)
        return connection

    def query(self, sql: str, parameters: Iterable[Any] | None = None, extra_views: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
        """执行只读查询；DuckDB 不可用时明确报错，不静默退回。"""
        connection = self.connect(extra_views)
        try:
            return connection.execute(sql, list(parameters or [])).fetch_df()
        finally:
            connection.close()

    def cross_section_zscore(
        self,
        frame: pd.DataFrame,
        columns: list[str],
        minimum_stocks: int = 5,
        minimum_coverage: float = 0.8,
        output_prefix: str = "xs_",
    ) -> pd.DataFrame:
        """按交易日做截面 Z 分数；DuckDB 完成分组聚合，缺失依赖时退回等价 pandas 实现。"""
        if frame.duplicated(["date", "code"]).any():
            raise ValueError("截面数据中股票与交易日重复。")
        if not columns:
            return frame.copy()
        if not self.duckdb_available():
            return self._pandas_zscore(frame, columns, minimum_stocks, minimum_coverage, output_prefix)
        working = frame.copy()
        working["_row"] = np.arange(len(working))
        select_parts = ["_row", "date", "COUNT(DISTINCT code) OVER (PARTITION BY date) AS _size"]
        for position, column in enumerate(columns):
            quoted = '"' + str(column).replace('"', '""') + '"'
            select_parts.extend(
                [
                    f"AVG({quoted}) OVER w AS _m{position}",
                    f"STDDEV_POP({quoted}) OVER w AS _s{position}",
                    f"COUNT({quoted}) OVER w AS _c{position}",
                ]
            )
        sql = "SELECT " + ", ".join(select_parts) + " FROM data WINDOW w AS (PARTITION BY date)"
        connection = self.connect({"data": working})
        try:
            statistics = connection.execute(sql).fetch_df()
        except Exception as exc:  # DuckDB 运行期错误不应让整条扫描链路失败
            LOGGER.warning("DuckDB 截面聚合失败，改用 pandas 等价实现 detail=%s", exc)
            connection.close()
            return self._pandas_zscore(frame, columns, minimum_stocks, minimum_coverage, output_prefix)
        connection.close()
        statistics = statistics.set_index("_row").reindex(working["_row"]).reset_index(drop=True)
        result = frame.copy()
        total_codes = frame["code"].nunique()
        required = max(minimum_stocks, int(np.ceil(total_codes * minimum_coverage)))
        enough_size = statistics["_size"].ge(required)
        for position, column in enumerate(columns):
            values = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            deviation = statistics[f"_s{position}"]
            score = (values - statistics[f"_m{position}"]) / deviation.where(deviation > 1e-12)
            score = score.mask((deviation <= 1e-12) & values.notna(), 0.0)
            enough = enough_size & statistics[f"_c{position}"].ge(required)
            result[f"{output_prefix}{column}"] = score.where(enough)
        return result

    @staticmethod
    def _pandas_zscore(
        frame: pd.DataFrame,
        columns: list[str],
        minimum_stocks: int,
        minimum_coverage: float,
        output_prefix: str,
    ) -> pd.DataFrame:
        """DuckDB 缺失时的等价实现，口径与 DuckDB 路径保持一致。"""
        result = frame.copy()
        size = result.groupby("date")["code"].transform("nunique")
        required = max(minimum_stocks, int(np.ceil(result["code"].nunique() * minimum_coverage)))
        for column in columns:
            values = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            groups = values.groupby(result["date"])
            deviation = groups.transform("std", ddof=0)
            score = (values - groups.transform("mean")) / deviation.where(deviation > 1e-12)
            score = score.mask((deviation <= 1e-12) & values.notna(), 0.0)
            enough = size.ge(required) & groups.transform("count").ge(required)
            result[f"{output_prefix}{column}"] = score.where(enough)
        return result

    def daily_cross_section(self, date: str, version: str | None = None) -> pd.DataFrame:
        """读取某个交易日的完整截面，供截面扫描与排序复用。"""
        panel = self.load_panel(version)
        if panel is None or panel.empty:
            return pd.DataFrame()
        target = pd.Timestamp(date)
        return panel.loc[pd.to_datetime(panel["date"]).eq(target)].reset_index(drop=True)

    def summary(self) -> dict[str, Any]:
        """面板目录的规模与更新时间，用于页面展示存储状态。"""
        files = list(self.history_dir.rglob("*.parquet")) if self.history_dir.exists() else []
        panel_files = list(self.factor_dir.glob("panel_v*.parquet")) if self.factor_dir.exists() else []
        size = sum(path.stat().st_size for path in files + panel_files)
        latest = max((path.stat().st_mtime for path in files + panel_files), default=None)
        return {
            "history_files": len(files),
            "panel_files": len(panel_files),
            "bytes": int(size),
            "latest_write": datetime.fromtimestamp(latest, ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds") if latest else None,
            "engine": "DuckDB + Parquet" if self.duckdb_available() else "pandas（未安装 duckdb）",
        }
