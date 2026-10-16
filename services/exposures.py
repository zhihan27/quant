"""市值与行业暴露的获取、缓存与合并。

行业优先使用申万一级分类（AkShare 逐行业接口，结果缓存 7 天），失败时退回东方财富行业字段。
暴露数据保存为 Parquet，供截面中性化与页面追溯使用。
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from config import settings
from services.panel_store import PanelStore

LOGGER = logging.getLogger("quant.exposures")
EM_INDUSTRY_SOURCE = "东方财富行业（免费字段，未使用申万一级）"
SW_INDUSTRY_SOURCE = "申万一级行业（AkShare 行业成分）"
SW_SCRIPT = """
import json
import sys

import akshare as ak

output = sys.argv[1]
info = ak.sw_index_first_info()
with open(output, "a", encoding="utf-8") as handle:
    for _, row in info.iterrows():
        code = str(row["行业代码"]).split(".")[0]
        name = str(row["行业名称"])
        try:
            members = ak.index_component_sw(symbol=code)
            symbols = members["证券代码"].astype(str)
            for symbol in symbols:
                handle.write(json.dumps({"code": symbol.strip().zfill(6), "industry": name}, ensure_ascii=False) + "\\n")
        except Exception as exc:  # 单个行业失败不影响其余行业落盘
            handle.write(json.dumps({"industry": name, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\\n")
        handle.flush()
"""


def parse_sw_records(lines: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    """解析申万行业成分 JSONL，返回成员记录与逐行业失败原因。"""
    members: list[dict[str, str]] = []
    failures: list[str] = []
    industries: set[str] = set()
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            failures.append("行业成分返回内容无法解析")
            continue
        industry = str(record.get("industry", "")).strip()
        if record.get("error"):
            failures.append(f"{industry}：{record['error']}")
            continue
        code = str(record.get("code", "")).strip()
        if len(code) == 6 and code.isdigit() and industry:
            members.append({"code": code, "industry": industry})
            industries.add(industry)
        else:
            failures.append(f"{industry}：成分记录缺少有效六位代码")
    return members, failures


def fetch_sw_industry_map(store: PanelStore | None = None, progress: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """调用 AkShare 逐行业读取申万一级成分，子进程超时后仍使用已落盘的部分结果。"""
    store = store or PanelStore()
    target = store.exposure_dir / "industry_sw.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    output = target.with_suffix(".jsonl")
    if output.exists():
        output.unlink()
    meta: dict[str, Any] = {"source": SW_INDUSTRY_SOURCE, "complete": False, "failures": []}
    if progress:
        progress("正在读取申万一级行业成分，首次约需 2 至 4 分钟…")
    try:
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", SW_SCRIPT, str(output)],
            capture_output=True, check=True, encoding="utf-8", text=True,
            timeout=settings.SW_INDUSTRY_TOTAL_TIMEOUT_SECONDS, cwd=settings.PROJECT_ROOT,
        )
        del completed
    except subprocess.TimeoutExpired:
        meta["warning"] = f"申万行业接口超过 {settings.SW_INDUSTRY_TOTAL_TIMEOUT_SECONDS} 秒，仅使用已返回的行业"
    except (subprocess.CalledProcessError, OSError) as exc:
        meta["warning"] = f"申万行业接口调用失败：{type(exc).__name__}"
        LOGGER.warning("申万行业接口失败 detail=%s", exc)
    lines = output.read_text(encoding="utf-8").splitlines() if output.exists() else []
    members, failures = parse_sw_records(lines)
    meta["failures"] = failures[:20]
    meta["failed_industries"] = len(failures)
    if not members:
        meta["status"] = "申万一级行业不可用，将退回东方财富行业字段"
        return pd.DataFrame(columns=["code", "industry"]), meta
    frame = pd.DataFrame(members).drop_duplicates("code", keep="last").reset_index(drop=True)
    meta["industry_count"] = int(frame["industry"].nunique())
    meta["member_count"] = int(len(frame))
    meta["complete"] = meta["industry_count"] >= settings.SW_INDUSTRY_MINIMUM_INDUSTRIES
    meta["fetched_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    meta["status"] = (
        f"申万一级行业已获取 {meta['industry_count']} 个行业 / {meta['member_count']} 只成分"
        if meta["complete"]
        else f"申万一级行业仅获取 {meta['industry_count']} 个行业，低于完整性阈值，改用东方财富行业"
    )
    frame.to_parquet(target, index=False, compression=settings.PARQUET_COMPRESSION)
    store.save_exposures_meta({**store.exposures_meta(), "sw_industry": meta})
    return frame, meta


def load_sw_industry_map(store: PanelStore | None = None, force_refresh: bool = False, progress: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取缓存中的申万一级行业映射，缓存过期或强制刷新时重新获取。"""
    store = store or PanelStore()
    target = store.exposure_dir / "industry_sw.parquet"
    meta = store.exposures_meta().get("sw_industry", {})
    if target.exists() and not force_refresh and meta.get("complete"):
        fetched_at = meta.get("fetched_at")
        try:
            fetched = datetime.fromisoformat(fetched_at)
            age = (datetime.now(fetched.tzinfo) - fetched).total_seconds()
        except (TypeError, ValueError):
            age = settings.INDUSTRY_CACHE_SECONDS + 1
        if age < settings.INDUSTRY_CACHE_SECONDS:
            frame = pd.read_parquet(target)
            return frame, {**meta, "status": f"{meta.get('status', '申万一级行业本地缓存')}；缓存时间 {fetched_at}"}
    return fetch_sw_industry_map(store, progress)


def build_exposure_frame(universe: pd.DataFrame, industry_map: pd.DataFrame | None = None) -> pd.DataFrame:
    """把股票池市值字段与行业分类合并为中性化暴露表。"""
    if universe is None or universe.empty:
        return pd.DataFrame(columns=["code", "name", "market_cap_total", "market_cap_float", "industry", "industry_source"])
    frame = universe.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    for column in ("market_cap_total", "market_cap_float"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.NA
    frame["industry_em"] = frame["industry"].astype(object) if "industry" in frame else None
    frame["industry"] = frame["industry_em"]
    frame["industry_source"] = EM_INDUSTRY_SOURCE
    if industry_map is not None and not industry_map.empty:
        mapping = industry_map.drop_duplicates("code", keep="last").set_index("code")["industry"]
        matched = frame["code"].map(mapping)
        frame.loc[matched.notna(), "industry"] = matched[matched.notna()]
        frame["industry_source"] = frame["industry_source"].where(matched.isna(), SW_INDUSTRY_SOURCE)
    columns = ["code", "name", "price", "market_cap_total", "market_cap_float", "industry", "industry_em", "industry_source"]
    kept = [column for column in columns if column in frame]
    result = frame[kept].reset_index(drop=True)
    result["fetched_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return result


def load_exposures(
    service: Any,
    scope: str = "all",
    force_refresh: bool = False,
    use_sw_industry: bool = True,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """获取市值与行业暴露；优先本地 Parquet 缓存，申万行业缺失时退回东方财富行业。"""
    store: PanelStore = service.panels
    cached = store.load_exposures()
    cached_meta = store.exposures_meta().get("exposures", {})
    if cached is not None and not force_refresh:
        try:
            fetched = datetime.fromisoformat(cached_meta.get("fetched_at", ""))
            age = (datetime.now(fetched.tzinfo) - fetched).total_seconds()
        except (TypeError, ValueError):
            age = settings.EXPOSURE_CACHE_SECONDS + 1
        if age < settings.EXPOSURE_CACHE_SECONDS:
            return cached, {**cached_meta, "status": f"暴露数据使用本地缓存（{cached_meta.get('fetched_at')}）"}
    universe, universe_meta = service.get_universe(scope=scope, force_refresh=force_refresh)
    industry_map = pd.DataFrame(columns=["code", "industry"])
    industry_meta: dict[str, Any] = {"status": "未启用申万一级行业，使用东方财富行业字段"}
    if use_sw_industry:
        try:
            industry_map, industry_meta = load_sw_industry_map(store, force_refresh=force_refresh, progress=progress)
        except (OSError, ValueError, ImportError) as exc:
            industry_meta = {"status": f"申万一级行业不可用，改用东方财富行业：{type(exc).__name__}"}
            LOGGER.warning("申万行业映射读取失败 detail=%s", exc)
    frame = build_exposure_frame(universe, industry_map if industry_meta.get("complete") else None)
    meta = {
        "status": f"暴露数据已更新，共 {len(frame)} 只股票",
        "fetched_at": frame["fetched_at"].iloc[0] if not frame.empty else None,
        "universe_source": universe_meta.get("source"),
        "universe_fetched_at": universe_meta.get("fetched_at"),
        "industry": industry_meta,
        "industry_source": SW_INDUSTRY_SOURCE if industry_meta.get("complete") else EM_INDUSTRY_SOURCE,
        "codes": int(len(frame)),
        "with_market_cap": int(pd.to_numeric(frame["market_cap_float"], errors="coerce").notna().sum()) if not frame.empty else 0,
        "limitation": "市值为采集时点的流通市值，历史市值为流通股本×历史收盘价的近似，非严格点时数据。",
    }
    try:
        store.save_exposures(frame)
        store.save_exposures_meta({**store.exposures_meta(), "exposures": meta})
    except (OSError, ValueError) as exc:
        meta["status"] = f"暴露数据已获取，但 Parquet 缓存写入失败：{type(exc).__name__}"
        LOGGER.warning("暴露数据写入失败 detail=%s", exc)
    return frame, meta
