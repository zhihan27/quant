"""股票池状态分类与生存者偏差控制。

系统的横截面计算只在「当前及历史在市正常交易」的股票池内进行：已退市、退市整理期与被特别
处理（ST / *ST）的标的一律不进入选股扫描、因子计算与回测链路。这里不重新纳入历史已退市
标的，因此回测股票池仍是「本次采集到的在市标的」，不会被伪装成全历史成分股。

状态来源分三层，优先级由高到低：
1. 交易所退市公司名单（决定 DELISTED）；
2. 全市场 ST 名单（决定 ST）；
3. 证券简称启发式（兜底，覆盖尚未进入名单的退市整理期与 ST 标的）。
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from typing import Any

import pandas as pd

from config import settings
from services.panel_store import PanelStore

LOGGER = logging.getLogger("quant.universe")

STATUS_NORMAL = "NORMAL"
STATUS_ST = "ST"
STATUS_DELISTING = "DELISTING_PERIOD"
STATUS_DELISTED = "DELISTED"
STATUS_UNKNOWN = "UNKNOWN"

STATUS_LABELS = {
    STATUS_NORMAL: "正常交易",
    STATUS_ST: "特别处理（ST / *ST）",
    STATUS_DELISTING: "退市整理期",
    STATUS_DELISTED: "已退市",
    STATUS_UNKNOWN: "状态未知",
}
# 被排除出股票池的状态；UNKNOWN 不排除，交由行情可用性判定，避免误杀名称缺失的正常股票。
EXCLUDED_STATUSES = (STATUS_ST, STATUS_DELISTING, STATUS_DELISTED)

# 除权除息前缀会出现在简称最前面（如「XD大秦铁路」），判定前须剥离，否则会误判正常股票。
_EX_RIGHT_PREFIX = re.compile(r"^(?:XD|XR|DR)+", flags=re.IGNORECASE)
# 特别处理简称以 ST / *ST / SST / S*ST 开头；只在开头匹配，避免命中名称中间的字母组合。
_ST_PREFIX = re.compile(r"^(?:\*?ST|S\*?ST|SST|PT)", flags=re.IGNORECASE)

REFERENCE_SOURCE = "交易所退市名单 + 全市场 ST 名单（AkShare）"

REFERENCE_SCRIPT = """
import json
import sys

import akshare as ak

output = sys.argv[1]


def emit(record):
    handle.write(json.dumps(record, ensure_ascii=False) + "\\n")


with open(output, "a", encoding="utf-8") as handle:
    try:
        frame = ak.stock_info_sh_delist(symbol="全部")
        for _, row in frame.iterrows():
            emit({
                "kind": "delisted", "exchange": "上交所",
                "code": str(row.get("公司代码", "")).strip().zfill(6),
                "name": str(row.get("公司简称", "")).strip(),
                "delist_date": str(row.get("暂停上市日期", "")).strip() or None,
            })
    except Exception as exc:
        emit({"kind": "error", "source": "上交所退市名单", "error": f"{type(exc).__name__}: {exc}"})
    handle.flush()
    try:
        frame = ak.stock_info_sz_delist(symbol="终止上市公司")
        for _, row in frame.iterrows():
            emit({
                "kind": "delisted", "exchange": "深交所",
                "code": str(row.get("证券代码", "")).strip().zfill(6),
                "name": str(row.get("证券简称", "")).strip(),
                "delist_date": str(row.get("终止上市日期", "")).strip() or None,
            })
    except Exception as exc:
        emit({"kind": "error", "source": "深交所退市名单", "error": f"{type(exc).__name__}: {exc}"})
    handle.flush()
    try:
        frame = ak.stock_zh_a_st_em()
        for _, row in frame.iterrows():
            emit({
                "kind": "st",
                "code": str(row.get("代码", "")).strip().zfill(6),
                "name": str(row.get("名称", "")).strip(),
            })
    except Exception as exc:
        emit({"kind": "error", "source": "全市场 ST 名单", "error": f"{type(exc).__name__}: {exc}"})
    handle.flush()
"""


def classify_name_status(name: Any) -> str:
    """按证券简称判定状态；只做前缀匹配与「退」字样匹配，避免误伤名称中偶现的字母组合。"""
    text = str(name or "").strip()
    if not text or text.lower() in ("nan", "none"):
        return STATUS_UNKNOWN
    text = _EX_RIGHT_PREFIX.sub("", text)
    # 退市整理期简称形如「退市海润」「海润退」「国华退」，均含「退」字。
    if "退" in text:
        return STATUS_DELISTING
    if _ST_PREFIX.match(text):
        return STATUS_ST
    return STATUS_NORMAL


def parse_reference_records(lines: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """解析子进程落盘的 JSONL，返回 code -> 状态记录 与逐源失败原因。"""
    records: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            failures.append("股票池状态返回内容无法解析")
            continue
        if record.get("kind") == "error":
            failures.append(f"{record.get('source')}：{record.get('error')}")
            continue
        code = str(record.get("code", "")).strip()
        if not re.fullmatch(r"\d{6}", code):
            failures.append(f"{record.get('kind')} 记录缺少有效六位代码")
            continue
        kind = record.get("kind")
        if kind == "delisted":
            records[code] = {
                "code": code, "name": record.get("name") or "",
                "stock_status": STATUS_DELISTED, "status_source": record.get("exchange") or "交易所退市名单",
                "delist_date": record.get("delist_date"),
            }
        elif kind == "st" and code not in records:
            records[code] = {
                "code": code, "name": record.get("name") or "",
                "stock_status": STATUS_ST, "status_source": "全市场 ST 名单", "delist_date": None,
            }
    return records, failures


def fetch_reference(store: PanelStore | None = None, progress: Any = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """调用 AkShare 读取交易所退市名单与 ST 名单，子进程超时后仍使用已落盘的部分结果。"""
    store = store or PanelStore()
    store.universe_dir.mkdir(parents=True, exist_ok=True)
    output = store.universe_dir / "reference.jsonl"
    if output.exists():
        output.unlink()
    meta: dict[str, Any] = {"source": REFERENCE_SOURCE, "failures": []}
    if progress:
        progress("正在读取交易所退市名单与全市场 ST 名单…")
    try:
        subprocess.run(
            [sys.executable, "-X", "utf8", "-c", REFERENCE_SCRIPT, str(output)],
            capture_output=True, check=True, encoding="utf-8", text=True,
            timeout=settings.UNIVERSE_REFERENCE_TIMEOUT_SECONDS, cwd=settings.PROJECT_ROOT,
        )
    except subprocess.TimeoutExpired:
        meta["warning"] = f"股票池状态名单接口超过 {settings.UNIVERSE_REFERENCE_TIMEOUT_SECONDS} 秒，仅使用已返回的来源"
    except (subprocess.CalledProcessError, OSError) as exc:
        meta["warning"] = f"股票池状态名单接口调用失败：{type(exc).__name__}"
        LOGGER.warning("股票池状态名单接口失败 detail=%s", exc)
    lines = output.read_text(encoding="utf-8").splitlines() if output.exists() else []
    records, failures = parse_reference_records(lines)
    meta["failures"] = failures[:20]
    meta["failed_sources"] = len(failures)
    if not records:
        meta["status"] = "股票池状态名单不可用，本次仅按证券简称启发式过滤"
        return pd.DataFrame(columns=["code", "name", "stock_status", "status_source", "delist_date"]), meta
    frame = pd.DataFrame(list(records.values())).drop_duplicates("code", keep="last").reset_index(drop=True)
    meta.update(
        delisted_count=int(frame["stock_status"].eq(STATUS_DELISTED).sum()),
        st_count=int(frame["stock_status"].eq(STATUS_ST).sum()),
        codes=int(len(frame)),
        fetched_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    meta["status"] = f"股票池状态名单已获取：退市 {meta['delisted_count']} 只，ST {meta['st_count']} 只"
    store.save_universe_status(frame)
    return frame, meta


def load_reference(
    store: PanelStore | None = None, force_refresh: bool = False, progress: Any = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取股票池状态名单，本地缓存过期或强制刷新时重新获取。"""
    store = store or PanelStore()
    cached = store.load_universe_status()
    meta_path = store.universe_dir / "status.meta.json"
    cached_meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached_meta = {}
    if cached is not None and not force_refresh:
        try:
            fetched = datetime.fromisoformat(cached_meta.get("fetched_at", ""))
            age = (datetime.now(fetched.tzinfo) - fetched).total_seconds()
        except (TypeError, ValueError):
            age = settings.UNIVERSE_STATUS_CACHE_SECONDS + 1
        if age < settings.UNIVERSE_STATUS_CACHE_SECONDS:
            return cached, {**cached_meta, "status": f"股票池状态使用本地缓存（{cached_meta.get('fetched_at')}）"}
    frame, meta = fetch_reference(store, progress)
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError) as exc:
        LOGGER.warning("股票池状态元信息写入失败 detail=%s", exc)
    return frame, meta


def reference_status_map(reference: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    """把状态名单转换为 code -> 记录 的映射，便于逐行判定。"""
    if reference is None or reference.empty or "code" not in reference:
        return {}
    frame = reference.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return {row["code"]: row for row in frame.drop_duplicates("code", keep="last").to_dict("records")}


def status_for(code: str, name: Any, status_map: dict[str, dict[str, Any]] | None = None) -> tuple[str, str]:
    """按「名单优先、简称兜底」的顺序给出状态与来源说明。"""
    text = str(code).zfill(6)
    record = (status_map or {}).get(text)
    if record is not None:
        status = str(record.get("stock_status") or "")
        if status in STATUS_LABELS:
            return status, str(record.get("status_source") or "股票池状态名单")
    return classify_name_status(name), "证券简称启发式"


def is_excluded_status(status: str) -> bool:
    return str(status) in EXCLUDED_STATUSES


def exclude_reason(status: str, source: str | None = None) -> str:
    """把状态翻译成页面可直接展示的排除原因。"""
    label = STATUS_LABELS.get(str(status), str(status))
    return f"{label}已排除（{source}）" if source else f"{label}已排除"


def annotate(frame: pd.DataFrame, reference: pd.DataFrame | None = None, name_column: str = "name", code_column: str = "code") -> pd.DataFrame:
    """为股票池逐行标注状态；缺少代码列时原样返回，由调用方决定如何处理。"""
    if frame is None or frame.empty or code_column not in frame:
        return frame if frame is not None else pd.DataFrame()
    result = frame.copy()
    status_map = reference_status_map(reference)
    codes = result[code_column].astype(str).str.zfill(6)
    names = result[name_column] if name_column in result else pd.Series("", index=result.index)
    resolved = [status_for(code, name, status_map) for code, name in zip(codes, names)]
    result[code_column] = codes
    result["stock_status"] = [item[0] for item in resolved]
    result["status_source"] = [item[1] for item in resolved]
    result["excluded"] = result["stock_status"].map(is_excluded_status)
    result["exclude_reason"] = [
        exclude_reason(status, source) if is_excluded_status(status) else ""
        for status, source in zip(result["stock_status"], result["status_source"])
    ]
    return result


def filter_universe(
    frame: pd.DataFrame, reference: pd.DataFrame | None = None, name_column: str = "name", code_column: str = "code"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """过滤退市、退市整理期与 ST 标的，返回保留的股票池与可追溯报告。"""
    annotated = annotate(frame, reference, name_column, code_column)
    if annotated.empty:
        return annotated, {"status": "股票池为空，未执行状态过滤", "kept": 0, "excluded": 0, "by_status": {}}
    if "excluded" not in annotated:
        return annotated, {"status": "股票池缺少代码列，未执行状态过滤", "kept": int(len(annotated)), "excluded": 0, "by_status": {}}
    excluded = annotated.loc[annotated["excluded"]]
    kept = annotated.loc[~annotated["excluded"]].reset_index(drop=True)
    by_status = {str(key): int(value) for key, value in excluded["stock_status"].value_counts().items()}
    report = {
        "source": REFERENCE_SOURCE if reference is not None and not reference.empty else "证券简称启发式",
        "checked": int(len(annotated)),
        "kept": int(len(kept)),
        "excluded": int(len(excluded)),
        "by_status": by_status,
        "excluded_codes": excluded[code_column].tolist()[:200],
        "status": (
            f"股票池过滤：检查 {len(annotated)} 只，排除 {len(excluded)} 只，保留 {len(kept)} 只"
            if len(excluded) else f"股票池过滤：检查 {len(annotated)} 只，无需排除"
        ),
        "rule": "已退市、退市整理期与被特别处理（ST / *ST）标的在扫描、因子计算与回测前一律排除。",
        "limitation": "过滤基于当前在市状态与当前名单，回测股票池仍为本次采集到的在市标的，无法还原历史成分股。",
    }
    return kept, report
