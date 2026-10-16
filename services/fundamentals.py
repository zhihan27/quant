"""PIT（点时）财务数据：以交易所实际公告日为锚，构建不含未来函数的财务面板。

财务数据的生效时点不是报告期结束日，而是正式公告日。免费数据源里东财的 ``NOTICE_DATE`` 在
历史区间存在系统性错位（例如 2023 年年报被标为 2025-04-03，而巨潮显示实际公告日为
2024-04-03），因此本模块**不使用**该字段，公告日一律取巨潮资讯（证监会指定披露平台）的公告
发布时间，报告期由公告标题解析得到，避免用「报告期 +1 年」这类脆弱假设去凑。

数据分两层：

- 公告日历：巨潮 ``hisAnnouncement/query``，逐股逐类别拉取，得到 ``(代码, 报告期) → 公告日``。
- 报表数值：东财 F10 全量报表按报告期批量取数，字段覆盖 Altman Z 所需的流动资产、流动负债与
  未分配利润，这些字段在精简版业绩报表接口里并不存在。

两层按 ``(代码, 报告期)`` 内连接后落盘为 ``code=*.parquet`` 分区，由 DuckDB 以 ASOF JOIN 对齐到
每个交易日。只有同时拿到公告日与报表数值的记录才会保留，缺一边即视为不可用，不做任何填充。
"""

from __future__ import annotations

import logging
import re
import time
from collections import namedtuple
from typing import Any, Iterator, Sequence

import pandas as pd
import requests

from config import settings
from services.finance_network import FinanceNetwork
from services.market_data import MarketDataError
from services.panel_store import PanelStore

LOGGER = logging.getLogger("quant.fundamentals")

PARTITION_NAME = "fundamentals"
COLUMNS = ["SECURITY_CODE", "REPORT_DATE"]
# 与分钟线服务保持同一节流口径，避免逐股翻页时被接口限流。
_MINIMUM_REQUEST_INTERVAL = 0.25

# 巨潮定期报告类别 ID；标题仍会再解析一次，类别只用于缩小拉取范围。
ANNOUNCEMENT_CATEGORIES = {
    "annual": "category_ndbg_szsh",
    "semi": "category_bndbg_szsh",
    "q1": "category_yjdbg_szsh",
    "q3": "category_sjdbg_szsh",
}
_KIND_ORDER = ("q1", "semi", "q3", "annual")
_PERIOD_MONTH_DAY = {"q1": (3, 31), "semi": (6, 30), "q3": (9, 30), "annual": (12, 31)}
# 披露窗口按「可能出现的月份」放宽，报告期交由标题解析决定；晚披露与更正在窗口内也能覆盖。
_DISCLOSURE_RULE = {
    "annual": ((1, 3, 1), (1, 6, 30)),
    "q1": ((0, 3, 15), (0, 6, 30)),
    "semi": ((0, 6, 15), (0, 9, 30)),
    "q3": ((0, 9, 15), (0, 12, 31)),
}
_KIND_LABELS = {"annual": "年度报告", "semi": "半年度报告", "q1": "一季度报告", "q3": "三季度报告"}

ReportPeriod = namedtuple("ReportPeriod", "kind year end")

# 标题解析：只认中文定期报告标题；英文版与摘要会解析成同一报告期，去重时天然合并。
_TITLE_PATTERNS = (
    ("annual", r"(20\d{2})\s*年\s*年度报告"),
    ("semi", r"(20\d{2})\s*年\s*半年度报告"),
    ("q1", r"(20\d{2})\s*年\s*第?\s*[一1]\s*季度报告"),
    ("q3", r"(20\d{2})\s*年\s*第?\s*[三3]\s*季度报告"),
)

# 字段名严格照抄接口，其中 UNASSIGN_RPOFIT 是接口自身的拼写（未分配利润）。
_GENERAL_BALANCE_FIELDS = (
    "TOTAL_ASSETS",
    "TOTAL_LIABILITIES",
    "TOTAL_EQUITY",
    "TOTAL_PARENT_EQUITY",
    "TOTAL_CURRENT_ASSETS",
    "TOTAL_CURRENT_LIAB",
    "UNASSIGN_RPOFIT",
)
# 银行／保险／证券的资产负债表没有流动与非流动的划分，也就没有这两个科目。
_FINANCIAL_BALANCE_FIELDS = tuple(
    field for field in _GENERAL_BALANCE_FIELDS
    if field not in ("TOTAL_CURRENT_ASSETS", "TOTAL_CURRENT_LIAB")
)
_INCOME_FIELDS = (
    "TOTAL_OPERATE_INCOME",
    "OPERATE_INCOME",
    "TOTAL_PROFIT",
    "OPERATE_PROFIT",
    "FINANCE_EXPENSE",
    "PARENT_NETPROFIT",
    "DEDUCT_PARENT_NETPROFIT",
)
_CASHFLOW_FIELDS = ("NETCASH_OPERATE",)

# A 股定期报告按行业分四套模板（通用 G／银行 B／保险 I／证券 S），模板名后缀对应行业，四套**互不重叠**：
# 每只股票只会出现在其中一套里。利润表的通用模板覆盖全部行业，资产负债表与现金流量表则不然——
# 只取通用模板会让整个金融板块（银行／保险／证券，约 193 只）拿不到这两张表，财务健康与经营现金流
# 因子对它们全部落空，而且是静默落空。故这两张表按四个模板逐一抓取后纵向合并。
# 每套模板的科目也不一致，而东财对「点名请求该模板不存在的科目」会让**整次请求**失败，故按模板给出
# 各自的科目；模板缺某个科目是正常的，缺一整套模板才是要拦的问题。
STATEMENT_FIELDS = {
    "balance": (
        ("GBALANCE", _GENERAL_BALANCE_FIELDS),
        ("BBALANCE", _FINANCIAL_BALANCE_FIELDS),
        ("IBALANCE", _FINANCIAL_BALANCE_FIELDS),
        ("SBALANCE", _FINANCIAL_BALANCE_FIELDS),
    ),
    "income": (("GINCOME", _INCOME_FIELDS),),
    "cashflow": tuple(
        (f"{prefix}CASHFLOW", _CASHFLOW_FIELDS) for prefix in ("G", "B", "I", "S")
    ),
}
# 一张报表的字段并集：合并后列集合统一，个别模板缺的科目自然为 NaN。
STATEMENT_UNION_FIELDS = {
    statement: tuple(dict.fromkeys(field for _report, fields in templates for field in fields))
    for statement, templates in STATEMENT_FIELDS.items()
}

_EASTMONEY_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_CNINFO_SEARCH_URL = "http://www.cninfo.com.cn/new/information/topSearch/query"
_CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
_HEADERS = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}

VALUE_COLUMNS = [column for fields in STATEMENT_UNION_FIELDS.values() for column in fields]


# ---------------------------------------------------------------- 报告期与披露窗口
def report_periods(start_year: int, end_year: int) -> list[ReportPeriod]:
    """枚举 ``[start_year, end_year]`` 内全部定期报告期，按报告期结束日升序。"""
    if end_year < start_year:
        raise ValueError("结束年份不能早于开始年份。")
    return [
        ReportPeriod(kind, year, period_end_date(kind, year))
        for year in range(start_year, end_year + 1)
        for kind in _KIND_ORDER
    ]


def period_end_date(kind: str, year: int) -> pd.Timestamp:
    month, day = _PERIOD_MONTH_DAY[kind]
    return pd.Timestamp(year=year, month=month, day=day)


def disclosure_window(kind: str, year: int) -> tuple[str, str]:
    """返回该报告期可能的披露日期区间（含晚披露与更正），格式 ``YYYY-MM-DD``。"""
    (start_offset, start_month, start_day), (end_offset, end_month, end_day) = _DISCLOSURE_RULE[kind]
    start = pd.Timestamp(year=year + start_offset, month=start_month, day=start_day)
    end = pd.Timestamp(year=year + end_offset, month=end_month, day=end_day)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def parse_report_period(title: Any) -> ReportPeriod | None:
    """从公告标题解析报告期；无法识别（如英文版、临时公告）时返回 None，不猜测。"""
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return None
    text = str(title)
    for kind, pattern in _TITLE_PATTERNS:
        matched = re.search(pattern, text)
        if matched:
            year = int(matched.group(1))
            return ReportPeriod(kind, year, period_end_date(kind, year))
    return None


# ---------------------------------------------------------------- 网络出口
class FundamentalDataClient:
    """财务数据抓取的网络出口：沿用行情网络的代理回退，并做统一节流。"""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.trust_env = True
        self.session.headers.update(_HEADERS)
        self.network = FinanceNetwork(self.session)
        self._last_request = 0.0

    def close(self) -> None:
        self.session.close()
        self.network.close()

    def _throttle(self) -> None:
        wait = _MINIMUM_REQUEST_INTERVAL - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get_json(self, url: str, params: dict[str, Any], timeout: int) -> Any:
        """GET 走行情网络：代理异常断开时按域名回退直连。"""
        self._throttle()
        response = self.network.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def post_json(self, url: str, data: dict[str, Any], timeout: int) -> Any:
        """POST 用于巨潮公告查询，该接口只接受表单提交。"""
        self._throttle()
        response = self.session.post(url, data=data, timeout=timeout)
        response.raise_for_status()
        return response.json()


def _request_json(request: Any, url: str, timeout: int) -> Any:
    """带一次重试的 JSON 请求；仍失败则抛出统一异常，由调用方决定降级。

    ``request`` 已绑定好具体地址与参数，``url`` 只用于错误信息。
    """
    last_error: Exception | None = None
    for _ in range(2):
        try:
            return request(timeout=timeout)
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    raise MarketDataError(f"请求失败 {url}：{type(last_error).__name__}") from last_error


# ---------------------------------------------------------------- 公告日历（巨潮）
def resolve_org_id(code: str, client: FundamentalDataClient, timeout: int) -> str | None:
    """解析巨潮内部机构 ID；查询公告时需要与证券代码成对传入。"""
    code = str(code).zfill(6)
    payload = _request_json(
        lambda **kwargs: client.post_json(_CNINFO_SEARCH_URL, {"keyWord": code, "maxNum": 10}, **kwargs),
        _CNINFO_SEARCH_URL,
        timeout,
    )
    for item in payload or []:
        if str(item.get("code", "")).zfill(6) == code and item.get("orgId"):
            return str(item["orgId"])
    return None


def _announcement_rows(payload: Any) -> list[dict[str, Any]]:
    rows = (payload or {}).get("announcements") or []
    return [row for row in rows if isinstance(row, dict) and row.get("announcementTitle")]


def _epoch_to_day(values: pd.Series) -> pd.Series:
    """巨潮的发布时间为毫秒时间戳，按北京时间取自然日，避免跨零点的时区偏移。"""
    return (
        pd.to_datetime(values, unit="ms", utc=True, errors="coerce")
        .dt.tz_convert("Asia/Shanghai")
        .dt.normalize()
        .dt.tz_localize(None)
    )


def fetch_disclosure_calendar(
    codes: Sequence[str],
    periods: Sequence[ReportPeriod],
    client: FundamentalDataClient | None = None,
    timeout: int | None = None,
    progress: Any = None,
    page_size: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """逐股逐类别拉取定期报告公告，得到 ``(代码, 报告期) → 公告日``。

    同一 ``(代码, 报告期)`` 可能有多条公告（正文、摘要、英文版、更正版），统一取最早发布时间，
    即该期财务数据首次对市场公开的时点；更正版更晚，不影响首次公开日。
    """
    client = client or FundamentalDataClient()
    timeout = int(settings.FUNDAMENTAL_REFERENCE_TIMEOUT_SECONDS if timeout is None else timeout)
    wanted = {(period.end, period.kind) for period in periods}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, code in enumerate(codes):
        code = str(code).zfill(6)
        if progress:
            progress(f"正在获取 {code} 的公告日历（{index + 1}/{len(codes)}）")
        try:
            org_id = resolve_org_id(code, client, timeout)
        except MarketDataError as exc:
            failures.append(f"{code}：机构 ID 解析失败")
            LOGGER.warning("巨潮机构 ID 解析失败 code=%s detail=%s", code, exc)
            continue
        if not org_id:
            failures.append(f"{code}：巨潮无对应机构")
            continue
        for kind in _KIND_ORDER:
            targets = [period for period in periods if period.kind == kind]
            if not targets:
                continue
            start = min(disclosure_window(kind, period.year)[0] for period in targets)
            end = max(disclosure_window(kind, period.year)[1] for period in targets)
            try:
                rows.extend(
                    _fetch_announcement_pages(
                        client,
                        timeout,
                        f"{code},{org_id}",
                        ANNOUNCEMENT_CATEGORIES[kind],
                        f"{start}~{end}",
                        code,
                        wanted,
                        page_size,
                    )
                )
            except MarketDataError as exc:
                failures.append(f"{code}：{_KIND_LABELS[kind]}公告拉取失败")
                LOGGER.warning("巨潮公告拉取失败 code=%s kind=%s detail=%s", code, kind, exc)
    frame = pd.DataFrame(rows, columns=["code", "report_period", "report_kind", "ann_date", "title"])
    report = {
        "codes_requested": len(codes),
        "codes_with_calendar": int(frame["code"].nunique()) if len(frame) else 0,
        "records": int(len(frame)),
        "failures": failures[:20],
        "failed_codes": len(failures),
        "source": "巨潮资讯 hisAnnouncement（证监会指定披露平台）",
    }
    return frame, report


def _fetch_announcement_pages(
    client: FundamentalDataClient,
    timeout: int,
    stock: str,
    category: str,
    window: str,
    code: str,
    wanted: set[tuple[pd.Timestamp, str]],
    page_size: int,
) -> Iterator[dict[str, Any]]:
    """翻页拉取单只股票某类别在窗口内的公告，只保留目标报告期的记录。"""
    page = 1
    while True:
        data = {
            "pageNum": page,
            "pageSize": page_size,
            "column": "szse",
            "tabName": "fulltext",
            "stock": stock,
            "category": category,
            "seDate": window,
            "isHLtitle": "true",
        }
        payload = _request_json(
            lambda **kwargs: client.post_json(_CNINFO_QUERY_URL, data, **kwargs), _CNINFO_QUERY_URL, timeout
        )
        rows = _announcement_rows(payload)
        if not rows:
            return
        for row in rows:
            period = parse_report_period(row.get("announcementTitle"))
            if period is None or (period.end, period.kind) not in wanted:
                continue
            yield {
                "code": code,
                "report_period": period.end,
                "report_kind": period.kind,
                "ann_date": row.get("announcementTime"),
                "title": str(row.get("announcementTitle")),
            }
        total = int((payload or {}).get("totalRecordNum") or 0)
        if page * page_size >= total:
            return
        page += 1


# ---------------------------------------------------------------- 报表数值（东财 F10 批量）
def fetch_statement_values(
    periods: Sequence[ReportPeriod],
    client: FundamentalDataClient | None = None,
    timeout: int | None = None,
    progress: Any = None,
    page_size: int = 500,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按报告期批量抓取全市场三大报表关键字段，附带报告期列。"""
    client = client or FundamentalDataClient()
    timeout = int(settings.FUNDAMENTAL_REFERENCE_TIMEOUT_SECONDS if timeout is None else timeout)
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for index, period in enumerate(periods):
        if progress:
            progress(f"正在获取 {period.end:%Y-%m-%d} 的财务报表（{index + 1}/{len(periods)}）")
        report_date = period.end.strftime("%Y-%m-%d")
        merged: pd.DataFrame | None = None
        for statement, templates in STATEMENT_FIELDS.items():
            try:
                chunk = _fetch_statement_templates(
                    client, timeout, templates,
                    [*COLUMNS, *STATEMENT_UNION_FIELDS[statement]], report_date, page_size,
                )
            except MarketDataError as exc:
                failures.append(f"{report_date} {statement}：拉取失败")
                LOGGER.warning("东财报表拉取失败 period=%s statement=%s detail=%s", report_date, statement, exc)
                merged = None
                break
            if chunk.empty:
                merged = None
                break
            merged = chunk if merged is None else merged.merge(
                chunk, on=COLUMNS, how="outer", validate="one_to_one"
            )
        if merged is None or merged.empty:
            continue
        merged = merged.copy()
        merged["report_period"] = period.end
        merged["report_kind"] = period.kind
        frames.append(merged)
    empty = pd.DataFrame(columns=[*COLUMNS, "report_period", "report_kind"])
    combined = pd.concat(frames, ignore_index=True) if frames else empty
    report = {
        "periods_requested": len(periods),
        "periods_with_values": len(frames),
        "rows": int(len(combined)),
        "failures": failures[:20],
        "source": "东方财富 F10 全量报表（RPT_F10_FINANCE_*，利润表取通用模板，资产表与现金流表取通用/银行/保险/证券四套模板）",
    }
    return combined, report


def _fetch_report_pages(
    client: FundamentalDataClient,
    timeout: int,
    report_name: str,
    columns: list[str],
    report_date: str,
    page_size: int,
) -> pd.DataFrame:
    """翻页抓取某个报告期的全市场报表。"""
    frames: list[pd.DataFrame] = []
    page = 1
    while True:
        params = {
            "sortColumns": "SECURITY_CODE",
            "sortTypes": "1",
            "pageSize": page_size,
            "pageNumber": page,
            "reportName": report_name,
            "columns": ",".join(columns),
            "filter": f"(REPORT_DATE='{report_date}')",
        }
        payload = _request_json(
            lambda **kwargs: client.get_json(_EASTMONEY_URL, params, **kwargs), _EASTMONEY_URL, timeout
        )
        if not (payload or {}).get("success"):
            raise MarketDataError(f"东财接口返回失败 report={report_name} date={report_date}")
        result = (payload or {}).get("result") or {}
        rows = result.get("data") or []
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        total = int(result.get("count") or 0)
        if page * page_size >= total:
            break
        page += 1
    if not frames:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True)
    keep = [column for column in columns if column in combined.columns]
    combined = combined.loc[:, keep]
    # 东财对少数北交所代码会返回字段完全相同的重复行，按 (代码, 报告期) 去重保留首行。
    if combined.duplicated(COLUMNS).any():
        LOGGER.warning("报表存在重复行，已去重 report=%s date=%s", report_name, report_date)
        combined = combined.drop_duplicates(COLUMNS, keep="first")
    return combined


def _fetch_statement_templates(
    client: FundamentalDataClient,
    timeout: int,
    templates: Sequence[tuple[str, Sequence[str]]],
    columns: list[str],
    report_date: str,
    page_size: int,
) -> pd.DataFrame:
    """抓取一张报表的全部行业模板并纵向合并。

    任一模板失败都向上抛出、由调用方整期跳过：模板之间互不重叠，少取一套等于静默丢掉一整个板块，
    而这恰恰是要拦的问题——覆盖率门控看不出板块级缺失，只会照常认为覆盖充分。
    """
    frames: list[pd.DataFrame] = []
    for name, fields in templates:
        chunk = _fetch_report_pages(
            client, timeout, f"RPT_F10_FINANCE_{name}", [*COLUMNS, *fields], report_date, page_size
        )
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).drop_duplicates(COLUMNS, keep="first")


# ---------------------------------------------------------------- 合并与落盘
def merge_fundamental_panels(calendar: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """按 ``(代码, 报告期)`` 内连接公告日与报表数值，并统一类型、去重。"""
    if calendar is None or calendar.empty or values is None or values.empty:
        return pd.DataFrame()
    left = calendar.copy()
    left["code"] = left["code"].astype(str).str.zfill(6)
    left["report_period"] = pd.to_datetime(left["report_period"])
    left["ann_date"] = _epoch_to_day(left["ann_date"])
    left = left.dropna(subset=["ann_date"])
    # 同一报告期多条公告取最早发布时间：这才是数据首次公开的时点。
    left = left.sort_values("ann_date").drop_duplicates(["code", "report_period"], keep="first")

    right = values.copy()
    # 报表侧的代码列沿用东财原始字段名，与公告日历侧的 code 对齐后才能按 (代码, 报告期) 内连接。
    right = right.rename(columns={"SECURITY_CODE": "code"})
    if "code" not in right:
        return pd.DataFrame()
    right["code"] = right["code"].astype(str).str.zfill(6)
    right["report_period"] = pd.to_datetime(right["report_period"])
    right = right.drop(columns=["REPORT_DATE", "report_kind"], errors="ignore")
    right = right.drop_duplicates(["code", "report_period"], keep="last")

    merged = left.merge(right, on=["code", "report_period"], how="inner", validate="one_to_one")
    for column in VALUE_COLUMNS:
        if column not in merged:
            merged[column] = pd.NA
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged.sort_values(["code", "report_period"]).reset_index(drop=True)


def build_fundamental_panel(
    store: PanelStore,
    codes: Sequence[str],
    periods: Sequence[ReportPeriod],
    progress: Any = None,
    refresh: bool = False,
    client: FundamentalDataClient | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """构建并落盘 PIT 财务面板，返回长表与覆盖报告。

    已存在分区且未要求刷新的股票会被跳过，重复运行只补增量，避免每次都全量重拉。
    """
    codes = [str(code).zfill(6) for code in codes]
    pending = codes if refresh else [code for code in codes if store.read_partition(PARTITION_NAME, code) is None]
    if not pending:
        existing = load_fundamental_facts(store, codes)
        return (existing if existing is not None else pd.DataFrame()), {
            "status": f"本地已有 {len(codes)} 只股票的财务面板，未重新拉取",
            "codes_requested": len(codes),
            "codes_built": 0,
            "periods": len(periods),
        }
    if progress:
        progress(f"正在获取 {len(pending)} 只股票的公告日历与报表数值")
    client = client or FundamentalDataClient()
    calendar, calendar_report = fetch_disclosure_calendar(pending, periods, client=client, progress=progress)
    values, values_report = fetch_statement_values(periods, client=client, progress=progress)
    merged = merge_fundamental_panels(calendar, values)
    written: list[str] = []
    for code, group in merged.groupby("code", sort=True):
        store.write_partition(PARTITION_NAME, code, group.reset_index(drop=True))
        written.append(code)
    combined = load_fundamental_facts(store, codes)
    if combined is None:
        combined = pd.DataFrame()
    report = {
        "codes_requested": len(codes),
        "codes_built": len(written),
        "periods": len(periods),
        "rows": int(len(combined)),
        "calendar": calendar_report,
        "values": values_report,
        "announcement_lag_days": announcement_lag_summary(merged),
        "status": (
            f"PIT 财务面板已落盘 {len(written)} 只股票、{len(periods)} 个报告期"
            if written
            else "未取得可用的公告日与报表数值配对，未生成财务面板"
        ),
        "limitation": (
            "公告日取巨潮资讯首次披露时间；东财 NOTICE_DATE 在历史区间存在系统性错位，故未采用。"
            "仅有同时取得公告日与报表数值的 (代码, 报告期) 会被保留，缺一即视为不可用。"
        ),
    }
    return combined, report


def announcement_lag_summary(merged: pd.DataFrame) -> dict[str, Any]:
    """报告期结束日到公告日之间的天数分布，用于核对 PIT 时点是否合理。"""
    if merged is None or merged.empty:
        return {}
    lag = (pd.to_datetime(merged["ann_date"]) - pd.to_datetime(merged["report_period"])).dt.days
    return {
        "median_days": float(lag.median()),
        "minimum_days": float(lag.min()),
        "maximum_days": float(lag.max()),
    }


def load_fundamental_facts(store: PanelStore, codes: Sequence[str] | None = None) -> pd.DataFrame | None:
    """读取本地 PIT 财务长表；指定 codes 时只读这些股票的分区。"""
    if codes is None:
        return store.load_partition(PARTITION_NAME)
    frames = []
    for code in codes:
        frame = store.read_partition(PARTITION_NAME, str(code).zfill(6))
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- 点时对齐
def align_asof_pit(
    store: PanelStore | None,
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Sequence[str],
    asof_column: str = "available_from",
    left_date: str = "date",
) -> pd.DataFrame:
    """把财务事实按「生效日严格早于交易日」对齐到每个交易日截面。

    采用严格小于（``allow_exact_matches=False`` / ``>``）：公告日当天不生效，次一交易日起才计入，
    避免把当日盘后或盘中才披露的数据提前用在当日截面上。
    """
    if left is None or left.empty or right is None or right.empty:
        result = (left if left is not None else pd.DataFrame()).copy()
        for column in columns:
            if column not in result:
                result[column] = pd.NA
        return result
    keys = left[["date", "code"]].copy()
    keys["_row"] = range(len(keys))
    keys["date"] = pd.to_datetime(keys["date"])
    keys["code"] = keys["code"].astype(str).str.zfill(6)
    # 调用方可能要求把生效日本身也带回来（用于追溯用了哪一期），此时去重避免选出同名列。
    selected = [asof_column, *[column for column in columns if column != asof_column]]
    facts = right[["code", *selected]].copy()
    facts["code"] = facts["code"].astype(str).str.zfill(6)
    facts[asof_column] = pd.to_datetime(facts[asof_column])
    facts = facts.dropna(subset=[asof_column]).sort_values([asof_column, "code"])
    if store is not None and store.duckdb_available():
        try:
            aligned = _duckdb_asof(store, keys, facts, columns, asof_column)
        except Exception as exc:  # ASOF JOIN 失败不应中断整条链路，退回等价 pandas 实现
            LOGGER.warning("DuckDB ASOF JOIN 失败，改用 pandas 等价实现 detail=%s", exc)
            aligned = _pandas_asof(keys, facts, columns, asof_column)
    else:
        aligned = _pandas_asof(keys, facts, columns, asof_column)
    result = left.copy()
    for column in columns:
        result[column] = aligned[column].to_numpy()
    return result


def _duckdb_asof(
    store: PanelStore,
    keys: pd.DataFrame,
    facts: pd.DataFrame,
    columns: Sequence[str],
    asof_column: str,
) -> pd.DataFrame:
    quoted = ", ".join(f'f."{column}"' for column in columns)
    sql = (
        f"SELECT k._row, {quoted} FROM keys k "
        f'ASOF JOIN (SELECT * FROM facts ORDER BY code, "{asof_column}") f '
        f'ON k.code = f.code AND k.date > f."{asof_column}"'
    )
    aligned = store.query(sql, extra_views={"keys": keys, "facts": facts})
    return aligned.set_index("_row").reindex(keys["_row"]).loc[:, list(columns)].reset_index(drop=True)


def _pandas_asof(
    keys: pd.DataFrame,
    facts: pd.DataFrame,
    columns: Sequence[str],
    asof_column: str,
) -> pd.DataFrame:
    """DuckDB 缺失时的等价实现：``direction="backward"`` 且不允许精确匹配。"""
    aligned = pd.merge_asof(
        keys.sort_values("date"),
        facts.rename(columns={asof_column: "_asof"}).sort_values("_asof"),
        left_on="date",
        right_on="_asof",
        by="code",
        direction="backward",
        allow_exact_matches=False,
    )
    # 生效日被改名成临时列用于 ASOF，如果调用方要它，就按原列名还回去。
    if asof_column in columns:
        aligned[asof_column] = aligned["_asof"]
    return aligned.set_index("_row").reindex(keys["_row"]).loc[:, list(columns)].reset_index(drop=True)


def fundamental_panel_summary(store: PanelStore) -> dict[str, Any]:
    """本地财务面板的规模概览，供页面展示覆盖情况。"""
    codes = store.partition_codes(PARTITION_NAME)
    facts = load_fundamental_facts(store)
    if facts is None or facts.empty:
        return {"codes": len(codes), "rows": 0, "status": "尚未构建 PIT 财务面板"}
    return {
        "codes": len(codes),
        "rows": int(len(facts)),
        "first_period": str(pd.to_datetime(facts["report_period"]).min().date()),
        "last_period": str(pd.to_datetime(facts["report_period"]).max().date()),
        "status": f"本地 PIT 财务面板覆盖 {len(codes)} 只股票、{len(facts)} 条报告期记录",
    }
