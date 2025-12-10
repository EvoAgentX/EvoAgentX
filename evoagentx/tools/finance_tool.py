import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

from .tool import Tool, Toolkit
from ..core.logging import logger


class FinanceTool(Tool):
    name: str = "finance_tool"
    description: str = f"Retrieve financial price data for stocks, indexes, and crypto. Current date: {datetime.now().strftime('%Y-%m-%d')}. The tool automatically corrects future dates and wrong years."
    inputs: Dict[str, Dict[str, Any]] = {
        "trading_symbol": {
            "type": "string",
            "description": "Preferred stock symbol or pair identifier (e.g., MSFT, ^GSPC, BTCUSD); company name are also allowed"
        },
        "unit": {
            "type": "string",
            "description": "Data granularity (minute limited to single-day window)",
            "enum": ["daily", "minute"]
        },
        "start": {
            "type": "string",
            "description": f"Start time/date (daily: YYYY-MM-DD; minute: YYYY-MM-DDTHH:MM:SSZ). Today is {datetime.now().strftime('%Y-%m-%d')}. Use recent dates for relative queries."
        },
        "end": {
            "type": "string",
            "description": f"End time/date (daily: YYYY-MM-DD; minute: YYYY-MM-DDTHH:MM:SSZ). Today is {datetime.now().strftime('%Y-%m-%d')}. Use recent dates for relative queries."
        },
    }
    required: Optional[List[str]] = ["trading_symbol", "unit", "start", "end"]

    def __init__(self, financial_data_api_key: Optional[str] = None, openrouter_api_key: Optional[str] = None):
        super().__init__()
        self.financial_data_api_key = financial_data_api_key or os.getenv("FINANCIAL_DATA_KEY")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.stock_symbols = self._load_stock_symbols()
        self.openrouter_client = self._init_openrouter_client() if self.openrouter_api_key else None

    def __call__(
        self,
        trading_symbol: str,
        unit: str,
        start: str,
        end: str,
        format: str = "json",
    ) -> Dict[str, Any]:

        if format not in {"json", "csv"}:
            return {"error": "invalid format"}

        if unit not in {"daily", "minute"}:
            return {"error": "invalid unit"}

        if not self.financial_data_api_key:
            return {"error": "FINANCIAL_DATA_KEY not provided"}

        # 验证 trading_symbol 是否在 stock_symbols.json 中
        validated_symbol = self._validate_stock_symbol(trading_symbol)
        if validated_symbol.get("error"):
            return validated_symbol
        
        # 使用验证后的 trading_symbol
        trading_symbol = validated_symbol.get("trading_symbol", trading_symbol)

        start, end = self._normalize_input_dates(unit, start, end)
        
        try:
            if unit == "daily":
                start_dt = datetime.strptime(start, "%Y-%m-%d")
                end_dt = datetime.strptime(end, "%Y-%m-%d")
            else:
                start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ")
                end_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return {"error": "invalid date format"}

        if unit == "minute":
            if (start_dt.year, start_dt.month, start_dt.day) != (end_dt.year, end_dt.month, end_dt.day):
                return {"error": "minute data must be within a single calendar day"}

        endpoint, supports_minute = self._resolve_endpoint(trading_symbol, unit)
        if endpoint is None:
            return {"error": "unsupported combination"}

        if unit == "daily":
            result = self._fetch_daily(
                endpoint=endpoint,
                symbol=trading_symbol,
                start_dt=start_dt,
                end_dt=end_dt,
                fmt=format,
            )
            return result
        else:
            if not supports_minute:
                return {"error": "minute data not supported for this symbol, please use daily data"}
            result = self._fetch_minute_range(
                endpoint=endpoint,
                symbol=trading_symbol,
                start_dt=start_dt,
                end_dt=end_dt,
                fmt=format,
            )
            return result

    def _resolve_endpoint(self, symbol: str, unit: str) -> (Optional[str], bool):
        base = "https://financialdata.net/api/v1"
        if symbol.startswith("^"):
            if unit == "daily":
                return f"{base}/index-prices", False
            else:
                return None, False
        if unit == "daily":
            return f"{base}/stock-prices", False
        else:
            return f"{base}/minute-prices", True

    def _fetch_daily(
        self,
        endpoint: str,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        fmt: str,
    ) -> Dict[str, Any]:
        params = {"identifier": symbol, "format": fmt, "key": self.financial_data_api_key}
        if fmt == "json":
            items: List[Dict[str, Any]] = []
            offset = 0
            step = 300
            while True:
                page_params = dict(params)
                page_params["offset"] = offset
                try:
                    resp = self._http_get(endpoint, page_params, timeout=30)
                    data = resp.json()
                    if not isinstance(data, list):
                        return {"error": "unexpected response"}
                except Exception as e:
                    logger.error(f"daily fetch error: {e}")
                    return {"error": str(e)}
                items.extend(data)
                if len(data) < step:
                    break
                offset += step
            filtered = self._filter_json(items, unit="daily", start_dt=start_dt, end_dt=end_dt)
            return {"data": filtered}
        else:
            lines: List[str] = []
            header: Optional[str] = None
            offset = 0
            step = 300
            while True:
                page_params = dict(params)
                page_params["offset"] = offset
                try:
                    resp = self._http_get(endpoint, page_params, timeout=30)
                    text = resp.text
                except Exception as e:
                    logger.error(f"daily fetch error: {e}")
                    return {"error": str(e)}
                chunk_lines = text.splitlines()
                if not chunk_lines:
                    break
                if header is None and chunk_lines:
                    header = chunk_lines[0]
                body = chunk_lines[1:] if chunk_lines and chunk_lines[0] == header else chunk_lines
                lines.extend(body)
                if len(chunk_lines) < step:
                    break
                offset += step
            if header is None:
                return {"data": ""}
            filtered_lines = self._filter_csv(header, lines, unit="daily", start_dt=start_dt, end_dt=end_dt)
            csv_text = "\n".join([header] + filtered_lines)
            return {"data": csv_text}

    def _fetch_minute_range(
        self,
        endpoint: str,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        fmt: str,
    ) -> Dict[str, Any]:
        day_start = datetime(start_dt.year, start_dt.month, start_dt.day)
        day_end = datetime(end_dt.year, end_dt.month, end_dt.day)
        days: List[datetime] = []
        cur = day_start
        while cur <= day_end:
            days.append(cur)
            cur += timedelta(days=1)

        if fmt == "json":
            all_items: List[Dict[str, Any]] = []
            for d in days:
                params = {
                    "identifier": symbol,
                    "date": d.strftime("%Y-%m-%d"),
                    "format": "json",
                    "key": self.financial_data_api_key,
                }
                offset = 0
                step = 300
                while True:
                    page_params = dict(params)
                    page_params["offset"] = offset
                    try:
                        resp = self._http_get(endpoint, page_params, timeout=30)
                        data = resp.json()
                        if not isinstance(data, list):
                            return {"error": "unexpected response"}
                    except Exception as e:
                        logger.error(f"minute fetch error: {e}")
                        return {"error": str(e)}
                    all_items.extend(data)
                    if len(data) < step:
                        break
                    offset += step
            filtered = self._filter_json(all_items, unit="minute", start_dt=start_dt, end_dt=end_dt)
            return {"data": filtered}
        else:
            header: Optional[str] = None
            out_lines: List[str] = []
            for d in days:
                params = {
                    "identifier": symbol,
                    "date": d.strftime("%Y-%m-%d"),
                    "format": "csv",
                    "key": self.financial_data_api_key,
                }
                offset = 0
                step = 300
                while True:
                    page_params = dict(params)
                    page_params["offset"] = offset
                    try:
                        resp = self._http_get(endpoint, page_params, timeout=30)
                        text = resp.text
                    except Exception as e:
                        logger.error(f"minute fetch error: {e}")
                        return {"error": str(e)}
                    chunk_lines = text.splitlines()
                    if not chunk_lines:
                        break
                    if header is None and chunk_lines:
                        header = chunk_lines[0]
                    body = chunk_lines[1:] if chunk_lines and chunk_lines[0] == header else chunk_lines
                    out_lines.extend(body)
                    if len(chunk_lines) < step:
                        break
                    offset += step
            if header is None:
                return {"data": ""}
            filtered_lines = self._filter_csv(header, out_lines, unit="minute", start_dt=start_dt, end_dt=end_dt)
            csv_text = "\n".join([header] + filtered_lines)
            return {"data": csv_text}

    def _filter_json(
        self,
        items: List[Dict[str, Any]],
        unit: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[Dict[str, Any]]:
        key_candidates = ["datetime", "date", "timestamp", "time", "date_time"]
        out: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            dt_val = None
            for k in key_candidates:
                if k in it and isinstance(it[k], str):
                    dt_val = it[k]
                    break
            if dt_val is None:
                continue
            try:
                if unit == "daily":
                    dt = datetime.strptime(dt_val[:10], "%Y-%m-%d")
                else:
                    dt = self._parse_iso(dt_val)
            except Exception:
                continue
            if start_dt <= dt <= end_dt:
                out.append(it)
        return out

    def _filter_csv(
        self,
        header: str,
        lines: List[str],
        unit: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[str]:
        cols = [c.strip() for c in header.split(",")]
        idx = -1
        for i, c in enumerate(cols):
            lc = c.lower()
            if lc in {"datetime", "date", "timestamp", "time", "date_time"}:
                idx = i
                break
        if idx < 0:
            return lines
        out: List[str] = []
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            if idx >= len(parts):
                continue
            val = parts[idx]
            try:
                if unit == "daily":
                    dt = datetime.strptime(val[:10], "%Y-%m-%d")
                else:
                    dt = self._parse_iso(val)
            except Exception:
                continue
            if start_dt <= dt <= end_dt:
                out.append(ln)
        return out

    def _parse_iso(self, s: str) -> datetime:
        try:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            try:
                return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")

    def _normalize_input_dates(self, unit: str, start: str, end: str) -> (str, str):
        if unit == "daily":
            start = start[:10] if len(start) >= 10 else start
            end = end[:10] if len(end) >= 10 else end
            return start, end
        def is_date_only(s: str) -> bool:
            return len(s) >= 10 and s[4] == "-" and s[7] == "-" and (len(s) == 10 or s[10] not in ["T", " "])
        def to_iso(s: str) -> str:
            if "T" in s and s.endswith("Z"):
                return s
            if "T" in s and not s.endswith("Z"):
                return s + "Z"
            if " " in s:
                return s.replace(" ", "T") + "Z"
            return s
        # 对于 minute 数据，如果输入是纯日期格式，添加默认时间
        if is_date_only(start) and is_date_only(end):
            # 提取日期部分并添加时间
            start = start[:10] + "T09:30:00Z"
            end = end[:10] + "T16:00:00Z"
            return start, end
        return to_iso(start), to_iso(end)

    def _load_stock_symbols(self) -> Dict[str, str]:
        """
        加载 stock_symbols.json 文件,创建 trading_symbol -> registrant_name 的映射
        返回字典: {trading_symbol: registrant_name}
        """
        try:
            # 获取当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "financial_materials", "stock_symbols.json")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                symbols_data = json.load(f)
            
            # 创建 trading_symbol -> registrant_name 的映射
            symbols_dict = {}
            for item in symbols_data:
                if isinstance(item, dict) and "trading_symbol" in item:
                    symbols_dict[item["trading_symbol"].upper()] = item.get("registrant_name", "")
            
            logger.info(f"成功加载 {len(symbols_dict)} 个股票代码")
            return symbols_dict
        except Exception as e:
            logger.error(f"加载 stock_symbols.json 失败: {e}")
            return {}

    def _init_openrouter_client(self) -> Optional[OpenAI]:
        """Initialize OpenRouter client"""
        try:
            client = OpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info("OpenRouter client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"OpenRouter client initialization failed: {e}")
            return None

    def _extract_symbol_with_llm(self, user_query: str, available_symbols: List[str]) -> Optional[str]:
        """
        Use OpenRouter's GPT-4o-mini to extract or convert trading_symbol from user query
        
        Args:
            user_query: User input query (maybe stock code or company name)
            available_symbols: Available stock code list (for context)
        
        Returns:
            Extracted stock code, if not extracted, return None
        """
        if not self.openrouter_client:
            logger.warning("OpenRouter client not initialized, cannot use LLM to extract stock code")
            return None
        
        try:
            # Build prompt
            prompt = f"""You are a stock code recognition expert. The user query may contain stock code or company name.

User query: "{user_query}"

Task:
1. If the user input is a standard stock code (e.g., AAPL, MSFT, TSLA), return the code directly
2. If the user input is a company name or description (e.g., "Apple", "Microsoft"), try to identify the corresponding stock code
3. If it cannot be determined, return "UNKNOWN"

Requirements:
- Only return the stock code itself, do not return any explanation
- The stock code should be all uppercase
- If it is a company name, return the main trading code in the US stock market

Please return the stock code directly (e.g., AAPL) or UNKNOWN:"""

            # Call OpenRouter API
            response = self.openrouter_client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional stock code recognition assistant, only return the stock code or UNKNOWN, do not return any other content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            extracted_symbol = response.choices[0].message.content.strip().upper()
            
            # Clean up the returned result (remove possible punctuation)
            extracted_symbol = extracted_symbol.replace(".", "").replace(",", "").strip()
            
            if extracted_symbol == "UNKNOWN":
                logger.info(f"LLM cannot extract stock code from query '{user_query}'")
                return None
            
            logger.info(f"LLM extracted stock code from query '{user_query}': {extracted_symbol}")
            return extracted_symbol
            
        except Exception as e:
            logger.error(f"Error extracting stock code using LLM: {e}")
            return None

    def _fuzzy_match_company_name(self, query: str) -> Optional[str]:
        """
        Fuzzy match in company name
        
        Args:
            query: User input query
        
        Returns:
            Matched stock code, if not matched, return None
        """
        query_lower = query.lower()
        
        # Try exact match
        for symbol, company_name in self.stock_symbols.items():
            if company_name and query_lower == company_name.lower():
                logger.info(f"Exact match to company name: {symbol} - {company_name}")
                return symbol
        
        # Try include match
        for symbol, company_name in self.stock_symbols.items():
            if company_name and query_lower in company_name.lower():
                logger.info(f"Fuzzy match to company name: {symbol} - {company_name}")
                return symbol
        
        # Try reverse include match (query contains company name)
        for symbol, company_name in self.stock_symbols.items():
            if company_name and company_name.lower() in query_lower:
                logger.info(f"Reverse match to company name: {symbol} - {company_name}")
                return symbol
        
        return None

    def _validate_stock_symbol(self, trading_symbol: str) -> Dict[str, Any]:
        """
        Validate if trading_symbol is in stock_symbols.json
        
        Logic:
        1. If the user directly inputs stock_symbol (all uppercase), check if it exists in the json
        2. If not in json, use OpenRouter GPT-4o-mini to extract/convert stock code
        3. If it is a company name, perform fuzzy matching
        4. If all methods fail, return error information
        
        返回:
        - Success: {"trading_symbol": "validated code"}
        - Failure: {"error": "error information"}
        """
        if not self.stock_symbols:
            logger.warning("stock_symbols not loaded, skip validation")
            return {"trading_symbol": trading_symbol}
        
        # Convert input to uppercase for comparison
        symbol_upper = trading_symbol.upper()
        
        # 1. Directly find trading_symbol
        if symbol_upper in self.stock_symbols:
            company_name = self.stock_symbols[symbol_upper]
            logger.info(f"✓ Found valid stock code: {symbol_upper} - {company_name}")
            return {"trading_symbol": symbol_upper}
        
        logger.info(f"Stock code '{trading_symbol}' not in database, try using LLM to extract...")
        
        # 2. Use OpenRouter LLM to extract stock code
        extracted_symbol = self._extract_symbol_with_llm(
            user_query=trading_symbol,
            available_symbols=list(self.stock_symbols.keys())[:100]  # Provide some examples
        )
        
        if extracted_symbol and extracted_symbol in self.stock_symbols:
            company_name = self.stock_symbols[extracted_symbol]
            logger.info(f"✓ LLM extracted valid stock code: {extracted_symbol} - {company_name}")
            return {"trading_symbol": extracted_symbol}
        
        # 3. Try fuzzy matching company name
        logger.info(f"LLM extraction failed, try fuzzy matching company name...")
        matched_symbol = self._fuzzy_match_company_name(trading_symbol)
        
        if matched_symbol:
            company_name = self.stock_symbols[matched_symbol]
            logger.info(f"✓ Fuzzy matching successful: {matched_symbol} - {company_name}")
            return {"trading_symbol": matched_symbol}
        
        # 4. All methods failed, return error information
        error_msg = (
            f"❌ Stock code '{trading_symbol}' not found.\n"
            f"Tried:\n"
            f"  1. Direct match stock code\n"
            f"  2. Use AI model extraction{' (extracted: ' + extracted_symbol + ')' if extracted_symbol else ''}\n"
            f"  3. Fuzzy matching company name\n"
            f"Please confirm if the input stock code or company name is correct.\n"
            f"Currently {len(self.stock_symbols)} stock codes are supported."
        )
        logger.error(error_msg)
        return {"error": error_msg}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4), retry=retry_if_exception_type(Exception), reraise=True)
    def _http_get(self, url: str, params: Dict[str, Any], timeout: int = 30):
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp


class FinanceToolkit(Toolkit):
    def __init__(self, name: str = "FinanceToolkit", financial_data_api_key: Optional[str] = None, openrouter_api_key: Optional[str] = None):
        tool = FinanceTool(financial_data_api_key=financial_data_api_key, openrouter_api_key=openrouter_api_key)
        super().__init__(name=name, tools=[tool])
