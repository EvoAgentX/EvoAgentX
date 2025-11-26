#!/usr/bin/env python3

"""
Condensed API Converter Examples (Safe, No API Keys)

This module provides:
- A quick, low-cost smoke test that builds and inspects a toolkit (no network calls)
- A single real call extracted from real_world_api_example.py (executes only if OPENWEATHER_API_KEY is set)

It mirrors the structure and style of other example modules in examples/tools.
"""

from typing import Dict, Any
import os
import json
import time
from dotenv import load_dotenv

from evoagentx.tools.api_converter import (
    create_openapi_toolkit,
)

load_dotenv()


financial_data_api_spec = json.load(open("evoagentx/tools/api_pool/Financial Data API.json"))


def openapi_test() -> None:
    print("\n===== SINGLE REAL CALL: OpenWeatherMap (extracted) =====\n")

    api_key = os.getenv("FINANCIAL_DATA_API_KEY")
    if not api_key or api_key.strip().lower() in {"", "your-api-key"}:
        print("Skipping real call: set FINANCIAL_DATA_API_KEY to run this test.")
        return

    toolkit = create_openapi_toolkit(
        schema_path_or_dict=financial_data_api_spec,
        service_name="Financial Data API"
    )

    # Inject default appid for all tools (pattern extracted from real_world_api_example.py)
    print("____________ Executing financial data querying ____________")
    # for financial_data_tool in toolkit.get_tools():
    #     example_query = {"key": api_key, "format": "csv"}
    #     print("Tool: ", financial_data_tool.name)
    #     print("Query inputs: \n", example_query)
    #     result = financial_data_tool(**example_query)
    #     print("Result: \n", result)
    #     print("-" * 100)
    #     time.sleep(2)

    financial_data_tool = toolkit.get_tool("getMinutePrices")
    example_query = {"identifier": "MSFT", "date": "2020-01-15", "key": api_key, "format": "csv"}
    # example_query = {"identifier": "MSFT", "key": api_key, "format": "csv"}
    print("Tool: ", financial_data_tool.name)
    print("Query inputs: \n", example_query)
    result = financial_data_tool(**example_query)
    print("Result: \n", result)
    print("-" * 100)
        
    

def main() -> None:
    """Main function to run condensed converter examples"""
    print("===== API CONVERTER EXAMPLES (CONDENSED) =====")
    # Single extracted real-world call (requires OPENWEATHER_API_KEY)
    openapi_test()
    print("\n===== ALL CONDENSED CONVERTER TESTS COMPLETED =====")


if __name__ == "__main__":
    main()