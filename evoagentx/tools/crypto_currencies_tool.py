import requests
from typing import Optional
from .api_converter import APITool, APIToolkit


def create_crypto_symbols_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get a list of cryptocurrency pair symbols
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/crypto-symbols"
        params = {}
        
        # Add API key to parameters
        if api_key:
            params["key"] = api_key
        
        if "offset" in kwargs:
            params["offset"] = kwargs["offset"]
        if "format" in kwargs:
            params["format"] = kwargs["format"]
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response
    
    return APITool(
        name="get_crypto_symbols",
        description="Get a list of cryptocurrency pair symbols and related information. Cryptocurrency is a digital currency that is secured through cryptography and exists on decentralised networks utilising blockchain technology. There is a limit of 500 records per API call.",
        inputs={
            "offset": {
                "type": "integer",
                "description": "Optional. The initial position of the record subset, which indicates how many records to skip. Defaults to 0.",
                "example": 500
            },
            "format": {
                "type": "string",
                "description": "Optional. The format of the returned data, either JSON or CSV. Defaults to JSON.",
                "example": "json",
                "enum": ["json", "csv"]
            }
        },
        required=[],
        endpoint_config={
            "url": "https://financialdata.net/api/v1/crypto-symbols",
            "method": "GET"
        },
        function=execute_api
    )


def create_crypto_information_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to retrieve basic information about a cryptocurrency
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/crypto-information"
        params = {"identifier": kwargs["identifier"]}
        
        # Add API key to parameters
        if api_key:
            params["key"] = api_key
        
        if "format" in kwargs:
            params["format"] = kwargs["format"]
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response
    
    return APITool(
        name="get_crypto_information",
        description="Retrieve basic information about the cryptocurrency, such as its market cap, total supply, ledger start date, and various other key facts. The API endpoint provides basic information for major cryptocurrencies.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The symbol (code) for a cryptocurrency.",
                "example": "BTC"
            },
            "format": {
                "type": "string",
                "description": "Optional. The format of the returned data, either JSON or CSV. Defaults to JSON.",
                "example": "json",
                "enum": ["json", "csv"]
            }
        },
        required=["identifier"],
        endpoint_config={
            "url": "https://financialdata.net/api/v1/crypto-information",
            "method": "GET"
        },
        function=execute_api
    )


def create_crypto_prices_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get daily historical cryptocurrency prices and volumes
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/crypto-prices"
        params = {"identifier": kwargs["identifier"]}
        
        # Add API key to parameters
        if api_key:
            params["key"] = api_key
        
        if "offset" in kwargs:
            params["offset"] = kwargs["offset"]
        if "format" in kwargs:
            params["format"] = kwargs["format"]
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response
    
    return APITool(
        name="get_crypto_prices",
        description="Retrieve daily historical cryptocurrency prices and trading volumes. The data covers major cryptocurrency pairs. There is a limit of 300 records per API call.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for the cryptocurrency pair.",
                "example": "BTCUSD"
            },
            "offset": {
                "type": "integer",
                "description": "Optional. The initial position of the record subset, which indicates how many records to skip. Defaults to 0.",
                "example": 300
            },
            "format": {
                "type": "string",
                "description": "Optional. The format of the returned data, either JSON or CSV. Defaults to JSON.",
                "example": "json",
                "enum": ["json", "csv"]
            }
        },
        required=["identifier"],
        endpoint_config={
            "url": "https://financialdata.net/api/v1/crypto-prices",
            "method": "GET"
        },
        function=execute_api
    )


def create_crypto_minute_prices_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get one-minute historical cryptocurrency prices and volumes
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/crypto-minute-prices"
        params = {
            "identifier": kwargs["identifier"],
            "date": kwargs["date"]
        }
        
        # Add API key to parameters
        if api_key:
            params["key"] = api_key
        
        if "offset" in kwargs:
            params["offset"] = kwargs["offset"]
        if "format" in kwargs:
            params["format"] = kwargs["format"]
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response
    
    return APITool(
        name="get_crypto_minute_prices",
        description="Retrieve one-minute historical cryptocurrency prices and volumes. The data covers major cryptocurrency pairs. There is a limit of 300 records per API call.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for the cryptocurrency pair.",
                "example": "BTCUSD"
            },
            "date": {
                "type": "string",
                "description": "The date in YYYY-MM-DD format.",
                "example": "2025-01-15"
            },
            "offset": {
                "type": "integer",
                "description": "Optional. The initial position of the record subset, which indicates how many records to skip. Defaults to 0.",
                "example": 300
            },
            "format": {
                "type": "string",
                "description": "Optional. The format of the returned data, either JSON or CSV. Defaults to JSON.",
                "example": "json",
                "enum": ["json", "csv"]
            }
        },
        required=["identifier", "date"],
        endpoint_config={
            "url": "https://financialdata.net/api/v1/crypto-minute-prices",
            "method": "GET"
        },
        function=execute_api
    )


def create_crypto_toolkit(api_key: Optional[str] = None) -> APIToolkit:
    """
    Create a Cryptocurrency Data Toolkit instance
    
    Provides comprehensive access to cryptocurrency market data, including:
    - Cryptocurrency pair symbols
    - Cryptocurrency information (market cap, supply, etc.)
    - Historical price data (daily and minute-level)
    - Trading volumes
    
    Args:
        api_key: Optional API key for authentication. When making requests, the API key 
                will be appended as ?key=YOUR_API_KEY or &key=YOUR_API_KEY depending on 
                whether other query parameters exist.
    
    Returns:
        APIToolkit instance
    
    Example:
        >>> # Create toolkit with API key
        >>> toolkit = create_crypto_toolkit(api_key="your_api_key_here")
        >>> 
        >>> # Get a specific tool
        >>> crypto_symbols = toolkit.get_tool("get_crypto_symbols")
        >>> result = crypto_symbols(offset=0, format="json")
        >>> 
        >>> # Get crypto information
        >>> crypto_info = toolkit.get_tool("get_crypto_information")
        >>> info = crypto_info(identifier="BTC")
        >>> 
        >>> # Get crypto prices
        >>> crypto_prices = toolkit.get_tool("get_crypto_prices")
        >>> prices = crypto_prices(identifier="BTCUSD", offset=0)
    """
    tools = [
        create_crypto_symbols_tool(api_key=api_key),
        create_crypto_information_tool(api_key=api_key),
        create_crypto_prices_tool(api_key=api_key),
        create_crypto_minute_prices_tool(api_key=api_key)
    ]
    
    auth_config = {}
    if api_key:
        auth_config = {"api_key": api_key}
    
    return APIToolkit(
        name="crypto_data_toolkit",
        tools=tools,
        base_url="https://financialdata.net/api/v1",
        auth_config=auth_config,
        common_headers={
            "Content-Type": "application/json"
        }
    )

