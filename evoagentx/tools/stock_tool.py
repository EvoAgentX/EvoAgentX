import requests
from typing import Optional
from .api_converter import APITool, APIToolkit


def create_stock_symbols_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get a list of stock symbols for publicly traded US and international companies
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/stock-symbols"
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
        name="get_stock_symbols",
        description="Get a list of stock symbols for publicly traded US and international companies. The list contains thousands of trading symbols as well as the names of the companies. There is a limit of 500 records per API call.",
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
            "url": "https://financialdata.net/api/v1/stock-symbols",
            "method": "GET"
        },
        function=execute_api
    )


def create_international_stock_symbols_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to retrieve a list of stock symbols for publicly traded international companies
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/international-stock-symbols"
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
        name="get_international_stock_symbols",
        description="Retrieve a list of stock symbols for publicly traded international companies. Data is available for Toronto, London, Frankfurt, Euronext Paris, Euronext Amsterdam, Tokyo, Hong Kong, Singapore, Indonesia, Malaysia, Korea, Brazil, Mexico, India, Bombay stock exchanges. There is a limit of 500 records per API call.",
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
            "url": "https://financialdata.net/api/v1/international-stock-symbols",
            "method": "GET"
        },
        function=execute_api
    )


def create_stock_prices_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get daily historical stock prices and volumes
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/stock-prices"
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
        name="get_stock_prices",
        description="Get more than 10 years of daily historical stock prices and volumes. The data covers several thousand US and international companies. There is a limit of 300 records per API call.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for a security.",
                "example": "MSFT"
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
            "url": "https://financialdata.net/api/v1/stock-prices",
            "method": "GET"
        },
        function=execute_api
    )


def create_international_stock_prices_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get daily historical international stock prices and volumes
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/international-stock-prices"
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
        name="get_international_stock_prices",
        description="Get more than 10 years of daily historical stock prices and volumes. Data is available for Toronto, London, Frankfurt, Euronext Paris, Euronext Amsterdam, Tokyo, Hong Kong, Singapore, Indonesia, Malaysia, Korea, Brazil, Mexico, India, Bombay stock exchanges. There is a limit of 300 records per API call.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for a security.",
                "example": "SHEL.L"
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
            "url": "https://financialdata.net/api/v1/international-stock-prices",
            "method": "GET"
        },
        function=execute_api
    )


def create_minute_prices_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get one-minute historical prices and volumes
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/minute-prices"
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
        name="get_minute_prices",
        description="Get more than 7 years of one-minute historical prices and volumes. The data is available for over 10,000 securities, including US stocks, international stocks, and exchange-traded funds. The timezone used for time values is UTC. There is a limit of 300 records per API call.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for a security.",
                "example": "MSFT"
            },
            "date": {
                "type": "string",
                "description": "The date in YYYY-MM-DD format.",
                "example": "2020-01-15"
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
            "url": "https://financialdata.net/api/v1/minute-prices",
            "method": "GET"
        },
        function=execute_api
    )


def create_index_symbols_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get a list of market index trading symbols and names
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/index-symbols"
        params = {}
        
        # Add API key to parameters
        if api_key:
            params["key"] = api_key
        
        if "format" in kwargs:
            params["format"] = kwargs["format"]
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response
    
    return APITool(
        name="get_index_symbols",
        description="Get a list of the trading symbols and names of the major market indexes. A market index measures the value of a portfolio of holdings with certain market characteristics.",
        inputs={
            "format": {
                "type": "string",
                "description": "Optional. The format of the returned data, either JSON or CSV. Defaults to JSON.",
                "example": "json",
                "enum": ["json", "csv"]
            }
        },
        required=[],
        endpoint_config={
            "url": "https://financialdata.net/api/v1/index-symbols",
            "method": "GET"
        },
        function=execute_api
    )


def create_index_prices_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get daily historical market index prices and volumes
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/index-prices"
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
        name="get_index_prices",
        description="Retrieve more than 10 years of daily historical market index prices and trading volumes. There is a limit of 300 records per API call.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for an index.",
                "example": "^GSPC"
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
            "url": "https://financialdata.net/api/v1/index-prices",
            "method": "GET"
        },
        function=execute_api
    )


def create_index_constituents_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get a list of constituents for a specific index
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/index-constituents"
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
        name="get_index_constituents",
        description="Get a list of constituents for a specific index. Index constituents are the individual components that comprise a market index. These can be stocks, bonds, or other financial instruments. There is a limit of 300 records per API call.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for an index.",
                "example": "^GSPC"
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
            "url": "https://financialdata.net/api/v1/index-constituents",
            "method": "GET"
        },
        function=execute_api
    )


def create_stock_toolkit(api_key: Optional[str] = None) -> APIToolkit:
    """
    Create a Stock Data Toolkit instance
    
    Provides comprehensive access to stock, index, and financial market data, including:
    - Stock symbols (US and international)
    - Historical price data (daily and minute-level)
    - Market index data
    - Index constituent information
    
    Args:
        api_key: Optional API key for authentication. When making requests, the API key 
                will be appended as ?key=YOUR_API_KEY or &key=YOUR_API_KEY depending on 
                whether other query parameters exist.
    
    Returns:
        APIToolkit instance
    
    Example:
        >>> # Create toolkit with API key
        >>> toolkit = create_stock_toolkit(api_key="your_api_key_here")
        >>> 
        >>> # Get a specific tool
        >>> stock_symbols = toolkit.get_tool("get_stock_symbols")
        >>> result = stock_symbols(offset=0, format="json")
        >>> 
        >>> # Or use tools directly
        >>> stock_prices = toolkit.get_tool("get_stock_prices")
        >>> prices = stock_prices(identifier="MSFT", offset=0)
    """
    tools = [
        create_stock_symbols_tool(api_key=api_key),
        create_international_stock_symbols_tool(api_key=api_key),
        create_stock_prices_tool(api_key=api_key),
        create_international_stock_prices_tool(api_key=api_key),
        create_minute_prices_tool(api_key=api_key),
        create_index_symbols_tool(api_key=api_key),
        create_index_prices_tool(api_key=api_key),
        create_index_constituents_tool(api_key=api_key)
    ]
    
    auth_config = {}
    if api_key:
        auth_config = {"api_key": api_key}
    
    return APIToolkit(
        name="stock_data_toolkit",
        tools=tools,
        base_url="https://financialdata.net/api/v1",
        auth_config=auth_config,
        common_headers={
            "Content-Type": "application/json"
        }
    )
