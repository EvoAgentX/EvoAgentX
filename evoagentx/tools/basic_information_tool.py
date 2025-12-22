import requests
from typing import Optional
from .api_converter import APITool, APIToolkit


def create_company_information_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get basic information about a company
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/company-information"
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
        name="get_company_information",
        description="Get basic information about the company, such as its LEI number, industry, contact information, and other key facts. The data covers a few thousand US and international companies.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for a security, or the central index key (CIK). The latter is assigned to the entity by the United States Securities and Exchange Commission.",
                "example": "MSFT"
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
            "url": "https://financialdata.net/api/v1/company-information",
            "method": "GET"
        },
        function=execute_api
    )


def create_international_company_information_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get basic information about an international company
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/international-company-information"
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
        name="get_international_company_information",
        description="Get basic information about the international company, such as its exchange, industry, employee count, and other key facts. Data is available for Toronto, London, Frankfurt, Euronext Paris, Euronext Amsterdam, Tokyo, Hong Kong, Singapore, Indonesia, Malaysia, Korea, Brazil, Mexico, India, Bombay stock exchanges.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for a security.",
                "example": "SHEL.L"
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
            "url": "https://financialdata.net/api/v1/international-company-information",
            "method": "GET"
        },
        function=execute_api
    )


def create_key_metrics_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get key financial metrics for a company
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/key-metrics"
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
        name="get_key_metrics",
        description="Get key financial metrics such as price-to-earnings ratio, price-to-book ratio, free cash flow, etc. This information is particularly important for value investors looking to identify undervalued stocks. Data is available for several thousand US and international companies.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for a security, or the central index key (CIK). The latter is assigned to the entity by the United States Securities and Exchange Commission.",
                "example": "MSFT"
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
            "url": "https://financialdata.net/api/v1/key-metrics",
            "method": "GET"
        },
        function=execute_api
    )


def create_market_cap_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get historical market capitalization data
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/market-cap"
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
        name="get_market_cap",
        description="Get historical market capitalization data. Market cap is calculated by multiplying the market price per common share by the total number of common shares outstanding. The API provides historical market cap data for a few thousand companies.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for a security, or the central index key (CIK). The latter is assigned to the entity by the United States Securities and Exchange Commission.",
                "example": "MSFT"
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
            "url": "https://financialdata.net/api/v1/market-cap",
            "method": "GET"
        },
        function=execute_api
    )


def create_employee_count_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get historical employee count data
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/employee-count"
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
        name="get_employee_count",
        description="Get the total number of company employees for a particular year. The historical data covers several thousand US and international companies.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for a security, or the central index key (CIK). The latter is assigned to the entity by the United States Securities and Exchange Commission.",
                "example": "MSFT"
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
            "url": "https://financialdata.net/api/v1/employee-count",
            "method": "GET"
        },
        function=execute_api
    )


def create_executive_compensation_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get historical executive compensation data
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/executive-compensation"
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
        name="get_executive_compensation",
        description="Get historical executive compensation data including salary, bonus, stock awards, and other benefits. Executive compensation includes both financial and non-financial benefits provided by their employer. The API endpoint provides data for several thousand US and international companies. There is a limit of 100 records per API call.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "The trading symbol for a security, or the central index key (CIK). The latter is assigned to the entity by the United States Securities and Exchange Commission.",
                "example": "MSFT"
            },
            "offset": {
                "type": "integer",
                "description": "Optional. The initial position of the record subset, which indicates how many records to skip. Defaults to 0.",
                "example": 100
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
            "url": "https://financialdata.net/api/v1/executive-compensation",
            "method": "GET"
        },
        function=execute_api
    )


def create_securities_information_tool(api_key: Optional[str] = None) -> APITool:
    """
    Create a tool to get basic information about securities
    
    Args:
        api_key: API key for authentication
        
    Returns:
        APITool instance
    """
    def execute_api(**kwargs):
        url = "https://financialdata.net/api/v1/securities-information"
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
        name="get_securities_information",
        description="Get basic information about securities (tradable financial instruments). A security may refer to stocks, bonds, notes, limited partnership interests, investment contracts, and others. This endpoint provides trading symbol, issuer, local and international identification numbers, and other details.",
        inputs={
            "identifier": {
                "type": "string",
                "description": "One of the following values: a security's trading symbol, the CUSIP (Committee on Uniform Securities Identification Procedures) number, or the ISIN (International Securities Identification Number).",
                "example": "AAPL"
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
            "url": "https://financialdata.net/api/v1/securities-information",
            "method": "GET"
        },
        function=execute_api
    )


def create_basic_information_toolkit(api_key: Optional[str] = None) -> APIToolkit:
    """
    Create a Basic Company Information Toolkit instance
    
    Provides comprehensive access to company and securities information, including:
    - Company information (US and international)
    - Key financial metrics
    - Market capitalization history
    - Employee count history
    - Executive compensation data
    - Securities information
    
    Args:
        api_key: Optional API key for authentication. When making requests, the API key 
                will be appended as ?key=YOUR_API_KEY or &key=YOUR_API_KEY depending on 
                whether other query parameters exist.
    
    Returns:
        APIToolkit instance
    
    Example:
        >>> # Create toolkit with API key
        >>> toolkit = create_basic_information_toolkit(api_key="your_api_key_here")
        >>> 
        >>> # Get company information
        >>> company_info = toolkit.get_tool("get_company_information")
        >>> result = company_info(identifier="MSFT")
        >>> 
        >>> # Get key metrics
        >>> key_metrics = toolkit.get_tool("get_key_metrics")
        >>> metrics = key_metrics(identifier="MSFT")
        >>> 
        >>> # Get executive compensation
        >>> exec_comp = toolkit.get_tool("get_executive_compensation")
        >>> compensation = exec_comp(identifier="MSFT", offset=0)
    """
    tools = [
        create_company_information_tool(api_key=api_key),
        create_international_company_information_tool(api_key=api_key),
        create_key_metrics_tool(api_key=api_key),
        create_market_cap_tool(api_key=api_key),
        create_employee_count_tool(api_key=api_key),
        create_executive_compensation_tool(api_key=api_key),
        create_securities_information_tool(api_key=api_key)
    ]
    
    auth_config = {}
    if api_key:
        auth_config = {"api_key": api_key}
    
    return APIToolkit(
        name="basic_information_toolkit",
        tools=tools,
        base_url="https://financialdata.net/api/v1",
        auth_config=auth_config,
        common_headers={
            "Content-Type": "application/json"
        }
    )

