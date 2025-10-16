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
from dotenv import load_dotenv

from evoagentx.tools.api_converter import (
    create_rapidapi_toolkit,
)

load_dotenv()



weather_api_spec = {
    "openapi": "3.0.0",
    "info": {
        "title": "Open Weather13 API",
        "version": "2.0",
        "description": "Complete weather data API via RapidAPI with 3 endpoints"
    },
    "servers": [
        {"url": "https://open-weather13.p.rapidapi.com"}
    ],
    "paths": {
        "/city": {
            "get": {
                "operationId": "getCityWeather",
                "summary": "Get weather by city name",
                "description": "Get current weather data by city name",
                "parameters": [
                    {
                        "name": "city",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "City name, e.g.: new york, beijing, tokyo"
                    },
                    {
                        "name": "lang",
                        "in": "query", 
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["AF", "AL", "AR", "AZ", "BG", "CA", "CZ", "DA", "DE", "EL", "EN", "EU", "FA", "FI", "FR", "GL", "HE", "HI", "HR", "HU", "ID", "IT", "JA", "KR", "LA", "LT", "MK", "NO", "NL", "PL", "PT", "PT_BR", "RO", "RU", "SE", "SK", "SL", "SP", "ES", "SR", "TH", "TR", "UK", "VI", "ZH_CN", "ZH_TW", "ZU"],
                            "default": "EN"
                        },
                        "description": "Language code, supports Simplified Chinese (ZH_CN), Traditional Chinese (ZH_TW), and many other languages"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "City weather data response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "coord": {
                                            "type": "object",
                                            "properties": {
                                                "lon": {"type": "number"},
                                                "lat": {"type": "number"}
                                            }
                                        },
                                        "weather": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "integer"},
                                                    "main": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "icon": {"type": "string"}
                                                }
                                            }
                                        },
                                        "main": {
                                            "type": "object",
                                            "properties": {
                                                "temp": {"type": "number"},
                                                "feels_like": {"type": "number"},
                                                "temp_min": {"type": "number"},
                                                "temp_max": {"type": "number"},
                                                "pressure": {"type": "number"},
                                                "humidity": {"type": "number"},
                                                "sea_level": {"type": "number"},
                                                "grnd_level": {"type": "number"}
                                            }
                                        },
                                        "visibility": {"type": "number"},
                                        "wind": {
                                            "type": "object",
                                            "properties": {
                                                "speed": {"type": "number"},
                                                "deg": {"type": "number"}
                                            }
                                        },
                                        "clouds": {
                                            "type": "object",
                                            "properties": {
                                                "all": {"type": "number"}
                                            }
                                        },
                                        "name": {"type": "string"},
                                        "sys": {
                                            "type": "object",
                                            "properties": {
                                                "type": {"type": "integer"},
                                                "id": {"type": "integer"},
                                                "country": {"type": "string"},
                                                "sunrise": {"type": "integer"},
                                                "sunset": {"type": "integer"}
                                            }
                                        },
                                        "timezone": {"type": "integer"},
                                        "id": {"type": "integer"},
                                        "cod": {"type": "integer"},
                                        "dt": {"type": "integer"},
                                        "base": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/latlon": {
            "get": {
                "operationId": "getWeatherByCoordinates",
                "summary": "Get weather by coordinates",
                "description": "Get current weather data using latitude and longitude",
                "parameters": [
                    {
                        "name": "latitude",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Latitude, e.g.: 40.730610"
                    },
                    {
                        "name": "longitude",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Longitude, e.g.: -73.935242"
                    },
                    {
                        "name": "lang",
                        "in": "query", 
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["AF", "AL", "AR", "AZ", "BG", "CA", "CZ", "DA", "DE", "EL", "EN", "EU", "FA", "FI", "FR", "GL", "HE", "HI", "HR", "HU", "ID", "IT", "JA", "KR", "LA", "LT", "MK", "NO", "NL", "PL", "PT", "PT_BR", "RO", "RU", "SE", "SK", "SL", "SP", "ES", "SR", "TH", "TR", "UK", "VI", "ZH_CN", "ZH_TW", "ZU"],
                            "default": "EN"
                        },
                        "description": "Language code"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Coordinate weather data response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "coord": {
                                            "type": "object",
                                            "properties": {
                                                "lon": {"type": "number"},
                                                "lat": {"type": "number"}
                                            }
                                        },
                                        "weather": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "integer"},
                                                    "main": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "icon": {"type": "string"}
                                                }
                                            }
                                        },
                                        "main": {
                                            "type": "object",
                                            "properties": {
                                                "temp": {"type": "number"},
                                                "feels_like": {"type": "number"},
                                                "temp_min": {"type": "number"},
                                                "temp_max": {"type": "number"},
                                                "pressure": {"type": "number"},
                                                "humidity": {"type": "number"},
                                                "sea_level": {"type": "number"},
                                                "grnd_level": {"type": "number"}
                                            }
                                        },
                                        "visibility": {"type": "number"},
                                        "wind": {
                                            "type": "object",
                                            "properties": {
                                                "speed": {"type": "number"},
                                                "deg": {"type": "number"}
                                            }
                                        },
                                        "clouds": {
                                            "type": "object",
                                            "properties": {
                                                "all": {"type": "number"}
                                            }
                                        },
                                        "name": {"type": "string"},
                                        "sys": {
                                            "type": "object",
                                            "properties": {
                                                "type": {"type": "integer"},
                                                "id": {"type": "integer"},
                                                "country": {"type": "string"},
                                                "sunrise": {"type": "integer"},
                                                "sunset": {"type": "integer"}
                                            }
                                        },
                                        "timezone": {"type": "integer"},
                                        "id": {"type": "integer"},
                                        "cod": {"type": "integer"},
                                        "dt": {"type": "integer"},
                                        "base": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/fivedaysforcast": {
            "get": {
                "operationId": "getFiveDayForecast",
                "summary": "Get 5-day weather forecast",
                "description": "Get 5-day weather forecast using latitude and longitude",
                "parameters": [
                    {
                        "name": "latitude",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Latitude, e.g.: 40.730610"
                    },
                    {
                        "name": "longitude",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Longitude, e.g.: -73.935242"
                    },
                    {
                        "name": "lang",
                        "in": "query", 
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["AF", "AL", "AR", "AZ", "BG", "CA", "CZ", "DA", "DE", "EL", "EN", "EU", "FA", "FI", "FR", "GL", "HE", "HI", "HR", "HU", "ID", "IT", "JA", "KR", "LA", "LT", "MK", "NO", "NL", "PL", "PT", "PT_BR", "RO", "RU", "SE", "SK", "SL", "SP", "ES", "SR", "TH", "TR", "UK", "VI", "ZH_CN", "ZH_TW", "ZU"],
                            "default": "EN"
                        },
                        "description": "Language code"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "5-day forecast data response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "cod": {"type": "string"},
                                        "message": {"type": "number"},
                                        "cnt": {"type": "integer"},
                                        "list": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "dt": {"type": "integer"},
                                                    "main": {
                                                        "type": "object",
                                                        "properties": {
                                                            "temp": {"type": "number"},
                                                            "feels_like": {"type": "number"},
                                                            "temp_min": {"type": "number"},
                                                            "temp_max": {"type": "number"},
                                                            "pressure": {"type": "number"},
                                                            "sea_level": {"type": "number"},
                                                            "grnd_level": {"type": "number"},
                                                            "humidity": {"type": "number"},
                                                            "temp_kf": {"type": "number"}
                                                        }
                                                    },
                                                    "weather": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "id": {"type": "integer"},
                                                                "main": {"type": "string"},
                                                                "description": {"type": "string"},
                                                                "icon": {"type": "string"}
                                                            }
                                                        }
                                                    },
                                                    "clouds": {
                                                        "type": "object",
                                                        "properties": {
                                                            "all": {"type": "number"}
                                                        }
                                                    },
                                                    "wind": {
                                                        "type": "object",
                                                        "properties": {
                                                            "speed": {"type": "number"},
                                                            "deg": {"type": "number"},
                                                            "gust": {"type": "number"}
                                                        }
                                                    },
                                                    "visibility": {"type": "number"},
                                                    "pop": {"type": "number"},
                                                    "rain": {
                                                        "type": "object",
                                                        "properties": {
                                                            "3h": {"type": "number"}
                                                        }
                                                    },
                                                    "sys": {
                                                        "type": "object",
                                                        "properties": {
                                                            "pod": {"type": "string"}
                                                        }
                                                    },
                                                    "dt_txt": {"type": "string"}
                                                }
                                            }
                                        },
                                        "city": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "integer"},
                                                "name": {"type": "string"},
                                                "coord": {
                                                    "type": "object",
                                                    "properties": {
                                                        "lat": {"type": "number"},
                                                        "lon": {"type": "number"}
                                                    }
                                                },
                                                "country": {"type": "string"},
                                                "population": {"type": "integer"},
                                                "timezone": {"type": "integer"},
                                                "sunrise": {"type": "integer"},
                                                "sunset": {"type": "integer"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}



def rapidapi_test() -> None:
    print("\n===== SINGLE REAL CALL: OpenWeatherMap (extracted) =====\n")

    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key or api_key.strip().lower() in {"", "your-api-key"}:
        print("Skipping real call: set RAPIDAPI_KEY to run this test.")
        return

    # Extracted (trimmed) OpenWeatherMap spec from real_world_api_example.py
    rapidapi_host = "open-weather13.p.rapidapi.com"
    toolkit = create_rapidapi_toolkit(
        schema_path_or_dict=weather_api_spec,
        rapidapi_key=api_key,
        rapidapi_host=rapidapi_host,
        service_name="Open Weather13"
    )

    # Inject default appid for all tools (pattern extracted from real_world_api_example.py)
    print("____________ Executing city weather querying ____________")
    city_weather_tool = toolkit.get_tools()[0]
    example_query = {"city": "new york"}
    print("Qeury inputs: \n", example_query)
    result = city_weather_tool(**example_query)
    print("Query result: \n", result)
    

def main() -> None:
    """Main function to run condensed converter examples"""
    print("===== API CONVERTER EXAMPLES (CONDENSED) =====")
    # Single extracted real-world call (requires OPENWEATHER_API_KEY)
    rapidapi_test()
    print("\n===== ALL CONDENSED CONVERTER TESTS COMPLETED =====")


if __name__ == "__main__":
    main()