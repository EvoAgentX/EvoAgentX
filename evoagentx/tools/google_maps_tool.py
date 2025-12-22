"""
Google Maps Platform API Tool for EvoAgentX

This module provides comprehensive Google Maps Platform integration including:
- Geocoding API: Convert addresses to coordinates and vice versa
- Places API: Search for places and get detailed information
- Routes API: Calculate directions and distance matrices  
- Time Zone API: Get time zone information for locations

Compatible with EvoAgentX tool architecture and follows the latest Google Maps Platform APIs.
"""

import requests
import json
import os
from typing import Dict, Any, List

from .tool import Tool, Toolkit
from ..core.module import BaseModule
from ..core.logging import logger


class GoogleMapsBase(BaseModule):
    """
    Base class for Google Maps Platform API interactions.
    Handles API key management, request formatting, and common utilities.
    """
    
    def __init__(self, api_key: str = None, timeout: int = 10, **kwargs):
        """
        Initialize the Google Maps base.
        
        Args:
            api_key (str, optional): Google Maps Platform API key. If not provided, will try to get from GOOGLE_MAPS_API_KEY environment variable.
            timeout (int): Request timeout in seconds
            **kwargs: Additional keyword arguments for parent class
        """
        super().__init__(**kwargs)
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        
        if not self.api_key:
            logger.warning(
                "No Google Maps API key provided. Please set GOOGLE_MAPS_API_KEY environment variable "
                "or pass api_key parameter. Get your API key from: https://console.cloud.google.com/apis/"
            )
        
        self.timeout = timeout
        self.base_url = "https://maps.googleapis.com/maps/api"
        
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to Google Maps Platform API.
        
        Args:
            endpoint (str): API endpoint
            params (dict): Request parameters
            
        Returns:
            dict: API response
        """
        # Check if API key is available
        if not self.api_key:
            return {
                "success": False,
                "error": "Google Maps API key not found. Please set GOOGLE_MAPS_API_KEY environment variable or pass api_key parameter."
            }
        
        try:
            # Add API key to parameters
            params['key'] = self.api_key
            
            # Build URL
            url = f"{self.base_url}/{endpoint}"
            
            # Make request
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Check API status
            status = data.get('status', 'UNKNOWN_ERROR')
            if status == 'OK':
                return {
                    "success": True,
                    "status": status,
                    "data": data
                }
            elif status == 'ZERO_RESULTS':
                return {
                    "success": True,
                    "status": status,
                    "data": data,
                    "message": "No results found"
                }
            else:
                error_message = data.get('error_message', f"API returned status: {status}")
                logger.error(f"Google Maps API error: {error_message}")
                return {
                    "success": False,
                    "status": status,
                    "error": error_message
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return {
                "success": False,
                "error": f"Invalid JSON response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def _format_coordinates(self, lat: float, lng: float) -> str:
        """Format coordinates for API requests."""
        return f"{lat},{lng}"


class GeocodeAddressTool(Tool):
    """Convert addresses to geographic coordinates (latitude/longitude)."""
    
    name: str = "geocode_address"
    description: str = "Convert a street address into geographic coordinates (latitude and longitude). Useful for finding exact locations of places."
    inputs: Dict[str, Dict[str, str]] = {
        "address": {
            "type": "string",
            "description": "The street address to geocode (e.g., '1600 Amphitheatre Parkway, Mountain View, CA')"
        },
        "components": {
            "type": "string", 
            "description": "Optional component filters (e.g., 'country:US|locality:Mountain View')"
        },
        "region": {
            "type": "string",
            "description": "Optional region code for biasing results (e.g., 'us', 'uk')"
        }
    }
    required: List[str] = ["address"]
    
    def __init__(self, google_maps_base: GoogleMapsBase):
        super().__init__()
        self.google_maps_base = google_maps_base
        
    def __call__(self, address: str, components: str = None, region: str = None) -> Dict[str, Any]:
        """
        Geocode an address to coordinates.
        
        Args:
            address: Street address to geocode
            components: Optional component filters
            region: Optional region bias
            
        Returns:
            Dictionary with geocoding results
        """
        params = {"address": address}
        
        if components:
            params["components"] = components
        if region:
            params["region"] = region
            
        result = self.google_maps_base._make_request("geocode/json", params)
        
        if result["success"] and result["data"].get("results"):
            # Extract first result
            geocode_result = result["data"]["results"][0]
            location = geocode_result["geometry"]["location"]
            
            return {
                "success": True,
                "address": address,
                "formatted_address": geocode_result.get("formatted_address"),
                "latitude": location["lat"],
                "longitude": location["lng"],
                "place_id": geocode_result.get("place_id"),
                "location_type": geocode_result["geometry"].get("location_type"),
                "address_components": geocode_result.get("address_components", [])
            }
        else:
            return {
                "success": False,
                "address": address,
                "error": result.get("error", "No results found")
            }


class ReverseGeocodeTool(Tool):
    """Convert geographic coordinates to a human-readable address."""
    
    name: str = "reverse_geocode"
    description: str = "Convert geographic coordinates (latitude and longitude) into a human-readable address."
    inputs: Dict[str, Dict[str, str]] = {
        "latitude": {
            "type": "number",
            "description": "Latitude coordinate"
        },
        "longitude": {
            "type": "number", 
            "description": "Longitude coordinate"
        },
        "result_type": {
            "type": "string",
            "description": "Optional filter for result types (e.g., 'street_address|route')"
        }
    }
    required: List[str] = ["latitude", "longitude"]
    
    def __init__(self, google_maps_base: GoogleMapsBase):
        super().__init__()
        self.google_maps_base = google_maps_base
        
    def __call__(self, latitude: float, longitude: float, result_type: str = None) -> Dict[str, Any]:
        """
        Reverse geocode coordinates to address.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            result_type: Optional result type filter
            
        Returns:
            Dictionary with reverse geocoding results
        """
        latlng = self.google_maps_base._format_coordinates(latitude, longitude)
        params = {"latlng": latlng}
        
        if result_type:
            params["result_type"] = result_type
            
        result = self.google_maps_base._make_request("geocode/json", params)
        
        if result["success"] and result["data"].get("results"):
            addresses = []
            for geocode_result in result["data"]["results"]:
                addresses.append({
                    "formatted_address": geocode_result.get("formatted_address"),
                    "place_id": geocode_result.get("place_id"),
                    "types": geocode_result.get("types", []),
                    "address_components": geocode_result.get("address_components", [])
                })
            
            return {
                "success": True,
                "latitude": latitude,
                "longitude": longitude,
                "addresses": addresses
            }
        else:
            return {
                "success": False,
                "latitude": latitude,
                "longitude": longitude,
                "error": result.get("error", "No results found")
            }


class PlacesSearchTool(Tool):
    """Search for places using text queries or nearby location."""
    
    name: str = "places_search"
    description: str = "Search for places (restaurants, shops, landmarks) using text queries. Can search near a specific location."
    inputs: Dict[str, Dict[str, str]] = {
        "query": {
            "type": "string",
            "description": "Text search query (e.g., 'pizza restaurants near Times Square')"
        },
        "location": {
            "type": "string",
            "description": "Optional location bias as 'latitude,longitude' (e.g., '40.7589,-73.9851')"
        },
        "radius": {
            "type": "number",
            "description": "Optional search radius in meters (max 50000)"
        },
        "type": {
            "type": "string", 
            "description": "Optional place type filter (e.g., 'restaurant', 'gas_station')"
        }
    }
    required: List[str] = ["query"]
    
    def __init__(self, google_maps_base: GoogleMapsBase):
        super().__init__()
        self.google_maps_base = google_maps_base
        
    def __call__(self, query: str, location: str = None, radius: float = None, type: str = None) -> Dict[str, Any]:
        """
        Search for places using text query.
        
        Args:
            query: Text search query
            location: Optional location bias as 'lat,lng'
            radius: Optional search radius in meters
            type: Optional place type filter
            
        Returns:
            Dictionary with search results
        """
        params = {"query": query}
        
        if location:
            params["location"] = location
        if radius:
            params["radius"] = min(radius, 50000)  # Max radius limit
        if type:
            params["type"] = type
            
        result = self.google_maps_base._make_request("place/textsearch/json", params)
        
        if result["success"]:
            places = []
            for place in result["data"].get("results", []):
                places.append({
                    "name": place.get("name"),
                    "place_id": place.get("place_id"),
                    "formatted_address": place.get("formatted_address"),
                    "rating": place.get("rating"),
                    "user_ratings_total": place.get("user_ratings_total"),
                    "price_level": place.get("price_level"),
                    "types": place.get("types", []),
                    "geometry": place.get("geometry", {}),
                    "business_status": place.get("business_status")
                })
            
            return {
                "success": True,
                "query": query,
                "places_found": len(places),
                "places": places
            }
        else:
            return {
                "success": False,
                "query": query,
                "error": result.get("error", "Search failed")
            }


class PlaceDetailsTool(Tool):
    """Get detailed information about a specific place using its Place ID."""
    
    name: str = "place_details"
    description: str = "Get comprehensive information about a specific place using its Place ID, including contact info, hours, reviews."
    inputs: Dict[str, Dict[str, str]] = {
        "place_id": {
            "type": "string",
            "description": "Unique Place ID from a place search"
        },
        "fields": {
            "type": "string",
            "description": "Optional comma-separated list of fields to return (e.g., 'name,rating,formatted_phone_number')"
        }
    }
    required: List[str] = ["place_id"]
    
    def __init__(self, google_maps_base: GoogleMapsBase):
        super().__init__()
        self.google_maps_base = google_maps_base
        
    def __call__(self, place_id: str, fields: str = None) -> Dict[str, Any]:
        """
        Get detailed place information.
        
        Args:
            place_id: Unique place identifier
            fields: Optional fields to return
            
        Returns:
            Dictionary with place details
        """
        params = {"place_id": place_id}
        
        # Default fields if none specified
        if not fields:
            fields = "name,formatted_address,formatted_phone_number,website,rating,user_ratings_total,opening_hours,price_level,types,geometry"
        
        params["fields"] = fields
        
        result = self.google_maps_base._make_request("place/details/json", params)
        
        if result["success"] and result["data"].get("result"):
            place = result["data"]["result"]
            
            return {
                "success": True,
                "place_id": place_id,
                "name": place.get("name"),
                "formatted_address": place.get("formatted_address"),
                "phone_number": place.get("formatted_phone_number"),
                "international_phone": place.get("international_phone_number"),
                "website": place.get("website"),
                "rating": place.get("rating"),
                "user_ratings_total": place.get("user_ratings_total"),
                "price_level": place.get("price_level"),
                "types": place.get("types", []),
                "opening_hours": place.get("opening_hours"),
                "geometry": place.get("geometry", {}),
                "business_status": place.get("business_status"),
                "reviews": place.get("reviews", [])
            }
        else:
            return {
                "success": False,
                "place_id": place_id,
                "error": result.get("error", "Place not found")
            }


class DirectionsTool(Tool):
    """Calculate driving, walking, bicycling, or transit directions between locations."""
    
    name: str = "directions"
    description: str = "Calculate directions between two or more locations with different travel modes (driving, walking, bicycling, transit)."
    inputs: Dict[str, Dict[str, str]] = {
        "origin": {
            "type": "string",
            "description": "Starting location (address, coordinates, or place ID)"
        },
        "destination": {
            "type": "string",
            "description": "Ending location (address, coordinates, or place ID)"
        },
        "mode": {
            "type": "string",
            "description": "Travel mode: 'driving', 'walking', 'bicycling', or 'transit' (default: driving)"
        },
        "waypoints": {
            "type": "string",
            "description": "Optional waypoints separated by '|' (e.g., 'via:San Francisco|via:Los Angeles')"
        },
        "alternatives": {
            "type": "boolean",
            "description": "Whether to return alternative routes (default: false)"
        }
    }
    required: List[str] = ["origin", "destination"]
    
    def __init__(self, google_maps_base: GoogleMapsBase):
        super().__init__()
        self.google_maps_base = google_maps_base
        
    def __call__(self, origin: str, destination: str, mode: str = "driving", 
                 waypoints: str = None, alternatives: bool = False) -> Dict[str, Any]:
        """
        Calculate directions between locations.
        
        Args:
            origin: Starting location
            destination: Ending location
            mode: Travel mode
            waypoints: Optional waypoints
            alternatives: Return alternative routes
            
        Returns:
            Dictionary with directions
        """
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "alternatives": alternatives
        }
        
        if waypoints:
            params["waypoints"] = waypoints
            
        result = self.google_maps_base._make_request("directions/json", params)
        
        if result["success"] and result["data"].get("routes"):
            routes = []
            for route in result["data"]["routes"]:
                # Extract route information
                legs = []
                total_distance = 0
                total_duration = 0
                
                for leg in route.get("legs", []):
                    leg_info = {
                        "start_address": leg.get("start_address"),
                        "end_address": leg.get("end_address"),
                        "distance": leg.get("distance", {}),
                        "duration": leg.get("duration", {}),
                        "steps": []
                    }
                    
                    # Add distance and duration to totals
                    if leg.get("distance", {}).get("value"):
                        total_distance += leg["distance"]["value"]
                    if leg.get("duration", {}).get("value"):
                        total_duration += leg["duration"]["value"]
                    
                    # Extract steps
                    for step in leg.get("steps", []):
                        leg_info["steps"].append({
                            "instructions": step.get("html_instructions", ""),
                            "distance": step.get("distance", {}),
                            "duration": step.get("duration", {}),
                            "travel_mode": step.get("travel_mode")
                        })
                    
                    legs.append(leg_info)
                
                routes.append({
                    "summary": route.get("summary"),
                    "legs": legs,
                    "total_distance_meters": total_distance,
                    "total_duration_seconds": total_duration,
                    "overview_polyline": route.get("overview_polyline", {}),
                    "warnings": route.get("warnings", []),
                    "copyrights": route.get("copyrights")
                })
            
            return {
                "success": True,
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "routes": routes
            }
        else:
            return {
                "success": False,
                "origin": origin,
                "destination": destination,
                "error": result.get("error", "No routes found")
            }


class DistanceMatrixTool(Tool):
    """Calculate travel times and distances between multiple origins and destinations."""
    
    name: str = "distance_matrix"
    description: str = "Calculate travel times and distances between multiple origins and destinations. Useful for finding the closest location."
    inputs: Dict[str, Dict[str, str]] = {
        "origins": {
            "type": "string",
            "description": "Origin locations separated by '|' (e.g., 'Seattle,WA|Portland,OR')"
        },
        "destinations": {
            "type": "string",
            "description": "Destination locations separated by '|' (e.g., 'San Francisco,CA|Los Angeles,CA')"
        },
        "mode": {
            "type": "string",
            "description": "Travel mode: 'driving', 'walking', 'bicycling', or 'transit' (default: driving)"
        },
        "units": {
            "type": "string",
            "description": "Unit system: 'metric' or 'imperial' (default: metric)"
        }
    }
    required: List[str] = ["origins", "destinations"]
    
    def __init__(self, google_maps_base: GoogleMapsBase):
        super().__init__()
        self.google_maps_base = google_maps_base
        
    def __call__(self, origins: str, destinations: str, mode: str = "driving", 
                 units: str = "metric") -> Dict[str, Any]:
        """
        Calculate distance matrix.
        
        Args:
            origins: Origin locations separated by '|'
            destinations: Destination locations separated by '|'
            mode: Travel mode
            units: Unit system
            
        Returns:
            Dictionary with distance matrix
        """
        params = {
            "origins": origins,
            "destinations": destinations,
            "mode": mode,
            "units": units
        }
        
        result = self.google_maps_base._make_request("distancematrix/json", params)
        
        if result["success"] and result["data"].get("rows"):
            origin_addresses = result["data"].get("origin_addresses", [])
            destination_addresses = result["data"].get("destination_addresses", [])
            
            matrix = []
            for i, row in enumerate(result["data"]["rows"]):
                origin_results = {
                    "origin_address": origin_addresses[i] if i < len(origin_addresses) else f"Origin {i+1}",
                    "destinations": []
                }
                
                for j, element in enumerate(row.get("elements", [])):
                    destination_result = {
                        "destination_address": destination_addresses[j] if j < len(destination_addresses) else f"Destination {j+1}",
                        "status": element.get("status"),
                        "distance": element.get("distance", {}),
                        "duration": element.get("duration", {}),
                        "duration_in_traffic": element.get("duration_in_traffic", {})
                    }
                    origin_results["destinations"].append(destination_result)
                
                matrix.append(origin_results)
            
            return {
                "success": True,
                "origins": origins.split("|"),
                "destinations": destinations.split("|"),
                "mode": mode,
                "units": units,
                "matrix": matrix
            }
        else:
            return {
                "success": False,
                "origins": origins,
                "destinations": destinations,
                "error": result.get("error", "Distance matrix calculation failed")
            }


class TimeZoneTool(Tool):
    """Get time zone information for a location."""
    
    name: str = "timezone"
    description: str = "Get time zone information for a specific location using coordinates."
    inputs: Dict[str, Dict[str, str]] = {
        "latitude": {
            "type": "number",
            "description": "Latitude coordinate"
        },
        "longitude": {
            "type": "number",
            "description": "Longitude coordinate"
        },
        "timestamp": {
            "type": "number",
            "description": "Optional Unix timestamp for the desired time (default: current time)"
        }
    }
    required: List[str] = ["latitude", "longitude"]
    
    def __init__(self, google_maps_base: GoogleMapsBase):
        super().__init__()
        self.google_maps_base = google_maps_base
        
    def __call__(self, latitude: float, longitude: float, timestamp: float = None) -> Dict[str, Any]:
        """
        Get time zone information.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            timestamp: Optional Unix timestamp
            
        Returns:
            Dictionary with time zone info
        """
        import time
        
        location = self.google_maps_base._format_coordinates(latitude, longitude)
        params = {
            "location": location,
            "timestamp": timestamp or int(time.time())
        }
        
        result = self.google_maps_base._make_request("timezone/json", params)
        
        if result["success"]:
            data = result["data"]
            return {
                "success": True,
                "latitude": latitude,
                "longitude": longitude,
                "time_zone_id": data.get("timeZoneId"),
                "time_zone_name": data.get("timeZoneName"),
                "dst_offset": data.get("dstOffset"),
                "raw_offset": data.get("rawOffset"),
                "status": data.get("status")
            }
        else:
            return {
                "success": False,
                "latitude": latitude,
                "longitude": longitude,
                "error": result.get("error", "Time zone lookup failed")
            }


class GoogleMapsToolkit(Toolkit):
    """
    Complete Google Maps Platform toolkit containing all available tools.
    """
    
    def __init__(self, api_key: str = None, timeout: int = 10, name: str = "GoogleMapsToolkit"):
        """
        Initialize the Google Maps toolkit.
        
        Args:
            api_key (str, optional): Google Maps Platform API key. If not provided, will try to get from GOOGLE_MAPS_API_KEY environment variable.
            timeout (int): Request timeout in seconds
            name (str): Toolkit name
        """
        # Create shared Google Maps base instance
        google_maps_base = GoogleMapsBase(api_key=api_key, timeout=timeout)
        
        # Create all tools with shared base
        tools = [
            GeocodeAddressTool(google_maps_base=google_maps_base),
            ReverseGeocodeTool(google_maps_base=google_maps_base),
            PlacesSearchTool(google_maps_base=google_maps_base),
            PlaceDetailsTool(google_maps_base=google_maps_base),
            DirectionsTool(google_maps_base=google_maps_base),
            DistanceMatrixTool(google_maps_base=google_maps_base),
            TimeZoneTool(google_maps_base=google_maps_base)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store base instance for access
        self.google_maps_base = google_maps_base
