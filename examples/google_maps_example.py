"""
Example usage of Google Maps Platform tools in EvoAgentX

This example demonstrates how to use the various Google Maps tools
for geocoding, places search, directions, and more.

Prerequisites:
1. Google Maps Platform API key with the following APIs enabled:
   - Geocoding API
   - Places API  
   - Directions API
   - Distance Matrix API
   - Time Zone API

2. Set your API key as an environment variable:
   export GOOGLE_MAPS_API_KEY="your_api_key_here"
"""

import os
from evoagentx.tools import GoogleMapsToolkit

def main():
    # Get API key from environment
    api_key = os.getenv("GOOGLE_MAPS_API_KEY") 
    if not api_key:
        print("Please set GOOGLE_MAPS_API_KEY environment variable")
        return
    
    # Initialize the toolkit
    gmaps_toolkit = GoogleMapsToolkit(api_key=api_key)
    
    print("=== Google Maps Platform Tools Demo ===\n")
    
    # 1. Geocoding - Convert address to coordinates
    print("1. Geocoding Address to Coordinates")
    geocode_tool = gmaps_toolkit.get_tool("geocode_address")
    result = geocode_tool(address="1600 Amphitheatre Parkway, Mountain View, CA")
    
    if result["success"]:
        print(f"Address: {result['formatted_address']}")
        print(f"Coordinates: {result['latitude']}, {result['longitude']}")
        print(f"Place ID: {result['place_id']}")
        
        # Store coordinates for other examples
        lat, lng = result['latitude'], result['longitude']
    else:
        print(f"Geocoding failed: {result['error']}")
        return
    
    print("\n" + "="*50 + "\n")
    
    # 2. Reverse Geocoding - Convert coordinates to address
    print("2. Reverse Geocoding Coordinates to Address")
    reverse_geocode_tool = gmaps_toolkit.get_tool("reverse_geocode")
    result = reverse_geocode_tool(latitude=lat, longitude=lng)
    
    if result["success"]:
        print(f"Coordinates: {result['latitude']}, {result['longitude']}")
        print("Addresses found:")
        for i, addr in enumerate(result['addresses'][:3]):  # Show first 3
            print(f"  {i+1}. {addr['formatted_address']}")
    else:
        print(f"Reverse geocoding failed: {result['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # 3. Places Search - Find nearby restaurants
    print("3. Places Search - Find Restaurants")
    places_search_tool = gmaps_toolkit.get_tool("places_search")
    result = places_search_tool(
        query="restaurants near Mountain View, CA",
        location=f"{lat},{lng}",
        radius=2000
    )
    
    if result["success"]:
        print(f"Found {result['places_found']} restaurants")
        for i, place in enumerate(result['places'][:3]):  # Show first 3
            print(f"  {i+1}. {place['name']}")
            print(f"     Address: {place['formatted_address']}")
            print(f"     Rating: {place.get('rating', 'N/A')}")
            print(f"     Place ID: {place['place_id']}")
        
        # Store a place ID for details example
        if result['places']:
            sample_place_id = result['places'][0]['place_id']
    else:
        print(f"Places search failed: {result['error']}")
        sample_place_id = None
    
    print("\n" + "="*50 + "\n")
    
    # 4. Place Details - Get detailed info about a place
    if sample_place_id:
        print("4. Place Details - Restaurant Information")
        place_details_tool = gmaps_toolkit.get_tool("place_details")
        result = place_details_tool(place_id=sample_place_id)
        
        if result["success"]:
            print(f"Name: {result['name']}")
            print(f"Address: {result['formatted_address']}")
            print(f"Phone: {result.get('phone_number', 'N/A')}")
            print(f"Website: {result.get('website', 'N/A')}")
            print(f"Rating: {result.get('rating', 'N/A')} ({result.get('user_ratings_total', 0)} reviews)")
            print(f"Price Level: {result.get('price_level', 'N/A')}")
        else:
            print(f"Place details failed: {result['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # 5. Directions - Get driving directions
    print("5. Directions - Driving Route")
    directions_tool = gmaps_toolkit.get_tool("directions")
    result = directions_tool(
        origin="San Francisco, CA",
        destination="Mountain View, CA",
        mode="driving"
    )
    
    if result["success"] and result['routes']:
        route = result['routes'][0]
        print(f"Route from {result['origin']} to {result['destination']}")
        print(f"Distance: {route['total_distance_meters']} meters")
        print(f"Duration: {route['total_duration_seconds']} seconds")
        print(f"Summary: {route.get('summary', 'N/A')}")
        
        # Show first few steps
        if route['legs'] and route['legs'][0]['steps']:
            print("First 3 steps:")
            for i, step in enumerate(route['legs'][0]['steps'][:3]):
                # Remove HTML tags from instructions (simple approach)
                instructions = step['instructions'].replace('<b>', '').replace('</b>', '').replace('<div>', ' ').replace('</div>', '')
                print(f"  {i+1}. {instructions}")
    else:
        print(f"Directions failed: {result['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # 6. Distance Matrix - Compare multiple routes
    print("6. Distance Matrix - Multiple Origins/Destinations")
    distance_matrix_tool = gmaps_toolkit.get_tool("distance_matrix")
    result = distance_matrix_tool(
        origins="San Francisco,CA|Oakland,CA",
        destinations="Mountain View,CA|Palo Alto,CA",
        mode="driving",
        units="imperial"
    )
    
    if result["success"]:
        print("Distance Matrix Results:")
        for origin_data in result['matrix']:
            print(f"\nFrom: {origin_data['origin_address']}")
            for dest in origin_data['destinations']:
                if dest['status'] == 'OK':
                    print(f"  To {dest['destination_address']}: {dest['distance'].get('text', 'N/A')} - {dest['duration'].get('text', 'N/A')}")
                else:
                    print(f"  To {dest['destination_address']}: {dest['status']}")
    else:
        print(f"Distance matrix failed: {result['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # 7. Time Zone - Get timezone info
    print("7. Time Zone Information")
    timezone_tool = gmaps_toolkit.get_tool("timezone")
    result = timezone_tool(latitude=lat, longitude=lng)
    
    if result["success"]:
        print(f"Location: {result['latitude']}, {result['longitude']}")
        print(f"Time Zone: {result['time_zone_name']} ({result['time_zone_id']})")
        print(f"UTC Offset: {result['raw_offset']} seconds")
        print(f"DST Offset: {result['dst_offset']} seconds")
    else:
        print(f"Time zone lookup failed: {result['error']}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()