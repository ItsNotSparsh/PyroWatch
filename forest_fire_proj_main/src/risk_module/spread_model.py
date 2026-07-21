import math

def calculate_spread_cone(lat, lon, wind_speed_ms, wind_deg):
    """
    Calculates a predictive 'Danger Zone' polygon based on wind speed and direction.
    Returns a list of [lat, lon] coordinates forming the spread shape.
    """
    # 1. Fire moves OPPOSITE to where the wind comes from
    spread_heading = (wind_deg + 180) % 360

    # 2. Calculate Distance (Simulated mapping scale for the UI)
    # The higher the wind speed, the longer the cone.
    # We use 0.002 as a multiplier to convert wind speed into map coordinate distances.
    spread_distance = wind_speed_ms * 0.002 
    if spread_distance == 0:
        spread_distance = 0.001 # Even with zero wind, fire spreads outward slightly

    # 3. Calculate Cone Width (Spread Angle)
    # High wind = narrow cone (15 degrees). Low wind = wide cone (up to 60 degrees).
    spread_angle = max(15, 60 - wind_speed_ms)

    # 4. Convert degrees to radians for Python's math library
    heading_rad = math.radians(spread_heading)
    left_rad = math.radians((spread_heading - spread_angle) % 360)
    right_rad = math.radians((spread_heading + spread_angle) % 360)

    # 5. Calculate the specific Map Coordinates for the polygon vertices
    # Navigational math: Y (Lat) uses Cosine, X (Lon) uses Sine
    tip_lat = lat + (spread_distance * math.cos(heading_rad))
    tip_lon = lon + (spread_distance * math.sin(heading_rad))

    left_lat = lat + (spread_distance * 0.6 * math.cos(left_rad))
    left_lon = lon + (spread_distance * 0.6 * math.sin(left_rad))

    right_lat = lat + (spread_distance * 0.6 * math.cos(right_rad))
    right_lon = lon + (spread_distance * 0.6 * math.sin(right_rad))

    # Return the polygon coordinates: [Start, Left Flank, Tip, Right Flank, Back to Start]
    return [
        [lat, lon],
        [left_lat, left_lon],
        [tip_lat, tip_lon], 
        [right_lat, right_lon],
        [lat, lon]
    ]

# --- Quick Test ---
if __name__ == "__main__":
    test_poly = calculate_spread_cone(34.05, -118.24, 15, 180)
    print("Generated Danger Cone Coordinates:")
    for point in test_poly:
        print(f"Lat: {point[0]:.4f}, Lon: {point[1]:.4f}")