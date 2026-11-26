from geopy.distance import geodesic


def calculate_spread_cone(lat, lon, wind_speed_ms, wind_deg, hours=1):
    """Illustrative downwind displacement at one percent of wind speed."""
    distance_m = wind_speed_ms * 0.01 * hours * 3600
    heading = (wind_deg + 180) % 360
    origin = (lat, lon)
    points = [origin]
    for offset, fraction in [(-30, 0.6), (0, 1), (30, 0.6)]:
        point = geodesic(meters=distance_m * fraction).destination(origin, heading + offset)
        points.append((point.latitude, point.longitude))
    return points + [origin]
