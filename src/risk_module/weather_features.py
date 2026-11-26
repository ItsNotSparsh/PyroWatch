from datetime import datetime, timezone
import os
import requests
from dotenv import load_dotenv

class WeatherFetcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OPENWEATHER_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
        if not self.api_key:
            print("Warning: OPENWEATHER_KEY not found in .env file.")

    def get_forecast(self, lat, lon):
        if not self.api_key:
            return []
        try:
            response = requests.get("https://api.openweathermap.org/data/2.5/forecast",
                                    params={"lat": lat, "lon": lon, "appid": self.api_key,
                                            "units": "metric", "cnt": 8}, timeout=10)
            response.raise_for_status()
            return response.json()["list"]
        except (requests.RequestException, ValueError, KeyError):
            return []

    def get_current_weather(self, lat, lon):
        reading = self.get_current_reading(lat, lon)
        return reading["values"] if reading else (None, None, None, None)

    def get_current_reading(self, lat, lon):
        if not self.api_key:
            return None
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"}
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            wind = data["wind"]
            values = (float(data["main"]["temp"]), float(data["main"]["humidity"]),
                      float(wind["speed"]), wind.get("deg"))
            timestamp = datetime.fromtimestamp(data["dt"], timezone.utc)
            return {"coordinates": (lat, lon), "place": data.get("name") or "Selected location",
                    "time": timestamp, "values": values}
        except (requests.RequestException, ValueError, TypeError, KeyError, OverflowError):
            return None

# --- Quick Test Block ---
if __name__ == "__main__":
    fetcher = WeatherFetcher()
    test_lat, test_lon = 34.05, -118.24
    print(f"Fetching weather for Lat: {test_lat}, Lon: {test_lon}...")
    
    t, h, ws, wd = fetcher.get_current_weather(test_lat, test_lon)
    if t is not None:
        print(f"Success! Temp: {t}°C | Hum: {h}% | Wind: {ws}m/s at {wd}°")