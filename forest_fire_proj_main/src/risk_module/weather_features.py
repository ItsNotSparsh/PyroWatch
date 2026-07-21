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

    def get_current_weather(self, lat, lon):
        """
        Fetches real-time temp, humidity, wind speed, and wind direction.
        """
        if not self.api_key:
            return None, None, None, None
            
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric' 
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status() 
            
            data = response.json()
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed'] # Meters per second
            wind_deg = data['wind'].get('deg', 0) # Degrees (0 is North)
            
            return temp, humidity, wind_speed, wind_deg
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None, None, None, None

# --- Quick Test Block ---
if __name__ == "__main__":
    fetcher = WeatherFetcher()
    test_lat, test_lon = 34.05, -118.24
    print(f"Fetching weather for Lat: {test_lat}, Lon: {test_lon}...")
    
    t, h, ws, wd = fetcher.get_current_weather(test_lat, test_lon)
    if t is not None:
        print(f"Success! Temp: {t}°C | Hum: {h}% | Wind: {ws}m/s at {wd}°")