import os
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

class ModisLoader:
    def __init__(self):
        # Load environment variables from the .env file
        load_dotenv()
        self.api_key = os.getenv('NASA_FIRMS_KEY')
        self.base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
        
        if not self.api_key:
            raise ValueError("NASA FIRMS API key not found. Please check your .env file.")

    def fetch_latest_data(self, bounding_box="-180,-90,180,90", days=1):
        """
        Fetches active fire data for a specific area.
        Default bounding box is the whole world (-180,-90,180,90).
        
        Note: NASA FIRMS NRT data is limited to 10 days maximum.
        For longer periods, the API will return no data.
        """
        # Cap days at 10 for NRT data source
        if days > 10:
            print(f"Warning: NASA FIRMS NRT data is limited to 10 days. Capping request to 10 days.")
            days = 10
        
        print(f"Fetching MODIS data for the last {days} day(s)...")
        
        # NASA API URL structure: base_url / api_key / source / coordinates / days
        url = f"{self.base_url}/{self.api_key}/MODIS_NRT/{bounding_box}/{days}"
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Check if response has actual data
                if not response.text or len(response.text.strip()) == 0:
                    print("API returned empty response. No fire data available for this period.")
                    return None
                
                # Convert the raw text response into a Pandas DataFrame
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)
                
                if df.empty:
                    print("No fire records found for the specified time period and area.")
                    return None
                
                print(f"Successfully retrieved {len(df)} fire records.")
                return df
            else:
                print(f"Failed to fetch data. HTTP Status: {response.status_code}")
                if response.status_code == 400:
                    print("Bad request - check API key and parameters.")
                elif response.status_code == 403:
                    print("Access forbidden - check your API key.")
                elif response.status_code == 429:
                    print("Rate limit exceeded - try again later.")
                return None
        except requests.exceptions.Timeout:
            print("Request timed out. NASA FIRMS server may be slow or unavailable.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Network error: {str(e)}")
            return None
        except Exception as e:
            print(f"Error parsing data: {str(e)}")
            return None

    def preprocess_and_save(self, df, output_path="../data/processed/latest_hotspots.csv", confidence_threshold=70):
        """
        Filters out low-confidence anomalies and saves the clean data.
        """
        if df is None or df.empty:
            print("No data to process.")
            return
            
        # Filter for higher confidence fires
        clean_df = df[df['confidence'] >= confidence_threshold]
        
        # Select only the features we care about for the models later
        columns_to_keep = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'acq_date', 'acq_time', 'confidence', 'frp']
        clean_df = clean_df[columns_to_keep]
        
        # Ensure the processed directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        clean_df.to_csv(output_path, index=False)
        print(f"Saved {len(clean_df)} high-confidence hotspots to {output_path}")

# --- Quick Test Block ---
if __name__ == "__main__":
    loader = ModisLoader()
    raw_data = loader.fetch_latest_data(days=1)
    
    if raw_data is not None:
        # Save it to the processed folder
        loader.preprocess_and_save(raw_data, output_path="data/processed/latest_hotspots.csv")