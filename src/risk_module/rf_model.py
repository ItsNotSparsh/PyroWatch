import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FireRiskModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def prepare_data(self, hotspots_path):
        print("Loading data and generating Wind features...")
        fires = pd.read_csv(hotspots_path)[['latitude', 'longitude']]
        fires['fire_occurred'] = 1
        
        # Simulating dangerous fire weather (Hot, Dry, HIGH WIND)
        fires['temperature_c'] = np.random.normal(35, 5, len(fires)) 
        fires['humidity_percent'] = np.random.normal(20, 10, len(fires))
        fires['wind_speed'] = np.random.normal(15, 5, len(fires)) # 15 m/s wind
        fires['wind_deg'] = np.random.uniform(0, 360, len(fires))
        
        no_fires = pd.DataFrame({
            'latitude': np.random.uniform(-90, 90, len(fires)),
            'longitude': np.random.uniform(-180, 180, len(fires)),
            'fire_occurred': 0,
            # Simulating safe weather (Cool, Wet, LOW WIND)
            'temperature_c': np.random.normal(20, 10, len(fires)),
            'humidity_percent': np.random.normal(60, 20, len(fires)),
            'wind_speed': np.random.normal(3, 2, len(fires)), # 3 m/s wind
            'wind_deg': np.random.uniform(0, 360, len(fires))
        })
        
        dataset = pd.concat([fires, no_fires]).sample(frac=1).reset_index(drop=True)
        
        # Now we have 6 distinct features!
        X = dataset[['latitude', 'longitude', 'temperature_c', 'humidity_percent', 'wind_speed', 'wind_deg']]
        y = dataset['fire_occurred']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        print("Training upgraded 6-Feature Random Forest...")
        self.model.fit(X_train, y_train)
        acc = accuracy_score(y_test, self.model.predict(X_test))
        print(f"Model Accuracy: {acc * 100:.2f}%")

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        print(f"Upgraded Model successfully saved to {save_path}")

if __name__ == "__main__":
    rf = FireRiskModel()
    # Path handling
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(ROOT_DIR, "data", "processed", "latest_hotspots.csv")
    model_save_path = os.path.join(ROOT_DIR, "models", "rf_risk_model.pkl")
    
    X_train, X_test, y_train, y_test = rf.prepare_data(data_path)
    rf.train_and_evaluate(X_train, X_test, y_train, y_test)
    rf.save_model(model_save_path)