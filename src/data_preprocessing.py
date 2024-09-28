import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def load_data(traffic_file, weather_file):
    """Load traffic and weather data from CSV files."""
    traffic_data = pd.read_csv(traffic_file, parse_dates=['timestamp'])
    weather_data = pd.read_csv(weather_file, parse_dates=['timestamp'])
    return traffic_data, weather_data

def preprocess_data(traffic_data, weather_data):
    """Preprocess the traffic and weather data."""
    # Merge datasets on timestamp
    data = pd.merge(traffic_data, weather_data, on='timestamp', how='inner')
    
    # Fill missing values
    data.fillna(method='ffill', inplace=True)

    # Normalize features
    scaler = MinMaxScaler()
    data[['traffic_volume', 'temperature', 'precipitation']] = scaler.fit_transform(
        data[['traffic_volume', 'temperature', 'precipitation']]
    )
    
    return data, scaler

if __name__ == '__main__':
    traffic_data, weather_data = load_data('data/traffic_data.csv', 'data/weather_data.csv')
    processed_data, scaler = preprocess_data(traffic_data, weather_data)
    processed_data.to_csv('data/processed_data.csv', index=False)
    print("Data preprocessing complete.")
