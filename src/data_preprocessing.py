import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def load_data(traffic_file, weather_file):
    """Load traffic and weather data from CSV files."""
    # Load traffic and weather data without parsing 'timestamp' since it doesn't exist
    traffic_data = pd.read_csv(traffic_file)
    
    # Parse dates from the "Date.Full" column in weather data
    weather_data = pd.read_csv(weather_file, parse_dates=['Date.Full'])
    
    return traffic_data, weather_data

def preprocess_data(traffic_data, weather_data):
    """Preprocess the traffic and weather data."""
    # Rename the 'Date.Full' column in weather data to 'timestamp' for easier merging
    weather_data.rename(columns={'Date.Full': 'timestamp'}, inplace=True)
    
    # Create a synthetic timestamp column in traffic_data
    # (You may need to adjust this part based on how traffic and weather data are related)
    traffic_data['timestamp'] = weather_data['timestamp']

    # Merge datasets on 'timestamp'
    data = pd.merge(traffic_data, weather_data, on='timestamp', how='inner')
    
    # Fill missing values
    data.ffill(inplace=True)

    # Normalize features
    scaler = MinMaxScaler()

    # Use the correct column names based on the CSV files
    # For example, 'all_motor_vehicles' could represent traffic volume
    data[['all_motor_vehicles', 'Data.Temperature.Avg Temp', 'Data.Precipitation']] = scaler.fit_transform(
        data[['all_motor_vehicles', 'Data.Temperature.Avg Temp', 'Data.Precipitation']]
    )
    
    return data, scaler

if __name__ == '__main__':
    traffic_data, weather_data = load_data('data/traffic_data.csv', 'data/weather_data.csv')
    processed_data, scaler = preprocess_data(traffic_data, weather_data)
    processed_data.to_csv('data/processed_data.csv', index=False)
    print("Data preprocessing complete.")
