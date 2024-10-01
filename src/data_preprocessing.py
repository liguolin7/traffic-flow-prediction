import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import joblib

def load_data(traffic_file, weather_file):
    """Load traffic and weather data from CSV files."""
    # Load traffic and weather data
    traffic_data = pd.read_csv(traffic_file)
    weather_data = pd.read_csv(weather_file, parse_dates=['Date.Full'])
    
    # Print the shape of the loaded data
    print("Traffic Data Shape:", traffic_data.shape)
    print("Weather Data Shape:", weather_data.shape)

    return traffic_data, weather_data

def preprocess_data(traffic_data, weather_data):
    """Preprocess the traffic and weather data."""
    # Rename the 'Date.Full' column in weather data to 'timestamp' for easier merging
    weather_data.rename(columns={'Date.Full': 'timestamp'}, inplace=True)

    # Create a synthetic timestamp column in traffic_data that matches the date range of weather_data
    start_date = weather_data['timestamp'].min()
    end_date = weather_data['timestamp'].max()
    traffic_data['timestamp'] = pd.date_range(start=start_date, end=end_date, periods=len(traffic_data))

    # Print the unique timestamps for debugging
    print("Traffic Data Timestamps:", traffic_data['timestamp'].unique())
    print("Weather Data Timestamps:", weather_data['timestamp'].unique())

    # Merge datasets on 'timestamp'
    data = pd.merge(traffic_data, weather_data, on='timestamp', how='inner')

    # Check if the merged data is empty
    if data.empty:
        raise ValueError("Merged data is empty. Please check your input files.")
    
    # Fill missing values
    data.ffill(inplace=True)

    # Check for missing values in the columns of interest
    if data[['all_motor_vehicles', 'Data.Temperature.Avg Temp', 'Data.Precipitation']].isnull().values.any():
        print("Warning: There are missing values in the features to be normalized. Filling them before scaling.")
        data.fillna(method='ffill', inplace=True)

    # Normalize features
    scaler = MinMaxScaler()
    data[['all_motor_vehicles', 'Data.Temperature.Avg Temp', 'Data.Precipitation']] = scaler.fit_transform(
        data[['all_motor_vehicles', 'Data.Temperature.Avg Temp', 'Data.Precipitation']]
    )
    
    return data, scaler
