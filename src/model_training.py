import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler  
import joblib

def create_dataset(data, time_step=1):
    """Create dataset with time steps."""
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :-1])  # all columns except the last
        y.append(data[i + time_step, -1])  # only the last column (target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_model(input_shape):
    """Build and compile LSTM model."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == '__main__':
    # Load processed data
    data = pd.read_csv('data/processed_data.csv')

    # Drop non-numeric columns
    non_numeric_columns = ['timestamp', 'region_name', 'Station.City', 
                           'Station.Code', 'Station.Location', 
                           'Station.State', 'region_ons_code', 
                           'Date.Month', 'Date.Week of', 'Date.Year','Data.Precipitation']
    data = data.drop(columns=non_numeric_columns, errors='ignore')  # Drop non-numeric columns

    # Initialize and fit the scaler
    scaler = MinMaxScaler()  # Create a scaler object
    data_scaled = scaler.fit_transform(data)  # Fit and transform the data

    # Create the dataset for LSTM
    X, y = create_dataset(data_scaled)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model((X_train.shape[0], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Save the model
    model.save('models/traffic_model.h5')
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)

    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')  # Save the fitted scaler
