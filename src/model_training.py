import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def create_dataset(data, time_step=1):
    """Create dataset with time steps."""
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :-1])
        y.append(data[i + time_step, -1])
    return np.array(X), np.array(y)

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
    data = pd.read_csv('data/processed_data.csv').values
    X, y = create_dataset(data)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Save the model
    model.save('models/traffic_model.h5')
    
    # Evaluate the model
    predictions = model.predict(X_test)
    print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
    print("Root Mean Squared Error:", mean_squared_error(y_test, predictions, squared=False))

    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
