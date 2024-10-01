import pandas as pd
import numpy as np
from keras.models import load_model
import joblib

def load_and_prepare_data(scaler, new_data):
    """Prepare new data for prediction."""
    new_data = new_data.reshape(1, -1)  # Reshape to (1, number_of_features)
    new_data_scaled = scaler.transform(new_data)  # Scale the data
    return new_data_scaled.reshape(1, 1, new_data_scaled.shape[1])  # Reshape for LSTM input

if __name__ == '__main__':
    # Load the trained model and scaler
    model = load_model('models/traffic_model.h5')
    scaler = joblib.load('models/scaler.pkl')

    # Print model summary to check input shape
    print("Model Summary:")
    model.summary()

    # Prepare new_data with exactly the same number of features used for training
    # Make sure this array has 17 features
    new_data = np.array([300, 15, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,0])  # This should have 17 features

    

    # Prepare the data
    prepared_data = load_and_prepare_data(scaler, new_data)

    # Make prediction
    prediction = model.predict(prepared_data)
    print("Predicted traffic volume:", prediction[0][0])
