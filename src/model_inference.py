import pandas as pd
import numpy as np
from keras.models import load_model
import joblib

def load_and_prepare_data(scaler, new_data):
    """Prepare new data for prediction."""
    new_data_scaled = scaler.transform(new_data)
    return new_data_scaled.reshape(1, new_data_scaled.shape[0], new_data_scaled.shape[1])

if __name__ == '__main__':
    # Load the trained model and scaler
    model = load_model('models/traffic_model.h5')
    scaler = joblib.load('models/scaler.pkl')

    # Example input data for prediction (customize as needed)
    new_data = np.array([[300, 15, 0]])  # traffic_volume, temperature, precipitation
    prepared_data = load_and_prepare_data(scaler, new_data)

    # Make prediction
    prediction = model.predict(prepared_data)
    print("Predicted traffic volume:", prediction[0][0])
