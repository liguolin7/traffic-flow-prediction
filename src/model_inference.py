import pandas as pd
import numpy as np
from keras.models import load_model
import joblib

def load_and_prepare_data(scaler, new_data):
    """Prepare new data for prediction."""
    # Ensure new_data is 2D before scaling
    new_data = new_data.reshape(1, -1)  # Reshape to (1, number_of_features)
    new_data_scaled = scaler.transform(new_data)  # Scale the data
    # Ensure shape is (1, 1, number_of_features)
    return new_data_scaled.reshape(1, 1, new_data_scaled.shape[1])  

if __name__ == '__main__':
    # Load the trained model and scaler
    model = load_model('models/traffic_model.h5')
    scaler = joblib.load('models/scaler.pkl')

    # Print model summary to check input shape
    print("Model Summary:")
    model.summary()

    # Adjust new_data to have exactly 17 features
    new_data = np.array([300.0, 15.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 20.0])  # 17 features

    # Check the shape of new_data
    print(f"Shape of new_data: {new_data.shape}")  # Should print (17,)

    prepared_data = load_and_prepare_data(scaler, new_data)

    # Print prepared data shape
    print(f"Shape of prepared_data: {prepared_data.shape}")  # Should print (1, 1, 17)

    # Make prediction
    prediction = model.predict(prepared_data)
    print("Predicted traffic volume:", prediction[0][0])
