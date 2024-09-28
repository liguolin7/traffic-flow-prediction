# Intelligent Traffic Flow Prediction System

## Overview
The **Intelligent Traffic Flow Prediction System** leverages deep learning techniques to predict traffic flow in urban environments. By analyzing historical traffic data and weather conditions, the system helps in optimizing traffic management, reducing congestion, and promoting sustainable mobility.

## Table of Contents
- [Project Description](#project-description)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Streamlit App](#streamlit-app)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Project Description
The project predicts future traffic flow using a Long Short-Term Memory (LSTM) model. The model is trained on historical traffic data and weather conditions to make real-time predictions. The resulting predictions can help city planners and traffic management systems improve traffic efficiency and reduce travel times.

## Objectives
- Develop a predictive model for traffic flow using deep learning techniques.
- Integrate historical traffic data with real-time weather data.
- Visualize predictions and historical trends using a user-friendly web application.

## Key Features
- **Data Preprocessing**: Scripts to clean and prepare the dataset for training.
- **Model Development**: A robust LSTM model for time series forecasting.
- **Model Evaluation**: Performance metrics to assess prediction accuracy.
- **Web Application**: An interactive Streamlit app for traffic flow predictions.

## Data Sources
- Historical traffic data can be sourced from publicly available datasets such as:
  - [UCI Machine Learning Repository - Traffic Data](https://archive.ics.uci.edu/ml/datasets/traffic+volume+data+from+Caltrans)
  - [City of Chicago Data Portal](https://data.cityofchicago.org/)
- Weather data can be obtained from APIs like [OpenWeatherMap](https://openweathermap.org/api) or historical weather datasets.

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/traffic-flow-prediction.git
   cd traffic-flow-prediction
2. **Create a virtual environment:**

       python -m venv venv
       source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install dependencies:**

        pip install -r requirements.txt


## Usage

 ### Data Preparation

       python src/data_preprocessing.py
       
 ### Model Training

     python src/model_training.py

 ### Streamlit App

     streamlit run src/streamlit_app.py

# Model Training
The model uses an LSTM architecture to predict traffic flow. The training process involves:

1. Splitting the data into training and testing sets.
2. Fitting the model on the training data and evaluating it on the test data.
3. Calculating Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) as evaluation metrics.
   
# Streamlit App

The Streamlit app allows users to input traffic volume, temperature, and precipitation, and receive real-time predictions of future traffic flow. This interactive interface enhances usability for traffic planners and researchers.

# Future Enhancements

1. Incorporate real-time traffic data from sensors or APIs for live predictions.
2. Extend the model to include more features like public transportation data, special events, and road construction.
3. Develop visualizations for historical traffic trends using libraries like Matplotlib or Seaborn.
4. Implement more complex models such as GRU or CNN for time series forecasting.
   
# License

This project is licensed under the MIT License. See the LICENSE file for more details.
