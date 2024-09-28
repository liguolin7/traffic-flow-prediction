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

