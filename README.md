
# Stock-Prediction-Model

## Overview

Stock-Prediction-Model is a full-stack, production-ready application that leverages advanced machine learning and deep learning techniques to forecast stock prices. Designed for both technical excellence and user experience, this project demonstrates expertise in Python, React, RESTful API design, and modern data science workflows. 

## Key Features

- **End-to-End ML Pipeline:** Integrates data acquisition, feature engineering, model training, and prediction in a seamless workflow.
- **Dual-Model Approach:** Implements both Linear Regression and LSTM (Long Short-Term Memory) neural networks for robust time series forecasting.
- **Dynamic Feature Engineering:** Utilizes technical indicators (RSI, Bollinger Bands, lagged features) to enhance predictive power.
- **Interactive React Frontend:** Clean, modern UI for parameter selection and real-time results visualization.
- **RESTful Flask API:** Efficient, scalable backend for serving predictions and handling user input.
- **Customizable Parameters:** Users can tune ticker, training period, epochs, batch size, and more for experimentation.
- **Data Visualization Ready:** (Optional) Easily extendable for charting and analytics.
- **Production-Grade Practices:** CORS-enabled, modular code, and clear separation of concerns.

## Architecture

**Backend:**
- Python (Flask, scikit-learn, TensorFlow/Keras, yfinance, tulipy)
- REST API endpoint `/api/predict` for model inference
- Real-time data fetching and feature engineering
- Model training (Linear Regression & LSTM) and next-day price prediction

**Frontend:**
- React (Functional Components, Hooks)
- Axios for API communication
- Responsive, user-friendly forms for parameter input
- Real-time display of model results (MSE, predictions)

## How It Works

1. **User Input:**
	- Enter stock ticker, start year, analysis period, and model hyperparameters in the React UI.
2. **API Request:**
	- Frontend sends a POST request to the Flask backend with user parameters.
3. **Data Pipeline:**
	- Backend fetches historical stock data, engineers features (RSI, Bollinger Bands, lagged values), and splits data for training/testing.
4. **Model Training:**
	- Trains both a Linear Regression and an LSTM model on the fly.
5. **Prediction:**
	- Returns next-day price predictions and model performance (MSE) for both models.
6. **Results Display:**
	- UI displays results instantly, enabling rapid experimentation and learning.

## Technologies Used

- **Frontend:** React, Axios, CSS
- **Backend:** Python, Flask, scikit-learn, TensorFlow/Keras, yfinance, tulipy, pandas, numpy
- **DevOps:** CORS, REST API, modular codebase

## Project Structure

```
src/
  main.py         # Flask API & ML pipeline
  home.jsx        # React UI (main page)
  index.jsx       # React entry point
  ...
public/           # Static assets (images, html)
```

## Quick Start

1. **Backend:**
	- Install Python dependencies: `pip install -r requirements.txt`
	- Run the Flask server: `python src/main.py`
2. **Frontend:**
	- Install Node dependencies: `npm install`
	- Start the React app: `npm start`