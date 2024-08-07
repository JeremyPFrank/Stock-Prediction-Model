import yfinance as yf
import pandas as pd
import numpy as np
import tulipy as ti
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

import sys
import io
#fix char encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

#Flask RESTful API
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes

#@app.route('/api/predict', methods=['GET'])
@app.route('/api/predict', methods=['POST'])

#def input():
#    userInput = request.json
#    ticker = userInput.get(ticker,'aapl')
#    start = userInput.get(start,'2000')  
#    period = userInput.get(period,10)

def predict():
    #ticker = input("Enter Valid Stock Ticker: ")
    #start = input("Enter Start Year for Data: ")
    #start = start + '-01-01'
    #perioderiod = input("Enter Period Length For Financial Analysis (14 is Recommended): ")
    #period = 14
    #ticker = 'aapl'
    #start = '2004-01-01'
    
    userInput = request.get_json()
    ticker = userInput.get('ticker','aapl')
    if ticker is None:
        ticker = 'aapl'
        
    start = userInput.get('start','2000') 
    if start is None:
        start = '2000' 
        
    period = userInput.get('period',14)
    if period is None:
        period = '14'
    period = int(period)
    
    epochs = userInput.get('epoch',32)
    if epochs is None:
        epochs = '32'
    epochs = int(epochs)
    
    batchSize = userInput.get('batch', 32)
    if batchSize is None:
        batchSize = '32'
    batchSize = int(batchSize)
    
    trainingSize = userInput.get('setSize',0.5)
    if trainingSize is None:
        trainingSize = '0.5'
    trainingSize = float(trainingSize)
    testingSize = 1-trainingSize
    
    randomSeed = userInput.get('random',88)
    if randomSeed is None:
        randomSeed = None
    else:
        randomSeed = int(randomSeed)
    

    
    start = str(start) + '-01-01'
    current = datetime.now().date()

    data = yf.download(ticker,start,current)
    data = data[['Close','Volume']]

    tommorows_features = pd.DataFrame()

    RsiCalc = ti.rsi(data['Close'].values, period)

    #Fill first period Spaces with nulls then add rsi for rest of values
    data['RSI'] = pd.Series(np.concatenate([np.full(period, np.nan), RsiCalc]), data.index)

    #bband with user given period and default std deviation of 2
    upper_band, middle_band, lower_band = ti.bbands(data['Close'].values, period, 2)
    bollinger_gap = upper_band - lower_band

    data['upper_band'] = pd.Series(np.concatenate([np.full(period-1, np.nan), upper_band]), data.index)
    data['middle_band'] = pd.Series(np.concatenate([np.full(period-1, np.nan), middle_band]), data.index)
    data['lower_band'] = pd.Series(np.concatenate([np.full(period-1, np.nan), lower_band]), data.index)
    data['bollinger_gap'] = pd.Series(np.concatenate([np.full(period-1, np.nan), bollinger_gap]), data.index)

    #Work with lagged features from last x days to correlate with next days price
    #Create period periods of Lagged Features
    for lag in range(period):
        data['Close Lagged ' + str(lag+1) + ' Periods'] = data['Close'].shift(lag+1)
        data['Volume Lagged ' + str(lag+1) + ' Periods'] = data['Volume'].shift(lag+1)
        data['RSI Lagged ' + str(lag+1) + ' Periods'] = data['RSI'].shift(lag+1)
        data['Bband Gap Lagged ' + str(lag+1) + ' Periods'] = data['bollinger_gap'].shift(lag+1)

    #Drop all rows with NaN
    data = data.dropna()

    ###Begin Machine Learning###

    #Create test-train split with sklearn
    X = data.drop(columns=['Close','Volume','RSI','upper_band','middle_band','lower_band','bollinger_gap'])
    #X = data[['RSI Lagged 1 Periods','Volume Lagged 1 Periods']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testingSize,shuffle=False,random_state=randomSeed)

    #Linear Regression Model
    model = LinearRegression()
    model.fit(X_train,y_train)
    linear_predictions = model.predict(X_test)
    mse = mean_squared_error(y_test,linear_predictions)

    #Get Tommorrows Prediction
    current_day = data.iloc[-1:]

    tomorrow = current_day.copy()
    tomorrow.index = [current + timedelta(days=1)]

    tomorrow_features = tomorrow.drop(columns=['Close','Volume','RSI','upper_band','middle_band','lower_band','bollinger_gap'])
    #tomorrow_features = data[['RSI','Volume']]
    tomorrow_prediction_linear = model.predict(tomorrow_features)
    #Will be printed at bottom


    #Lets Try a Time Series LSTM Model Now!


    #X = data.drop(columns = ['Close']).values
    X = data.drop(columns=['Close','Volume','RSI','upper_band','middle_band','lower_band','bollinger_gap']).values
    y = data['Close'].values

    #Scale Data
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1))

    #Create sequences

    X_seq = []
    y_seq = []
    for i in range(len(data) - period):
        end = i + period
        X_seq.append(X[i:end])
        y_seq.append(y[end])
    X_seq,y_seq = np.array(X_seq), np.array(y_seq)


    X_train, X_test, y_train, y_test = train_test_split(X_seq,y_seq,test_size=testingSize,shuffle=False,random_state=randomSeed)

    LSTM_model = Sequential()
    LSTM_model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    LSTM_model.add(Dense(1))
    LSTM_model.compile(optimizer='adam', loss='mean_squared_error')

    history = LSTM_model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, validation_data=(X_test, y_test), verbose=2)

    # Make predictions
    predictions = LSTM_model.predict(X_test)

    #reverse scaling
    y_test_inv = scaler_y.inverse_transform(y_test)
    predictions_inv = scaler_y.inverse_transform(predictions)

    # Evaluate model
    LSTM_mse = mean_squared_error(y_test_inv, predictions_inv)
    
    retString = ""
    print('Linear Model MSE: ', mse)
    retString = retString + 'Linear Model MSE: ' + str(mse) + '\nLSTM Model MSE: ' + str(LSTM_mse)
    print('LSTM Model MSE: ', LSTM_mse)

    #Get Tommorrows Prediction

    current_day = X[-period:] #Get last sequence from X - X_seq???

    current_day = np.vstack([current_day, scaler_X.transform(tomorrow_features)])

    # Reshape to (1, period, number_of_features)
    current_day = current_day.reshape((1, current_day.shape[0], current_day.shape[1]))

    # Make LSTM prediction
    tomorrow_prediction_LSTM = LSTM_model.predict(current_day)

    # Reverse scaling of the LSTM prediction
    tomorrow_prediction_LSTM = scaler_y.inverse_transform(tomorrow_prediction_LSTM)



    print('Tomorrow\'s Linear Price: ' + str(tomorrow_prediction_linear))
    print('Tomorrow\'s LSTM Price: ' + str(tomorrow_prediction_LSTM))
    retString = retString + '\nTomorrow\'s Linear Price: ' + str(tomorrow_prediction_linear) + '\nTomorrow\'s LSTM Price: ' + str(tomorrow_prediction_LSTM)
    
    mse = np.float64(mse)
    LSTM_mse = np.float64(LSTM_mse)
    tomorrow_prediction_linear = np.float64(tomorrow_prediction_linear[0])
    tomorrow_prediction_LSTM = np.float64(tomorrow_prediction_LSTM[0][0])
    
    retList = {
        "lin_mse": mse,
        "lstm_mse": LSTM_mse,
        "lin_pred": tomorrow_prediction_linear,
        "lstm_pred": tomorrow_prediction_LSTM
    }
    #Plot results
    #plt.figure(figsize=(14, 5))
    #plt.plot(y_test_inv, label='True Closing Price')
    #plt.plot(predictions_inv, label='LSTM Model Prediction')
    #plt.plot(linear_predictions, label='Linear Model Prediction')
    #plt.legend()
    #plt.show()
    #print(data.tail(1))
    
    #return jsonify(message = retString)
    return Response(json.dumps(retList),  mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)
