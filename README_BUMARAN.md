**This project is about forecasting temperature values using RNN**

1.) For this assignment , below libraries are used :
    -import numpy as np
    -import pandas as pd
    -import matplotlib.pyplot as plt
    -import tensorflow as tf
    -import random
    -from sklearn.preprocessing import MinMaxScaler
    -from tensorflow.keras import Sequential#T o deal with sequential data
    -from tensorflow.keras.layers import SimpleRNN, Dense
    -from sklearn.model_selection import train_test_split

2.) Create a dummy time series data as below and create a DataFrame (2D array) :
    days = np.arange(400) # 400 points/days
    temp = 28 + 3*np.sin(2*np.pi*days/30) # base signal
    temp += 0.8*np.random.randn(len(days)) # noise
    df = pd.DataFrame({"day": days, "temp": temp}) # data frame

3.) Next we have to scale it using MinMaxScaler. MinMaxScaer expects 2D array data. So we fed in with df[["temp"]]
    scaler = MinMaxScaler()
    temp_scaled = scaler.fit_transform(df[['temp']].values)

4.) Next, we create make_sequence function as below :
    def make_sequence(arr,win=30)
    - This function is passed with 2D data (arr), shape 400,1
    - The window is set to 30 (win = 30 -> sliding window size (monthly = 30 days, often used for “last 30 days” in monthly data).
    - First we create empty list of X,y
    - During slicing X.append(arr[i:i+win]) , the shape is (30,1) --> (2D) array
    - During indexing y.append(arr[i+win]), the shape is (1,) --> (1D) array
    - After the loop and when performing this  X = np.array(X), the shape becomes (370,30,1) --> 3D array (exactly what needed by RNN, 3D)
    - when performing this y = np.array(y).flatten(), the shape changes from (370,1) to (370,) --> (1D) array (exactly the target value needed by RNN, 1D)

5.) Next, we create the input features and the output labels with the time series data.
    -X,y = make_sequence(temp_scaled) # keep the array returned by the function

6.) Then, we perform Train validation split

7.) Next, we Build the RNN Model

8.) Followed by Compile the function

9.) Next we Train the model

10.) Then, we evaluate the model. Based on evaluation, the MSE value is 0.0072. MSE value near to zero indicates Predictions are very accurate (errors are very small)

11.) Next we plot Plot training and validation loss / metrics. It is normal for the plot looks different in the beginning because at the beginning the model is untrained, so        it has no knowledge of patterns in either train set or val set. As training progresses Training loss decreases and Validation loss decreases. Eventually, if model is           stable, train and val curves converge.

12.) Next, we create the forecast from last window. We create a function like below :

    def forcast_multi(last_window_scaled,d) --> This function returns preds

    d = 50
    last_window_scaled = X_test[-1].flatten() --> X_test[-1] retrieves the last window size of 30 (samples) from the test dataset -->2D array (30,1), after flatten, it becomes     1D
    future = forcast_multi(last_window_scaled,d) # future becomes preds ( the return of function forecast_multi)

13.) Next, plot the value and look at the graph for visualization. The visualization looks perfect with all past data linked together.
