# Simple-Time-Series-Forecasting-Using-RNN
Create a simple forecast for temperature prediction


#Simulation of 400 days of temperature data. -> using the same example as attached create a time series forcast using the temperature data 

days = np.arange(400)
temp = 28 + 3*np.sin(2*np.pi*days/30) + 0.8*np.random.randn(len(days))  # ~monthly seasonality
df = pd.DataFrame({"day": days, "temp": temp})

print("Normal DF (head):")
print(df.head())
