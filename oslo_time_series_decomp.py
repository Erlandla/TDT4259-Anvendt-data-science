import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt

data = pd.read_csv("consumption_temp.csv")

data = data[data["location"] == "oslo"]

data.set_index("time", inplace=True)
data.index = pd.to_datetime(data.index)

# Drop null values
data.dropna(inplace=True)

result = seasonal_decompose(
    data["consumption"], model="multiplicative", period=24)
result.seasonal.plot()
result.trend.plot()
fig = result.plot()
plt.show()
