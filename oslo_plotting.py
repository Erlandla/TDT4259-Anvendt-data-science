import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("consumption_temp.csv")

# Focus on data from Oslo
data = data[data["location"] == "oslo"]

dateTime = data["time"]
temp = data["temperature"]
consumption = data["consumption"]

data.set_index("time", inplace=True)
data.index = pd.to_datetime(data.index)


def plot_temp():
    temp_data = data.copy()
    temp_data = temp_data.drop(["consumption"], axis=1)
    temp_data.plot()


def plot_consumption():
    consumption_data = data.copy()
    consumption_data = consumption_data.drop(["temperature"], axis=1)
    consumption_data.plot()


def plot_temp_consumption():
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Consumption (kWh1)")
    plt.scatter(x=temp, y=consumption, alpha=0.8, linewidths=0.2)
    plt.show()


def main():
    plot_temp()
    plot_consumption()
    plot_temp_consumption()
    plt.show()


if __name__ == "__main__":
    main()
