import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("consumption_temp.csv")
data_copy = data.copy()

data = data[data["location"] == "oslo"]
data = data.drop(["temperature"], axis=1)

data.set_index("time", inplace=True)
data.index = pd.to_datetime(data.index)


def moving_average(x, D):
    """
    Simple moving average taken from the slides of IT3212 Datadrevet programvare.

    Takes a column-dataframe `x` and a window size `D` and smooths out the values by taking the 
    average of every value datapoint within that window

    ### Params
    x: column, preferrably a DataFrame
    D: window size

    ### Returns
    arraylike consisting of the moved average values of the given column
    """
    h = [v for v in x[:D]]
    for p in range(len(x) - D):
        h.append(sum(x[p:p+D])/float(D))
    return h


def test():
    figure, axis = plt.subplots(4, 1)

    # Important resource for determining window size: https://medium.com/@thedatabeast/time-series-part-4-determining-the-window-size-for-moving-averages-a07c5cfcfac9
    # Some points from it:
    #   - Optimal window size based on trial and error. Using the validation set may help in this endeavour
    #   - Window size should be a multiple of the seasonal period to ensure that the seasonal pattern is captured
    #      properly
    #   - The window size should be chosen based on the characteristics of the time series and the forecasting
    #      problem at hand. There is no one-size-fits-all approach, and the optimal window size can vary depending
    #      on the data.
    #   - Longer window size: capture long-term trends
    #   - Small size: capture short-term fluctuations and changes
    #   - Ex.: If the time series has a yearly seasonality, a 12-month window size might be appropriate

    # --- Default ---
    axis[0].set_title("Default consumption values")
    axis[0].plot(data["consumption"])

    # --- Window size = 24 ---
    # This graph looks frighteningly similar to `seasonal_decompose()`'s trend-component
    axis[1].set_title("Consumption after moving average, window size = 24")
    axis[1].plot(data.index, moving_average(data["consumption"], 24))

    # --- Window size = 100 ---
    # Quite heavy smoothing. Notice the big spike on Christmas eve
    # (potential outlier mentioned in `oslo_outlier_detection.py`) is no longer there
    axis[2].set_title("Consumption after moving average, window size = 100")
    axis[2].plot(data.index, moving_average(data["consumption"], 100))

    # --- Window size = 500 ---
    # This seems to be too big of a window, as it smooths it wayy too much
    axis[3].set_title("Consumption after moving average, window size = 500")
    axis[3].plot(data.index, moving_average(data["consumption"], 500))

    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == "__main__":
    test()
