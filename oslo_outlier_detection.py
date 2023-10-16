import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("consumption_temp.csv")

data = data[data["location"] == "oslo"]
data.reset_index(inplace=True)
data = data.drop(["index"], axis=1)
print(data)

column = data["consumption"]

# Window size
window_percentage = 3
k = int(len(column) * (window_percentage/2/100))

N = len(column)


def get_bands(column):
    """Creates upper and lower boundaries on consumption based on the mean and standard 
        deviation of the data and returns it as a tuple: `(upper, lower)`"""
    return (np.mean(column) + 3 * np.std(column), np.mean(column) - 3 * np.std(column))


bands = [get_bands(column[range(0 if i-k < 0 else i-k, i+k if i+k < N else N)])
         for i in range(0, N)]
upper, lower = zip(*bands)

# Creates a list consisting containing True or False for every value in the assigned column,
# e.g., consumption value at index i in `column` is an outlier if `anomalies[i]` is True
anomalies = (column > upper) | (column < lower)

# Using the mean±3*std-method of creating bands creates two outliers on the 24.12.22, at 14:00 and 15:00
for i in range(len(anomalies)):
    if (anomalies[i] == True):
        print("ANOMALY on time " + data["time"][i] +
              " with value " + str(data["consumption"][i]))
