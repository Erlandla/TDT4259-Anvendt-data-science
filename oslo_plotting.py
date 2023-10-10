import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("consumption_temp.csv")

# Dicts used for plotting
oslo_time_consumption = dict()
oslo_time_temp = dict()
oslo_temp_consumption = dict()

# Not used, as I focus on Oslo
bergen_consumption = dict()
bergen_temp = dict()
trondheim_consumption = dict()
trondheim_temp = dict()
stavanger_consumption = dict()
stavanger_temp = dict()
tromso_consumption = dict()
tromso_temp = dict()
helsingfors_consumption = dict()
helsingfors_temp = dict()

# Setup dicts
for i in range(len(data["time"])):
    if (data["location"][i] == "oslo"):
        oslo_time_temp[data["time"][i]] = data["temperature"][i]
        oslo_temp_consumption[data["temperature"][i]] = data["consumption"][i]
        oslo_time_consumption[data["time"][i]] = data["consumption"][i]
    elif (data["location"][i] == "bergen"):
        bergen_temp[data["time"][i]] = data["temperature"][i]
        bergen_consumption[data["time"][i]] = data["consumption"][i]
    elif (data["location"][i] == "trondheim"):
        trondheim_temp[data["time"][i]] = data["temperature"][i]
        trondheim_consumption[data["time"][i]] = data["consumption"][i]
    elif (data["location"][i] == "stavanger"):
        stavanger_temp[data["time"][i]] = data["temperature"][i]
        stavanger_consumption[data["time"][i]] = data["consumption"][i]
    elif (data["location"][i] == "tromsø"):
        tromso_temp[data["time"][i]] = data["temperature"][i]
        tromso_consumption[data["time"][i]] = data["consumption"][i]
    elif (data["location"][i] == "helsingfors"):
        helsingfors_temp[data["time"][i]] = data["temperature"][i]
        helsingfors_consumption[data["time"][i]] = data["consumption"][i]


def plot_temperature_per_city():
    time = oslo_time_temp.keys()
    temp = oslo_time_temp.values()

    plt.title("Temperature in Oslo, 07.04.22 21:00 to 02.04.22 21:00")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")

    ticks = ["07.04.22", "07.05.22", "07.06.22", "07.07.22", "07.08.22",
             "07.09.22", "07.10.22", "07.11.22", "07.12.22", "07.01.23",
             "07.02.23", "07.03.23", "02.04.23"]
    # magic numbers here are the indices of the seventh day of each month
    plt.xticks(
        labels=ticks,
        ticks=[0, 720, 1464, 2184,
               2928, 3672, 4392,
               5136, 5856, 6600,
               7344, 8016, 8641],
    )

    plt.plot(time, temp)
    plt.show()


def plot_consumption_per_city():
    time = oslo_time_consumption.keys()
    consumption = oslo_time_consumption.values()

    plt.title("Energy consumption in Oslo, 07.04.22 21:00 to 02.04.22 21:00")
    plt.xlabel("Time")
    plt.ylabel("Energy consumption (kWh)")

    ticks = ["07.04.22", "07.05.22", "07.06.22", "07.07.22", "07.08.22",
             "07.09.22", "07.10.22", "07.11.22", "07.12.22", "07.01.22",
             "07.02.22", "07.03.22", "02.04.22"]

    plt.xticks(
        labels=ticks,
        ticks=[0, 720, 1464, 2184,
               2928, 3672, 4392,
               5136, 5856, 6600,
               7344, 8016, 8641]
    )

    plt.plot(time, consumption)
    plt.show()


def plot_temperature_consumption():
    temp = oslo_temp_consumption.keys()
    consumption = oslo_temp_consumption.values()

    plt.title("Temperature-consumption graph")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Consumption (kWh)")
    plt.scatter(x=temp, y=consumption, alpha=0.8, linewidths=0.2)
    plt.show()


def main():
    # plot_temperature_per_city()
    # plot_consumption_per_city()
    plot_temperature_consumption()


if __name__ == "__main__":
    main()
