import preprocessing
import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


def random_forest(df, location):

    y = df["consumption"]
    X = df.drop("consumption", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=50,
        min_samples_split=2,
        min_samples_leaf=1
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mse)
    accuracy = 100 - np.mean(mape)

    smoother = 0.2
    plt.figure(figsize=(5, 7))
    ax = sns.kdeplot(y, color="r", label="Actual values", bw_adjust=smoother)
    sns.kdeplot(predictions, color="b",
                label="Predicted values", ax=ax, bw_adjust=smoother)
    plt.title(
        f"Kernel Density Estimate of real vs. predicted values for {location}")
    plt.ylabel("Density of values")
    plt.legend()
    plt.show()

    return f"""+------------ Random Forest: {location} ------------+
    Mean Absolute Error: {round(mae, 3)}
    Mean Absolute Percentage Error: {round(mape, 3)}
    Root Mean Square Error: {round(rmse, 3)}
    Accuracy score: {round(accuracy, 3)} %
    """


def main():
    df = pd.read_csv('consumption_temp.csv', delimiter=',')
    locations = ["oslo", "stavanger", "tromsø", "bergen", "trondheim"]
    results = []
    for location in locations:
        print(location)
        df_one_city = df[df["location"] == location]
        df_one_city = preprocessing.preprocess(df_one_city)
        results.append(random_forest(df_one_city, location))

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
