import preprocessing
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


def random_forest(df, location):

    y = df["consumption"]
    X = df.drop("consumption", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False)

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

    # Plotting real vs predicted values for a time period
    fig = plt.figure(figsize=(16, 8))
    plt.title(f'Real vs Predicted - {location} - MAE {mae}', fontsize=20)
    plt.plot(y_test, color='red')
    plt.plot(pd.Series(predictions, index=y_test.index), color='green')
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Consumption', fontsize=16)
    plt.legend(labels=['Real', 'Prediction'], fontsize=16)
    plt.grid()
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
