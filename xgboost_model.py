import preprocessing
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
def xgboost(df):

    
    scaler = StandardScaler()
    y = df["consumption"]
    X = df.drop("consumption", axis=1)
    tscv = TimeSeriesSplit(gap=120, n_splits=5, test_size=24)

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = xgb.XGBRegressor(eta= 0.3, min_child_weight= 1, max_depth= 6, gamma= 0, colsample_bytree= 1,subsample= 1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"MAE: {mae}")


    '''
    #Plotting real vs predicted values for a time period
    fig = plt.figure(figsize=(16,8))
    plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
    plt.plot(y_test, color='red')
    plt.plot(pd.Series(predictions, index=y_test.index), color='green')
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Consumption', fontsize=16)
    plt.legend(labels=['Real', 'Prediction'], fontsize=16)
    plt.grid()
    plt.show()

    '''
def main():
    df = pd.read_csv('consumption_temp.csv', delimiter=',')
    df_oslo = df[df["location"] == "oslo"]
    df_oslo = preprocessing.preprocess(df_oslo)
    xgboost(df_oslo)

if __name__ == '__main__':
    main()  
    