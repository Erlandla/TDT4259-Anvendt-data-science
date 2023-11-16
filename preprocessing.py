import pandas as pd
import basic_moving_average as bma

def preprocess(df):
    
    # Outlier detection using z-score
    # Har oppdaget at det varierende mengder outliers. Oslo har ingen f.eks, mens de fleste andre har ~10 stk.
    z_scores = (df['temperature'] - df['temperature'].mean()) / df['temperature'].std()
    #print(df.shape[0])
    df = df[(z_scores < 3) & (z_scores > -3)]
    #print(df.shape[0])
    
    

    # Feature engineering
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.iloc[:, 0])
    df.drop("time", axis=1, inplace=True)
    df.drop("location", axis=1, inplace=True)
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df['dayofmonth'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    df["temperature"] = bma.moving_average(df["temperature"], 12)
    df["prev_consumption"] = df["consumption"].shift(168)
    df.dropna(how='any', axis=0, inplace=True)
    # Tror kanskje ikke det er lov å bruke denne
    # df["consumption2"] = bma.moving_average(df["consumption"], 3)


    return df


def main():
    df = pd.read_csv('consumption_temp.csv', delimiter=',')
    df_oslo = df[df["location"] == "trondheim"]
    preprocess(df_oslo)


if __name__ == "__main__":
    main()
