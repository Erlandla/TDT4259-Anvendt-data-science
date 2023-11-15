import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('consumption_temp.csv', delimiter=',')
# Create model predicting power consumption in oslo in the first instance
df = df[df["location"] == "oslo"]
# Convert first column to datetime. Additionaly set it as index and drop it from the dataframe
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
df = df.set_index(df.iloc[:, 0])
df.drop("time", axis=1, inplace=True)
df.drop("location", axis=1, inplace=True)

import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('consumption_temp.csv', delimiter=',')
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Create model predicting power consumption in oslo in the first instance
df = df[df["location"] == "oslo"]
# Convert first column to datetime. Additionaly set it as index and drop it from the dataframe
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
df = df.set_index(df.iloc[:, 0])
df.drop("time", axis=1, inplace=True)
df.drop("location", axis=1, inplace=True)

# Create new features based on timestamp. Important with e.g. hour to adhere to seasonality of power consumption. 
# Also important with month as the general consumption is higher in winter than in summer. 
df["hour"] = df.index.hour
df["month"] = df.index.month

#Lagged feature. Creates better predictions as the consumption is dependent on the previous consumption. 
#Currently shifting by 1 hour.
df["prev_consumption"] = df["consumption"].shift(1)
df.dropna(how='any', axis=0, inplace=True)

# Split into X and y and training and testing sets
y = df["consumption"]
X = df.drop("consumption", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=42, stratify=None)


# Testing with two different models, XGBoost and Random Forest. Both performing well
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=0, min_samples_split=3, min_samples_leaf=3)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#Calculating mean absolute error
mae = mean_absolute_error(y_test, predictions)

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


#Creating a dataframe with the feature importances
df_importances = pd.DataFrame({
    'feature':  model.feature_name_,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)


#plot variable importances of the model
plt.title('Variable Importances', fontsize=16)
sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
plt.show()