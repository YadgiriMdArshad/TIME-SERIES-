# TIME-SERIES-
 ## time series forecasting techniques using Python. It includes data preprocessing, exploratory analysis, model building, and evaluation for forecasting future trends.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.api as smt
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

df = pd.read_csv("exchange_rate.csv", parse_dates=True, index_col='date')

df

df.isnull().sum()  # Checking for Null Values

df.plot(figsize=(20,10))  
plt.xlabel('Date')
plt.ylabel('Ex_rate')  # Checking the model whether it is multiplicative or additive

def test_stationary(timeseries):
    # Calculating the Rolling Mean and Rolling Standard Deviation
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # Plotting the figure to check if the data is stationary
    plt.plot(timeseries, color='black', label='Original')
    plt.plot(rolmean, color='blue', label='Rolling Mean')
    plt.plot(rolstd, color='grey', label='Rolling Std Dev')
    plt.legend(loc='best')

    # Hypothesis testing (ADF Test) for stationarity
    df_test = adfuller(timeseries)
    my_out = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for i, j in df_test[4].items():
        my_out["Critical Value (%s)" % i] = j
    print(my_out)

test_stationary(df['Ex_rate'])

df['Diff'] = df['Ex_rate'] - df['Ex_rate'].shift(30)  # Differencing to make data stationary

df.dropna(inplace=True)  # Removing missing values

# Checking stationarity again
test_stationary(df['Diff'])

fig, axes = plt.subplots(1, 2)
fig.set_figwidth(12)
fig.set_figheight(4)
smt.graphics.plot_pacf(df['Diff'], lags=20, ax=axes[0])
smt.graphics.plot_acf(df['Diff'], lags=20, ax=axes[1])
plt.tight_layout()

model = ARIMA(df['Ex_rate'], order=(2,2,1))  # Building ARIMA Model

result = model.fit()  # Fitting the model

forecast = result.forecast(20)  
forecast  # Predicting Future Values

df["Forecast"] = result.predict()  
df.dropna(inplace=True)

mean_absolute_error(df['Ex_rate'], df['Forecast'])

simple = SimpleExpSmoothing(df['Ex_rate']).fit(smoothing_level=0.5)

df["SES"] = simple.fittedvalues  

plt.plot(df['Ex_rate'], label='Actual')
plt.plot(df['SES'], label='Predicted')  
plt.plot(simple.forecast(20), label='Forecasting')
plt.legend(loc='best')
plt.show()

double = ExponentialSmoothing(df['Ex_rate'], trend='multiplicative').fit(smoothing_level=0.5, smoothing_trend=0.2)

df['DES'] = double.fittedvalues  

plt.plot(df['Ex_rate'], label='Actual')
plt.plot(df['DES'], label='Predicted')
plt.plot(double.forecast(20), label='Forecasting')
plt.legend(loc='best')
plt.show()

triple = ExponentialSmoothing(df['Ex_rate'], trend='multiplicative', seasonal='multiplicative', seasonal_periods=12).fit(
    smoothing_level=0.2, smoothing_trend=0.5, smoothing_seasonal=0.4)

df['TES'] = triple.fittedvalues  

rmse_triple = np.sqrt(mean_squared_error(df['Ex_rate'], df['TES'])).round(2)
mae_triple = mean_absolute_error(df['Ex_rate'], df['TES']).round(2)
mape_triple = round(mean_absolute_percentage_error(df['Ex_rate'], df['TES']) * 100, 2)

rmse_arima = np.sqrt(mean_squared_error(df['Ex_rate'], df['Forecast'])).round(2)
mae_arima = mean_absolute_error(df['Ex_rate'], df['Forecast']).round(2)
mape_arima = round(mean_absolute_percentage_error(df['Ex_rate'], df['Forecast']) * 100, 2)

results = pd.DataFrame({
    'Method': ['RMSE', 'MAE', 'MAPE'],
    'Exponential_Smoothing': [rmse_triple, mae_triple, mape_triple],
    'ARIMA': [rmse_arima, mae_arima, mape_arima]
})

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Evaluation and Comparison', size=20, y=1, fontweight='bold', alpha=0.7)
sns.barplot(ax=axes[0], x=results.Method, y=results.Exponential_Smoothing)
axes[0].set_title('Exponential Smoothing')
sns.barplot(ax=axes[1], x=results.Method, y=results.ARIMA)
axes[1].set_title('ARIMA')
plt.tight_layout()
