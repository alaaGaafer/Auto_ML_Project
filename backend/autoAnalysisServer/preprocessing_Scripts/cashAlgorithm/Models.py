import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
# from prophet import Prophet
# Suppress warnings
warnings.filterwarnings("ignore")

# ARIMA model class
class ARIMAModel:
    def __init__(self):
        self.model = None

    @staticmethod
    def Arimasmac(train, test, p, d, q, freq='D'):
        
        train = train.asfreq(freq)
        test = test.asfreq(freq)
        


        model_fit = ARIMA(train['y'], order=(p, d, q)).fit()
        forecast_values = model_fit.forecast(steps=len(test))
        # print ("the forecast values: ",forecast_values)
        # print("the test values: ",test['y'])

        # print("the forcast abs minus the test: ",np.abs((forecast_values - test['y'])))
        print("the meanloss: ",np.mean(np.abs((forecast_values - test['y']) / test['y']) * 100))
        return np.mean(np.abs((forecast_values - test['y']) / test['y'])) * 100

    def fit(self, train, p, d, q, freq='D'):
        train = train.asfreq(freq)
        model = ARIMA(train['y'], order=(p, d, q))
        self.model = model.fit()
        return self.model

    def predict(self, steps):
        return self.model.forecast(steps=steps)

# SARIMA model class
class SARIMAModel:
    def __init__(self):
        self.model = None
    
    @staticmethod
    def Sarimasmac(train, test, p, q, d, P, Q, D, s, freq='D'):
        train = train.asfreq(freq)
        test = test.asfreq(freq)
        model = SARIMAX(train, order=(p, q, d), seasonal_order=(P, Q, D, s), freq=freq)
        model_fit = model.fit(disp=False)  # Suppress the fitting output
        forecast_values = model_fit.forecast(steps=len(test))
        return np.mean(np.abs((forecast_values - test['y']) / test['y'])) * 100
    # @staticmethod
    def fit_with_tests(self, train_data, p, q, d, P, Q, D, s, freq='D'):
        try:
            # Perform Augmented Dickey-Fuller test for stationarity
            # print("order are",p, q, d,P, Q, D, s)
            adf_test = adfuller(train_data)
            adf_statistic = adf_test[0]
            adf_pvalue = adf_test[1]
            enforce_stationarity = adf_pvalue < 0.05

            train_data = train_data.asfreq(freq)

            # Fit the SARIMAX model
            model = SARIMAX(train_data['y'], order=(p, d, q), seasonal_order=(P, Q, D, s),
                            freq=freq, enforce_stationarity=enforce_stationarity)
            self.model = model.fit(disp=False)  # Suppress the fitting output
            return self.model
        except Exception as e:
            print(f"Error in fit_with_tests: {e}")
            return None

    def predict(self, steps):
        if self.model is not None:
            return self.model.forecast(steps=steps)
        else:
            raise ValueError("Model has not been fitted yet. Call `fit_with_tests` before predicting.")

# Prophet model class
class ProphetModel:
    def __init__(self, holidays=None, seasonality_mode=None, seasonality_prior_scale=None):
        self.model = Prophet(holidays=holidays, seasonality_mode=seasonality_mode,
                             seasonality_prior_scale=seasonality_prior_scale)

    def add_country_holidays(self, country_name):
        self.model.add_country_holidays(country_name=country_name)

    def fit(self, train_data):
        self.model.fit(train_data)

    def predict(self, future_data):
        forecast = self.model.predict(future_data)
        return forecast[['ds', 'yhat']]

# if __name__ == "__main__":
#     path = r"C:\Users\Alaa\Downloads\daily-minimum-temperatures-in-me.csv"
#     df = pd.read_csv(path)
#     df.rename(columns={'Date': 'ds', 'Daily minimum temperatures': 'y'}, inplace=True)
#     df['ds'] = pd.to_datetime(df['ds'], format='%m/%d/%Y')
#     df['y'] = df['y'].str.replace('[^0-9\.]', '', regex=True)
#     df['y'] = pd.to_numeric(df['y'], errors='coerce')
#     df['y'] = df['y'].astype(float)
#     df.set_index('ds', inplace=True)
#     split_date = pd.to_datetime('1990-12-15')
#     train_data = df[df.index <= split_date]
#     test_data = df[df.index > split_date]
#
#     sarima_model = SARIMAModel()
#     loss = sarima_model.Sarimasmac(train_data, test_data, 1, 1, 1, 1, 1, 1, 7, freq='D')
#     print(f"Loss: {loss}")
#
#     model = sarima_model.fit_with_tests(train_data, 1, 1, 1, 1, 1, 1, 7, freq='D')
#     if model is not None:
#         print("Model fitted successfully.")
#     else:
#         print("Model fitting failed.")
