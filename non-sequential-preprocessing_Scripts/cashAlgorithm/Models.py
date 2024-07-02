import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from prophet import Prophet
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBClassifier, XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ARIMA model class
class ARIMAModel:
    def __init__(self):
    #     self.order = order
        self.model = None

    def Arimasmac(train,test, p, d, q,freq='d'):
        train = train.asfreq(freq)
        test = test.asfreq(freq)
        
        
        model_fit = ARIMA(train['y'], order=(p, d, q))
        model_fit = model_fit.fit()
        forecast_values = model_fit.forecast(steps=len(test))
            
        return  np.mean(np.abs((forecast_values - test['y']) / test['y'])) * 100
    def fit(self, train,p,d,q,freq='d'):
        # print(train)
        # print(q)
        train = train.asfreq(freq)
        model = ARIMA(train['y'], order=(p,d,q))
        modelfit = model.fit()
        self.model = modelfit
        return self.model
    def predict(self, steps):
        return self.model.forecast(steps=steps)


# SARIMA model class
class SARIMAModel:
    def __init__(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.enforce_stationarity = False
        self.enforce_invertibility = False

    def fit(self, train_data):
        # Perform Augmented Dickey-Fuller test for stationarity
        adf_test = adfuller(train_data)
        adf_statistic = adf_test[0]
        adf_pvalue = adf_test[1]

        # Check if stationarity should be enforced
        if adf_pvalue < 0.05:
            self.enforce_stationarity = True
            print("Augmented Dickey-Fuller Test: Series is not stationary. Enforcing stationarity.")
        else:
            print("Augmented Dickey-Fuller Test: Series is stationary.")

        # Fit the SARIMAX model
        self.model = SARIMAX(train_data, order=self.order, seasonal_order=self.seasonal_order,
                             enforce_stationarity=self.enforce_stationarity,
                             enforce_invertibility=self.enforce_invertibility).fit(disp=False)

        # Perform Ljung-Box test for residuals autocorrelation
        residuals = self.model.resid
        lb_test = acorr_ljungbox(residuals, lags=10)
        lb_statistic = lb_test[0][0]
        lb_pvalue = lb_test[1][0]

        # Check if invertibility should be enforced
        if lb_pvalue < 0.05:
            self.enforce_invertibility = True
            print("Ljung-Box Test: Residuals exhibit autocorrelation. Enforcing invertibility.")
        else:
            print("Ljung-Box Test: Residuals do not exhibit significant autocorrelation.")

        # Print Durbin-Watson statistic for residual autocorrelation
        dw_statistic = durbin_watson(residuals)
        print(f"Durbin-Watson Statistic: {dw_statistic}")

    def predict(self, steps):
        return self.model.forecast(steps=steps)


# Prophet model class
class ProphetModel:
    # ['additive', 'multiplicative']
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




