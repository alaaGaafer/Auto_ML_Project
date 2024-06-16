import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

# Base class for all models
class BaseModel:
    def fit(self, train_data):
        raise NotImplementedError

    def predict(self, test_data):
        raise NotImplementedError

# ARIMA model class
class ARIMAModel(BaseModel):
    def __init__(self, order):
        self.order = order
        self.model = None

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=self.order).fit()

    def predict(self, steps):
        return self.model.forecast(steps=steps)

# SARIMA model class
class SARIMAModel(BaseModel):
    def __init__(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def fit(self, train_data):
        self.model = SARIMAX(train_data, order=self.order, seasonal_order=self.seasonal_order,
                             enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    def predict(self, steps):
        return self.model.forecast(steps=steps)

# Prophet model class
class ProphetModel(BaseModel):
    def __init__(self):
        self.model = Prophet()

    def fit(self, train_data):
        self.model.fit(train_data)

    def predict(self, future_data):
        forecast = self.model.predict(future_data)
        return forecast[['ds', 'yhat']]

# Model Manager class
class ModelManager:
    def __init__(self):
        self.models = {}

    def register_model(self, name, model):
        self.models[name] = model

    def get_model(self, name):
        model = self.models.get(name)
        if not model:
            raise ValueError(f"Model {name} is not registered.")
        return model

    def fit_model(self, name, train_data):
        model = self.get_model(name)
        model.fit(train_data)

    def predict_model(self, name, test_data):
        model = self.get_model(name)
        return model.predict(test_data)

def initialize_and_register_arima(model_manager):
    arima_order = (1, 1, 1)
    arima_model = ARIMAModel(arima_order)
    model_manager.register_model("ARIMA", arima_model)

def initialize_and_register_sarima(model_manager):
    sarima_order = (1, 1, 1)
    sarima_seasonal_order = (1, 1, 1, 12)
    sarima_model = SARIMAModel(sarima_order, sarima_seasonal_order)
    model_manager.register_model("SARIMA", sarima_model)

def initialize_and_register_prophet(model_manager):
    prophet_model = ProphetModel()
    model_manager.register_model("Prophet", prophet_model)

def fit_and_predict_arima(model_manager, train_data, test_length):
    initialize_and_register_arima(model_manager)
    model_manager.fit_model("ARIMA", train_data)
    return model_manager.predict_model("ARIMA", test_length)

def fit_and_predict_sarima(model_manager, train_data, test_length):
    initialize_and_register_sarima(model_manager)
    model_manager.fit_model("SARIMA", train_data)
    return model_manager.predict_model("SARIMA", test_length)

def fit_and_predict_prophet(model_manager, train_data, future_data):
    initialize_and_register_prophet(model_manager)
    model_manager.fit_model("Prophet", train_data)
    return model_manager.predict_model("Prophet", future_data)

def prepare_prophet_data(train_df, test_df, product):
    prophet_train_data = train_df[train_df['Product Description'] == product][['Date', 'value']].rename(
        columns={'Date': 'ds', 'value': 'y'})
    prophet_future_data = test_df[test_df['Product Description'] == product][['Date']].rename(columns={'Date': 'ds'})
    return prophet_train_data, prophet_future_data

#usage in the other script
# model_manager = ModelManager()

# ARIMA model
# arima_predictions = fit_and_predict_arima(model_manager, train_data, len(test_data))
# print("ARIMA Predictions:", arima_predictions)
#
# SARIMA model
# sarima_predictions = fit_and_predict_sarima(model_manager, train_data, len(test_data))
# print("SARIMA Predictions:", sarima_predictions)
#
# Prophet model
# prophet_train_data, prophet_future_data = prepare_prophet_data(train_df, test_df, product)
# prophet_predictions = fit_and_predict_prophet(model_manager, prophet_train_data, prophet_future_data)
# print("Prophet Predictions:", prophet_predictions)
