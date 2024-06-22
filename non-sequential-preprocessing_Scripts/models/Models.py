import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
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
    def __init__(self, order):
        self.order = order
        self.model = None

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=self.order).fit()

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


# Classification model classes
class LogisticRegressionModel:
    def __init__(self, C=None, penalty=None, solver=None):
        self.model = LogisticRegression(C=C, penalty=penalty, solver=solver)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class RidgeClassifierModel:
    def __init__(self, alpha=None):
        self.model = RidgeClassifier(alpha=alpha)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class DecisionTreeClassifierModel:
    def __init__(self, criterion=None, splitter=None, max_depth=None, min_samples_leaf=None):
        self.model = DecisionTreeClassifier(criterion=criterion, splitter=splitter,
                                            max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class RandomForestClassifierModel:
    def __init__(self, n_estimators=None, criterion=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class SVCModel:
    def __init__(self, C=None, gamma=None, kernel=None):
        self.model = SVC(C=C, gamma=gamma, kernel=kernel)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class KNeighborsClassifierModel:
    def __init__(self, n_neighbors=None, weights=None, algorithm=None, leaf_size=None):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,
                                          algorithm=algorithm, leaf_size=leaf_size)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class XGBClassifierModel:
    def __init__(self, learning_rate=None, n_estimators=None, max_depth=None, min_child_weight=None,
                 gamma=None, subsample=None, colsample_bytree=None):
        self.model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                   max_depth=max_depth, min_child_weight=min_child_weight,
                                   gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class NaiveBayesModel:
    def __init__(self):
        self.model = GaussianNB()

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


# Regression model classes
class LassoModel:
    def __init__(self, alpha=None):
        self.model = Lasso(alpha=alpha)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class DecisionTreeRegressorModel:
    def __init__(self, criterion=None, splitter=None, max_depth=None, min_samples_leaf=None):
        self.model = DecisionTreeRegressor(criterion=criterion, splitter=splitter,
                                           max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class RidgeModel:
    def __init__(self, alpha=None):
        self.model = Ridge(alpha=alpha)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class RandomForestRegressorModel:
    def __init__(self, n_estimators=None, criterion=None):
        self.model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class PolynomialRegressionModel:
    def __init__(self, degree=None):
        self.degree = degree
        self.model = None

    def fit(self, train_data, train_labels):
        poly_features = PolynomialFeatures(degree=self.degree)
        X_poly = poly_features.fit_transform(train_data)
        self.model = LinearRegression().fit(X_poly, train_labels)

    def predict(self, test_data):
        poly_features = PolynomialFeatures(degree=self.degree)
        X_poly = poly_features.transform(test_data)
        return self.model.predict(X_poly)


class XGBRegressorModel:
    def __init__(self, learning_rate=None, n_estimators=None, max_depth=None, min_child_weight=None,
                 gamma=None, subsample=None, colsample_bytree=None):
        self.model = XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators,
                                  max_depth=max_depth, min_child_weight=min_child_weight,
                                  gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


class KNeighborsRegressorModel:
    def __init__(self, n_neighbors=None, weights=None, algorithm=None, leaf_size=None):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights,
                                         algorithm=algorithm, leaf_size=leaf_size)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)


# Testing functions
def test_classification_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    Accuracy = accuracy_score(test_labels, predictions)
    Cm = confusion_matrix(test_labels, predictions)
    Cr = classification_report(test_labels, predictions, zero_division=1)
    return Accuracy, Cm, Cr


def test_regression_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    MAE = mean_absolute_error(test_labels, predictions)
    MSE = mean_squared_error(test_labels, predictions)
    RMSE = np.sqrt(MSE)
    return MAE, MSE, RMSE


