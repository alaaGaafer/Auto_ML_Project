import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import seaborn as sn
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from ConfigSpace.conditions import InCondition,EqualsCondition
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class Models:
    def __init__(self,similarModels,problemType):
        self.Models=similarModels
        self.Problemtype=problemType
    def configspace(self):
        confs = ConfigurationSpace(seed=0)
        #HPOs
        if self.Problemtype=='Classification':
            # models are ['KNN','LR',"RF",'SVC']
            models=Categorical('Models',self.Models)
            #KNN parameters
            Kneighbors=Integer('Ks',(1,10),default=1)
            #LR  and svc Parameters
            rc=Float('regularizationStre',(0.01,1))
            #RF parameters
            nestimators=Integer('n_estimators',(1,20),default=10)
            #SVC parameters
            kernel=Categorical('kernel',['linear','rbf'])
            #dependencies

            useks=InCondition(child=Kneighbors,parent=models,values=['KNN'])
            userc=InCondition(child=rc,parent=models,values=['LR','SVC'])
            usekernel=InCondition(child=kernel,parent=models,values=['SVC'])
            useEst=EqualsCondition(child=nestimators,parent=models,value='RF')



            #adding conditions and HPs
            confs.add_hyperparameters([models,Kneighbors,rc,nestimators,kernel])
            confs.add_conditions([useks,userc,usekernel,useEst])
        elif self.Problemtype=='Regression':
            models=Categorical('Models',self.Models)
            #linear regression parameters
            #lasso and ridge regression parameters
            alpha=Float('alpha',(0.01,100))
            
            #random forest and XGboost parameters
            nestimators=Integer('n_estimators',(1,20),default=10)
            #dependencies 
            usealpha=InCondition(child=alpha,parent=models,values=['Lasso','Ridge'])
            useEst=InCondition(child=nestimators,parent=models,values=['RF','XGboost'])
            #adding conditions and HPs
            confs.add_hyperparameters([models,alpha,nestimators])
            confs.add_conditions([usealpha,useEst])



            
        return confs
    def train(self,config:Configuration,seed: int=0):
        
        config_dict=config.get_dictionary()
        model=config_dict['Models']
        print(f"config_dict:{config_dict}")
        if self.Problemtype=='Classification':
            if model =='KNN':
                Classifier=KNeighborsClassifier(n_neighbors=config_dict['Ks'])
                Classifier.fit(X_train_raw,y_train)
                y_pred = Classifier.predict(X_test)
                loss=1-accuracy_score(y_test,y_pred)
                print("the loss is: ",loss)
                return loss
            elif model=='LR':
                Classifier=LogisticRegression(C=config_dict['regularizationStre'])
                Classifier.fit(X_train_raw,y_train)
                y_pred = Classifier.predict(X_test)
                loss=1-accuracy_score(y_test,y_pred)
                print("the losss is: ",loss)
                return loss
            elif model=='RF':
                Classifier=RandomForestClassifier(n_estimators=config_dict['n_estimators'])
                Classifier.fit(X_train_raw,y_train)
                y_pred = Classifier.predict(X_test)
                loss=1-accuracy_score(y_test,y_pred)
                print("the loss is: ",loss)
                return loss
            elif model=='SVC':
                Classifier=SVC(C=config_dict['regularizationStre'],kernel=config_dict['kernel'])
                Classifier.fit(X_train_raw,y_train)
                y_pred = Classifier.predict(X_test)
                loss=1-accuracy_score(y_test,y_pred)
                print("the loss is: ",loss)
                return loss
        elif self.Problemtype=='Regression':
            return self.regression(config_dict)
    def regression(self, configDict):
            model=configDict['Models']
            if model=='LR':
                regressor=LinearRegression()
                regressor.fit(X_train,y_train)
                y_pred = regressor.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                print("Mean Squared Error:", mse)
                return mse
            elif model=='Lasso':
                regressor=Lasso(alpha=configDict['alpha'])
                regressor.fit(X_train,y_train)
                y_pred = regressor.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                print("Mean Squared Error:", mse)
                return mse
            elif model=='Ridge':
                regressor=Ridge(alpha=configDict['alpha'])
                regressor.fit(X_train,y_train)
                y_pred = regressor.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                print("Mean Squared Error:", mse)
                return mse
            elif model=='RF':
                regressor=RandomForestRegressor(n_estimators=configDict['n_estimators'])
                regressor.fit(X_train,y_train)
                y_pred = regressor.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                print("Mean Squared Error:", mse)
                return mse
            elif model=='XGboost':
                regressor=XGBRegressor(n_estimators=configDict['n_estimators'])
                regressor.fit(X_train,y_train)
                y_pred = regressor.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                print("Mean Squared Error:", mse)
                return mse