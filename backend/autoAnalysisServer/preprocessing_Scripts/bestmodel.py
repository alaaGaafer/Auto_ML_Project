
from .cashAlgorithm import smacClass
from .cashAlgorithm.smacClass import ProblemType
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from .cashAlgorithm.Models import ARIMAModel, SARIMAModel
import joblib
import pandas as pd
class Bestmodel:
    def __init__(self,problemtype,choosednModels=None,X_train=None,X_text=None,y_train=None,y_test=None,freq='D',m=7) -> None:
        
        self.problemtype=problemtype
        self.X_train=X_train
        self.X_test=X_text
        self.y_train=y_train
        self.y_test=y_test
        self.choosenModels=choosednModels
        self.freq=freq
        self.m=m
        self.accuracy=None
        self.mse=None
    def Getincumbent(self):
        # print("the ytrain is: ",self.y_train.head())
        Facadee=smacClass.Facade(self.problemtype,self.choosenModels,self.X_train,self.X_test1,self.y_train,self.y_test1,self.freq,self.m)
        incumbent=Facadee.chooseFacade()
        # print(incumbent)
        return incumbent
    def TrainModel(self):
        incumbent=self.Getincumbent()
        HPOdict=incumbent.get_dictionary()
        if self.problemtype ==ProblemType.CLASSIFICATION:
            if HPOdict['Models'] == 'KNN':
                model = KNeighborsClassifier(n_neighbors=HPOdict.get('Ks', 1))
            elif HPOdict['Models'] == 'LR':
                model = LogisticRegression(C=HPOdict.get('regularizationStre', 1.0), max_iter=1000)
            elif HPOdict['Models'] == 'RF':
                model = RandomForestClassifier(n_estimators=HPOdict.get('n_estimators', 10))
            elif HPOdict['Models'] == 'SVC':
                model = SVC(C=HPOdict.get('regularizationStre', 1.0), kernel=HPOdict.get('kernel', 'rbf'))

            model.fit(self.X_train,self.y_train)
            y_pred = model.predict(self.X_test1)
            accuracy = accuracy_score(self.y_test1, y_pred)
            print(f"Model accuracy: {accuracy * 100:.2f}%")
            self.accuracy=accuracy
            self.modelstr=HPOdict['Models']
            self.modelobj=model
        elif self.problemtype == ProblemType.REGRESSION:
            if HPOdict['Models'] == 'LinearRegression':
                model = LinearRegression()
            elif HPOdict['Models'] == 'Lasso':
                model = Lasso(alpha=HPOdict.get('alphalas', 1.0))
            elif HPOdict['Models'] == 'Ridge':
                model = Ridge(alpha=HPOdict.get('alpharid', 1.0))
            elif HPOdict['Models'] == 'RF':
                model = RandomForestRegressor(n_estimators=HPOdict.get('n_estimatorsrf', 10))
            elif HPOdict['Models'] == 'XGboost':
                model = XGBRegressor(n_estimators=HPOdict.get('n_estimatorsxg', 10))

            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test1)
            mse = mean_squared_error(self.y_test1, y_pred)
            print(f"Model MSE: {mse}")
            self.mse=mse
            self.modelstr=HPOdict['Models']
            self.modelobj=model
        elif self.problemtype == ProblemType.TIME_SERIES:
            if HPOdict['Models'] == 'Arima':
                

                armodelobj = ARIMAModel()
                armodelobj.fit(self.y_train, HPOdict.get('p', 1), HPOdict.get('q', 1), HPOdict.get('d', 1), self.freq)
                # self.modelobj=arRes
            elif HPOdict['Models'] == 'Sarima':
                armodelobj = SARIMAModel()
                armodelobj.fit_with_tests(self.y_train, HPOdict.get('p', 1), HPOdict.get('q', 1), HPOdict.get('d', 1),
                                          HPOdict.get('sp', 1), HPOdict.get('sq', 1), HPOdict.get('sd', 1),HPOdict.get('ss', 7), self.freq) 
                
            # print("theytest",self.y_test1)
            # print(len(self.y_test1))
            y_pred = armodelobj.predict(len(self.y_test1))
            # print("the ypred ", y_pred)
            # print(y_pred)
            # self
            self.modelobj=armodelobj

            mse = mean_squared_error(self.y_test1, y_pred)
            print(f"Model MSE: {mse}")
            self.mse=mse
            self.modelstr=HPOdict['Models']
            # self.modelobj=model


    def PredictModel(self,xtopred):
        if self.problemtype == ProblemType.CLASSIFICATION:
            y_pred = self.modelobj.predict(xtopred)
            Concatedyandx = pd.concat([xtopred, y_pred], axis=1)
            return Concatedyandx
        elif self.problemtype == ProblemType.REGRESSION:
            y_pred = self.modelobj.predict(xtopred)
            concatedyandx=pd.concat([xtopred,y_pred],axis=1)
            return Concatedyandx
        elif self.problemtype == ProblemType.TIME_SERIES:
            y_pred=self.modelobj.predict(steps=len(xtopred))
            concatedyandx=pd.concat([xtopred,y_pred],axis=1)
            return concatedyandx
    def saveModel(self,dsid):
        path=f"preprocessing_Scripts/models/{dsid}.pkl"
        joblib.dump(self.modelobj,path)
        return True
    def loadModel(self,dsid):
        path=f"preprocessing_Scripts/models/{dsid}.pkl"
        model=joblib.load(path)
        self.modelobj=model
        return True
    def splitTestData(self):
        #split self.X_test and self.y_test into two parts
        if self.problemtype!=ProblemType.TIME_SERIES:

            self.X_test1, self.X_test2, self.y_test1, self.y_test2 = train_test_split(self.X_test, self.y_test, test_size=0.5, random_state=42)
        elif self.problemtype==ProblemType.TIME_SERIES:
            split_ratio = 0.5  
            split_index = int(len(self.y_test) * split_ratio)

            self.y_test1 = self.y_test[:split_index]
            self.y_test2 = self.y_test[split_index:]
            self.X_test1='lolll'
    

if __name__ =="__main__":
#test classifcation
    classincummbet=Facade(ProblemType.CLASSIFICATION,['KNN','LR','RF'],X_train_raw,X_test,y_train,y_test)
    classincummbet.chooseFacade()