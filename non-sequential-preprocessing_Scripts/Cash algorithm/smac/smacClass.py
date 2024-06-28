from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from ConfigSpace.conditions import InCondition,EqualsCondition
from smac import HyperparameterOptimizationFacade, Scenario,Callback
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import time

class CustomCallback(Callback):
    def __init__(self, max_same_incumbent_trials=30):
        self.trials_counter = 0
        self.incumbent_counter = 0
        self.last_incumbent = None
        self.max_same_incumbent_trials = max_same_incumbent_trials

    def on_start(self, smbo):
        print("let's start the optimization")

    def on_tell_end(self, smbo, info, value):
        self.trials_counter += 1
        print(f"The number of trials is: {self.trials_counter}")

        incumbents = smbo.intensifier.get_incumbents()
        if incumbents:
            current_incumbent = incumbents[0]  # Assuming there's at least one incumbent
            current_incumbent_dict = current_incumbent.get_dictionary()

            if self.last_incumbent == current_incumbent_dict:
                self.incumbent_counter += 1
            else:
                self.incumbent_counter = 0  # Reset the counter if incumbent changes
                self.last_incumbent = current_incumbent_dict

            if self.incumbent_counter >= self.max_same_incumbent_trials:
                print(f"The incumbent has remained the same for {self.max_same_incumbent_trials} trials.")
                print("Stopping the optimization process.")
                return False  # This stops the optimization process

            if self.trials_counter % 10 == 0:
                print(f"The incumbent is: {current_incumbent_dict}")
                print(f"The incumbent loss is: {smbo.runhistory.get_cost(current_incumbent)}")

            if self.trials_counter == 100:
                print("Let's stop the optimization at trial 100")
                return False 

        return None

class Models:
    def __init__(self,similarModels,problemType,X_train,Y_train,X_test,Y_test,freq='d',m=7):
        self.Models=similarModels
        self.Problemtype=problemType
        self.X_train=X_train
        self.Y_train=Y_train
        self.X_test=X_test
        self.Y_test=Y_test
        self.freq=freq
        self.m=m
    def configspace(self):
        confs = ConfigurationSpace(seed=0)
        #HPOs
        if self.Problemtype=='Classification':

            # models are ['KNN','LR',"RF",'SVC']
            models=Categorical('Models',self.Models)
            parameters=[]
            conditions=[]
            parameters.append(models)
            if 'KNN' in self.Models:
                Kneighbors=Integer('Ks',(1,10),default=1)
                useks=InCondition(child=Kneighbors,parent=models,values=['KNN'])
                parameters.append(Kneighbors)
                conditions.append(useks)
            if 'SVC' in self.Models:
                kernel=Categorical('kernel',['linear','rbf'])
                usekernel=InCondition(child=kernel,parent=models,values=['SVC'])
                parameters.append(kernel)
                conditions.append(usekernel)
                if 'LR' in self.Models:
                    rc=Float('regularizationStre',(0.01,1))
                    userc=InCondition(child=rc,parent=models,values=['LR','SVC'])
                    parameters.append(rc)
                    conditions.append(userc)
                else:
                    rc=Float('regularizationStre',(0.01,1))
                    userc=InCondition(child=rc,parent=models,values=['SVC'])
                    parameters.append(rc)
                    conditions.append(userc)
            elif 'LR' in self.Models:
                rc=Float('regularizationStre',(0.01,1))
                userc=InCondition(child=rc,parent=models,values=['LR'])
                parameters.append(rc)
                conditions.append(userc)
            if 'RF' in self.Models:
                nestimators=Integer('n_estimators',(1,20),default=10)
                useEst=InCondition(child=nestimators,parent=models,values=['RF'])
                parameters.append(nestimators)
                conditions.append(useEst)
            confs.add_hyperparameters(parameters)
            confs.add_conditions(conditions)
            #models=["Lasso","Ridge","RF","XGboost"]
        elif self.Problemtype=='Regression':
            models=Categorical('Models',self.Models)
            conditions=[]
            parameters=[models]
            if 'Lasso' in self.Models:
                alphalasso=Float('alphalas',(0.01,100))
                usealphaLASSO=InCondition(child=alphalasso,parent=models,values=['Lasso'])
                parameters.append(alphalasso)
                conditions.append(usealphaLASSO)
            if 'Ridge' in self.Models:
                alpharidge=Float('alpharid',(0.01,100))
                usealphaRidge=InCondition(child=alpharidge,parent=models,values=['Ridge'])
                parameters.append(alpharidge)
                conditions.append(usealphaRidge)
            if 'RF' in self.Models:
                nestimatorsrf=Integer('n_estimatorsrf',(1,20),default=10)
                useEstRf=InCondition(child=nestimatorsrf,parent=models,values=['RF'])
                parameters.append(nestimatorsrf)
                conditions.append(useEstRf)
            if 'XGboost' in self.Models:
                nestimators=Integer('n_estimatorsxg',(1,20),default=10)
                useEstXG=InCondition(child=nestimators,parent=models,values=['XGboost'])
                parameters.append(nestimators)
                conditions.append(useEstXG)
            confs.add_hyperparameters(parameters)
            confs.add_conditions(conditions)
            # models=['Sarima','Arima]
        elif self.Problemtype=='TimeSeries':
            # ['arima',sarima,'prophet']
            models=Categorical("Models",self.Models)
            
            p=Integer('p',(0,3))
            q=Integer('q',(0,3))
            d=Integer('d',(0,3))

            conditions=[]
            parameters=[models,p,q,d]
            if 'Sarima' in self.Models:
                SarimaP=Integer('sp',(0,3))
                sarimaq=Integer('sq',(0,3))
                sarimad=Integer('sd',(0,3))
                
                usecondip=EqualsCondition(child=SarimaP,parent=models,value='Sarima')
                usecondiq=EqualsCondition(child=sarimaq,parent=models,value='Sarima')
                usecondid=EqualsCondition(child=sarimad,parent=models,value='Sarima')
                parameters.append(SarimaP)
                parameters.append(sarimaq)
                parameters.append(sarimad)
                conditions.append(usecondip)
                conditions.append(usecondiq)
                conditions.append(usecondid)

            confs.add_hyperparameters(parameters)
            confs.add_conditions(conditions)
            
        return confs
            
        
    def train(self,config:Configuration,seed: int=0):
        start_time=time.time()
        config_dict=config.get_dictionary()
        model=config_dict['Models']
        print(f"config_dict:{config_dict}")
        if self.Problemtype=='Classification':
            return self.classification(config_dict,start_time)
        elif self.Problemtype=='Regression':
            return self.regression(config_dict,start_time)
        elif self.Problemtype=='TimeSeries':
            return self.timeser(config_dict)
    def classification(self,configDict,start_time):
        model=configDict['Models']
        if model=='KNN':
            Classifier=KNeighborsClassifier(n_neighbors=configDict['Ks'])
        elif model=='LR':
            Classifier=LogisticRegression(C=configDict['regularizationStre'])
        elif model=='RF':
            Classifier=RandomForestClassifier(n_estimators=configDict['n_estimators'],random_state=42)
        elif model=='SVC':
            Classifier=SVC(C=configDict['regularizationStre'],kernel=configDict['kernel'])
        print(f"the type of the classifier is: {type(Classifier)}")
        Classifier.fit(self.X_train,self.Y_train)
        y_pred = Classifier.predict(self.X_test)
        loss=1-accuracy_score(self.Y_test,y_pred)
        print("the loss is: ",loss)
        return {'loss':loss,'time':time.time()-start_time}
    def regression(self, configDict,start_time):
            model=configDict['Models']
            if model=='LR':
                regressor=LinearRegression()
            elif model=='Lasso':
                regressor=Lasso(alpha=configDict['alphalas'])
            elif model=='Ridge':
                regressor=Ridge(alpha=configDict['alpharid'])
            elif model=='RF':
                regressor=RandomForestRegressor(n_estimators=configDict['n_estimatorsrf'],random_state=42)
            elif model=='XGboost':
                regressor=XGBRegressor(n_estimators=configDict['n_estimatorsxg'],random_state=42)
            regressor.fit(self.X_train,self.Y_train)
            y_pred = regressor.predict(self.X_test)
            mse = mean_squared_error(self.Y_test, y_pred)
            print("Mean Squared Error:", mse)
            return {'loss':mse,'time':time.time()-start_time}
    def timeser(self,configDict):
        model=configDict['Models']
        if model=='Arima':
            return Arima(self.Y_train,self.Y_test,p=configDict['p'],d=configDict['d'],q=configDict['q'],freq=self.freq)
        elif model=='Sarima':
            return 10000
from enum import Enum

class ProblemType(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    TIME_SERIES = 'time series'
    UNBALANCED = 'unbalanced'

class Facade:
    def __init__(self, problem_type,Models,X_train,X_test,Y_train,Y_test,freq="d",m=7):
        if isinstance(problem_type, ProblemType):
            self.problem_type = problem_type
            self.models=Models
            self.X_train=X_train
            self.X_test=X_test
            self.Y_train=Y_train
            self.Y_test=Y_test
            self.freq=freq
            self.m=m
        else:
            raise ValueError("problem_type must be a ProblemType Enum")
    def chooseFacade(self):
        if self.problem_type==ProblemType.CLASSIFICATION:
            return  self.ClassificationFacade()
        elif self.problem_type==ProblemType.REGRESSION:
            return self.RegressionFacade()
        elif self.problem_type==ProblemType.TIME_SERIES:
            return self.TimeSeriesFacade()
        elif self.problem_type==ProblemType.UNBALANCED:
            return self.UnbalancedFacade()
    def ClassificationFacade(self):
        classifier=Models(self.models,'Classification',self.X_train,self.Y_train,self.X_test,self.Y_test)
        scenario = Scenario(classifier.configspace(), deterministic=True,objectives=['loss','time'], n_trials=100)
        smac = HyperparameterOptimizationFacade(scenario, classifier.train,overwrite=True,callbacks=[CustomCallback()],
                                                multi_objective_algorithm=HyperparameterOptimizationFacade.get_multi_objective_algorithm(scenario,objective_weights=[2, 1]))
        incumbents = smac.optimize()
        for incumbent in incumbents:
            print(incumbent)
        return incumbents
    def RegressionFacade(self):
        Regressor=Models(self.models,'Regression',self.X_train,self.Y_train,self.X_test,self.Y_test)
        scenario = Scenario(Regressor.configspace(), deterministic=True,objectives=['loss','time'], n_trials=100)
        smac = HyperparameterOptimizationFacade(scenario, Regressor.train,overwrite=True,callbacks=[CustomCallback()],
                                                multi_objective_algorithm=HyperparameterOptimizationFacade.get_multi_objective_algorithm(scenario,objective_weights=[2, 1]))
        incumbents = smac.optimize()
        for incumbent in incumbents:
            print(incumbent)
        return incumbents
    def TimeSeriesFacade(self):
        timeclassifier=Models(self.models,'TimeSeries',self.X_train,self.Y_train,self.X_test,self.Y_test,freq=self.freq,m=self.m)
        scenario = Scenario(timeclassifier.configspace(), deterministic=True, n_trials=100)
        smac = HyperparameterOptimizationFacade(scenario, timeclassifier.train,overwrite=True,callbacks=[CustomCallback()])
        incumbent=smac.optimize()
        print(incumbent)
        return incumbent
        
    def UnbalancedFacade(self):
        pass
            
