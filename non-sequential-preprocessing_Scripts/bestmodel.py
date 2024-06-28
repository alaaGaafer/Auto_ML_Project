from similaritySearch.sim_function import MetaFeatureSimilarity
from cashAlgorithm import smacClass
from cashAlgorithm.smacClass import ProblemType
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
metaFeatureobj=MetaFeatureSimilarity("lol", "lol")
choosenModels=metaFeatureobj.pseudomodels()
print(choosenModels)
print ("hello world")
# self.models,'Classification',self.X_train,self.Y_train,self.X_test,self.Y_test
#note the problem type must be enum object from problem type
class Bestmodel:
    def __init__(self,problemtype,X_train,X_text,y_train,y_test) -> None:
        
        self.problemtype=problemtype
        self.X_train=X_train
        self.X_test=X_text
        self.y_train=y_train
        self.y_test=y_test
    def Getincumbent(self):
        choosenModels=metaFeatureobj.pseudomodels()
        Facadee=smacClass.Facade(self.problemtype,choosenModels,self.X_train,self.X_test1,self.y_train,self.y_test1)
        return Facadee.chooseFacade()
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
            y_pred = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test1, y_pred)
            print(f"Model MSE: {mse}")
            self.modelstr=HPOdict['Models']
            self.modelobj=model
        elif self.problemtype == ProblemType.TIME_SERIES:
            pass

    def PredictModel(self,xtopred):
        if self.problemtype == ProblemType.CLASSIFICATION:
            y_pred = self.modelobj.predict(xtopred)
            return y_pred
        elif self.problemtype == ProblemType.REGRESSION:
            y_pred = self.modelobj.predict(xtopred)
            return y_pred
        elif self.problemtype == ProblemType.TIME_SERIES:
            pass
    def saveModel(self):
        pass
    def splitTestData(self):
        pass
    

if __name__ =="__main__":
#test classifcation
    classincummbet=Facade(ProblemType.CLASSIFICATION,['KNN','LR','RF'],X_train_raw,X_test,y_train,y_test)
    classincummbet.chooseFacade()