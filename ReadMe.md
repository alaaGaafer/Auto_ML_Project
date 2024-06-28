# our tasks to track feel free to choose one

- is there a better way instead CSV nn
- modify the cash algo (assem) finished
- use the combinations between the cleaned datasets with similarity search and get the best model(asseM) (finished)
- train the best model with high fidelity(assem)(finished)
- modify the smac to get only the loss not the time(callbacks,loss,scenario) (assem) finished
- test the results(assem) (semifinished, test the regresstion and SVM and timeseries)
- connect preprocessing to data frontend-backend (assem)
- add multifidelity facade to the logreg smac(assem) NN for MVP
- add the sarima to cash (assem and alaa)
- use sarima and arima in best model (assem and alaa)
- connect the similarity search to feature extractor(alaa))
- create search on by key knowledge base(alaa)
- get cleaned datasets(alaa)
- add time series to best models (assem and alaa)
- complete the database (nada and zeina)
- connect front end with back end to (all team)

# models

_Classification_
['KNN',"LR","RF","SVC"]
hyperparameters

- Number of neighbors (KNN)
- Regularization Strength (logisting regression)
- N Estimators( random forest)
- Regularization strength, kernel(linear, or rbf in svc)

# regression

['LinearRegression','Lasso','Ridge','RF',XGboost]

hyperparameters = {
'LinearRegression': {}, # No specific hyperparameters
'Lasso': {
'alpha': [1.0] # Regularization strength
},
'Ridge': {
'alpha': [1.0] # Regularization strength
},
'RF': {
'n_estimators': [10] # Number of trees in the forest
},
'XGboost': {
'n_estimators': [10] # Number of boosting rounds
}
}
