# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 20:59:11 2018

@author: Shahid
"""
#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor 


#Read dataset
mydata= pd.read_csv("movie_metadata.csv")

#Setting up data for model

X=mydata.iloc[:,[2,4,5,7,8,12,13,18,22,24,27]].values
y=mydata.iloc[:,25].values

#Test for the missing value in the dataset.

mydata.isnull().sum()



#replacing the missing value with the mean of the column

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy= 'mean', axis = 0)
imputer=imputer.fit(X[:,0:12])
X[:,0:12] = imputer.transform(X[:,0:12])
X
z=pd.DataFrame(X)
print(z.head(10))

#Data Visualization

#histograms
z.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show()

# density plot

z.plot(kind='density', subplots= True,layout=(4,4), sharex=False, sharey=False,
       fontsize=1)
plt.show()

# Spliting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X =StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Test options and evaluation metric
num_folds=10
seed=0
scoring = 'neg_mean_squared_error'

#Spot-Check Algorithm
models = [] 
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso())) 
models.append(('EN', ElasticNet())) 
models.append(('KNN', KNeighborsRegressor())) 
models.append(('CART', DecisionTreeRegressor())) 
models.append(('SVR', SVR()))

#Evaluate each model

results=[]
names=[]
for name, model in models:
    kfold=KFold(n_splits=num_folds,random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#trying to improve model performance by ensembles method
ensembles=[]
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())]))) 
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())]))) 
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())]))) 
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
results=[]
names=[]
for name, model in ensembles:
    kfold=KFold(n_splits=num_folds,random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Tuning GBM
param_grid = dict(n_estimators=np.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 
means = grid_result.cv_results_['mean_test_score'] 
stds = grid_result.cv_results_['std_test_score'] 
params = grid_result.cv_results_['params'] 
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#Finalize the model
# The best model looks like GBM with n_estimators = 200

model = GradientBoostingRegressor(random_state=seed, n_estimators=200) 
model.fit(X_train,y_train)

# Use the model on test set
predictions=model.predict(X_test)
print('mean_squared_error = ',mean_squared_error(y_test,predictions))


# Neural Network model

#define standard model

def standard_model():
    model=Sequential()
    model.add(Dense(13, input_dim=11, kernel_initializer='normal', activation='relu')) 
    model.add(Dense(1, kernel_initializer='normal')) 
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
seed=0
np.random.seed(seed)

#evaluate model
#estimator = KerasRegressor(build_fn=standard_model, epochs=100, batch_size=5, verbose=0) 
#kfold = KFold(n_splits=10, random_state=seed) 
#results = cross_val_score(estimator, X_train, y_train, cv=kfold) 
#print("Standard: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    
  


