# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:08:37 2019

@author: Ezone
"""

import pandas as pd

###############################################################################
#importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
   
############################################################################### 
#To check correlation in the dataset
import seaborn as sns

corr = train.corr()
ax = sns.heatmap(corr,vmin=-1,vmax = 1,center = 0,cmap= sns.diverging_palette(20,220,n = 200),square = True)

###############################################################################
#Assigning X and y
X = train.iloc[:,0:6]
y = train.iloc[:,6:7]

del X['Username']
del X['ID']

###############################################################################
#Making dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder()
X['Tag'] = labelencoder_X.fit_transform(X['Tag'])

onehotencoder = OneHotEncoder(categorical_features = [0])

X = onehotencoder.fit_transform(X).toarray()
#len(X['Tag'].unique())

###############################################################################
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_r = StandardScaler()
X = sc_r.fit_transform(X)

###############################################################################
#Splitting the dataset just to check the accuracy of the model
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y, test_size = 0.25,random_state = 42)

###############################################################################
#Implementing the Random Forest Classifier
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators = 250,random_state = 42)
reg.fit(X_train,y_train)
pred = reg.predict(X_test)

###############################################################################
#for test data
del test['Username']
idTest = test['ID']
del test['ID']


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder()
test['Tag'] = labelencoder_X.fit_transform(test['Tag'])

onehotencoder = OneHotEncoder(categorical_features = [0])

test = onehotencoder.fit_transform(test).toarray()
#len(X['Tag'].unique())

from sklearn.preprocessing import StandardScaler
sc_r = StandardScaler()
test = sc_r.fit_transform(test)

###############################################################################
#Making prediction File in .csv
predictionTest = reg.predict(test)

predictionTest = {'ID':idTest, 'Upvotes':predictionTest}
predictionAns = pd.DataFrame(predictionTest)
predictionAns = predictionAns.to_csv("predictionAns.csv",index = False)
###############################################################################