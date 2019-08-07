# Importing the libraries
import numpy as np
import pandas as pd

import pickle
import requests
import json

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

#print(dataset.head())
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#print (X)

#
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder = LabelEncoder()
# X[:, 3] = labelencoder.fit_transform(X[:, 3])
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()
#
#
# X = X[:, 1:]

# #newer version of encoding the input variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
     [('one_hot_encoder', OneHotEncoder(), [3])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
     remainder='passthrough'                         # Leave the rest of the columns untouched
 )

X = np.array(ct.fit_transform(X), dtype=np.float)
#encodedate(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

print(y_pred)

#create the model file. Export the model to a file
pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict(X_test))

