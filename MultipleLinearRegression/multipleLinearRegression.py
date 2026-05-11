# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

# Importing dataset and creating the matrices
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Encoding the categorical data - State
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[-1])],remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Splitting training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Training the Model
multipleLinearRegressor = LinearRegression()
multipleLinearRegressor.fit(X_train,y_train)
y_pred = multipleLinearRegressor.predict(X_test)
print(y_pred)
print(y_test)
