import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR

dataset = pd.read_csv("Position_Salaries.csv")
print(dataset.corr(numeric_only = True)["Salary"].sort_values(ascending=False))

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# print(X)

y = y.reshape(len(y),1)

plt.scatter(X,y,color = "red");


# print(y)

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

# print(X)
# print(y)

regressor = SVR(kernel="rbf")
regressor.fit(X,y)
y_pred = regressor.predict(X)
print(y_scaler.inverse_transform([y_pred]))

plt.scatter(x_scaler.inverse_transform(X),y_scaler.inverse_transform([y_pred]),color = "blue")

plt.show()
