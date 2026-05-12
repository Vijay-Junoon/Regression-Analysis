import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression



dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

linear_reg = LinearRegression()
linear_reg.fit(X,y)

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)


lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)


print(linear_reg.predict([[6.5]]))
print(lin_reg.predict(poly_reg.transform([[6.5]])))
plt.scatter(X,y,color="red")
plt.plot(X,linear_reg.predict(X),color="blue")
plt.plot(X,lin_reg.predict(X_poly),color="indigo")
plt.show()
