import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

linearRegressor = LinearRegression()
linearRegressor.fit(X_train,y_train)

y_pred = linearRegressor.predict(X_test)

# Visualizing training results
# plt.scatter(X_train,y_train,color="red")
# plt.plot(X_train,linearRegressor.predict(X_train),color="blue")
# plt.title("Experience vs Salary")
# plt.xlabel("Experience")
# plt.ylabel("Salary")
# plt.show()

# Visualizing Testing results
plt.scatter(X_test,y_test,color="red")
plt.plot(X_test,y_pred,color="blue")
plt.show()