# Regression-Analysis

## Day -  01
### Simple Linear Regression
 - Learnt what is Simple Linear Regression
 - Straight line equation for Simple Linear Regression
 - Graph for Simple Linear Regression
 - Ordinary Least Squares method to determine the Best Fit Line
 - Used a simple dataset and LinearRegression() method in sklearn to conduct regression analysis.

### Multiple Linear Regression
  - Learnt what is Multiple Linear Regression
  - Equation for Multiple Linear Regression
  - Assumptions made for Multiple Linear Regression - Linearity, Normality, Homoschedasticity, Independence, Lack of Multicolinearity
  - Methods used for Multiple Linear Regression - All in, Backward Elimination, Forward Selection, Bidirectional Elimination
  - Used a simple dataset and LinearRegression() method in sklearn to conduct regression analysis.

## Day - 02

### Polynomial Regression
- Learnt what is Polynomial Regression  
- Understood why Polynomial Regression is used when Linear Regression cannot capture non-linear relationships  
- Learnt the polynomial equation used in regression analysis  
- Understood how polynomial features transform a linear model into a non-linear model  
- Learnt the concept of degree (`x²`, `x³`, `x⁴`, etc.) and its impact on model fitting  
- Visualized Polynomial Regression using graphs and compared it with Linear Regression  
- Learnt about underfitting and overfitting in Polynomial Regression  
- Used a sample dataset and `PolynomialFeatures()` along with `LinearRegression()` from sklearn to perform regression analysis

### Support Vector Regression (SVR)
- Learnt what is Support Vector Regression (SVR)  
- Understood the concept of support vectors and margin in SVR  
- Learnt how SVR fits data within an acceptable error boundary (`epsilon margin`)  
- Understood the importance of feature scaling using `StandardScaler()` before training an SVR model  
- Learnt about different kernels used in SVR — Linear, Polynomial, and Radial Basis Function (RBF)  
- Visualized SVR predictions and compared them with actual data points  
- Used a dataset and `SVR()` from sklearn to perform regression analysis  
- Evaluated model performance using metrics such as **R² Score**, **Mean Absolute Error (MAE)**, and **Mean Squared Error (MSE)**

## Day - 03  
### Decision Tree Regression  
- Learnt what is Decision Tree Regression  
- Understood how Decision Tree Regression splits data into regions based on conditions  
- Learnt how predictions are made using the average value in leaf nodes  
- Understood overfitting in Decision Tree Regression and the importance of controlling tree depth  
- Visualized the non-linear relationship and step-like prediction graph of Decision Tree Regression  
- Used a simple dataset and `DecisionTreeRegressor()` method in sklearn to perform regression analysis  
- Learnt the importance of feature scaling (usually not required for Decision Trees)  
- Compared Decision Tree Regression with Simple Linear Regression for non-linear data handling  

### Random Forest Regression  
- Learnt what is Random Forest Regression  
- Understood how Random Forest combines multiple Decision Trees to improve prediction accuracy  
- Learnt the concept of ensemble learning and bagging technique  
- Understood how Random Forest reduces overfitting compared to a single Decision Tree  
- Learnt how predictions are made by averaging outputs from multiple trees  
- Visualized prediction results and understood non-linear regression behavior  
- Used a simple dataset and `RandomForestRegressor()` method in sklearn to perform regression analysis  
- Learnt the importance of tuning parameters like `n_estimators` and `random_state`  
- Compared Random Forest Regression with Decision Tree Regression for better generalization  
