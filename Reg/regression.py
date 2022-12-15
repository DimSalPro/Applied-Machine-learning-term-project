import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import model_selection
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import regression_assumptions
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Read encoded csv
print('reading data...')
df = pd.read_csv('cars__test_py_encoded.csv')
# df = df.sample(frac=0.01)

print('columns: ',df.columns)
print(df)

# Preprocessing
print('preprocessing...\n')
y = np.array(df['price'])
x = np.array(df.drop('price', axis=1))

# Scale Data
x = preprocessing.scale(x)
# scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# scaler.fit_transform(x)

# Define Models

# Split Data
print('splitting data...\n')
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.5, random_state=1)

def linear_regression(x_train,x_test,y_train,y_test):
	print('--------------------------------------------------------')
	# Linear regression
	model_lr = LinearRegression()
	print('train linear regression...')
	model_lr.fit(x_train, y_train, sample_weight=2)
	print('linear model score on test:')
	print('score of test:', model_lr.score(x_test, y_test))
	print('score of train:', model_lr.score(x_train, y_train), '\n')
	y_lr_pred = model_lr.predict(x_test)
	print('mse : ',mean_squared_error(y_test,y_lr_pred,squared=False))
	coefs = model_lr.coef_
	print('linear regression intercept', model_lr.intercept_)
	counter = 1
	
	return y_lr_pred

def lasso_regression(x_train,x_test,y_train,y_test):
	print('--------------------------------------------------------')
	model_lasso = Lasso(alpha=0.01)

	# Lasso Regression
	print('train lasso...')
	model_lasso.fit(x_train, y_train)
	print('lasso score:')
	print('test:',model_lasso.score(x_test,y_test))
	print('train',model_lasso.score(x_train,y_train),'\n')
	y_lasso_pred = model_lasso.predict(x_test)
	print('mse : ',mean_squared_error(y_test,y_lasso_pred,squared=False))
	#coefs = model_lasso.coef_
	# for coef in coefs:
	# 	print(coef)

	return y_lasso_pred#, coefs

def random_forest_reg(x_train,x_test,y_train,y_test):
	print('--------------------------------------------------------')
	model_rfr = RandomForestRegressor(n_estimators=200,max_depth=10, random_state=1)

	# Random Forest Regressor
	print('training random forest regressor...')
	model_rfr.fit(x_train, y_train)
	print('random forest regressor model score on test:')
	print('score of test:', model_rfr.score(x_test, y_test))
	print('score of train:', model_rfr.score(x_train, y_train), '\n')
	y_rfr_pred = model_rfr.predict(x_test)
	print('mse : ',mean_squared_error(y_test,y_rfr_pred,squared=False))
	return	y_rfr_pred

def poynomial_regression(x_train,x_test,y_train,y_test):
	print('--------------------------------------------------------')
	# Polynomial Regression
	print('training Polynomial regression model...')
	model_pr = LinearRegression()
	model_pr2 = LinearRegression()
	poly = PolynomialFeatures(degree=2)

	a = poly.fit_transform(x_train)
	a2 = poly.fit_transform(x_test)

	model_pr.fit(x_train,y_train)
	model_pr2.fit(a,y_train)

	y_pred = model_pr.predict(x_test)
	y_pred_poly = model_pr2.predict(a2)

	print('test score:',r2_score(y_test,y_pred_poly),'\n')
	print('mse : ',mean_squared_error(y_test,y_pred_poly,squared=False))

	return y_pred_poly

def ridge_regression(x_train,x_test,y_train,y_test):
	print('--------------------------------------------------------')
	print('Training ridge regression...')
	# Ridge Regressor
	ridge = Ridge(alpha=100)
	ridge.fit(x_train,y_train)
	print('test score: ',ridge.score(x_test,y_test))
	print('train score: ',ridge.score(x_train,y_train))
	y_ridge_pred = ridge.predict(x_test)
	print('mse : ',mean_squared_error(y_test,y_ridge_pred,squared=False))
	return y_ridge_pred


# Uncommend the desired algoriths bellow to see the results
# Warning do not run polynomial and ridge in high dimension dataset


# Train models
lr_p=linear_regression(x_train,x_test,y_train,y_test)
# poly_p=poynomial_regression(x_train,x_test,y_train,y_test)
# lasso_p=lasso_regression(x_train,x_test,y_train,y_test)
# ridge_p=ridge_regression(x_train,x_test,y_train,y_test)
rfr_p=random_forest_reg(x_train,x_test,y_train,y_test)




# Residuals
residuals_rfr = pd.Series(rfr_p) - pd.Series(y_test)
residuals_lr = pd.Series(rfr_p) - pd.Series(y_test)
# residuals_lasso = pd.Series(rfr_p) - pd.Series(y_test)
# residuals_ridge = pd.Series(rfr_p) - pd.Series(y_test)
# residuals_poly = pd.Series(rfr_p) - pd.Series(y_test)


# regression_assumptions.normal_errors_assumption('linear_regression',residuals_lr)
# regression_assumptions.normal_errors_assumption('random_forest_regressor',residuals_rfr)
# regression_assumptions.normal_errors_assumption('lasso_regressor',residuals_lasso)

# regression_assumptions.homoscedasticity_assumption('linear_regression',residuals_lr)
# regression_assumptions.homoscedasticity_assumption('random_forest_regressor',residuals_rfr)
# regression_assumptions.homoscedasticity_assumption('lasso_regressor',residuals_lasso)



# Uncomment the following lines of code for feature selection based on lasso coefficients
#--------------------------
# foo = df.drop('price',axis=1)
# foo2 = zip(foo.columns,coefs)
# for key,value in foo2:
# 	if value == 0:
# 		print(key,value)
# 		df.drop(key,axis=1,inplace=True)
#
# print(df)
# exit(-1)
# # Preprocessing
# print('preprocessing...\n')
# y = np.array(df['price'])
# x = np.array(df.drop('price', axis=1))
#
# # Scale Data
# x = preprocessing.scale(x)
# scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# scaler.fit_transform(x)
#
# lr_p=linear_regression(x_train,x_test,y_train,y_test)
# poly_p=poynomial_regression(x_train,x_test,y_train,y_test)
# lasso_p, coefs=lasso_regression(x_train,x_test,y_train,y_test)
# ridge_p=ridge_regression(x_train,x_test,y_train,y_test)
# rfr_p=random_forest_reg(x_train,x_test,y_train,y_test)