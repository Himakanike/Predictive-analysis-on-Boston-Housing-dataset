#Loading dataset
from sklearn.datasets import load_boston
boston=load_boston()

#Dataframe
import pandas as pd
db=pd.DataFrame(boston.data,columns=boston.feature_names)
db1=pd.DataFrame(boston.data,columns=boston.feature_names)
print(db)
db['MEDV']=boston.target

#Linearity check with Scatterplot
import matplotlib.pyplot as plt 
x=db['RM']
y=db['MEDV']
plt.scatter(x,y)
x=db['LSTAT']
y=db['MEDV']
plt.scatter(x,y)
x=db['CRIM']
y=db['MEDV']
plt.scatter(x,y)
x=db['NOX']
y=db['MEDV']
plt.scatter(x,y)
x=db['ZN']
y=db['MEDV']
plt.scatter(x,y)

#correlation matrix
import seaborn as sb
correlation_matrix=db.corr().round(2)
sb.heatmap(data=correlation_matrix,annot=True)

#Normal distribution of predictors
ax = sb.distplot(db['RM'])
ax = sb.distplot(db['LSTAT'])
ax = sb.distplot(db['CRIM'])
ax = sb.distplot(db['NOX'])
ax = sb.distplot(db['ZN'])
ax = sb.distplot(db['B'])
ax = sb.distplot(db['INDUS'])
ax = sb.distplot(db['AGE'])

#Building mode 1)Splitting 2)Train
import statsmodels.api as sm
import numpy as np
import statsmodels.api as sm 
import numpy as np
X = db1
Y = db['MEDV']
print(X.head())
X=np.append(arr=np.ones((506,1)).astype(int),values=X,axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
X_train = np.append (arr=np.ones([X_train.shape[0],1]).astype(int), values = X_train, axis = 1)
X_opt = X_train[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
regressor = sm.OLS(endog=Y_train,exog=X_opt).fit()
regressor.summary()
Y = db['MEDV']
X = pd.DataFrame(np.c_[db['LSTAT'], db['RM']], columns = ['LSTAT','RM'])
print(X.head())
X=np.append(arr=np.ones((506,1)).astype(int),values=X,axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
X_train = np.append (arr=np.ones([X_train.shape[0],1]).astype(int), values = X_train, axis = 1)
X_opt = X_train[:,[0,1,2]]
regressor = sm.OLS(endog=Y_train,exog=X_opt).fit()
regressor.summary()

#homoscedasticity
y_predict=regressor.predict(X_test)

#Evaluate rmse, r2  (On test dataset as well)
y_predict=regressor.predict(X_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import math
model_mae=mean_absolute_error(Y_test,y_predict)
model_mse=mean_squared_error(Y_test,y_predict)
model_rmse=math.sqrt(model_mse)
print("MAE {:.3}".format(model_mae))
print("MSE {:.3}".format(model_mse))
print("RMSE {:.3}".format(model_rmse))
model_r2=r2_score(Y_test,y_predict)
model_r2