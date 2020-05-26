from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import statsmodels.api as sm 
import numpy as np

#DATASET LOADING
boston=load_boston()
print("DATASET KEYS ARE",boston.keys())
print("DATASET DESCRIPTION",boston.DESCR)
print("FEATURE NAMES",boston.feature_names)
print("DATA",boston.data)
#CREATING DATAFRAMES
boston1=pd.DataFrame(boston.data,columns=boston.feature_names)
boston2=pd.DataFrame(boston.data,columns=boston.feature_names)
boston1['MEDV']=boston.target
boston2['MEDV']=boston.target
#CHECKING LINEARITY WITH SCATTERPLOTS
x=boston1['RM']
y=boston1['MEDV']
plt.scatter(x,y)
plt.xlabel('RM')
plt.ylabel('MEDV')
#correlation matrix and heatmap
correlation_matrix=boston1.corr().round(2)
sb.heatmap(data=correlation_matrix,annot=True)
X = boston1
Y = boston1['MEDV']
print(boston1.head())
print(X.head())
X=np.append(arr=np.ones((506,1)).astype(int),values=X,axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
X_train = np.append (arr=np.ones([X_train.shape[0],1]).astype(int), values = X_train, axis = 1)
X_opt = X_train[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
regressor = sm.OLS(endog=Y_train,exog=X_opt).fit()
regressor.summary()
