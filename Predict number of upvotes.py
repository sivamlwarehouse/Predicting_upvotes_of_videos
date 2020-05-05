# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:15:38 2019

@author: admin
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

os.chdir("D:/siva/Deployment/upvotes/")
train=pd.read_csv("train.csv")

columns= train.columns
 
lis = []
for i in range(0, train.shape[1]):
    #print(i)
    if(train.iloc[:,i].dtypes == 'object'):
        train.iloc[:,i] = pd.Categorical(train.iloc[:,i])
        #print(marketing_train[[i]])
        train.iloc[:,i] = train.iloc[:,i].cat.codes 
        train.iloc[:,i] = train.iloc[:,i].astype('object')    
        lis.append(train.columns[i])
        
cat_data=train.loc[:,lis] 
num_data=train[['ID', 'Reputation', 'Answers', 'Username', 'Views', 'Upvotes']]
num_cols=num_data.columns


###missing values
missing_values=train.isnull().sum()
##no missing values in the dataset

####outliers check
import matplotlib.pylab  as pal
 
fig,(ax1,ax2)=plt.subplots(1,2)

ax1.boxplot(train.iloc[:,6])
ax2.hist(train.iloc[:,6])


from scipy import stats
z=np.abs(stats.zscore(train['Reputation']))


for i in range(0,len(num_cols)):
    print(num_cols[i])
    q75, q25 = np.percentile(train[num_cols[i]], [75 ,25])
    iqr=q75-q25
    min=q25-(iqr*1.5)
    max=q75+(iqr*1.5)
    print(max)
    print(min)
    train.loc[train.loc[:,num_cols[i]] < min,num_cols[i]]=np.nan
    train.loc[train.loc[:,num_cols[i]] > max,num_cols[i]]=np.nan



#Create dataframe with missing percentage
missing_val = pd.DataFrame(train.isnull().sum())

#Reset index
missing_val = missing_val.reset_index()

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

#Calculate percentage
missing_val['Missing_percentage']= (missing_val['Missing_percentage']/len(train))*100

#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)

#save output results 
#missing_val.to_csv("Miising_perc.csv", inex = False)


#####imputing mean values
train=train.dropna(axis=0)

#from sklearn.preprocessing import Imputer
#values = train.values
#imputer = Imputer(strategy='median')
#transformed_values = imputer.fit_transform(values)
#train=pd.DataFrame(transformed_values).round(2)
#train.columns=columns







import numpy as np
import statsmodels.api as sm
import pylab

#test = np.random.normal(0,1, 1000)

sm.qqplot(num_data.iloc[:,3], line='45')
pylab.show()



from sklearn import preprocessing 
  
""" MIN MAX SCALER """
from sklearn.preprocessing import MinMaxScaler  
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
  
# Scaled feature 
x_after_min_max_scaler = min_max_scaler.fit_transform(num_data) 
num_data=pd.DataFrame(x_after_min_max_scaler)  





#train['Reputation']=(train['Reputation'] - train['Reputation'].astype(int).min())/(train['Reputation'].astype(int).max() - train['Reputation'].astype(int).min())

#train['Reputation'] = (train['Reputation'] - train['Reputation'].mean())/train['Reputation'].std()

####feature selection
df_corr=train.loc[:,num_cols]
#set the widhth and height of the plot
f,ax=plt.subplots(figsize=(7,5))
    
corr=df_corr.corr()             

 #Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)    
###features reduction
train=train.drop(['ID','Username'],axis=1)
            
  ##########splitting data 
X_train=train.iloc[:,0:4] 
y_train=train.iloc[:,4] 
from sklearn.model_selection import train_test_split               
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)



#####mulitlinear regression
#from sklearn.linear_model import LinearRegression
#regressor=LinearRegression()
#regressor.fit(X_train,y_train)
####predicting 
#y_pred_ml=regressor.predict(X_test)
##regressor.summary()
#
#########building the optimal model
#
#import statsmodels.formula.api as sm
#X=np.append(arr=X,values=np.ones((737453,1)).astype(int),axis=1)
#
#X1=X.iloc[:,:].values
#X_opt=X1[:,:]
#
#import statsmodels.formula.api as sm
#
#regressor_OLS=sm.OLS(endog=y,exog=X).fit()
#regressor_OLS.summary()
#
#################
#from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,regression,mean_squared_error
#from math import sqrt
#
#
#
#def metrics(y_test,y_pred):
#    MSE=mean_squared_error(y_test,y_pred)
#    
#    MAE=mean_absolute_error(y_test,y_pred)
#    rmse=sqrt(mean_squared_error(y_test, y_pred))
#    r2=r2_score(y_test,y_pred)
#    adj_r2=1-(1-r2**2)*((X_test.shape[1]-1)/(X_test.shape[0]-X_test.shape[1]-1))
#    return MSE,MAE,rmse,r2,adj_r2
#    
#
#metrics(y_test,y_pred_ml)
###############decision tree
#from sklearn.tree import DecisionTreeRegressor
#tree_regressor=DecisionTreeRegressor()
#tree_regressor.fit(X_train,y_train)
#y_dec_pre=tree_regressor.predict(X_test)
##metrics(y_test,y_dec_pre)


#####random forest regressor
from sklearn.ensemble import RandomForestRegressor
tree_rf=RandomForestRegressor()
tree_rf=tree_rf.fit(X_train,y_train)
#y_pre_rf=tree_rf.predict(X_test)
##metrics(y_test,y_pre_rf)



####saving to disc
model=pickle.dump(tree_rf,open('upvotes.pkl','wb'))


###loading model to compare the results
model=pickle.load(open('upvotes.pkl','rb'))

print(tree_rf.predict([[0,3942.0,2.0,7855.0]]))














