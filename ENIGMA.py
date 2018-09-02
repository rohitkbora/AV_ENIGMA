import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline 
import seaborn as sns
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
dummies = pd.get_dummies(df['Tag'])
df = df.join(dummies)
dummies_test = pd.get_dummies(df_test['Tag'])
df_test = df_test.join(dummies_test)
df.drop('Tag',inplace=True,axis=1)
df_test.drop('Tag',inplace=True,axis=1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
ss = StandardScaler()
#################  basice random forest 500trees  with filtering ####################################
x,x_test,y,y_test=train_test_split(ss.fit_transform(df.iloc[:,[1,2,4]]),df.iloc[:,5])
rf = RandomForestRegressor(n_estimators=500,random_state=1234)
rf.fit(x,y)
pred = rf.predict(x_test)
np.sqrt(mean_squared_error(y_test,pred))

#############  final file submission #################################

pred_final = rf.predict(ss.fit_transform(df_test.iloc[:,[1,2,4]]))
df_final =pd.DataFrame({'ID':df_test.iloc[:,0],'Upvotes':pred_final}).set_index('ID')
df_final.to_csv('D:\\AV\\ITBHU\\basic_RFregressor_500_trees.csv')
