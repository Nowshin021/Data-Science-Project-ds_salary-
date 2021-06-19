# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 00:55:49 2021

@author: nowshin
"""

import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import *
import numpy as np



#%%

df=pd.read_csv('salary_data_cleaned.csv')
print(df.head(3))
print(df['Rating'].dtype)

#%%
df=df[['avg_salary', 'Rating', 'company_age','Industry', 'Type of ownership','Sector','job_state',
               'Hourly', 'employer provided','python_jobs','ai_jobs']]

print(df.head(5))

#%%

#the catagorical data 

categorical_cols=[]
for column_name in df.columns:
    if df[column_name].dtype == object:
        categorical_cols.append(column_name)
#%%

#encoding 
df_encoded=pd.get_dummies(df, columns=categorical_cols)

#%%

#spliting x and y 
x=df_encoded.drop('avg_salary', axis=1)
y=df[['avg_salary']]

#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state = 42)

#%%
#making the model

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred_train = lr.predict(x_train)
y_pred = lr.predict(x_test)


#%%


from sklearn.metrics import *
accurecy_score_train = lr.score(x_train,y_train)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Acuurecy scores : ")
print("Score train dataset : ", accurecy_score_train*100)
print('MAE is %.3f'% mae)
print('MSE is %.3f'% mse)
























