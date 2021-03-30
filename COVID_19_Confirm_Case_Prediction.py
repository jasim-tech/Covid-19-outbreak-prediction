#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARY

# In[1]:


import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


# # DATA LOAD

# In[2]:


#Data read using Pandas
data=pd.read_csv("D:\projects\Covid-19 outbreak prediction\COVID-19 Dataset\coronaCases.csv",sep=",")

data=data[['id','cases']] 
data.tail()


# # DATA PREPROCESSING

# In[11]:


x=np.array(data['id']).reshape(-1,1) #Data coversion to numpy array
y=np.array(data['cases']).reshape(-1,1)
plt.plot(y,'-m')
polyfeat=PolynomialFeatures(degree=4) # 4 degree polynomial
x=polyfeat.fit_transform(x)
plt.xlabel('X')
plt.ylabel('Y')


# # TRAINING DATA

# In[12]:


model=linear_model.LinearRegression()
model.fit(x,y)
accuracy=model.score(x,y) #Accuracy Calculation
print(f'Accuracy:{round(accuracy*100,3)}%')

Test=model.predict(x) #prediction
plt.plot(Test,'--b')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# # PREDICTION

# In[48]:


Days=3
print(f'Predicted Case after {Days} days:',end='')
print(round(int(model.predict(polyfeat.fit_transform([[234+Days]])))/1000000,3),'Million')


#                             # Thanks

# In[ ]:




