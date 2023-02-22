#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the dependencies
import numpy as np
import pandas as pd
#split data to train and test
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#Data collection and processing
heart_data=pd.read_csv("Documents/ml/heart/heart_disease_data.csv")


# In[3]:


heart_data.head()


# In[4]:


#to check rows and columns in the data set
heart_data.shape


# In[5]:


#getting some info about the dataset
heart_data.info()


# In[6]:


#we can also check null value 
heart_data.isnull().sum()


# In[7]:


#stastical measures about the data
heart_data.describe()
# %does not represent percentage it means percentile


# In[8]:


#checking the distribution of target variable
heart_data['target'].value_counts()


# In[9]:


#The target column is what the machine learning model is trying to predict.
#The feature columns are what the machine learning model uses to make the prediction. 


# In[10]:


#splitting the features and target to predict the target
x=heart_data.drop(columns='target',axis=1)
#for column axis=1 and axis=0 for row
y=heart_data['target']


# In[11]:


print(x)


# In[12]:


print(y)


# In[13]:


#they are seperated now
#now splitting the data into train data and test data


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[15]:


print(x.shape,x_train.shape,x_test.shape)


# In[16]:


#model training


# In[17]:


#logicstic regression model
model=LogisticRegression()


# In[18]:


#training the Logistic model with training data
model.fit(x_train,y_train)


# In[19]:


#model evaluation
#accuracy score on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print("accuracy on training data",training_data_accuracy)


# In[20]:


#now accuracy score in test data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("accuracy on test data",test_data_accuracy)


# In[24]:


#building a predictive system

input_data=(55,1,0,160,289,0,0,145,1,0.8,1,1,3)

#changing a input data into numpy array

ipt_data_np_array=np.asarray(input_data)

#reshape the np array as we are predicting for only one data
ipt_data_reshape=ipt_data_np_array.reshape(1,-1)
prediction=model.predict(ipt_data_reshape)
print(prediction)
if(prediction[0]==0):
    print("the person does not have heart disease")
else:
        print("the person have heart disease")


# In[ ]:




