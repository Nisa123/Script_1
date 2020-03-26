#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


#LOADING THE DATA From data.csv
df = pd.read_csv('breast_cancer.csv')
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

#Complete data set:
X = df[df.columns[2:32]]
Y = df['diagnosis']
Y = Y.values.reshape(Y.shape[0],1)


#train set (80%):
train_X = X.loc[0:454,X.columns[0:]]
train_Y = Y[0:455]

#test set (20%):
test_X = X.loc[0:143,X.columns[0:]]
test_Y = Y[0:144]


# In[7]:


#training set:

mean = train_X.mean()
std_error = train_X.std()
train_X = (train_X - mean)/std_error

#test set:
mean = test_X.mean()
std_error = test_X.std()
test_X = (test_X - mean)/std_error


# In[4]:


#hypothesis i.e. y = a = sigmoid(z) where z = w^TX + b
def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[5]:


def random_init(dim):
    w = np.zeros((dim,1))
    b = 0
    
    return w,b


# In[6]:


#hypothesis in logistic regression is y = a = sigmoid(z) = sigmoid(w^TX + b)
def propo(w,b,X,Y):
    
    m = X.shape[0]
    
    #forward propogation
    z = np.dot(X,w) + b
    a = sigmoid(z)
    cost = -np.sum(Y*np.log(a) - (1-Y)*np.log(1-a))/m
    
    
    #backpropogation:
    dz = a-Y
    dw  = np.dot(np.transpose(X),dz)/m
    db = np.sum(dz)/m
    
    grad = {
        "dw":dw,
        "db":db
    }
    return grad,cost


# In[7]:


#Gradient descent
def optim(w,b,X,Y,learning_rate,num_iteration):
    costs = []
    
    for i in range(num_iteration):
        grads, cost=propo(w,b,X,Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        #updating w and b
        w  = w - learning_rate*dw
        b  = b - learning_rate*db
          
        if(i%100==0):
            costs.append(cost)
        
    params= {
        "w":w,
        "b":b
    }
    grads = {
        "dw":dw,
        "db":db
    }
    return params,grads,costs


# In[10]:


#random init of w,b
w,b = random_init(train_X.shape[1])

#forward, backward & grad. descent:
params,grads,costs = optim(w,b,train_X,train_Y,0.01,2000)

def predict(w,b,X):
    a = sigmoid(np.dot(X,w) + b)
    return a

def oneORzero(x):
    if(x>=0.5):
        return 1
    elif(x<0.5):
        return 0


# In[11]:


# Accuracy for test set:
temp = predict(params["w"],params["b"],test_X)
test_prediction = np.array(list(map(oneORzero,temp)))
test_prediction = test_prediction.reshape((test_prediction.shape[0],1))


# In[13]:


print("Test Accuracy = ",(100 - np.mean(np.abs(test_prediction - test_Y))*100))


# In[ ]:




