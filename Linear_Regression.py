#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


from sklearn.datasets import make_regression


# In[3]:


from matplotlib import pyplot as plt


# # DataSet

# In[4]:


x, y=make_regression(n_samples=100,n_features=1,noise=10)
plt.scatter(x,y)


# In[5]:


print(x.shape)
y=y.reshape(y.shape[0],1)
print(y.shape)


# In[6]:


X=np.hstack((x,np.ones(x.shape)))
X.shape
X


# In[7]:


O=np.random.randn(2,1)
O


# # Modele

# In[8]:


def modele(x,o):
    return x.dot(o)
modele(X,O)


# In[9]:


plt.scatter(x,y)
plt.plot(x,modele(X,O),c='r')


# # Fonction Cout

# In[10]:


def const_function(x,y,o):
    m=len(y)
    return 1/(2*m)*np.sum((modele(X,o)-y)**2)


# In[11]:


const_function(X,y,O)


# # gradient descent 

# In[12]:


def grad(x,o,y):
    m=len(y)
    return 1/m*((X.T).dot((modele(X,O)-y)))
grad(X,O,y)


# In[13]:


def gradiant_desc(x,y,o,learning_rate,n_iterations):
    const_history=np.zeros(n_iterations)
    for i in range(n_iterations):
        o=o-learning_rate*grad(x,o,y)
        const_history[i]=const_function(x,y,o)
    return [o,const_history]


# In[14]:


O_final=gradiant_desc(X,y,O,0.001,1000)[0]


# In[15]:


O_final


# In[16]:


const_history=gradiant_desc(X,y,O,0.001,1000)[1]


# In[17]:


prediction=modele(X,O_final)
plt.scatter(x,y,c='b')
plt.plot(x,prediction,c='r')


# # Evaluation

# In[19]:


plt.plot(range(1000),const_history)


# In[21]:


def coef_determination(y,pred):
    u=((y-pred)**2).sum()
    v=((y-y.mean())**2).sum()
    return 1-(u/v)


# In[22]:


coef_determination(y,prediction)

