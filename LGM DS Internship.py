#!/usr/bin/env python
# coding: utf-8

# # Name: Utsav Barman
# # Task 1 - Iris Flower Classification ML Project

# In[1]:


#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# In[3]:


#Importing DataSet
df = pd.read_csv('iris.data', sep=',', names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
df.to_csv("iris.csv", sep=',')


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.describe()


# In[9]:


df.tail()


# In[10]:


df.isnull()


# In[11]:


df.isnull().sum()


# In[12]:


df['species'].value_counts()


# In[13]:


sb.pairplot(df, hue='species')
plt.show()


# In[14]:


df.hist(edgecolor='red',figsize=(12,12))
plt.show()


# In[15]:


df.corr()


# In[16]:


sb.heatmap(df.corr(method='pearson'), cmap="BuPu")
plt.show()


# In[17]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
df.head(10)


# In[18]:


from sklearn.model_selection import train_test_split

X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 30,random_state=0)


# In[19]:


X_Train.shape


# In[20]:


X_Test.shape


# In[21]:


Y_Train.shape


# In[22]:


Y_Test.shape


# In[23]:


# Feature Scaling to bring all the variables in a single scale.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.transform(X_Test)

# Importing some metrics for evaluating  models.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[24]:


# Model Creation
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)

# Model Training
classifier.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Log = classifier.predict(X_Test)


# In[25]:


print("Accuracy of Log_Reg:", accuracy_score(Y_Test, Y_Pred_Log)*100)


# In[26]:


print(classification_report(Y_Test, Y_Pred_Log))


# In[27]:


confusion_matrix(Y_Test,Y_Pred_Log)


# In[ ]:




