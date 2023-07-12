#!/usr/bin/env python
# coding: utf-8

# Name = Pawan Kumar Yadav
# Task 3 = Success of an upcoming movie
# Bharat Intern
# Business Analytics Intern
Import the libraries and load the dataset
# In[2]:


import numpy as np 
import pandas as pd


# In[4]:


df = pd.read_csv("C:/Users/pawan/Downloads/movie_success_rate.csv")


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.columns


# In[8]:


df['Genre'].value_counts()


# In[9]:


df['Director'].value_counts()


# In[10]:


df['Actors'].value_counts()


# In[11]:


import seaborn as sns
sns.heatmap(df.isnull())


# In[27]:


df = df.fillna(df.median())


# In[13]:


df.columns


# In[14]:


x = df[['Year',
       'Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)',
       'Metascore', 'Action', 'Adventure', 'Aniimation', 'Biography', 'Comedy',
       'Crime', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War',
       'Western']]
y = df['Success']


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.1,stratify=y)


# In[16]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train,y_train)


# In[17]:


log.score(x_test,y_test)


# In[18]:


from sklearn.metrics import confusion_matrix
clf = confusion_matrix(y_test,log.predict(x_test))


# In[19]:


sns.heatmap(clf,annot=True)


# In[21]:


#normalising all columns
x_train_opt = x_train.copy()
x_test_opt = x_test.copy()


# In[22]:


from sklearn.preprocessing import StandardScaler
x_train_opt = StandardScaler().fit_transform(x_train_opt)
x_test_opt = StandardScaler().fit_transform(x_test_opt)


# In[24]:


log.fit(x_train_opt,y_train)


# In[25]:


log.score(x_test_opt,y_test)


# In[32]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train,y_train)
tree.score(x_test,y_test)


# In[34]:


from sklearn.metrics import confusion_matrix
clf = confusion_matrix(y_test,tree.predict(x_test))


# In[35]:


clf


# In[36]:


sns.heatmap(clf,annot=True)


# In[ ]:




