#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#from nltk.tokenize import RegexpTokenizer  
#from nltk.corpus import stopwords
#from nltk.stem.snowball import SnowballStemmer


# 2. Import data

# In[2]:


data = pd.read_csv("./spam.csv", encoding='latin-1', usecols=["v1","v2"])


# In[3]:


data.head()


# In[4]:


data = data.rename(columns={"v1":"label", "v2":"text"})


# In[5]:


data.head()


# In[6]:


data.label.value_counts()


# Convert labels to numerical variables

# In[7]:


data['label_num'] = data.label.map({'ham':0, 'spam':1})
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#le.fit(data['label'])
#label = le.transform(data['label'])
#print(np.unique(label))
#print(np.unique(data['label_num']))


# In[8]:


data.head()


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train,X_test,y_train,y_test = train_test_split(data["text"],data["label_num"], test_size = 0.2, random_state = 10)


# In[11]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[13]:


vect = TfidfVectorizer()


# In[14]:


vect.fit(X_train)


# In[15]:


print(vect.get_feature_names()[0:10])
print(vect.get_feature_names()[-10:])


# In[16]:


print("Vocabulary size: {}".format(len(vect.vocabulary_)))
#print("Vocabulary content:\n {}".format(vect.vocabulary_))


# In[17]:


X_train_df = vect.fit_transform(X_train)


# In[18]:


X_train_df[:3].nonzero()


# In[19]:


prediction = dict()
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_df,y_train)


# In[20]:


X_test_df = vect.transform(X_test)
prediction["Multinomial"] = model.predict(X_test_df)


# In[21]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[22]:


accuracy_score(y_test,prediction["Multinomial"])


# In[23]:


print(classification_report(y_test,prediction["Multinomial"]))


# In[24]:


conf_mat = confusion_matrix(y_test, prediction['Multinomial'])
print(conf_mat)
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]


# In[25]:


print(conf_mat_normalized)


# In[26]:


print("train score:", model.score(X_train_df, y_train))
print("test score:", model.score(X_test_df, y_test))

