#!/usr/bin/env python
# coding: utf-8

# ### Importing basic libraries

# In[40]:


import numpy as np
import pandas as pd


# ### Reading dataset

# In[41]:


df=pd.read_csv("/content/drive/MyDrive/Precily_Text_Similarity.csv")


# In[42]:


df.head()


# In[43]:


df.info()


# ### Dataset contain two columns and 3000 rows

# ### Checking for null values

# In[44]:


empty_idx=[]
for text in df.itertuples():
  if type(text)==str:
    if text.isspace():
      empty_idx.append(indx)
print(empty_idx)


# ### No null values present in the dataset.

# In[45]:


import nltk

from nltk.tokenize import word_tokenize
nltk.download("punkt")

from nltk.corpus import stopwords
nltk.download("stopwords")

from nltk.stem import PorterStemmer,WordNetLemmatizer
nltk.download("wordnet")
nltk.download('omw-1.4')


# ### Preprocessing the dataset

# In[46]:


def clean_text(text):
  token=word_tokenize(text.lower()) #case conversion + tokenization.


  #non alpha removal.
  ftoken=[i for i in token if i.isalpha()]

  #stop words removal
  stpwd=stopwords.words("english")
  stoken=[i for i in ftoken if i not in stpwd]

  #lemma.
  lemma=WordNetLemmatizer()
  ltoken=[lemma.lemmatize(i) for i in stoken]

  #joining list of msgs
  return " ".join(ltoken)


# In[47]:


df["text1"]=df["text1"].astype(str)
df["text2"]=df["text2"].astype(str)



# In[48]:


df["clean_text1"]=df["text1"].apply(clean_text)


# In[49]:


df["clean_text2"]=df["text2"].apply(clean_text)


# In[50]:


x=df["clean_text1"]
y=df["clean_text2"]


# ### Vectorization

# In[51]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[52]:


vec=TfidfVectorizer()
x=vec.fit_transform(x).toarray()
x


# In[53]:


y=vec.transform(y).toarray()
y


# ### Finding Similarities

# In[54]:


from sklearn.metrics.pairwise import cosine_similarity


# In[55]:


similarity_score = cosine_similarity(x, y)


# In[80]:


similarity_score


# In[82]:


similarity_score.min()


# In[83]:


similarity_score.max()


# In[84]:


similarity_score.max(axis=1)


# In[85]:


df['similarity_score']=similarity_score.max(axis=1)


# In[86]:


df.head()


# In[87]:


df.tail()


# In[ ]:




