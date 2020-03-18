```python
import numpy as np
import tensorflow  as tf

import numpy as np
import tarfile
from six.moves import urllib
import os


from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from gensim.models import KeyedVectors
import pandas as pd
from builtins import range
import sys
import matplotlib.pyplot as plt
import warnings
from google.colab import drive

drive.mount('/content/drive')

train=pd.read_csv('/content/drive/My Drive/natural_language_processing/train.txt',header=None,sep='\t')
test=pd.read_csv('/content/drive/My Drive/natural_language_processing/test.txt',header=None,sep='\t')
train.columns=['label','content']
test.columns=['label','content']
import gensim.downloader as api
warnings.filterwarnings("ignore")
wv = api.load('word2vec-google-news-300')

class GloveVectorizer:
    def __init__(self):
        word2Vec={}
        embedding=[]
        idx2word=[]
        with open(r'/content/glove.6B.50d.txt') as f:
            for line in f:
                value=line.split()
                word=value[0]
                vec=np.array(value[1:],dtye=np.float32)
                word2Vec[word]=vec
                embedding.append(vec) 
                idx2word.append(word)
        #save for later
        self.word2Vec=word2Vec
        self.embedding=embedding
        self.word2idx={word:idx for idx,word in enumerate(idx2word)}
        self.V,self.D=self.embedding.shape
        
    def fit(self,data):
        pass
    
    
    '''
    empty counts tells us how many sentences did't have any words we could find vectors for
    That is how we are going to deal witht words that do not appear in the preacher in word vector's list
    '''
    def transform(self,data):
        X=np.zeros((len(data),self.D))
        n=0
        emptycount=0
        for sentence in data:
            tokens=setence.lower().split()
            vecs=[]
            for word in tokens:
                if word in self.word2Vec:
                    vec=self.word2Vec[word]
                    vecs.append(vec)
            if len(vecs)>0:
                vecs=np.array(vecs)
                # if vecs is grater than zero, we will assign the mean to the X of n. If it does not, we increment empty count
                X[n]=vecs.mean(axis=0)
            else:
                emptycount+=1
            n+=1
        print(f"Number of samples with no words found:{emptycount}/{len(data)}")
        return X
'''
unlike gloverVectorizer,Word2Vectorizer does have both a lower and upper case. 
We can not use the wordvector likea a dicitonary but still obtain the vector by implemeting get_vector fucntion
If the word is not found, there is nothing we can do.
'''

class Word2Vectorizer:
  def __init__(self,wv):
    self.word_vectors=wv
  def fit(self,data):
    pass

  def transform(self,data):
    v=self.word_vectors.get_vector('king')
    self.D=v.shape[0]
    X=np.zeros((len(data),self.D))
    n=0
    emptycount=0
    for setence in data:
      token=setence.split()
      vecs=[]
      m=0
      for word in token:
        try:
          vec=self.word_vectors.get_vector(word)
          vecs.append(vec)
          m+=1
        except KeyError:
          pass
      if len(vecs)>0:
        vecs=np.array(vecs)
        #note that vecs is [d,]
        X[n]=vecs.mean(axis=0)
      else:
        emptycount+=1
      n+=1
    print(f"Number of samples with no words founds {emptycount}/{len(data)}")
    return X
  def fit_transform(self,data):
    self.fit(data)
    return self.transform(data)

vectorizer=Word2Vectorizer(wv)
Xtrain=vectorizer.fit_transform(train.content)
ytrain=train.label

Xtest=vectorizer.transform(test.content)
ytest=test.label

vectorizer=Word2Vectorizer(wv)
Xtrain=vectorizer.fit_transform(train.content)
ytrain=train.label

Xtest=vectorizer.transform(test.content)
ytest=test.label
```
      
    

