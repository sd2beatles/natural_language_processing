import numpy as np
import tensorflow  as tf
from __future__ import print_function,division
from future.utils import iteritems
from builtins import range
from sklearn.metrics.pairwise import pairwise_distances
from google.colab import drive
from zipfile import ZipFile
fileName='...../glove.6B.50d.txt.zip'
with ZipFile(fileName,'r') as zipfile:
  zipfile.extractall()
  
word2Vec={}
embedding=[]
idx2word=[]
with open('/content/glove.6B.50d.txt') as f:
  for line in f:
    #each line in the file is just a space separated list of items 
    #first we should make it into the single list 
    values=line.split()
    #each list contains word in the first index and the rest are the vector components
    word=values[0]
    vec=np.array(values[1:],dtype=np.float32)
    word2Vec[word]=vec
    embedding.append(vec)
    idx2word.append(word)
  print(f'Found {len(word2Vec)}')

#to convert the list of embedding into numpy.array
embedding=np.array(embedding)
V,D=embedding.shape

def dist1(a,b):
  return np.linalg.norm(a-b)

def dist2(a,b):
  return 1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

dist,metric=dist2,'cosine'


def find_analogies(w1,w2,w3):
  for w in (w1,w2,w3):
    if w not in word2Vec: #note that word2Vec is the dictionary with keys for word and values for its vector
      print(f"{w} is not avialbe")
      return
  #to get the vector for each given word
  king=word2Vec[w1]
  man=word2Vec[w2]
  woman=word2Vec[w3]
  v0=king-man+woman
   
  #the first vector of v0 contains D*1 vector while the second list contains all the word vectors in our embedding matrix
  #Therefore,we could estimate that v0=1*D  and our distance function to compute the distance v0 and v(i) in our embedding matrix for n times
  #The results [[d1,d2,d3,......dn]]. If you print out the shape of the result,you will soon find that it has an unnecessarydimension
  #There are many ways we could do to remove this extra dimension. 
  distance=pairwise_distances(v0.reshape(1,D),embedding,metric=metric)
  distance=np.squeeze(distance)
  #to find the index which has the shortest distance between two words
  idx=distance.argmin()
  #this index refers back to the closes word so we can use the index to word dictionary to map
  best_word=idx2word[idx]
  print(w1,"-",w2,"=",best_word,"-",w3)


def nearest_neighbors(w,n=5):
  if w not in word2Vec:
    print(f"{w} is not in the dictionary")
    return
  v=word2Vec[w]
  distances=pairwise_distances(v.shape(1,D),embedding,metric=metirc)
  distances=distances.squeeze()
  idxs=distances.argsort()[1:n+1]
  for idx in idxs:
    print(f"{idx2word[idx]}")

find_analogies('king','man','woman')
find_analogies('france', 'paris', 'london')
find_analogies('france', 'paris', 'rome')
find_analogies('paris', 'france', 'italy')
find_analogies('france', 'french', 'english') 
find_analogies('japan', 'japanese', 'chinese')
find_analogies('japan', 'japanese', 'italian')
find_analogies('japan', 'japanese', 'australian') 
find_analogies('december', 'november', 'june')
