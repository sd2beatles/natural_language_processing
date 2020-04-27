## 1. Data preprocessing
### 1.1 Data Import 

We should make a separate dictionary to store the unique code for each movie since there are duplicated entries in it. 
The key is code and  its corresponding values contain english title and rating. Right after we have our movie codes in use, 
then we assign the extra information to the columns of comment_data if id codes are matched.   
```python
stop_words=pd.read_csv(ret[0],sep='\s+',header=None)
comment_data=pd.read_csv(ret[1])
movie_data=pd.read_csv(ret[2])
```
We should make a separate dictionary to store the unique code for each movie since there are duplicated entries in it. 
The key is code and  its corresponding values contain english title and rating. Right after we have our movie codes in use, then
we assign the extra information to the columns of comment_data if id codes are matched.   

```python
storage={code:{'eng':eng,'rating':user} for eng,code,user in zip(movie_data.title_ENG,movie_data.link_code,movie_data.user_rating)}
```

### 1.2 Data Preprocessing

First, we need to look into the measurements of scores from three different groups. The column score refers to the rate assigned 
by an individual leaving the comment whereas the others are floating numbers , indirectly indicating that they are average based values.   
Furthermore, the evaluation of critics on the movies is measured as a percentage.  

We definitely take some measures to tackle these mismatch problems. We should take % away from the avg of critics score and convert it 
into the floating type

```python
comment_data['critics_score']=comment_data.critics_score.map(lambda x: float(x.split('%')[0])/10)
```

## 2. Natural Language Processing

## 2.1 Preprocessing 


Let's label the score into the binary. If the score ranges between 1 and 4, it is now labeld as 0 indicating 'unsatisfactory' or 'negative' while labled as 1 
if it belongs to 9 and 10. Any score apart from the ranges are automatically excluded. 

```python
nlp_data=comment_data[['text','score']]
nlp_data=nlp_data[(nlp_data.score<5)|(nlp_data.score>8)]
#double check if the scores are properly selected

nlp_data.score=nlp_data.score.map(lambda x: 1 if x>8 else 0)
nlp_data.columns=['document','label']


from sklearn.model_selection import train_test_split
from collections import Counter
from gensim.models import Word2Vec


train_data,test_data=train_test_split(nlp_data,test_size=0.2,stratify=nlp_data.label)
```

## 2.2  Removing the missing values

Since our attention is called to the prediction of users given the comments, we should get rid of 
any instance with missing values of commetnts.

```python
def remove_empty_review(X):
  empty_idx=[]
  for index,value in enumerate(X):
    if len(str(value))==0:
      empty_idx.append(index)
  #to avoid the disruption to the orders of index,
  #we will delete the highest index.To do so,we should rearrange our empty_index
  #from the highest to the lowest number
  if len(empty_idx)==0:
    print('No missing values for the comments are detected!') 
  else:
    empty_idx=sorted(empty_idx,reverse=True)
    for index in empty_idx:
      del X[index]
  
  return X
  
train_data=remove_empty_review(train_data)
test_data=remove_empty_review(test_data)
  ```
  
  ```python
 #only extract the first column which represetns the list of stopwords
stopwords=stop_words.iloc[:,0]
okt=Okt()


def tokenized(sentence,max_char=10):
  tokenizer=[]
  sentence=str(sentence)
  sentence=re.sub('[ ]+',' ',sentence)
  sentence=re.sub('[^\w]+|ㅋ+|ㅎ+',' ',sentence)
  if (len(sentence))>max_char:
        sentence=sentence[:max_char]
  token_ls=okt.pos(sentence,stem=True)
  for token in token_ls:
    if token[0] not in stopwords and len(token[0])>1:
      tokenizer.append(token[0])
  return tokenizer


def prepare_csv(train_data):
  train_data,val_data=train_test_split(train_data,test_size=0.2)
  train_data.to_csv('cache/train_data.csv',index=False) 
  val_data.to_csv('cache/val_data.csv',index=False)
  test_data.to_csv('cache/test_data.csv',index=False)


@memory.cache
def read_fields(train_data,fix_length=10):
  prepare_csv(train_data)
  text=data.Field(sequential=True,tokenize=tokenized,batch_first=True,pad_first=True,fix_length=fix_length)
  label=data.Field(sequential=False,use_vocab=False)
  data_fields=[('document',text),('label',label)]
  trn,vld=data.TabularDataset.splits(path='cache/',train='train_data.csv',validation='val_data.csv',
                                 format='csv',skip_header=True,fields=data_fields)
  test=data.TabularDataset(path='cache/test_data.csv',format='csv',skip_header=True,fields=data_fields)
  
  text.build_vocab(
           trn,
           max_size=10,
           min_freq=2)
  label.build_vocab(trn)
  print("print the size of the stored word set")
  print(f"the size of vocab:{len(text.vocab)}")
  print('-'*1000)
  print(f'the dictionary of vocabuarary:\n{text.vocab.stoi}')
  print('-'*1000)
  print(f'the first instance of train data:\n{vars(trn[0])}')
  return trn,vld,test
 
 ```
 Let's check out whether every rows of each dataset are saved properly. We have two columns one of which is a text field storing 
 comments and the other saving lables. 

We will do the following tasks ,
 -  print the first instance of the trainset
 -  list all the stored vocabulary and each index number
 
 ```python
 
 trn,vld,test=read_fields(train_data)
 train_iter,val_iter=BucketIterator.splits((trn,vld)
                                          ,batch_size=3
                                          ,shuffle=True
                                          ,repeat=False)
                                          
#dl must be an itertor
#x_var be the name of input_feature
#y_var be the name of output

class BatchWrapper:
    def __init__(self,dl,x_var,y_var):
        self.dl=dl
        self.x_var=x_var
        self.y_var=y_var
    def __iter__(self):
        for batch in self.dl:
            x=getattr(batch,self.x_var)
            if type(self.y_var)==list and len(self.y_var)>0:
                y=torch.cat([getattr(batch,feat).unsqueeze(1) for feat in self.y_var],dim=1).float()
            elif self.y_var==None:
                y=torch.zeros(1)
            else: 
                y=getattr(batch,self.y_var)
            yield(x,y)
    def __len__(self):
        return len(self.dl)

train_dl = BatchWrapper(train_iter, 'document','label')
valid_dl=BatchWrapper(val_iter,'document','label')


import torch

class GRU(nn.Module):
  def __init__(self,n_layers,hidden_dim,n_vocab,embed_dim,n_classes,dropout_p=0.2):
    super(GRU,self).__init__()
    self.n_layers=n_layers
    self.hidden_dim=hidden_dim
    self.embed=nn.Embedding(n_vocab,embed_dim)
    self.dropout=dropout_p
    self.gru=nn.GRU(embed_dim,hidden_dim,num_layers=n_layers,batch_first=True)
    self.out=nn.Linear(hidden_dim,n_classes)
  
  def __init__state(self,batch_size=1):
    #first create a weight tensor
    weight=next(self.parameters()).data
    #we should reshape our weight to the format of (the number of layers,batch_size, the number of hidden dimenison) by using .new
    #And next since the weight consists of the initialied weights and we have to set them all to zeros by .zero_()
    return weight.new(self.n_layers,batch_size,self.hidden_dim).zero_()

  def forward(self,x):
    x=self.embed(x)
    #setting all the hidden states to zeros
    h_0=self.__init__state(batch_size=x.size(0))
    #GRU returns the following outcomes:batch size,the length of sequence(or time steps),and the size of hidden state
    #Therfore,the shape of x is (batch size,the length of sequence,the number of hidden states)
    x,_=self.gru(x,h_0)
    #extract only the final states sent to the ones in the fully connected layers
    h_t=x[:,-1,:]
    logit=self.out(h_t)
    return logit
 
import time
import sys
import tqdm
start=time.time()
EPOCH=3
start=time.time()
EPOCH=3
for epoch in range(1,EPOCH+1):
    running_loss=0
    model.train()
    for x,y in tqdm.tqdm(train_dl):
        opt.zero_grad()
        pred=model(x)
        #In our case, each iteration returns b*c matrix where b represents the number of batch while c refers to the number of classes
        #Since the function we are dealing with cross entropy, we should pick the maxium value for each instance ie) logit.max(1) 
        #tensor.max(axis=) or tensor.min(axis=) returns values and indices of the chosen operation. Note that 0 is for indicies
        #whiel 1 for indices
        #Now,we need to compare our predicted classes with the actual ones. Before the comparison, make sure that the shapes of 
        #both must match to each other. logit.max(1)[1].view(y.size())
        loss=F.cross_entropy(pred,y)
        loss.backward()
        opt.step()
        running_loss+=loss.data*x.size(0)
      except:
        pass
    epoch_loss=running_loss/len(trn)
  
    running_corrects=0
    #Evaluation begins
    val_loss=0
    corrects=0
    model.eval()
    for x,y in valid_dl:
        pred=model(x)
        loss=F.cross_entropy(pred,y)
        val_loss+=loss.data*x.size(0)
        running_corrects+=(pred.max(1)[1].view(y.size()).data==y.data).sum()
    val_loss/=len(vld)
    epoch_accuracy=running_corrects/len(trn)
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
    print('Corrects: {}'.format(epoch_accuracy))
    print(f'time taken:{time.time()-start}')
    print('-'*100)
 ```




