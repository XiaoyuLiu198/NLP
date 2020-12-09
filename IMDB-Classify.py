#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import warnings
import re
warnings.filterwarnings('ignore')
trainset=pd.read_csv('D:\PyCharm Community Edition 2020.1.3\datasets\\train.csv')
testset=pd.read_csv('D:\PyCharm Community Edition 2020.1.3\datasets\\test.csv')
trainset.head()


# In[4]:


testset.head()


# In[5]:


#split and extract words
train_word_senti=trainset.copy(deep=True)
for i,comments in enumerate(trainset['text']):
    train_word_senti['text'].iloc[i]=re.sub("[^a-zA-Z]"," ",comments).split()
    train_word_senti['sentiment'].iloc[i]=trainset['sentiment'].iloc[i]
train_word_senti.head()


# In[6]:


test_word_senti=testset.copy(deep=True)
for i,comments in enumerate(testset['text']):
    test_word_senti['text'].iloc[i]=re.sub("[^a-zA-Z]"," ",comments).split()
    test_word_senti['sentiment'].iloc[i]=testset['sentiment'].iloc[i]
test_word_senti.head()


# In[7]:


import nltk
#nltk.download('stopwords')


# In[8]:


from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))


# In[9]:


train_o=train_word_senti.copy(deep=True)
test_o=test_word_senti.copy(deep=True)
def delet_stw(sets,n_ds):
    for i,strings in enumerate(sets['text']):
        n_string=[]
        for words in strings:
            if words not in en_stops:
                n_string.append(words)
        n_ds['text'].iloc[i]=n_string
    return n_ds
trainset=delet_stw(train_word_senti,train_o)
testset=delet_stw(test_word_senti,test_o)


# In[10]:


testset.head()


# Now lets compare the words left after deleting the stopwords. Focus on the first string in trainset. I listed the deleted words as followed.

# difference=[]
# for words in train_word_senti['text'].iloc[0]:
#     if words not in train['text'].iloc[0]:
#         difference.append(words)
# print(difference)

# import sklearn
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidfvectorizer = TfidfVectorizer(analyzer='word')
# tfidf_wm=train.copy(deep=True)
# tfidf_wm.columns=['word_vec','label']
# 
# for i in range(len(train)):
#     tfidf_wm['word_vec'].iloc[i] = tfidfvectorizer.fit_transform(train['text'].iloc[i])
# 

# embedding_dim=16
# 
# model = Sequential([
#   vectorize_layer,
#   Embedding(vocab_size, embedding_dim, name="embedding"),
#   GlobalAveragePooling1D(),
#   Dense(16, activation='relu'),
#   Dense(1)
# ])

# #!pip install hub
# import hub
# embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
# hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                            dtype=tf.string, trainable=True)

# sets=pd.concat([trainset,testset]) 
# sets

# In[11]:


#import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
  
warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec 
sets=pd.concat([trainset,testset],axis=0,ignore_index=True)   
# Replaces escape character with space 
f0 = sets['text'].replace(",", " ") 
#f1 = f.replace("<br/>", " ")
data = [] 
  
# iterate through each sentence in the file 
for m in range(len(f0)-1):
    f2=str(f0[m+1])
    for i in sent_tokenize(f2): 
        temp = [] 
        # tokenize the sentence into words 
        for j in word_tokenize(i): 
            temp.append(j.lower()) 
        data.append(temp) 
  
  
# Create Skip Gram model 
#model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
#                                             window = 5, sg = 1) 
#v1 = word2vec.wv['impressed']
#print(v1)


# In[12]:


sets=pd.concat([trainset,testset],axis=0,ignore_index=True)
sets


# In[13]:


model2 = gensim.models.Word2Vec(data, min_count = 2, size = 20, 
                                             window = 5, sg = 1) 


# In[14]:


train=trainset.copy(deep=True)
test=testset.copy(deep=True)


# In[15]:


def get_set(dataset,targetset):
    f1 = dataset['text'].replace(",", " ") 
#f1 = f.replace("<br/>", " ")
    for m in range(len(f1)-1):
        f2=str(f1[m+1])
        vector = []
        for i in sent_tokenize(f2):
            for j in word_tokenize(i): 
                try:
                    vector_i=model2.wv[j.lower()]
                    vector.append(vector_i)
                except:
                    continue
        vec=np.mean([vector[token] for token in range(len(vector))], axis=0).tolist()
        targetset.iloc[m,0]=vec
        targetset.iloc[m,1]=dataset.iloc[m,1]
        targetset=targetset.iloc[0:-1,]
    return targetset


# def get_mean_vector(word2vec_model, words):
#     # remove out-of-vocabulary words
#     words_kept = [word for word in words if word in word2vec_model.vocab]
#     if len(words_kept) >= 1:
#         return np.mean(word2vec_model[words], axis=0)
#     else:
#         return []

# In[16]:


f1 = trainset['text'].replace(",", " ")
for m in range(len(f1)-1):
    f2=str(f1[m+1])
    vector = []
    for i in sent_tokenize(f2):
        for j in word_tokenize(i): 
            try:
                vector_i=model2.wv[j.lower()]
                vector.append(vector_i)
            except:
                continue
    vec=np.mean([vector[token] for token in range(len(vector))], axis=0).tolist()
    train.iloc[m,0]=vec
    train.iloc[m,1]=trainset.iloc[m,1]
f1 = testset['text'].replace(",", " ")
for m in range(len(f1)-1):
    f2=str(f1[m+1])
    vector = []
    for i in sent_tokenize(f2):
        for j in word_tokenize(i): 
            try:
                vector_i=model2.wv[j.lower()]
                vector.append(vector_i)
            except:
                continue
    vec=np.mean([vector[token] for token in range(len(vector))], axis=0).tolist()
    test.iloc[m,0]=vec
    test.iloc[m,1]=trainset.iloc[m,1]


# In[17]:


#train=get_set(trainset,train)
#test=get_set(testset,test)
train.head()


# In[18]:


train=train.replace("neg",0)
train=train.replace("pos",1)
test=test.replace("neg",0)
test=test.replace("pos",0)


# In[19]:


test_1=pd.DataFrame(test['text'].tolist(), index= test.index)
test_1=test_1.iloc[:,0:20]
test_1["sentiment"]=test["sentiment"]
train_1=pd.DataFrame(train['text'].tolist(), index= train.index)
train_1=train_1.iloc[:,0:20]
train_1["sentiment"]=train["sentiment"]


# In[20]:


x_train=train_1.iloc[:-1,0:20]
y_train=train_1.iloc[:-1,20]
x_test=test_1.iloc[:-1,0:20]
y_test=test_1.iloc[:-1,20]


# In[74]:


##check if data is balanced
len(train_1[train_1['sentiment']==0])


# In[40]:


import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import GaussianNB
lists=np.arange(0,1,0.05)
scoreList = []
accuracies = {}
for i in lists:
    clf = GaussianNB(var_smoothing=i)  # n_neighbors means k
    clf.fit(x_train,y_train)
    scoreList.append(clf.score(x_test,y_test))
    
plt.plot(np.arange(0,1,0.05), scoreList)
plt.xticks(np.arange(0,1,0.05))
plt.xlabel("var_smoothing")
plt.ylabel("Score")
plt.show()


# In[44]:


from sklearn.metrics import classification_report
clf = GaussianNB(var_smoothing=0.7)
clf.fit(x_train,y_train)
predictions = clf.predict(x_train)
print(classification_report(y_train,predictions))


# In[32]:


##SVM
from sklearn import svm
param = {'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,3,5,7,9],'gamma' : ('auto','scale')}
grids = GridSearchCV(svm.SVC(),param,cv=5)
grids.fit(x_train,y_train)
grid_search.best_params_


# In[ ]:


#clf = svm.SVC(C=)
clf.fit(x_train,y_train)
predictions = model.predict(X_test) 
print(classification_report(y_test, predictions))


# In[ ]:


from sklearn.model_selection import KFold

ntrain = x_train.shape[0]
ntest = x_test.shape[0]
SEED = 0
NFOLDS = 7 
kf = KFold(n_splits = NFOLDS, shuffle=False)

def get_kfold_predict(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[ ]:


rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=13, 
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)

ada = AdaBoostClassifier(n_estimators=10, learning_rate=0.01)

svm=SVC(C=0.03,gamma= 1, kernel='poly')
#gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)

rf_oof_train, rf_oof_test = get_kfold_predict(rf, x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_kfold_predict(ada, x_train, y_train, x_test) # AdaBoost 
svm_oof_train, svm_oof_test = get_kfold_predict(svm, x_train, y_train, x_test) # Gradient Boost

x_train_f = np.concatenate((rf_oof_train, ada_oof_train, svm_oof_train), axis=1)
x_test_f = np.concatenate((rf_oof_test, ada_oof_test, svm_oof_test), axis=1)

from xgboost import XGBClassifier

gbm = XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, 
                     colsample_bytree=0.8, nthread= -1, scale_pos_weight=1).fit(x_train_f, y_train)
predictions = gbm.predict(x_test_f)
prediction_eva=f1_score(y_test[:86], predictions, average='weighted')
print(prediction_eva)

