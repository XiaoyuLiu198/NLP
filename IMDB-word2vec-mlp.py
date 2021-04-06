#!/usr/bin/env python
# coding: utf-8

# In[167]:


import pandas as pd
import numpy as np
import warnings
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
warnings.filterwarnings(action = 'ignore') 
import gensim 
from gensim.models import Word2Vec,Phrases 
import keras
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
nltk.download('wordnet')


# In[118]:


trainset=pd.read_csv('/Users/molly1998/Desktop/python//train.csv')
testset=pd.read_csv('/Users/molly1998/Desktop/python//test.csv')
dataset=pd.concat([trainset,testset],ignore_index=True)
dataset_comm=np.concatenate([trainset['text'],testset['text']],axis=0)
dataset_tar=np.concatenate([trainset['sentiment'],testset['sentiment']],axis=0)
len(dataset_comm)
len(dataset_tar)


# In[82]:


dataset_comm[1]


# In[83]:


en_stops = set(stopwords.words('english'))


# In[104]:


def clean_review(comments: str) -> str:
    #Remove non-letters
    letters_only = re.compile(r'[^A-Za-z\s]').sub(" ", comments)
    #Convert to lower case
    lowercase_letters = letters_only.lower()
    return lowercase_letters


def lemmatize(tokens: list) -> list:
    #Lemmatize
    tokens = list(map(WordNetLemmatizer().lemmatize, tokens))
    lemmatized_tokens = list(map(lambda x: WordNetLemmatizer().lemmatize(x,"v"), tokens))
    #Remove stop words
    meaningful_words = list(filter(lambda x: not x in en_stops, lemmatized_tokens))
    return meaningful_words


def preprocess(review: str, total: int, show_progress: bool = True) -> list:
    if show_progress:
        global counter
        counter += 1
        print('Processing... %6i/%6i'% (counter, total), end='\r')
    review = clean_review(review)
    tokens = word_tokenize(review)
    lemmas = lemmatize(tokens)
    return lemmas
counter=0
all_comments=list(map(lambda x: preprocess(x,len(dataset_comm)),dataset_comm))


# In[107]:


all_comments[1]


# In[121]:


dataset_tar[dataset_tar=="neg"]=0
dataset_tar[dataset_tar=="pos"]=1

##check if data is balanced
sns.countplot(x='sentiment', data=dataset)


# In[141]:


##count words in each comment
dataset['count_words']=list(map(lambda x: len(x),all_comments))
dataset['count_words']
fig, ax = plt.subplots()
sns.distplot(dataset['count_words'], bins=dataset['count_words'].max(),
            hist_kws={"alpha": 0.9, "color": "red"}, ax=ax,
            kde_kws={"color": "black", 'linewidth': 3})
ax.set_xlim(left=0, right=np.percentile(dataset['count_words'], 95))
ax.set_xlabel('Words in Comments')
ymax = 0.014
plt.ylim(0, ymax)

ax.set_title('Words per comments distribution', fontsize=20)
plt.legend()
plt.show()


# data=data_senti.copy(deep=True)
# X_train=data.loc[:35000,'text']
# y_train=data.loc[:35000,'sentiment']
# X_test=data.loc[35000:,'text']
# y_test=data.loc[35000:,'sentiment']
# x_word2_train=X_train.copy()
# x_word2_test=X_test.copy()

# In[146]:


bigrams = Phrases(sentences=all_comments)
trigrams = Phrases(sentences=bigrams[all_comments])


# In[147]:


print(trigrams[bigrams[all_comments[1]]])


# In[148]:


w2v_model = word2vec.Word2Vec(sentences = trigrams[bigrams[all_comments]], min_count = 35, size = 256, 
                                             window = 8, workers=5, sample=1e-3)
w2v_model.init_sims(replace=True)


# In[153]:


w2v_model.wv.most_similar("love")


# In[163]:


print(list(w2v_model.wv.vocab.keys()).index("love"))
print(list(w2v_model.wv.vocab.keys()).index("fell_love"))


# In[156]:


get_ipython().run_cell_magic('time', '', "def vectorize(text,vocabulary):\n    keys = list(vocabulary.keys())\n    filter_unknown = lambda x: vocabulary.get(x, None) is not None\n    encode = lambda x: list(map(keys.index, filter(filter_unknown, x)))\n    vectorized = list(map(encode, text))\n    return vectorized\npadded=pad_sequences(vectorize(trigrams[bigrams[all_comments]],w2v_model.wv.vocab),maxlen=250,padding='post')")


# In[200]:


X_train, X_test, y_train, y_test = train_test_split(padded,dataset_tar,test_size=0.15,shuffle=True,random_state=42)


# In[201]:


X_train=np.asarray(X_train).astype(np.int)
X_test=np.asarray(X_test).astype(np.int)
y_test=np.asarray(y_test).astype(np.int)
y_train=np.asarray(y_train).astype(np.int)


# In[235]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional,BatchNormalization,Flatten
from keras.layers.embeddings import Embedding

def build_model(embedding_matrix: np.ndarray, input_length: int):
    model = Sequential()
    model.add(Embedding(
        input_dim = embedding_matrix.shape[0],
        output_dim = embedding_matrix.shape[1], 
        input_length = input_length,
        weights = [embedding_matrix],
        trainable=False))
    #model.add(Bidirectional(LSTM(88, recurrent_dropout=0.1)))
    #model.add(Dense(32))
    #model.add(BatchNormalization())
    
    model.add(Dense(64))
    model.add(Dropout(0.25))
    model.add(Flatten())
    #model.add(Dense(128))
    #model.add(Dropout(0.15))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model

model = build_model(
    embedding_matrix=w2v_model.wv.vectors,
    input_length=250)
model.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy'])


# In[236]:


mlp_model = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    batch_size=100,
    epochs=20)


# In[237]:


y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)
def plot_confusion_matrix(y_true, y_pred, ax, class_names, vmax=None,
                          normed=True, title='Confusion matrix'):
    matrix = confusion_matrix(y_true,y_pred)
    if normed:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(matrix, vmax=vmax, annot=True, square=True, ax=ax,
               cmap=plt.cm.Blues_r, cbar=False, linecolor='black',
               linewidths=1, xticklabels=class_names)
    ax.set_title(title, y=1.20, fontsize=16)
    #ax.set_ylabel('True labels', fontsize=12)
    ax.set_xlabel('Predicted labels', y=1.10, fontsize=12)
    ax.set_yticklabels(class_names, rotation=0)
fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2)
plot_confusion_matrix(y_test, y_test_pred, ax=axis2,
                      title='Confusion matrix (test data)',
                      class_names=['Positive', 'Negative'])
plot_confusion_matrix(y_train, y_train_pred, ax=axis1,
                      title='Confusion matrix (train data)',
                      class_names=['Positive', 'Negative'])


# In[ ]:




