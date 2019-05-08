#%% 
#========================== Include library =====================================
from scipy.linalg import svd
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
import re
import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.corpus import stopwords 
from sklearn.ensemble import RandomForestClassifier 
from tqdm import tqdm
import re
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import multiprocessing
from sklearn import utils
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
from numpy import array
import numpy as np
from numpy import linalg as LA
import math

np.set_printoptions(threshold=np.inf)

print("===================== library import =======================", "\n")

train_size = 20000
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#%%
#============================= Define Function ===============================

#calculate cosine simularuty
def score(q_vec, d_vec):
    q_vec = np.array(q_vec)
    d_vec = np.array(d_vec)
    norm_q = np.sqrt(np.sum(q_vec**2))
    norm_d = np.sqrt(np.sum(d_vec**2))
    #cosine similarity
    return np.dot(q_vec, d_vec) / (norm_d * norm_q)

#convert counter-vec to operational query array
def query_conv(U, s, q):
    #s_inv = np.linalg.inv(s)
    UT = U.T
    q_arr = UT.dot(q)
    #print(q_arr)
    #print(q)
    #q_arr = q_arr.dot(q)
    return q_arr

#svd calculation
# def mysvd(A, rk):

#     N = len(A)
#     M = len(A[0])
#     R = LA.matrix_rank(A)
#     square = np.dot(A.transpose(),A)
#     w, v = LA.eig(square)
#     a = sorted(list(zip(w,v.transpose())),key =lambda x: x[0],reverse=True)
#     V_t = []

#     for i in range(rk):
#         V_t.append(list(a[i][1]))

#     V_t = np.array(V_t)

#     sigma = []

#     for i in range(rk):
#         temp = [0] * i + [math.sqrt(a[i][0])] + [0] * (rk-i-1)
#         sigma.append(temp)

#     sigma = np.array(sigma)
#     V = V_t.transpose()

#     sigma_inv = np.linalg.inv(sigma)
#     U = np.dot(A, V)
#     U = np.dot(U, sigma_inv)

#     return(U,sigma,V_t)


#%%
#================== read file & pre processing ======================
train = pd.read_csv('train.csv',nrows=20000)
test = pd.read_csv('test.csv',nrows=5000)

r1 = '[0-9’!１２３４５６７８９０"#$%&\'()（）*+,-/:;<=>?@，。?★、…【】《》＊;「」？“”‘’！：[\\]^_`{|}~‧]+'
train[train.columns[2]] = train[train.columns[2]].apply(lambda x: re.sub(r1,'',x))

#%%
# ============== pos tagging ==============
train_split = []
test_split = []

for i in range(train_size):
    row = train.loc[i]
    temp = nltk.pos_tag(word_tokenize(row[2]))
    nounVerb = []
    required_type = ['JJ', 'JJR', 'JJS', 'NN','NNS','NNP','NNPS']
    for i in temp:
        if i[1] in required_type:
            nounVerb.append(lemmatizer.lemmatize(i[0]))
    train_split.append(nounVerb)

# test part
for i in range(5000):
    row = test.loc[i]
    temp = nltk.pos_tag(word_tokenize(row[2]))
    nounVerb = []
    required_type = ['JJ', 'JJR', 'JJS', 'NN','NNS','NNP','NNPS']
    for i in temp:
        if i[1] in required_type:
            nounVerb.append(lemmatizer.lemmatize(i[0]))
    test_split.append(nounVerb)
print("===================== POS tagged =======================", "\n")
#%%
#=========================== Adjusting data=================================== 
D_train = []
for i in range(len(train_split)):
	str1 = ' '.join(train_split[i])
	D_train.append(str1)

D_test = []
for i in range(len(test_split)):
	str1 = ' '.join(test_split[i])
	D_test.append(str1)

train['if positive'] = train[train.columns[0]].apply(lambda x: 1 if x >= 3 else 0)
test['if positive'] = test[test.columns[0]].apply(lambda x: 1 if x >= 3 else 0)

y_train = list(train[train.columns[3]])
y_test = list(test[test.columns[3]])

#%%
#======================= CountVector Feature Selection =================================
countvectorizer = CountVectorizer(max_features = 800, min_df=1, max_df=0.9, stop_words=stopwords.words('english'))
X_train = countvectorizer.fit_transform(D_train).toarray()
features = countvectorizer.get_feature_names()
#print(countvectorizer.get_feature_names())
temp = CountVectorizer(vocabulary=features)
X_test = temp.fit_transform(D_test).toarray()
print(temp.get_feature_names())
#print("\n",X_train)
print("===================== text vectorized =======================", "\n")
#%%
#==================== SVD - Create query=========================
X_train = np.array(X_train)
X_train = X_train.T
X_test = np.array(X_test)
#U, s, VT = mysvd(X_train, 500)
rk = 400
u_1, sigma_1, V_1 = LA.svd(X_train, full_matrices = True)

sigma = []

for i in range(rk):
    temp = [0] * i + [sigma_1[i]] + [0] * (rk-i-1)
    sigma.append(temp)

s = np.array(sigma)

u_1 = u_1.T
for i in range(train_size - rk):
    u_1 = np.delete(u_1, rk, 0)

U = np.array(u_1)
U = U.T

for i in range(800 - rk):
    V_t1 = np.delete(V_t1, rk, 0)

VT = np.array(V_t1)


VT = VT.real
VT = np.dot(s, VT)
VT = VT.real

print("\n",s)
X_test = X_test.T
query = query_conv(U, s, X_test)
query = query.real
#print("\n",query)
print("===================== SVD calculated =======================", "\n")
#%%
#========================similarity calculate========================
#calculate vec to vec simularity

X_train = VT.T
X_test = query.T
classifier = RandomForestClassifier(n_estimators=10, random_state=0)  
classifier.fit(X_train, y_train[:train_size])  


y_pred = classifier.predict(X_test)
print("===================== Random forest finished ready to predict =======================", "\n")
#========================== Evaluation ====================================
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))