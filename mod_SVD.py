#%%
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

print("===================== library import =======================", "\n")

train_size = 80000
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# news data 

def score(q_vec, d_vec):
    q_vec = np.array(q_vec)
    d_vec = np.array(d_vec)
    norm_q = np.sqrt(np.sum(q_vec**2))
    norm_d = np.sqrt(np.sum(d_vec**2))
    #cosine similarity
    return np.dot(q_vec, d_vec) / (norm_d * norm_q)

#convert counter-vec to operational query array
def query_conv(U, s, q):
    s_inv = np.linalg.inv(s)
    UT = U.T
    q_arr = s_inv.dot(UT)
    print(q_arr)
    print(q)
    q_arr = q_arr.dot(q)
    return q_arr

#svd calculation
def mysvd(A, rk):

    M = len(A[0])
    square = np.dot(A.transpose(),A)
    w, v = LA.eig(square)
    a = sorted(list(zip(w,v.transpose())),key =lambda x: x[0],reverse=True)
    V_t = []

    for i in range(rk):
        V_t.append(list(a[i][1]))

    for i in range(rk):
        flag = 0
        for j in range (M):
            if V_t[i][j] > 0: flag = 1
        if flag == 0:
            for j in range (M):
                if V_t[i][j] != 0:
                    V_t[i][j] = -1 * V_t[i][j]

    V_t = np.array(V_t)

    sigma = []

    for i in range(rk):
        temp = [0] * i + [math.sqrt(a[i][0])] + [0] * (rk-i-1)
        sigma.append(temp)

    sigma = np.array(sigma)
    V = V_t.transpose()

    sigma_inv = np.linalg.inv(sigma)
    U = np.dot(A, V)
    U = np.dot(U, sigma_inv)

    return(U,sigma,V_t)



train = pd.read_csv('train1.csv')
test = pd.read_csv('test1.csv')

feature_num = 0
required_type = []
reply = 0

pos_n = ['NN','NNS','NNP','NNPS']
pos_v = ['VB','VBD','VBG','VBN','VBP','VBZ']
pos_adj = ['JJ', 'JJR', 'JJS']
pos_adv = ['RB','RBR','RBS']


method = input("enter 1 for TfidfVectorizer, 2 for CountVectorizer, 3 for HashingVectorizer, 4 for tfidf with chi2 selection : ")
feature_num = int(input("please enter the numbers of features : "))
# ============= reloading features ==========
reply = input("Would you like to have NOUNS as your features? (Y/N) : ")
if reply == 'Y':
    required_type  += pos_n

reply = input("Would you like to have VERBS as your features? (Y/N) : ")
if reply == 'Y':
    required_type  += pos_v

reply = input("Would you like to have ADJ as your features? (Y/N) : ")
if reply == 'Y':
    required_type  += pos_adj

reply = input("Would you like to have ADV as your features? (Y/N) : ")
if reply == 'Y':
    required_type  += pos_adv

print(required_type)
# ========================================

#%%
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
    for i in temp:
        if i[1] in required_type and i[0] != 'Reuters':
            nounVerb.append(lemmatizer.lemmatize(i[0]))
    train_split.append(nounVerb)

# test part
for i in range(7599):
    row = test.loc[i]
    temp = nltk.pos_tag(word_tokenize(row[2]))
    nounVerb = []
    for i in temp:
        if i[1] in required_type and i[0] != 'Reuters':
            nounVerb.append(lemmatizer.lemmatize(i[0]))
    test_split.append(nounVerb)

#%%
print("===================== POS tagged =======================", "\n")
D_train = []
for i in range(len(train_split)):
    str1 = ' '.join(train_split[i])
    D_train.append(str1)

D_test = []
for i in range(len(test_split)):
    str1 = ' '.join(test_split[i])
    D_test.append(str1)

#%%
y_train = list(train[train.columns[0]])
y_test = list(test[test.columns[0]])

#%%
#======================= Feature Selection =================================
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from nltk.corpus import stopwords 




if method == '1':
    tfidfconverter = TfidfVectorizer(max_features = feature_num, min_df=5, max_df=0.6, stop_words=stopwords.words('english')) 
    print("======== method : tfidfconverter =========")
    X_train = tfidfconverter.fit_transform(D_train).toarray()
    features = tfidfconverter.get_feature_names()
    print("======== features : tfidfconverter =========")
    print(tfidfconverter.get_feature_names())
    temp = TfidfVectorizer(vocabulary=features)
    X_test = temp.fit_transform(D_test).toarray()
    #print(temp.get_feature_names())
if method == '2':
    Countconverter = CountVectorizer(max_features = feature_num, min_df=5, max_df=0.6, stop_words=stopwords.words('english'))
    print("======== method : Countconverter =========")
    X_train = Countconverter.fit_transform(D_train).toarray()
    features = Countconverter.get_feature_names()
    print("======== features : Countconverter =========")
    print(Countconverter.get_feature_names())
    temp = CountVectorizer(vocabulary=features)
    X_test = temp.fit_transform(D_test).toarray()
    #print(temp.get_feature_names())
if method == '3':
    Hashingconverter = HashingVectorizer(n_features = feature_num, stop_words=stopwords.words('english'))
    print("======== method : Hashingconverter =========")
    X_train = Hashingconverter.fit_transform(D_train).toarray()
    
    temp = HashingVectorizer(n_features = feature_num)
    X_test = temp.fit_transform(D_test).toarray()
    
if method == '4':
    print("======== method : tfidf chi2 =========")
    tfidfconverter =  CountVectorizer(lowercase=True, stop_words=stopwords.words('english'))
    X_train = tfidfconverter.fit_transform(D_train).toarray()
    chi2_score = chi2(X_train, y_train[:train_size])[0]
    wscores = zip(tfidfconverter.get_feature_names(),chi2_score)
    wchi2 = sorted(wscores, key = lambda x:x[1]) 
    features = []
    for i in range(feature_num) :
        features.append(wchi2[i][0])
    print(features)
    # temp = TfidfVectorizer(vocabulary=features)
    # X_train = temp.fit_transform(D_train).toarray()
    # X_test = temp.fit_transform(D_test).toarray()


#%%
#======================= model training ====================================

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

print("===================== text vectorized =======================", "\n")
#%%
#==================== SVD - Create query =========================
X_train = np.array(X_train)
X_train = X_train.T
X_test = np.array(X_test)
U, s, VT = mysvd(X_train, 250)
X_test = X_test.T
query = query_conv(U, s, X_test)

print("===================== SVD calculated =======================", "\n")
#%%
#======================== similarity calculate ========================
#calculate vec to vec simularity
ans = score(query.T, VT)

result = []
for i in ans:
    sim_table = []
    for j, k in enumerate(i):
        sim_table.append((j, k))
    #get max similar index
    result.append(max(sim_table,key = itemgetter(1))[0])

y_pred = []
for j in result:
    #y-pred = the result of most similar y_train
    y_pred.append(y_train[j])

print("===================== ready to predict =======================", "\n")

#%%
y_pred = classifier.predict(X_test) 

#%%
#print(len(y_pred))
#%%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))
