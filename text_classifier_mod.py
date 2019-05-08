
#%%
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context

import re
import pandas as pd
import nltk
import numpy as np
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.corpus import stopwords 
from sklearn.ensemble import RandomForestClassifier 

train_size = 80000
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# news data 
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

classifier = RandomForestClassifier(n_estimators=10, random_state=0)  
classifier.fit(X_train, y_train[:train_size])

cnt = 0
for i in range(7599):
    if np.sum(X_test[i]) == 0:
        cnt = cnt+1
cntB = 0
for i in range(train_size):
    if np.sum(X_train[i]) == 0:
        cntB = cntB+1

#%%
y_pred = classifier.predict(X_test) 

#%%
#print(len(y_pred))
#%%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))
print("=============== cnt =================", "\n", "test = ", cnt, "\n", "train = ", cntB)
