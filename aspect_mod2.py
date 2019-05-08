import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context


import re
import nltk
import spacy
nlp = spacy.load('en_core_web_lg')
import pandas as pd
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.stem import WordNetLemmatizer     ##詞性還原
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.corpus import stopwords 
from sklearn.ensemble import RandomForestClassifier 
from nltk.corpus import sentiwordnet as swn
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords 

print("============= library implemented ==============", "\n")
#%%
#data = pd.read_csv("restaurant.csv", encoding='UTF-8')
raw1 = open('foursquare_raw_reviews.txt', encoding = 'UTF-8')#完整的資料
raw = open('small_data.txt', encoding = 'UTF-8')#只有八筆資料，用來測試用的
seed_pos = open('positive_word.txt', encoding = 'UTF-8')#正面的種子字
seed_neg = open('negative_word.txt', encoding = 'UTF-8')#負面的種子字

t1 = raw.read()
#t = "the meat is really good, but the soup is really disgusting."
seed_pos = seed_pos.read()
seed_neg = seed_neg.read()
sentence = t1.split('\n')

#print(t1)
seed_pos = nlp(seed_pos)#這個函數可以把字變成可以算spaCy similarity的狀態。
seed_neg = nlp(seed_neg)
#%%
def sentiment(w, feat_opinion, idx):
    if w.text in seed_pos.text:#如果直接在正面種子字裡面找到。
        print("feature : ", feat_opinion[idx][1], "\n","opinion word : ", w.text, "\n", "ans : pos", "\n", "    similarity: 1.00", "\n", "    similar word: ", w.text, "\n")
    elif w.text in seed_neg.text:#如果直接在負面種子字裡面找到。
        print("feature : ", feat_opinion[idx][1], "\n","opinion word : ", w.text, "\n","ans : neg", "\n", "    similarity: 1.00", "\n", "    similar word: ", w.text, "\n")

    else:
        max_w = ''
        max_sim = 0 #seed與詞之間最大的相似度
        flag_a = 0 #有沒有任何負面字相似度大於最大的正義字
        for tok in seed_pos:#算出最接近的正面種子字
            sim = w.similarity(tok) #相似度
            if (max_sim < sim):
                max_sim = sim
                max_w = tok.text

                
        for tok in seed_neg:#看看有沒有更接近的負面種子字
            sim = w.similarity(tok)
            if (max_sim < sim):
                max_sim = sim
                max_w = tok.text
                ans = 'neg'
                # seed_neg.append(w)
                flag_a = 1
        if(flag_a == 0): 
            ans = 'pos'
            # seed_pos.append(w)
        if(max_sim <= 0.7):#如果沒有超過0.7的相似度，則判斷為中立字詞
            ans = 'obj'

        print("feature : ", feat_opinion[idx][1], "\n","opinion word : ", w.text, "\n", "ans : "
            , ans, "\n", "    similarity: ", max_sim, "\n","    similar word: ", max_w, "\n")
        return(ans)

def pos_sentence(s, lexical):
    #print(s)
    train_split = []
    if(lexical == 'n'):#只有用到名詞跟形容詞那兩排。形容詞那排加入了所有opinion word可能有的詞性。
        required_type = ['NN','NNS','NNP','NNPS']
    elif(lexical == 'v'):
        required_type = ['VB','VBD','VBG','VBN','VBP','VBZ']
    elif(lexical == 'a'):
        required_type = ['JJ', 'JJR', 'JJS', 'VB','VBD','VBG','VBN','VBP','VBZ']
    elif(lexical == 'r'):
        required_type = ['RB','RBR','RBS']
    for i in range(len(s)):
        temp = nltk.pos_tag(word_tokenize(s[i]))

        nounVerb = []
        for i in temp:
            if i[1] in required_type:
                nounVerb.append(str.lower(lemmatizer.lemmatize(i[0])))#轉小寫
        train_split.append(nounVerb)
    return (train_split)

def get_counted_feature(D_train, sen):
    feat_location = []
    countvectorizer = CountVectorizer(max_features = 50, min_df=1, max_df=0.7, stop_words=stopwords.words('english'))
    X_train = countvectorizer.fit_transform(D_train).toarray()
    features = countvectorizer.get_feature_names()

    for i in sen:#製作出一個有三個元素的東西（feature或是opinion, 原本的整句評論, 該字詞在原本的評論中的位置）
        for j in features:
            if j in i:
                temp = (j, i, i.index(j))
                feat_location.append(temp)


    return(features, feat_location)
    # temp = CountVectorizer(vocabulary=features)



#sentiment('yummy')


#%%
lemmatizer = WordNetLemmatizer()



#print(t)
feature_num = 0
required_type = []
reply = 0
r1 = '[0-9’!１２３４５６７８９０"#$%&\'()（）*+,-/:;<=>?@，。?★、…【】《》＊;「」？“”‘’！：[\\]^_`{|}~‧]+'
sen1 = [re.sub(r1, '', i) for i in sentence]
#print(sentence)#形式：['good food Parking can be a real pain', 'The red beans and rice are DELICIOUS better than Pat OBriens']
sentence = []

for i in sen1:
    sentence.append(str.lower(lemmatizer.lemmatize(i)))#轉小寫

#print(sentence)

#%%
# ============== pos tagging ==============
adj = pos_sentence(sentence, 'a')

#verb = pos_sentence(sentence, 'v')
noun = pos_sentence(sentence, 'n')

print("============= pos tagged ==============", "\n")
# print(adj)
openion_adj = []
for i in range(len(adj)):
	str1 = ' '.join(adj[i])
	openion_adj.append(str1)
#print(openion_adj)
# openion_verb = []
# for i in range(len(verb)):
# 	str1 = ' '.join(verb[i])
# 	openion_verb.append(str1)

target_noun = []
for i in range(len(noun)):
	str1 = ' '.join(noun[i])
	target_noun.append(str1)

#print(openion_adj)

getadj, adj_loc = get_counted_feature(openion_adj, sentence)
target, n_loc = get_counted_feature(target_noun, sentence)
#print(target)。   #形式：['juice', 'lighting', 'love', 'mango', 'meal', 'nice', 'pizza', 'place', 'price', 'rip', 'road', 'service', 'small', 'sol', 'south', 'tiny', 'total', 'tradition', 'tried', 'try']

feat_opinion = []
for i in range(len(adj_loc)):
    min_diff = 9999
    min_feat = ''
    for j in range(len(n_loc)):
        if (n_loc[j][1] == adj_loc[i][1]) & (abs(n_loc[j][2] - adj_loc[i][2]) < min_diff):
            min_diff = abs(n_loc[j][2] - adj_loc[i][2])
            min_feat = n_loc[j][0]
    pair = (adj_loc[i][0], min_feat)#把對應的opinion word跟feature綁在一起
    feat_opinion.append(pair)
#print(feat_opinion) #形式：[('try', 'juice'), ('change', 'rip'), ('double', 'service'), ('get', 'road'), ('small', 'pizza'), ('tiny', 'pizza'), ('tried', 'rip'), ('best', 'everything'), ('find', 'everything'), ('authentic', 'tradition'), ('good', 'lighting'), ('mango', 'juice')]

print("============= feature get ==============", "\n")

#list_adj = ' '.join(getadj) #形式：authentic best change double find get good indian mango small tiny total tried try
#list_verb = ' '.join(getverb)
list_tar = ' '.join(target)

list_adj = []
for i in range(len(feat_opinion)):
    list_adj.append(feat_opinion[i][0])

list_adj = ' '.join(list_adj)



#print(getadj)#形式：['authentic', 'best', 'change', 'double', 'find', 'get', 'good', 'indian', 'mango', 'small', 'tiny', 'total', 'tried', 'try']
#for i in range(getadj)

list_adj = nlp(list_adj)
#list_verb = nlp(list_verb)
list_tar = nlp(list_tar)
# X_test = temp.fit_transform(D_test).toarray()

#%% 

# adj_sentiment = []
# for i in range(len(list_adj)):
#     adj_sentiment.append(sentiment(list_adj[i]))


adj_sentiment = []
for i in range(len(list_adj)):
    sentiment(list_adj[i], feat_opinion, i)
#%%
# verb_sentiment = []
# for i in range(len(list_verb)):
#     verb_sentiment.append(sentiment(list_verb[i]))
# print(verb_sentiment)

