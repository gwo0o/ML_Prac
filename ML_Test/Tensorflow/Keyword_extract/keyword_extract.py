from add_keyword import add_keyword
from del_keyword import del_keyword
from article import article
from ckonlpy.tag import Twitter
from collections import Counter
import tfidf as idf
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict

tw = Twitter()

for i in range(len(add_keyword)):
    tw.add_dictionary(add_keyword[i], 'Noun')

article_keyword = []
for i in range(len(article)):
    key_array = tw.pos(article[i])
# print(key_array)
    keyword_array = []
    pri_keyword = []

    for i in range(len(key_array)):
        if key_array[i][1] in('Noun', 'Hashtag') and len(key_array[i][0]) > 1 and key_array[i][0] not in del_keyword:
            keyword_array.append(key_array[i][0])

    keyword_array = " ".join(keyword_array)
    article_keyword.append(keyword_array)

vectorizer = TfidfVectorizer()
sp_matrix = vectorizer.fit_transform(article_keyword)

word2id = defaultdict(lambda : 0)
for idx, feature in enumerate(vectorizer.get_feature_names()):  
    word2id[feature] = idx

# print(word2id)

tdidf = {}
for i, sent in enumerate(article_keyword):
    tdidf[i] = [(token, sp_matrix[i, word2id[token]]) for token in sent.split()]

# print(sp_matrix[2, word2id['word']])      # 각 항목목들의 가중치를 직접적으로 조회할 수 있다.
# print(sp_matrix[1, word2id['word']])      # 1번 문서의 '감소' 단어의 가중치를 조회한다.

for i in range(len(article_keyword)):    
    tdidf[i].sort(key = lambda x: -x[1])  # 각 파일을 역순으로 정렬
    result = []
    for v in tdidf[i]:
        if v not in result:
            result.append(v)               # 중복 제거
    # print(result)    # 각 항목들의 가중치를 볼 수 있다.
    try:
        for j in range(5):
            print(result[j][0], end = ' ')     # 상위 10개의 단어만 키워드로 추출하고자한다.
    except:
        continue
    print('\n', end = '')