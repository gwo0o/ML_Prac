from add_keyword import add_keyword
from del_keyword import del_keyword
from vote_list import _id, post_title, post_content
from ckonlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

tw = Twitter()

for i in range(len(add_keyword)):
    tw.add_dictionary(add_keyword[i], 'Noun')

article_keyword = []
post_res = []

for i in range(len(post_title)): # 데이터 전처리 과정
    temp = post_title[i] + ' ' +  post_content[i]
    post_res.append(temp) 

for i in range(len(post_res)):
    key_array = tw.pos(post_res[i])
    # print(key_array)
    keyword_array = []
    pri_keyword = []

    for i in range(len(key_array)):
        if key_array[i][1] in('Noun', 'Hashtag') and len(key_array[i][0]) > 1 and key_array[i][0] not in del_keyword:
            keyword_array.append(key_array[i][0])

    keyword_array = " ".join(keyword_array)
    article_keyword.append(keyword_array)

# print(article_keyword)  # 추출된 단어들의 항목

vectorizer = TfidfVectorizer()
sp_matrix = vectorizer.fit_transform(article_keyword)

word2id = defaultdict(lambda : 0)
for idx, feature in enumerate(vectorizer.get_feature_names()):  
    word2id[feature] = idx

# print(word2id)  # {'a': 1, 'b': 2, 'c': 3 ...}

tdidf = {}
for i, sent in enumerate(article_keyword):
    tdidf[i] = [(token, sp_matrix[i, word2id[token]]) for token in sent.split()]

print(tdidf)

for i in range(len(article_keyword)):
    tdidf[i].sort(key = lambda x: -x[1])
    print(tdidf[i])