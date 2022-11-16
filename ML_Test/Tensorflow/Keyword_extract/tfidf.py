# import pandas as pd # 데이터프레임 사용을 위해
# from math import log # IDF 계산을 위해

# docs = [
#   '먹고 싶은 사과',
#   '먹고 싶은 바나나',
#   '길고 노란 바나나 바나나',
#   '저는 과일이 좋아요'
# ] 
# vocab = list(set(w for doc in docs for w in doc.split()))
# vocab.sort()

# N = len(docs) 

# def tf(t, d):
#   return d.count(t)

# def idf(t):
#   df = 0
#   for doc in docs:
#     df += t in doc
#   return log(N/(df+1))

# def tfidf(t, d):
#   return tf(t,d)* idf(t)

# result = []

# # 각 문서에 대해서 아래 연산을 반복
# for i in range(N):
#   result.append([])
#   d = docs[i]
#   for j in range(len(vocab)):
#     t = vocab[j]
#     result[-1].append(tf(t, d))

# tf_ = pd.DataFrame(result, columns = vocab)
# print(tf_)

# result = []
# for j in range(len(vocab)):
#     t = vocab[j]
#     result.append(idf(t))

# idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])
# print(idf_)

# result = []
# for i in range(N):
#   result.append([])
#   d = docs[i]
#   for j in range(len(vocab)):
#     t = vocab[j]
#     result[-1].append(tfidf(t,d))

# tfidf_ = pd.DataFrame(result, columns = vocab)
# print(tfidf_)

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

def temp(article):

    vectorizer = TfidfVectorizer()
    print(article)
    sp_matrix = vectorizer.fit_transform(article)

    print(sp_matrix)