import pandas as pd
from gensim.summarization.summarizer import summarize
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import gensim
from gensim.models.fasttext import FastText
import gensim.models.word2vec
import gensim.downloader as api

from PIL import Image
import urllib.request
import time
from io import BytesIO
import requests

df = pd.read_csv('../db/editor_2.csv', low_memory=False)
df2 = pd.read_csv('../db/pic_db_7.csv', low_memory=False)

df['Keywords'] = df['Keywords'].str.replace("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣]", "")
df2['Keywords'] = df2['Keywords'].str.replace("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣]", "")
df['Keywords']
df2['Keywords']
title_array = []
keyword_array = []
picture_array = []
for i in range(len(df['Keywords'])):
    temp = df['Keywords'][i].split()
    keyword_array.append(temp)
    title_array.append(df['Title'][i])
    picture_array.append(df['14'][i])
for i in range(len(df2['Keywords'])):
    temp2 = df2['Keywords'][i].split()
    keyword_array.append(temp2)
    title_array.append(df2['Title'][i])
    picture_array.append(df2['14'][i])
print(keyword_array[0])
print(title_array[0])
print(len(keyword_array))
# print(keyword_array)
print(keyword_array[214403], picture_array[214403])

url = picture_array[2000]

res = requests.get(url)

#Img open
request_get_img = Image.open(BytesIO(res.content))
request_get_img

stopwords = ['으로', '하다', '지난해', '선인', '어치', '구역', '방문', '손님', '오전', '오후', '스도', '사비', '이씨', '제이',
            '천가', '디케', '이락', '올해', '고자', '일간', '계단', '향년', '자릿수', '사액', '현재', '이달', '바사', '와이', '바보',
            '앞으로', '데시', '대비', '니어', '반시', '나흘', '최승', '하루', '이스', '아너', '아난', '열자', '시노', '시스', '리프',
            '레시', '다시', '리보', '기존', '한시', '제가', '루타', '이난', '김광', '성제', '과기', '만지', '분분', '어스', '마다',
            '분이', '마의', '윤규', '장재', '프롬', '모어', '망가', '익스', '그니', '이드', '이지', '리오', '갈라치', 
            '아비', '그나', '유부', '안나', '티스', '종업', '맞이', '디파', '인트', '멀다', '수고', '대니', '빌다', '르망', '라보','장보',
            '아무', '푸라', '부음', '브리', '여태', '레비', '그니', '러시', '김강', '쏘다', '여지', '보다', '이다']
tokenized_data = []

# 형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
okt = Okt()

for sentence in df['Keywords']:
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords and len(word) != 1] # 불용어 제거
#     print(stopwords_removed_sentence)
    tokenized_data.append(stopwords_removed_sentence)

for sentence2 in df2['Keywords']:
    if len(sentence2) != 0:
        tokenized_sentence2 = okt.morphs(sentence2, stem=True) # 토큰화
        stopwords_removed_sentence2 = [word for word in tokenized_sentence2 if not word in stopwords and len(word) != 1] # 불용어 제거
    #     print(stopwords_removed_sentence)
        tokenized_data.append(stopwords_removed_sentence2)

from gensim.models import Word2Vec

model = Word2Vec(sentences = tokenized_data, size = 100, window = 1, min_count = 10, workers = 4, sg = 0)
# Word2Vec model 은 단어간의 벡터적 유사도만을 값으로 가진다.

# model = FastText(sentences = tokenized_data, size = 100, window = 5, min_count = 5, workers = 4, sg = 0)
# FastText의 경우 단어간의 벡터 유사도 뿐만아니라 형태 유사도를 동시에 따진다 ex) 카카오톡과 카카오, 사탕, 코코아를 비슷한 단어로 분류한다.

# model = api.load("glove-wiki-gigaword-50")
# model
# model = gensim.models.KeyedVectors.load_word2vec_format('~/gensim-data/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz')
# print(model)
# GloVe model 의 경우 LAS(Latent Semantic Analysis)와 Word2Vec의 단점을 보완하는 목적으로 나왔고 성능이 우수한 편이다.
# 다만 한국어 버전을 제공하지 않는다.

# print(model.wv.similarity('비트코인', '하락'))
# print(model.wv.similarity('비트코인', '상승'))
# print(model.wv.similarity('비트코인', '도지'))
# print(model.wv.similarity('공유', '상승'))  # 상관관계 확인 가능

# print(model.wv.most_similar('비트코인'))  # 가장 유사한 단어 확인 가능
# find_p = input().split('#')
# print(find_p)

find_p = ['비트코인']