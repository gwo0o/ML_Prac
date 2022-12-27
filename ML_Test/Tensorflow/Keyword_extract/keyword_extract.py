from add_keyword import add_keyword
from del_keyword import del_keyword
from article import article
from article import category_ids
from article import article_key
from ckonlpy.tag import Twitter
from collections import Counter
import tfidf as idf
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, Dense, GRU, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import input_new_tags as update
# from tensorflow.keras.preprocessing.text import Tokenizer
# from cnn import cnn

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

# print('sp_matrix', sp_matrix)

word2id = defaultdict(lambda : 0)
for idx, feature in enumerate(vectorizer.get_feature_names()):  
    word2id[feature] = idx

# print(word2id)  # {'a': 1, 'b': 2, 'c': 3 ...}

tdidf = {}
for i, sent in enumerate(article_keyword):
    tdidf[i] = [(token, sp_matrix[i, word2id[token]]) for token in sent.split()]

# print('tdidf', tdidf)  # {0: [('파이낸셜뉴스', 0.04911270611173677), ('애경', 0.24811880389895863), ('산업', 0.24811880389895863), ('지난', 0.054687767085255384), ('메가박스', 0.16541253593263908), ('상암', 0.08270626796631954), ('월드컵경기', 0.08270626796631954), ('장점', 0.0703078838142362), ... }

res = []
res2 = []

for i in range(len(article_keyword)):    
    tdidf[i].sort(key = lambda x: -x[1])  # 각 파일을 역순으로 정렬
    result = []
    temp = []
    temp2 = []
    for v in tdidf[i]:
        if v not in result:
            result.append(v)   # 중복 제거
    # print(result)    # 각 항목들의 가중치를 볼 수 있다.
    try:
        for j in range(10):
            temp.append(word2id[result[j][0]])
            temp2.append(result[j][0])
            # print(result[j][0], end = ' ')     # 상위 5개의 단어만 키워드로 추출하고자한다.
    except:
        temp.append('NOT_EXIST_DATA')
        temp2.append('NOT_EXIST_DATA')  # update.temp 수행하기 위해서 임시방편
    # print('\n', end = '')

    res.append(temp)
    res2.append(temp2)

# print(res)  # [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]]
for i in range(len(res2)):
    print(res2[i], article_key[i])  # [['a', 'b', 'c', 'd', 'e'], ['f', 'g', 'h', 'i', 'j']]
    update.temp(res2[i], article_key[i])

total_data = category_ids

for i in range(len(total_data)):
    total_data[i] = np.select([#total_data[i] == '경제', 
                                    #  total_data[i] == '정치',
                                    #  total_data[i] == '사회',
                                    #  total_data[i] == '문화',
                                    #  total_data[i] == 'IT',
                                    #  total_data[i] == '과학',
                                    #  total_data[i] == '국제',
                                    #  total_data[i] == '스포츠',
                                    #  total_data[i] == '연예',
                                    total_data[i] == '001001000',
                                    total_data[i] == '001001001',
                                    total_data[i] == '001001002',
                                    total_data[i] == '001001003',
                                    total_data[i] == '001001004',
                                    total_data[i] == '001001005',
                                    total_data[i] == '001001006',
                                    total_data[i] == '001001007',
                                    total_data[i] == '001002000',
                                    total_data[i] == '001002001',
                                    total_data[i] == '001002002',
                                    total_data[i] == '001002003',
                                    total_data[i] == '001002004',
                                    total_data[i] == '001002005',
                                    total_data[i] == '001002006',
                                    total_data[i] == '001002007',
                                    total_data[i] == '001002008',
                                    total_data[i] == '001002009',
                                    total_data[i] == '001003000',
                                    total_data[i] == '001003001',
                                    total_data[i] == '001003002',
                                    total_data[i] == '001003003',
                                    total_data[i] == '001003004',
                                    total_data[i] == '001003005',
                                    total_data[i] == '001003006',
                                    total_data[i] == '001004001',
                                    total_data[i] == '001004002',
                                    total_data[i] == '001005000',
                                    total_data[i] == '001005001',
                                    total_data[i] == '001005002',
                                    total_data[i] == '001005003',
                                    total_data[i] == '001005004',
                                    total_data[i] == '002001000',
                                    total_data[i] == '002001001',
                                    total_data[i] == '002001002',
                                    total_data[i] == '002001003',
                                    total_data[i] == '002001004',
                                    total_data[i] == '002001005',
                                    total_data[i] == '002001006',
                                    total_data[i] == '002001007',
                                    total_data[i] == '002001010',
                                    total_data[i] == '002001011',
                                    total_data[i] == '002001012',
                                    total_data[i] == '002002000',
                                    total_data[i] == '002002001',
                                    total_data[i] == '002002002',
                                    total_data[i] == '002002003',
                                    total_data[i] == '002002004',
                                    total_data[i] == '002002005',
                                    total_data[i] == '002002006',
                                    total_data[i] == '002002007',
                                    total_data[i] == '002003000',
                                    total_data[i] == '002003001',
                                    total_data[i] == '002003002',
                                    total_data[i] == '002003003',
                                    total_data[i] == '002003004',
                                    total_data[i] == '002003005',
                                    total_data[i] == '002004000',
                                    total_data[i] == '002004001',
                                    total_data[i] == '002004004',
                                    total_data[i] == '002004005',
                                    total_data[i] == '002004006',
                                    total_data[i] == '002004007',
                                    total_data[i] == '002004008',
                                    total_data[i] == '002004009',
                                    total_data[i] == '002004010',
                                    total_data[i] == '002004011',
                                    total_data[i] == '002004012',
                                    total_data[i] == '002005000',
                                    total_data[i] == '002005001',
                                    total_data[i] == '002005002',
                                    total_data[i] == '002005003',
                                    total_data[i] == '002005004',
                                    total_data[i] == '002005005',
                                    total_data[i] == '002005006',
                                    total_data[i] == '002006000',
                                    total_data[i] == '002006001',
                                    total_data[i] == '002006002',
                                    total_data[i] == '002006003',
                                    total_data[i] == '002006004',
                                    total_data[i] == '002006005',
                                    total_data[i] == '002007000',
                                    total_data[i] == '002007001',
                                    total_data[i] == '002007002',
                                    total_data[i] == '002007003',
                                    total_data[i] == '002008000',
                                    total_data[i] == '002008001',
                                    total_data[i] == '002008002',
                                    total_data[i] == '002009000',
                                    total_data[i] == '002009001',
                                    total_data[i] == '002009002',
                                    total_data[i] == '002009004',
                                    total_data[i] == '002010002',
                                    total_data[i] == '002010003',
                                    total_data[i] == '002010004',
                                    total_data[i] == '002010005',
                                    total_data[i] == '002010006',
                                    total_data[i] == '002010007',
                                    total_data[i] == '003001001',
                                    total_data[i] == '003001006',
                                    total_data[i] == '003001008',
                                    total_data[i] == '003001009',
                                    total_data[i] == '003001010',
                                    total_data[i] == '003001011',
                                    total_data[i] == '003001012',
                                    total_data[i] == '003001016',
                                    total_data[i] == '003001017',
                                    total_data[i] == '003001019',
                                    total_data[i] == '003001022',
                                    total_data[i] == '003002001',
                                    total_data[i] == '003002002',
                                    total_data[i] == '003002004',
                                    total_data[i] == '003002005',
                                    total_data[i] == '003002006',
                                    total_data[i] == '004001001',
                                    total_data[i] == '004001002',
                                    total_data[i] == '004001003',
                                    total_data[i] == '004001004',
                                    total_data[i] == '004002000',
                                    total_data[i] == '004002001',
                                    total_data[i] == '004002002',
                                    total_data[i] == '005001001',
                                    total_data[i] == '005001002',
                                    total_data[i] == '005002001',
                                    total_data[i] == '005002002',
                                    total_data[i] == '005003001',
                                    total_data[i] == '005003002',
                                    total_data[i] == '005005001',
                                    total_data[i] == '005005002',
                                    total_data[i] == '005005003',
                                    total_data[i] == '005006001',
                                    total_data[i] == '005006003',
                                    total_data[i] == '008001001',
                                    total_data[i] == '009001001',
                                    ],
                                    [#1, 2, 3, 4, 5, 6, 7, 8, 9, 
                                    # 경제 정치 사회 문화 IT 과학 국제 스포츠 연예
                                    2, 2, 2, 2, 2, 2, 
                                    2, 2, 7, 7, 7, 7, 7, 7, 7, 7, 
                                    7, 7, 3, 3, 3, 3, 3, 3, 3, 3,
                                    3, 3, 3, 3, 3, 3, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 5, 5, 5, 5, 5, 5, 5, 3,
                                    3, 6, 6, 6, 6, 1, 1, 1, 1, 0,
                                    1, 1, 1, 1, 1, 8, 5, 1, 1, 1,
                                    1, 6, 3, 3, 2, 3, 1, 7, 2, 3,
                                    5, 3, 2, 3, 3, 1, 3, 3, 9, 9,
                                    9, 9, 8, 8, 8, 1, 6, 4, 1, 4,
                                    1, 4, 4, 4, 4, 3, 4, 1,], default=0)

# print(total_data)  # [0, 1, 0, 1, 2, 0, 1, ...]

# all_data = []
# for i in range(len(res)):
#     temp = {}
#     temp['Keyword'] = res[i]
#     temp['Category'] = total_data[i]
#     all_data.append(temp)

# print(all_data)

key_train_data, key_test_data = train_test_split(res, test_size = 0.25, random_state = 20)
cate_train_data, cate_test_data = train_test_split(total_data, test_size = 0.25, random_state = 20)


X_train = np.array(key_train_data)
Y_train = np.array(cate_train_data)
X_test= np.array(key_test_data)
Y_test = np.array(cate_test_data)
print(Y_test)

try:
    loaded_model = load_model('best_model.h5')
except:
    print('load fail')
    max_len = 5

    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    embedding_dim = 100
    hidden_units = 128
    vocab_size = len(word2id)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(LSTM(hidden_units))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(X_train, Y_train, epochs=10, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model('best_model.h5')
# print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, Y_test)[1]))