from add_keyword import add_keyword
from del_keyword import del_keyword
from vote_list import post_id, post_title, post_content
from ckonlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from page_list import page_id, page_title, page_summary, page_content
import numpy as np

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

# print(tdidf)
word_impact = {}  # 각 단어들이 가진 가중치를 저장합니다.
main_keyword = []

for i in range(len(article_keyword)):
    tdidf[i].sort(key = lambda x: -x[1])
    # print(tdidf[i])  # [('날씨', 0.6060433187339399), ('가요', 0.6060433187339399), ('오늘', 0.5151921890284284)]
    result = []  # 중복 제거된 tdidf 를 담기 위한 array
    for v in tdidf[i]:
        if v not in result:
            result.append(v)               # 중복 제거
    # print(result)

    for word, impact in result:
        if word not in word_impact:
            word_impact[word] = impact

    temp_main_keyword = []
    try:
        for j in range(5):
            temp_main_keyword.append(result[j][0]) 
    except:
        for k in range(j, 5):
            temp_main_keyword.append('')

    item = {
        'post_id': post_id[i],
        'key_list': temp_main_keyword
    }
    main_keyword.append(item)

# print(main_keyword)  # [{'post_id': 85661, 'key_list': ['지난', '배추', '소매', '김치', '보쌈']}, {'post_id': 85592, 'key_list': ['햄버거', '브랜드', '선호', '', '']},
# print(word_impact)  # {'지난': 0.3333333333333333, '배추': 0.3333333333333333, '소매': 0.3333333333333333, '김치': 0.16666666666666666, '보쌈': 0.16666666666666666, '생각': 0.16666666666666666, '농산물': 0.16666666666666666, '유통': 0.16666666666666666, '정보': 0.16666666666666666, '포기': 0.16666666666666666, '평균': 0.16666666666666666, ... }

page_keyword_array = []

for i in range(len(page_title)):
    page_array = tw.pos(page_content[i])
    page_key_array = []

    # print('-------------', len(page_array))
    for i in range(len(page_array)):
        if page_array[i][1] in ('Noun', 'Hashtag') and len(page_array[i][0]) > 1 and page_array[i][0] not in del_keyword:
            page_key_array.append(page_array[i][0])

    page_keyword_array.append(page_key_array)

# print(page_keyword_array)
# print(main_keyword)

# for sub_keyword in main_keyword:  # 가장 유사도가 높은 아이템 찾기
#     score = 0
#     for one_key in sub_keyword['key_list']:
#         if one_key != '' and one_key in word_impact:
#             for i in range(len(page_keyword_array)):
#                 for j in range(len(page_keyword_array[i])):
#                     if page_keyword_array[i][j] == one_key:
#                         score += word_impact[one_key]
# sub_keyword['key_list'].insert(0, score)

# temp = []

# for sub_keyword in main_keyword:
#     temp.append(sub_keyword['key_list'][0])

# take_post = []
# take_post.append(main_keyword[temp.index(max(temp))]['post_id'])
# print(take_post)


for i in range(len(page_keyword_array)):  # 가장 유사도가 높은 아이템 찾기
    for sub_keyword in main_keyword:
        score = 0
        for one_key in sub_keyword['key_list']:
            if one_key != '' and one_key in word_impact:
                for j in range(len(page_keyword_array[i])):
                    if page_keyword_array[i][j] == one_key:
                        score += word_impact[one_key]
        sub_keyword['key_list'].insert(0, score)

temp_all = []

for i in range(5):
    temp = []
    for sub_keyword in main_keyword:
        temp.append(sub_keyword['key_list'][i])
    temp_all.append(temp)

take_post = []
for i in range(5):
    if max(temp_all[i]) != 0 :
        take_post.append(main_keyword[temp_all[i].index(max(temp_all[i]))]['post_id'])
    else:
        take_post.append(0)

print(take_post)