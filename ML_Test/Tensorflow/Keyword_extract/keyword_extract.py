from add_keyword import add_keyword
from del_keyword import del_keyword
from article import article
from ckonlpy.tag import Twitter
from collections import Counter
import tfidf as idf

tw = Twitter()

for i in range(len(add_keyword)):
    tw.add_dictionary(add_keyword[i], 'Noun')

article_keyword = []
for i in range(len(article)):
    key_array = tw.pos(article[0])
# print(key_array)
    keyword_array = []
    pri_keyword = []

    for i in range(len(key_array)):
        if key_array[i][1] in('Noun', 'Hashtag') and len(key_array[i][0]) > 1 and key_array[i][0] not in del_keyword:
            keyword_array.append(key_array[i][0])

    print(keyword_array)
    print('-------------------------------------------------')
    article_keyword.append(keyword_array)

# count = dict(Counter(article_keyword))
# print(count)

idf.temp(article_keyword)

