from add_keyword import add_keyword
from del_keyword import del_keyword
from article import article
from article import category_ids
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

tw = Twitter()

for i in range(len(add_keyword)):
    tw.add_dictionary(add_keyword[i], 'Noun')

article_keyword = []

for i in range(len(article)):
    key_array = tw.pos(article[i])
    print(key_array)
    keyword_array = []
    pri_keyword = []

    for i in range(len(key_array)):
        if key_array[i][1] in('Noun', 'Hashtag') and len(key_array[i][0]) > 1 and key_array[i][0] not in del_keyword:
            keyword_array.append(key_array[i][0])

    keyword_array = " ".join(keyword_array)
    article_keyword.append(keyword_array)