from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

from keras.models import load_model
model = load_model('best_model.h5')

encoded = [[6866, 6089, 3106, 1853, 9839]]
max_len = 5

pad_new = pad_sequences(encoded, maxlen = max_len)
classifier = model.predict(pad_new)

np.set_printoptions(precision=3, suppress=True)
print('predict', model.predict(pad_new))

temp = max(classifier[0].tolist())
index = classifier[0].tolist().index(temp)
# print(index)

if index == 1:
    print("카테고리는 경제입니다.")
elif index == 2:
    print("카테고리는 정치입니다.")
elif index == 3:
    print("카테고리는 사회입니다.")
elif index == 4:
    print("카테고리는 문화입니다.")
elif index == 5:
    print("카테고리는 IT입니다.")
elif index == 6:
    print("카테고리는 과학입니다.")
elif index == 7:
    print("카테고리는 국제입니다.")
elif index == 8:
    print("카테고리는 스포츠입니다.")
elif index == 9:
    print("카테고리는 연예입니다.")