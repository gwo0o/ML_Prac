from keras.models import load_model
model = load_model('best_model.h5')

encoded = [[10, 33, 17, 11, 13], [100,100,100,100,100], [10,10,10,10,10], [1,1,1,1,1]]

model.predict(encoded)

pad_new = pad_sequences(encoded, maxlen = max_len)
classifier = model.predict(pad_new)

np.set_printoptions(precision=3, suppress=True)
print('predict', model.predict(pad_new))

# classifier = model.predict(encoded)
# print(type(classifier))
# classifier = list(classifier)

for i in range(len(classifier)):
    temp = max(classifier[i].tolist())
    index = classifier[i].tolist().index(temp)

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
    elif index == 0:
        print("카테고리 분류 실패")