b1 = [[1,2,3,4,5,6,7,8], [1,3,5,7]]  # page_keyword_array
b2 = [[2,3,4,5,19], [1,3,4,5,6]]  # main_keyword
dic = {'1' : 0.1, '2' : 0.2, '3' : 0.4}

res = []
for i in range(len(b1)):
    temp = []
    c3 = [filter(lambda x: x in b1[i], sublist) for sublist in b2]
    for j in range(len(c3)):
        temp.append(list(c3[j]))
    res.append(temp)
print(res)

res3 = []
for res2 in res:  # res2 = [[2,3,4,5], [1,3,4,5,6]]
    for j in res2:  # j = [2,3,4,5]
        if len(j) > 2:
            score = 0
            for k in j:  # k = 2 ... 3 ... 4 ... 5
                if str(k) in dic:
                    score += dic[str(k)]
            j.insert(0, score)
            print(j)
            res3.append(j)
print(res3)