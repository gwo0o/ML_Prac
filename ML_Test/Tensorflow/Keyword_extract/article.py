import mysql.connector

mydb = mysql.connector.connect(
    host = 'www.devkids.co.kr',
    user = 'newming',
    port = 14402,
    database = 'ds_lab',
    password = 'u8Yto93qrAgW',
)

mycursor = mydb.cursor()

mycursor.execute("SELECT category_ids, content FROM nm_press_article_20220622 WHERE LENGTH(category_ids) = 9 LIMIT 0, 100")

myres = mycursor.fetchall()

article = []
category_ids = []
for i in range(0, len(myres)):
    article.append(myres[i][1].decode())
    category_ids.append(myres[i][0].decode())
# print(article) 