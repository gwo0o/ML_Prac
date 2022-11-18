import mysql.connector

mydb = mysql.connector.connect(
    host = 'www.devkids.co.kr',
    user = 'newming',
    port = 14402,
    database = 'ds_lab',
    password = 'u8Yto93qrAgW',
)

mycursor = mydb.cursor()

mycursor.execute("SELECT title, content FROM nm_articles LIMIT 0, 500")

myres = mycursor.fetchall()

article = []
for i in range(0, len(myres)):
    article.append(myres[i][1].decode())    