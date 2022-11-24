import mysql.connector

mydb = mysql.connector.connect(
    host = 'www.devkids.co.kr',
    user = 'newming',
    port = 14402,
    database = 'ds_lab',
    password = 'u8Yto93qrAgW',
)

mycursor = mydb.cursor()

mycursor.execute("SELECT _id, post_title, post_content FROM nm_post WHERE post_type = 'vote' ORDER BY _id DESC LIMIT 0, 10000")

myres = mycursor.fetchall()

post_id = []
post_title = []
post_content = []

for i in range(0, len(myres)):
    post_id.append(myres[i][0])
    post_content.append(myres[i][2].decode())
    post_title.append(myres[i][1].decode())
# print(article)

