import mysql.connector
from find_vote import take_post

mydb = mysql.connector.connect(
    host = 'www.devkids.co.kr',
    user = 'newming',
    port = 14402,
    database = 'ds_lab',
    password = 'u8Yto93qrAgW',
)

find_post_id = []
find_post_title = []
find_post_content = []
res = []

for j in range(5):
    mycursor = mydb.cursor()

    mycursor.execute("SELECT _id, post_title, post_content FROM nm_post WHERE _id = %s", (take_post[j], ))

    myres = mycursor.fetchall()

    res.append({
        '_id': myres[0][0],
        'title': myres[0][1].decode(),
        'content': myres[0][2].decode(),
    })    
print(res)