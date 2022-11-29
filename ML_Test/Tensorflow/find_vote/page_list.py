import mysql.connector

mydb = mysql.connector.connect(
    host = 'www.devkids.co.kr',
    user = 'newming',
    port = 14402,
    database = 'ds_lab',
    password = 'u8Yto93qrAgW',
)

mycursor = mydb.cursor()

mycursor.execute("SELECT _id, title, summary FROM nm_page WHERE page_parent_id = 2598 ORDER BY _id DESC LIMIT 0, 10")

myres = mycursor.fetchall()

page_id = []
page_title = []
page_summary = []
page_content = []

for i in range(0, len(myres)):
    page_id.append(myres[i][0])
    page_title.append(myres[i][1].decode())
    page_summary.append(myres[i][2].decode())
    page_content.append(myres[i][1].decode() + ' ' + myres[i][2].decode() )

# print(page_content)
