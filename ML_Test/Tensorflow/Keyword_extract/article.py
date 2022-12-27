import mysql.connector

mydb = mysql.connector.connect(
    host = 'localhost',
    user = 'root',
    port = 3306,
    database = 'newming_press_article',
    password = '1234',
)

mycursor = mydb.cursor()

mycursor.execute("""SELECT category_ids, text, 60days._key from nm_interest_press_article_60days as 60days,
              (select _key , text from nm_press_article_20221201
              union  all
              select _key , text from nm_press_article_20221202
                                 union  all
              select _key , text from nm_press_article_20221203
                                 union  all
              select _key , text from nm_press_article_20221204
                                 union  all
              select _key , text from nm_press_article_20221205
							                   union  all
              select _key , text from nm_press_article_20221206
							                   union  all
              select _key , text from nm_press_article_20221207
							                   union  all
              select _key , text from nm_press_article_20221208
							                   union  all
              select _key , text from nm_press_article_20221209
							                   union  all
              select _key , text from nm_press_article_20221211
							                   union  all
              select _key , text from nm_press_article_20221212
							                   union  all
              select _key , text from nm_press_article_20221213
							                   union  all
              select _key , text from nm_press_article_20221214
							                   union  all
              select _key , text from nm_press_article_20221215
							                   union  all
              select _key , text from nm_press_article_20221216
							                   union  all
              select _key , text from nm_press_article_20221217
							                   union  all
              select _key , text from nm_press_article_20221218
							                   union  all
              select _key , text from nm_press_article_20221219
							                   union  all
              select _key , text from nm_press_article_20221220
							                   union  all
              select _key , text from nm_press_article_20221221
							                   union  all
              select _key , text from nm_press_article_20221222
							                   union  all
              select _key , text from nm_press_article_20221223
							                   union  all
              select _key , text from nm_press_article_20221224
							                   union  all
              select _key , text from nm_press_article_20221225
							                   union  all
              select _key , text from nm_press_article_20221226
										
                                  ) as article_date
                                  where 60days._key  = article_date._key""")

myres = mycursor.fetchall()

article = []
category_ids = []
article_key = []
for i in range(0, len(myres)):
    article.append(myres[i][1].decode())
    category_ids.append(myres[i][0].decode())
    article_key.append(myres[i][2].decode())
# print(article) 