import mysql.connector

def temp(text, key):
    mydb = mysql.connector.connect(
        host = 'localhost',
        user = 'root',
        port = 3306,
        database = 'newming_press_article',
        password = '1234',
    )

    mycursor = mydb.cursor()


    mycursor.execute(f'UPDATE nm_interest_press_article_60days set new_tags = \"{text}\" where _key = \'{key}\'')

    mydb.commit()

# def temp(text):
#     print('temp')
#     print(text)





