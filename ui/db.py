import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="mysql.railway.internal",
        port=3306,
        user="root",
        password="katRoJeQMIdzImFzDndYCoxRqqeDaxvH",
        database="railway"
    )
