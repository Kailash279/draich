import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="your_actual_password",   # ← Replace this
        database="your_database_name"      # ← Replace this
    )
    print("✅ Connection successful")
    conn.close()
except mysql.connector.Error as err:
    print("❌ Error:", err)
