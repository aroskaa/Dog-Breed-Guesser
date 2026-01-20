import sqlite3

conn = sqlite3.connect("users.db")
cursor = conn.execute("SELECT id, username FROM users")

print("Users yang terdaftar:")
for row in cursor:
    print(f"ID: {row[0]}, Username: {row[1]}")

conn.close()