import sqlite3

# Connect to SQLite database (or create it)
conn = sqlite3.connect('food_safety.db')

# Create a cursor
cursor = conn.cursor()

# Create the restaurants table
cursor.execute('''
CREATE TABLE IF NOT EXISTS restaurants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurant_name TEXT,
    dining_rating REAL,
    delivery_rating REAL,
    dining_votes INTEGER,
    delivery_votes INTEGER,
    cuisine TEXT,
    place_name TEXT,
    city TEXT,
    item_name TEXT,
    best_seller TEXT,
    votes INTEGER,
    prices REAL
);
''')

# Commit and close the connection
conn.commit()
conn.close()

print("Database and table created successfully!")
