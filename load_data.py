import pandas as pd
import sqlite3

# Load your dataset (CSV file)
data = pd.read_csv('restaurants_data.csv')

# Connect to SQLite database
conn = sqlite3.connect('food_safety.db')

# Insert data into the restaurants table
data.to_sql('restaurants', conn, if_exists='replace', index=False)

# Commit and close the connection
conn.commit()
conn.close()

print("Data loaded into the database successfully!")
