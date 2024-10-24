import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # Import joblib to save the model

# Connect to the SQLite database
conn = sqlite3.connect('food_safety.db')

# Query the data
query = """
SELECT 
    Dining_Rating AS dining_rating, 
    Delivery_Rating AS delivery_rating, 
    "Dining Votes" AS dining_votes, 
    Delivery_Votes AS delivery_votes, 
    Votes AS votes, 
    Prices AS prices 
FROM 
    restaurants
"""

# Load the data into a pandas DataFrame
data = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# Ensure column names are stripped and lowercased
data.columns = data.columns.str.strip().str.lower()

# Create a binary target variable
data['compliant'] = (data['dining_rating'] > 4.0).astype(int)  # Modify this condition as needed

# Check for missing values
print(data.isnull().sum())

# Fill NaN values with the mean of each column
data.fillna(data.mean(), inplace=True)

# Define features and target variable
X = data[['dining_rating', 'delivery_rating', 'dining_votes', 'delivery_votes', 'votes', 'prices']]
y = data['compliant']

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Machine Learning Model
# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Save the Trained Model
joblib.dump(model, 'food_safety_model.pkl')  # Save the trained model

# Load the model from the file
loaded_model = joblib.load('food_safety_model.pkl')

# Use the loaded model to make predictions
y_loaded_pred = loaded_model.predict(X_test)

# Evaluate the loaded model
loaded_accuracy = accuracy_score(y_test, y_loaded_pred)
print(f"Loaded Model Accuracy: {loaded_accuracy:.2f}")
