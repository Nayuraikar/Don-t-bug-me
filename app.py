from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd  # Import pandas here

app = Flask(__name__)

# Load the trained model
model = joblib.load('food_safety_model.pkl')

@app.route('/')
def home():
    return "Welcome to the Food Safety Violation Detector!"

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure data is provided
    data = request.json
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # Extract features
    features = np.array([[data['dining_rating'], 
                          data['delivery_rating'], 
                          data['dining_votes'], 
                          data['delivery_votes'], 
                          data['votes'], 
                          data['prices']]])

    # Make a prediction
    prediction = model.predict(features)

    # Return the prediction as a JSON response
    return jsonify({'compliant': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Set debug=False for production

