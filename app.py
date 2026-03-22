import os
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('insurance_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the encoder
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
@app.route('/')
def home():
    return "Insurance Prediction API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([data])

        # Apply encoder (IMPORTANT FIX)
        input_processed = encoder.transform(input_df)

        # Prediction
        prediction = model.predict(input_processed)

        return jsonify({'prediction': max(prediction[0], 0)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)