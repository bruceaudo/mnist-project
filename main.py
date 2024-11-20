import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
port = os.getenv('PORT')
host = os.getenv('HOST')

app = Flask(__name__)

# Load the trained MNIST model
model = tf.keras.models.load_model('mnist_model_v2.keras')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    # Open the image and preprocess it
    img = Image.open(file).convert('L').resize((28, 28))  # Convert to grayscale and resize
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 28, 28)  # Add batch dimension

    # Predict using the model
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions)
    confidence = float(np.max(predictions) * 100)  # Extract the maximum probability as confidence

    return jsonify({
        'digit': int(predicted_digit),
        'confidence': f"{confidence:.2f}%"
    })

if __name__ == '__main__':
    app.run(host=host, port=port)
