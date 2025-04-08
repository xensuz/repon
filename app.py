from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import cv2

app = Flask(__name__)

# Load the trained model
model_path = os.path.abspath('mask_detector.h5').encode('utf-8')
model = load_model(model_path)

def preprocess_image(image):
    # Resize to match model input shape
    image = cv2.resize(image, (128, 128))
    # Convert to RGB if needed
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Scale the image
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected"
    
    # Read and preprocess the image
    image = Image.open(file)
    image_array = np.array(image)
    processed_image = preprocess_image(image_array)
    
    # Make prediction
    prediction = model.predict(processed_image)
    result = "With Mask" if np.argmax(prediction) == 0 else "Without Mask"
    confidence = float(np.max(prediction))
    
    return render_template('output.html', 
                         result=result, 
                         confidence=f"{confidence:.2%}")

if __name__ == '__main__':
    app.run(debug=True)