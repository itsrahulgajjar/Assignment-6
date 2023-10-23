from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)

# Load the pre-trained MNIST classifier model
model = keras.models.load_model("mnist_classifier_model.h5")

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # Process the uploaded image and get predictions
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 (MNIST image size)
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model
    img_array = img_array / 255.0  # Normalize pixel values

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    return render_template('results.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
