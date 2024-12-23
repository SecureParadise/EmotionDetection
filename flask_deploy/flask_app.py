import os
from flask import Flask, request, redirect, url_for, render_template
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import keras

# Load pre-trained model
model = keras.models.load_model('/home/seanet/Documents/Learning/AI/CnnProject/DataSets/Resnet_model_version_2.keras')

# Emotion labels
emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
index_to_emotion = {v: k for k, v in emotion_labels.items()}

app = Flask(__name__)

# Directory to store uploaded images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Route to display the upload form
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the image file is in the request
        if 'image' not in request.files:
            return "No file part", 400
        file = request.files['image']

        # If no file is selected, return an error
        if file.filename == '':
            return "No file selected", 400

        if file:
            # Save the file temporarily
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # Redirect to the prediction route with the image filename
            return redirect(url_for('prepare_image', filename=file.filename))

    return render_template('index.html')

@app.route('/prep/<filename>', methods=['GET'])
def prepare_image(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        # Process image
        img_pil = Image.open(filepath)
        img = img_pil.convert('RGB')
        img = img.resize((224, 224))  # Resize image to match model input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_emotion = index_to_emotion.get(predicted_class[0], "Unknown Emotion")
    except Exception as e:
        return f"Error processing image: {str(e)}", 500

    return render_template(
        'result.html',
        filename=filename,
        emotion=predicted_emotion
    )

if __name__ == '__main__':
    app.run(debug=True)
