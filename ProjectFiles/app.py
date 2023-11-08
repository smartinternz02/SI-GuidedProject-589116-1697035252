from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/cog')
def cog():
    return render_template('cog.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict_image_file', methods=['POST'])
def predict_image_file():
    if request.method == 'POST':
        img_file = request.files['file']
        img = Image.open(img_file)
        img = img.resize((150, 150))
        img = img.convert("RGB")  # Convert to RGB
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        
        model = load_model("C:/Users/angad/Downloads/MindSync+/xception.h5")
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        class_labels = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        predicted_label = class_labels[predicted_class]

        if (predicted_label == 'VeryMildDemented'):
            return render_template('vmild_dem.html')
        elif (predicted_label == 'MildDemented'):
            return render_template('mild_dem.html')
        elif (predicted_label == 'ModerateDemented'):
            return render_template('mod_dem.html')
        elif (predicted_label == 'NonDemented'):
            return render_template('non_dem.html')

if __name__ == '__main__':
    app.run(debug=True)