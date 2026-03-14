from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

import json

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# keys come as strings → convert to int
class_names = {int(k): v for k, v in class_names.items()}
# Load model once (important!)
model = load_model('healthy_vs_rotten.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Image preprocessing
            img = load_img(filepath, target_size=(224, 224))
            x = img_to_array(img)
            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)

            preds = model.predict(x)
            class_id = np.argmax(preds)
            label = class_names[class_id]
            confidence = float(np.max(preds)) * 100
            # label = class_names.get(class_id, "Unknown")

            return render_template(
                'predict.html',
                prediction=label,
                confidence=round(confidence, 2),
                image_path=filepath
            )

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)