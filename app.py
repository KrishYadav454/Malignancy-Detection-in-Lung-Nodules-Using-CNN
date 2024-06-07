import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Load the pre-trained model
base_model = VGG19(include_top=False, input_shape=(224, 224, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)
model_03.load_weights('model_01.weights.h5')

app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')

# Function to determine the class name
def get_className(classNo):
    if classNo == 0:
        return "Cancerous"
    elif classNo == 1:
        return "Non Cancerous"

# Function to predict the result from the image
def getResult(img):
    image = cv2.imread(img)
    if not is_ct_scan(image):
        return "Invalid Image. Please upload a CT scan image."
    
    image = Image.fromarray(image, 'RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model_03.predict(input_img)
    result01 = np.argmax(result, axis=1)
    return get_className(result01[0])

# Enhanced heuristic to check if the image is a CT scan
def is_ct_scan(image):
    # Check if the image is grayscale (2D array or single channel)
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        return True
    
    # Additional heuristic: check if the image is predominantly grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        diff = cv2.absdiff(image, color_image)
        non_gray_count = np.count_nonzero(diff)
        total_count = diff.size
        gray_ratio = non_gray_count / total_count
        if gray_ratio < 0.1:  # 90% of the image is grayscale
            return True

    return False

# Routes for the Flask app
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        return value
    return None

if __name__ == '__main__':
    app.run(debug=True)
