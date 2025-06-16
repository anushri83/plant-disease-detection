import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# Load CSV data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load Model
model = CNN.CNN(39)
try:
    model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

# Function for Prediction with Exception Handling
def prediction(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = model(input_data)
        
        output = output.numpy()
        index = np.argmax(output)

        return index
    except Exception as e:
        print(f"Prediction Error: {e}")
        return -1  # Return an invalid index

# Flask App Initialization
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file uploaded!", 400  # Bad request

        image = request.files['image']
        
        if image.filename == '':
            return "No selected file!", 400

        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return "Invalid file format! Please upload a PNG or JPG image.", 400

        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        
        try:
            image.save(file_path)
            pred = prediction(file_path)

            if pred == -1:
                return "Error: Unable to process the image. Please try again.", 500  # Internal server error

            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            
            return render_template(
                'submit.html',
                title=title,
                desc=description,
                prevent=prevent,
                image_url=image_url,
                pred=pred,
                sname=supplement_name,
                simage=supplement_image_url,
                buy_link=supplement_buy_link
            )
        except Exception as e:
            return f"Error processing image: {str(e)}", 500  # Internal server error

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
