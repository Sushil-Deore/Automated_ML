# Importing Packages

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify
import logging
import os

# Application logging

logging.basicConfig(filename='deployment_logs.log', level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')

# Flask application
app = Flask(__name__)

# Loading the model from the File
model_load = joblib.load('')

# Enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


# Get the uploaded files
@app.route("/", methods=['POST'])
def uploadFiles():
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # set the file path
        uploaded_file.save(file_path)
    # save the file
    return redirect(url_for('index'))


@app.route('/predict', method=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        input_val = []
        final_features = [np.array(input_val)]
        df = pd.DataFrame(final_features)

        output = model_load.predict(df)
        result = "%.2f" % round(output[0], 2)

        # logging operation
        logging.info(f"Insurance Premium is {result}")

        logging.info('Prediction getting posted to the web page.')

        return render_template('index.html', prediction_text=f'Insurance Premium is $ {result} ')

    else:
        return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict_api():
    print("request.method :", request.method)
    if request.method == 'POST':
        input_val = []
        final_features = [np.array(input_val)]
        df = pd.DataFrame(final_features)

        output = model_load.predict(df)
        result = "%.2f" % round(output[0], 2)

        return jsonify(result)

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)