import os
from flask import Flask, request,jsonify, render_template, send_from_directory
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import math
import json
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Load the taxonomy CSV file
taxonomy_df = pd.read_csv('eBird_Taxonomy_v2021.csv')
species_info = taxonomy_df.set_index('SPECIES_CODE').to_dict('index')

# Load your data and create LabelEncoder
df = pd.read_csv('mapping.csv')
class_names = df['class'].unique()
le = LabelEncoder()
le.fit(class_names)

# Load the pre-trained model
model = joblib.load('Mark_1.pkl')

def clean_for_json(obj):
    """Clean dictionary values for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif pd.isna(obj):  # Handle NaN values
        return None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj) if not math.isnan(obj) else None
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return clean_for_json(obj.tolist())
    else:
        return obj

def process_audio(file_path):
    # Load the MP3 file using librosa
    y, sr = librosa.load(file_path, sr=None)

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    # Make prediction
    prediction = model.predict([mfccs_processed])
    predicted_code = le.inverse_transform(prediction)[0]

    # Get species information
    predicted_info = species_info.get(predicted_code, {})

    cleaned_info = clean_for_json(predicted_info)
    
    return cleaned_info

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and file.filename.lower().endswith('.mp3'):
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            predicted_info = process_audio(file_path)

            return render_template('index.html', info=predicted_info)
        else:
            return render_template('index.html', error='Invalid file format. Please upload an MP3 file.')

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith('.mp3'):
        try:
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            app.logger.info(f'Processing file: {filename}')
            predicted_info = process_audio(file_path)
            app.logger.info(f'Prediction result: {predicted_info}')
            
            # Clean the response data
            cleaned_response = {
                'PRIMARY_COM_NAME': predicted_info.get('PRIMARY_COM_NAME'),
                'SCI_NAME': predicted_info.get('SCI_NAME'),
                'ORDER1': predicted_info.get('ORDER1'),
                'FAMILY': predicted_info.get('FAMILY'),
                'CATEGORY': predicted_info.get('CATEGORY')
            }
            
            response = jsonify(cleaned_response)
            return response
            
        except Exception as e:
            app.logger.error(f'Error processing request: {str(e)}')
            return jsonify({'error': 'Internal server error', 'details': str(e)}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return jsonify({'error': 'Invalid file format. Please upload an MP3 file.'}), 400


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0')