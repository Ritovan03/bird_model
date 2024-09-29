import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)

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
    
    return predicted_info

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
        filename = file.filename
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        predicted_info = process_audio(file_path)
        
        return jsonify(predicted_info)
    else:
        return jsonify({'error': 'Invalid file format. Please upload an MP3 file.'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)