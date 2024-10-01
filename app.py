import os
from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model (make sure to adjust the path to your model)
model = joblib.load('Mark_1.pkl')

def process_audio(file):
    # Load the MP3 file using librosa directly from the file object
    y, sr = librosa.load(file, sr=None)

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    # Make prediction
    prediction = model.predict([mfccs_processed])
    
    return int(prediction[0])  # Return the predicted species code

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.lower().endswith('.mp3'):
        # Process the audio file in memory without saving to disk
        predicted_code = process_audio(file)
        
        return jsonify({'predicted_code': predicted_code})
    else:
        return jsonify({'error': 'Invalid file format. Please upload an MP3 file.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
