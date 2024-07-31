from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust CORS policy as needed

model_file_path = 'your_model.h5'
captions_file_path = 'captions.npy'

# Load the pre-trained model and captions
model = load_model(model_file_path)
captions = np.load(captions_file_path, allow_pickle=True).item()

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            filename = os.path.join('uploads', file.filename)
            file.save(filename)
            
            # Simulating caption generation
            caption = captions.get(file.filename, "Caption not found")
            return jsonify({'caption': caption, 'image_url': f"/uploads/{file.filename}"})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Vercel's entry point for the app
def handler(request):
    return app(request.environ, request.start_response)
