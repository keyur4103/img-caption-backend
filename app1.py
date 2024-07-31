from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Define file paths
model_file_path = 'your_model_caption.h5'
captions_file_path = 'captions.npy'
captions = {}

# Load the pre-trained model
try:
    model = load_model(model_file_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load captions from the file
try:
    captions = np.load(captions_file_path, allow_pickle=True).item()
    print("Captions loaded successfully.")
except Exception as e:
    print(f"Error loading captions: {e}")

def generate_caption(image_name):
    base_name = os.path.basename(image_name)
    caption = captions.get(base_name, "Caption not found")
    print(f"Looking for caption for: {base_name}")
    print(f"Caption found: {caption}")
    return caption

@app.route('/upload', methods=['POST'])
def upload_image():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Save the uploaded file
            filename = os.path.join('uploads', file.filename)
            file.save(filename)
            
            print(f"Uploaded file: {file.filename}")

            # Get the caption for the uploaded image
            caption = generate_caption(file.filename)
            print(f"Caption found: {caption}")

            image_url = f"/uploads/{file.filename}"
            
            return jsonify({'caption': caption, 'image_url': image_url})
        except Exception as e:
            print(f"Error saving file or generating caption: {e}")
            return jsonify({'error': 'Error processing file or generating caption'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        
    app.run(debug=True, port=5000)
