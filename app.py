from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch
import gc
import traceback
import sys
from ultralytics import YOLO

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = '/tmp/uploads' if os.environ.get('VERCEL') else 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Global variable to store the model
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_model_if_needed():
    global model
    try:
        model_path = '/tmp/yolov5n.pt' if os.environ.get('VERCEL') else 'yolov5n.pt'
        if not os.path.exists(model_path):
            print("Downloading YOLOv5 model...")
            # Clear memory before downloading
            gc.collect()
            torch.cuda.empty_cache()
            # Use a smaller model for Vercel
            model = YOLO('yolov5n.pt' if not os.environ.get('VERCEL') else 'yolov5n.pt')
            print("Model downloaded successfully!")
        else:
            print("Model already exists, skipping download.")
            model = YOLO(model_path)
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("Received upload request")
        if 'file' not in request.files:
            print("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            print(f"Processing file: {file.filename}")
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Use the global model
            global model
            if model is None:
                print("Model not loaded, downloading...")
                download_model_if_needed()
            
            # Process image with YOLO
            print("Running inference...")
            results = model(filepath)
            
            # Get the first result
            result = results[0]
            
            # Draw bounding boxes
            print("Processing image...")
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = box.cls[0].item()
                class_name = result.names[int(class_id)]
                
                # Draw rectangle
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(img, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                detections.append({
                    'class': class_name,
                    'confidence': float(confidence),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
            
            # Save processed image
            print("Saving processed image...")
            processed_filename = f'processed_{filename}'
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Clear memory after processing
            gc.collect()
            torch.cuda.empty_cache()
            
            response_data = {
                'success': True,
                'original_image': filename,
                'processed_image': processed_filename,
                'detections': detections
            }
            
            print("Sending response:", response_data)
            return jsonify(response_data)
        
        print("Invalid file type")
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Download model on startup
    download_model_if_needed()
    app.run(debug=True, port=5001) 