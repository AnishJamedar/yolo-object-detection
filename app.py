from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import gc
import traceback

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = '/tmp/uploads' if os.environ.get('VERCEL') else 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_model_if_needed():
    model_path = '/tmp/yolov8n.pt' if os.environ.get('VERCEL') else 'yolov8n.pt'
    if not os.path.exists(model_path):
        print("Downloading YOLOv8 model...")
        # Clear memory before downloading
        gc.collect()
        torch.cuda.empty_cache()
        # Use a smaller model for Vercel
        model = YOLO('yolov8n.pt' if not os.environ.get('VERCEL') else 'yolov8n.pt')
        print("Model downloaded successfully!")
    else:
        print("Model already exists, skipping download.")

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
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Download model if needed
            download_model_if_needed()
            
            # Process image with YOLO
            model_path = '/tmp/yolov8n.pt' if os.environ.get('VERCEL') else 'yolov8n.pt'
            # Clear memory before loading model
            gc.collect()
            torch.cuda.empty_cache()
            model = YOLO(model_path)
            results = model(filepath)
            
            # Get the first result
            result = results[0]
            
            # Draw bounding boxes
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
            processed_filename = f'processed_{filename}'
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Clear memory after processing
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
            return jsonify({
                'success': True,
                'original_image': filename,
                'processed_image': processed_filename,
                'detections': detections
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Download model on startup
    download_model_if_needed()
    app.run(debug=True, port=5001) 