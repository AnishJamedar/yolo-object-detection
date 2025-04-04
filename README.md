# YOLO Object Detection Web Application

This project demonstrates the implementation of YOLO (You Only Look Once) object detection in a web application. It allows users to upload images and see real-time object detection results with bounding boxes and confidence scores.

## Features

- Upload images for object detection
- Real-time processing using YOLOv5
- Display of original and processed images
- List of detected objects with confidence scores
- Modern, responsive web interface

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`

## How to Use

1. Click the "Choose File" button to select an image
2. Click "Detect Objects" to process the image
3. View the results:
   - Original image
   - Processed image with bounding boxes
   - List of detected objects with confidence scores

## Deployment

The application can be deployed on various platforms:
- Heroku
- PythonAnywhere
- AWS
- Google Cloud Platform

For deployment, you'll need to:
1. Set up the environment variables
2. Configure the web server
3. Install the required dependencies

## Technical Details

- Uses YOLOv5 for object detection
- Built with Flask for the backend
- Modern frontend using Bootstrap
- Asynchronous image processing
- Responsive design for all devices

## Learning Outcomes

This project demonstrates:
- Understanding of YOLO architecture
- Implementation of object detection
- Web application development
- Image processing techniques
- API design and implementation 