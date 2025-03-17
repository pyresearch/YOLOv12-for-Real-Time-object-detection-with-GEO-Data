import os
import cv2
import supervision as sv
from ultralytics import YOLO
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from collections import Counter
from werkzeug.exceptions import RequestEntityTooLarge
import pyresearch

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output_images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
app.config['MAX_CONTENT_LENGTH'] = 5000 * 1024 * 1024  # Set max upload size to 5GB (5,242,880,000 bytes)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Classes used by the model
CLASSES = ['airport', 'helicopter', 'oiltank', 'plane', 'warship']

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model(weights_path):
    """Load the YOLO model."""
    try:
        model = YOLO(weights_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_image(model, image_path, output_path):
    """Process an image and return detected class counts."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file {image_path}.")
        return Counter()

    # Perform object detection
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Annotate the image
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Save the annotated image
    cv2.imwrite(output_path, annotated_image)

    # Count detected classes
    class_ids = detections.class_id
    class_names = [CLASSES[class_id] for class_id in class_ids if class_id < len(CLASSES)]
    return Counter(class_names)

def analyze_folder(model, input_folder, output_folder):
    """Analyze all images in a folder and return results."""
    image_files = [f for f in os.listdir(input_folder) if allowed_file(f)]
    total_counts = Counter()
    image_results = []

    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_filename = f"annotated_{filename}"
        output_path = os.path.join(output_folder, output_filename)

        # Process image and get class counts
        counts = process_image(model, input_path, output_path)
        total_counts.update(counts)

        # Store result for display
        image_results.append({
            'original': url_for('uploaded_file', filename=filename),
            'annotated': url_for('output_file', filename=output_filename),
            'counts': dict(counts)
        })

    return image_results, dict(total_counts)

# Load the model once when the app starts
weights_path = "last.pt"  # Replace with your YOLO weights file path
model = load_model(weights_path)

@app.errorhandler(RequestEntityTooLarge)
def handle_entity_too_large(e):
    """Handle oversized upload errors."""
    print(f"Request too large. Content-Length: {request.content_length} bytes")
    return render_template('dashboard.html', error="File too large! Max size is 5GB. Please upload fewer or smaller files."), 413

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    """Handle file uploads and display dashboard."""
    error = None
    image_results = []
    total_counts = {}

    if request.method == 'POST':
        print(f"Received POST request. Content-Length: {request.content_length} bytes")
        if 'files[]' not in request.files:
            error = "No files uploaded"
        else:
            files = request.files.getlist('files[]')
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                else:
                    error = "Invalid file type detected"

            if not error and model:
                image_results, total_counts = analyze_folder(model, app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'])
            else:
                error = "Model not loaded or no valid files"

    # If no POST request, still show existing images in folder
    elif model:
        image_results, total_counts = analyze_folder(model, app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'])

    return render_template('dashboard.html', 
                         image_results=image_results, 
                         total_counts=total_counts, 
                         classes=CLASSES, 
                         error=error)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded original images."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output_images/<filename>')
def output_file(filename):
    """Serve annotated images."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    if model:
        print(f"Max content length set to: {app.config['MAX_CONTENT_LENGTH']} bytes")
        app.run(debug=True)
    else:
        print("Cannot start the application: Model failed to load.")