from flask import Flask, request, jsonify
import os
from image_segmentation import process_image  # Import the segmentation function
from flask_cors import CORS, cross_origin
import logging

app = Flask(__name__)
CORS(app, origin = "*")
app.config['CORS_HEADERS'] = 'Content-Type'


# Configure the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
SEGMENTED_FOLDER = 'segmented'  # Folder to store segmented images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTED_FOLDER'] = SEGMENTED_FOLDER

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image(file):
    """Save the uploaded file to the upload folder."""
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    logging.info("Input File Saved")
    return file_path

def segment_image(file_path):
    """Segment the image and save the output."""
    filename = os.path.basename(file_path)
    output_filename = 'segmented_' + filename
    output_path = os.path.join(app.config['SEGMENTED_FOLDER'], output_filename)
    process_image(file_path, output_path) # Call the segmentation function
    return output_filename

@app.route('/upload-image', methods=['POST'])
@cross_origin(origin = "*")
def upload_image():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        file_path = save_image(file)  # Save the uploaded file
        output_filename = segment_image(file_path)  # Perform image segmentation

        # Return the path or URL of the segmented image
        return jsonify({'message': 'File uploaded and segmented successfully', 'filename': output_filename}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/')
def home():
    return "hi"
if __name__ == '__main__':
    # Create the upload and segmented folders if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(SEGMENTED_FOLDER, exist_ok=True)
    app.run(debug=True)