from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
from skimage import exposure
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Create directories if they do not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def apply_gaussian_filter(image):
    ksize = (9, 9)
    return cv2.GaussianBlur(image, ksize, 0)

def apply_butterworth_filter(image):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Create a Butterworth filter mask
    D0 = 30  # Cutoff frequency
    n = 2  # Order of the filter
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = 1 / (1 + (distance / D0) ** (2 * n))

    # Apply DFT to the image
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Apply the mask to the DFT shifted image
    fshift = dft_shift * mask[:, :, np.newaxis]

    # Inverse DFT to get the image back
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize the image to the range 0-255
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)

def apply_laplacian_filter(image):
    return cv2.Laplacian(image, cv2.CV_64F, ksize=7)

def histogram_matching(source, template):
    matched = exposure.match_histograms(source, template)
    return matched

def save_image(image, folder, filename):
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, image)
    return filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filter_type = request.form.get('filter')
    if not file:
        return "No file uploaded.", 400

    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    original_filename = save_image(image, app.config['UPLOAD_FOLDER'], file.filename)

    if filter_type == 'gaussian':
        filtered_image = apply_gaussian_filter(image)
        filtered_filename = save_image(filtered_image, app.config['RESULT_FOLDER'], 'filtered_' + file.filename)
        return render_template('result.html', original_image=original_filename, filtered_image=filtered_filename)
    
    elif filter_type == 'butterworth':
        filtered_image = apply_butterworth_filter(image)
        filtered_filename = save_image(filtered_image, app.config['RESULT_FOLDER'], 'filtered_' + file.filename)
        return render_template('result.html', original_image=original_filename, filtered_image=filtered_filename)
    
    elif filter_type == 'laplacian':
        filtered_image = apply_laplacian_filter(image)
        filtered_filename = save_image(filtered_image, app.config['RESULT_FOLDER'], 'filtered_' + file.filename)
        return render_template('result.html', original_image=original_filename, filtered_image=filtered_filename)
    
    elif filter_type == 'histogram':
        template_file = request.files.get('template')
        if not template_file:
            return "No template file uploaded for histogram matching.", 400
        template_image = cv2.imdecode(np.frombuffer(template_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        template_filename = save_image(template_image, app.config['UPLOAD_FOLDER'], 'template_' + template_file.filename)
        
        filtered_image = histogram_matching(image, template_image)
        filtered_filename = save_image(filtered_image, app.config['RESULT_FOLDER'], 'filtered_' + file.filename)
        
        return render_template('result.html', original_image=original_filename, reference_image=template_filename, filtered_image=filtered_filename)
    else:
        return "Invalid filter type.", 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
