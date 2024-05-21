from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
from skimage import exposure
import os

flask_app = Flask(__name__)

UPLOAD_DIR = 'uploaded_images'
PROCESSED_DIR = 'processed_images'

flask_app.config['UPLOAD_DIR'] = UPLOAD_DIR
flask_app.config['PROCESSED_DIR'] = PROCESSED_DIR

# Ensure directories exist
os.makedirs(flask_app.config['UPLOAD_DIR'], exist_ok=True)
os.makedirs(flask_app.config['PROCESSED_DIR'], exist_ok=True)


def gaussian_blur(image):
    kernel_size = (7, 7)
    return cv2.GaussianBlur(image, kernel_size, 0)


def butterworth_filter(image):
    height, width = image.shape
    center_x, center_y = height // 2, width // 2

    # Butterworth filter mask creation
    cutoff_frequency = 30
    filter_order = 2
    filter_mask = np.zeros((height, width), np.float32)
    for y in range(height):
        for x in range(width):
            distance = np.sqrt((y - center_x) ** 2 + (x - center_y) ** 2)
            filter_mask[y, x] = 1 / (1 + (distance / cutoff_frequency) ** (2 * filter_order))

    # Apply Discrete Fourier Transform
    dft_result = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft_result)

    # Mask application in frequency domain
    filtered_shifted = dft_shifted * filter_mask[:, :, np.newaxis]

    # Inverse DFT to obtain filtered image
    inverse_dft_shifted = np.fft.ifftshift(filtered_shifted)
    inverse_img = cv2.idft(inverse_dft_shifted)
    inverse_img_magnitude = cv2.magnitude(inverse_img[:, :, 0], inverse_img[:, :, 1])

    # Normalizing to 0-255 range
    inverse_img_normalized = cv2.normalize(inverse_img_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(inverse_img_normalized)


def laplacian_edge_detection(image):
    return cv2.Laplacian(image, cv2.CV_64F, ksize=7)


def histogram_equalization(source_img, reference_img):
    matched_img = exposure.match_histograms(source_img, reference_img)
    return matched_img


def store_image(image, directory, filename):
    file_path = os.path.join(directory, filename)
    cv2.imwrite(file_path, image)
    return filename


@flask_app.route('/')
def home():
    return render_template('index.html')


@flask_app.route('/upload', methods=['POST'])
def handle_upload():
    uploaded_file = request.files['file']
    filter_selection = request.form.get('filter')
    if not uploaded_file:
        return "No file was uploaded.", 400

    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    original_file = store_image(img, flask_app.config['UPLOAD_DIR'], uploaded_file.filename)

    if filter_selection == 'gaussian':
        processed_img = gaussian_blur(img)
        processed_file = store_image(processed_img, flask_app.config['PROCESSED_DIR'],
                                     'processed_' + uploaded_file.filename)
        return render_template('result.html', original_image=original_file, processed_image=processed_file)

    elif filter_selection == 'butterworth':
        processed_img = butterworth_filter(img)
        processed_file = store_image(processed_img, flask_app.config['PROCESSED_DIR'],
                                     'processed_' + uploaded_file.filename)
        return render_template('result.html', original_image=original_file, processed_image=processed_file)

    elif filter_selection == 'laplacian':
        processed_img = laplacian_edge_detection(img)
        processed_file = store_image(processed_img, flask_app.config['PROCESSED_DIR'],
                                     'processed_' + uploaded_file.filename)
        return render_template('result.html', original_image=original_file, processed_image=processed_file)

    elif filter_selection == 'histogram':
        reference_file = request.files.get('template')
        if not reference_file:
            return "No template file uploaded for histogram matching.", 400
        template_img = cv2.imdecode(np.frombuffer(reference_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        template_file = store_image(template_img, flask_app.config['UPLOAD_DIR'], 'template_' + reference_file.filename)

        processed_img = histogram_equalization(img, template_img)
        processed_file = store_image(processed_img, flask_app.config['PROCESSED_DIR'],
                                     'processed_' + uploaded_file.filename)

        return render_template('result.html', original_image=original_file, reference_image=template_file,
                               processed_image=processed_file)
    else:
        return "Invalid filter selection.", 400


@flask_app.route('/uploaded_images/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(flask_app.config['UPLOAD_DIR'], filename)


@flask_app.route('/processed_images/<filename>')
def serve_result_file(filename):
    return send_from_directory(flask_app.config['PROCESSED_DIR'], filename)


if __name__ == '__main__':
    flask_app.run(debug=True)
