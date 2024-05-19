import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage import exposure

def gaussian_filter(image):
    ksize = (3, 3)
    result = cv2.GaussianBlur(image, ksize, 0)
    return result

def butterworth_filter(image):
    height, width = image.shape
    center_y, center_x = height // 2, width // 2

    cutoff_frequency = 30  # Define cutoff frequency
    filter_order = 2  # Define filter order
    butterworth_mask = np.zeros((height, width), np.float32)
    for y in range(height):
        for x in range(width):
            distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
            butterworth_mask[y, x] = 1 / (1 + (distance / cutoff_frequency) ** (2 * filter_order))

    # Perform DFT (Discrete Fourier Transform)
    dft_result = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft_result)
    
    # Apply Butterworth mask
    filtered_shifted = dft_shifted * butterworth_mask[:, :, np.newaxis]

    # Perform inverse DFT
    inv_dft_shifted = np.fft.ifftshift(filtered_shifted)
    filtered_image = cv2.idft(inv_dft_shifted)
    filtered_image_magnitude = cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])

    # Normalize the result to the range 0-255
    normalized_image = cv2.normalize(filtered_image_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(normalized_image)


def laplacian_filter(image):
    result = cv2.Laplacian(image, cv2.CV_64F, ksize=5)
    return result

def histogram_matching(img1, img2):
    matched = exposure.match_histograms(img1, img2)
    return matched

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.configure(bg="#f0f0f0")
        
        self.original_image = None
        self.filtered_image = None
        self.template_image = None
        
        self.create_widgets()

    def create_widgets(self):
        # Styles
        button_style = {"bg": "#4CAF50", "fg": "white", "font": ("Arial", 12), "relief": "raised"}
        label_style = {"bg": "#f0f0f0", "font": ("Arial", 14)}
        combobox_style = {"font": ("Arial", 12)}
        
        # Upload original image
        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image, **button_style)
        self.upload_button.pack(pady=10)

        # Upload template image (initially disabled)
        self.upload_template_button = tk.Button(self.root, text="Upload Template Image", command=self.upload_template_image, state=tk.DISABLED, **button_style)
        self.upload_template_button.pack(pady=10)

        # Filter selection
        self.filter_label = tk.Label(self.root, text="Choose a filter:", **label_style)
        self.filter_label.pack(pady=5)
        
        self.filter_var = tk.StringVar()
        self.filter_combobox = ttk.Combobox(self.root, textvariable=self.filter_var, **combobox_style)
        self.filter_combobox['values'] = ('Gaussian Filter', 'Butterworth Filter', 'Laplacian Filter', 'Histogram Matching')
        self.filter_combobox.current(0)
        self.filter_combobox.pack(pady=5)
        self.filter_combobox.bind("<<ComboboxSelected>>", self.on_filter_change)

        # Apply filter button
        self.apply_button = tk.Button(self.root, text="Apply Filter", command=self.apply_filter, **button_style)
        self.apply_button.pack(pady=10)

        # Display frames for images and their titles
        self.image_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.image_frame.pack(pady=10)

        # Original image frame
        self.original_frame = tk.Frame(self.image_frame, bg="#f0f0f0")
        self.original_frame.grid(row=0, column=0, padx=10)

        self.original_image_canvas = tk.Label(self.original_frame, bg="#f0f0f0")
        self.original_image_canvas.pack()

        self.original_image_label = tk.Label(self.original_frame, text="Original Image", **label_style)
        self.original_image_label.pack()

        # Template image frame
        self.template_frame = tk.Frame(self.image_frame, bg="#f0f0f0")
        self.template_frame.grid(row=0, column=1, padx=10)

        self.template_image_canvas = tk.Label(self.template_frame, bg="#f0f0f0")
        self.template_image_canvas.pack()

        self.template_image_label = tk.Label(self.template_frame, text="Template Image", **label_style)
        self.template_image_label.pack()

        # Initially hide template image and label
        self.template_frame.grid_remove()

        # Filtered image frame
        self.filtered_frame = tk.Frame(self.image_frame, bg="#f0f0f0")
        self.filtered_frame.grid(row=0, column=2, padx=10)

        self.filtered_image_canvas = tk.Label(self.filtered_frame, bg="#f0f0f0")
        self.filtered_image_canvas.pack()

        self.filtered_image_label = tk.Label(self.filtered_frame, text="Filtered Image", **label_style)
        self.filtered_image_label.pack()

    def on_filter_change(self, event):
        filter_type = self.filter_var.get()
        if filter_type == 'Histogram Matching':
            self.upload_template_button.config(state=tk.NORMAL)
        else:
            self.upload_template_button.config(state=tk.DISABLED)
            self.template_frame.grid_remove()  # Hide template frame if not needed

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.original_image, self.original_image_canvas)

    def upload_template_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.template_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.template_image, self.template_image_canvas)
            self.template_frame.grid()  # Show template frame when a template is uploaded

    def display_image(self, image, label):
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image=image)
        label.config(image=image_tk)
        label.image = image_tk

    def apply_filter(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        filter_type = self.filter_var.get()
        if filter_type == 'Gaussian Filter':
            self.filtered_image = gaussian_filter(self.original_image)
        elif filter_type == 'Butterworth Filter':
            self.filtered_image = butterworth_filter(self.original_image)
        elif filter_type == 'Laplacian Filter':
            self.filtered_image = laplacian_filter(self.original_image)
        elif filter_type == 'Histogram Matching':
            if self.template_image is None:
                messagebox.showerror("Error", "Please upload a template image for histogram matching.")
                return
            self.filtered_image = histogram_matching(self.original_image, self.template_image)
        else:
            messagebox.showerror("Error", "Invalid filter type.")
            return

        self.display_image(self.filtered_image, self.filtered_image_canvas)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
