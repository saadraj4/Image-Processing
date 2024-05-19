import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
#%%
def apply_gaussian_filter(image):
    ksize = (3, 3)
    return cv2.GaussianBlur(image, ksize, 0)
#%%
def apply_butterworth_filter(image):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = 1 / (1 + (distance / 30) ** (2 * 2))

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    fshift = dft_shift * mask[:, :, np.newaxis]
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back
#%%
def apply_laplacian_filter(image):
    return cv2.Laplacian(image, cv2.CV_64F,ksize=3)
#%%
def histogram_matching(source, template):
    matched = exposure.match_histograms(source, template, multichannel=False)
    return matched
#%%
def display_image(title, image):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
#%%
def main():
    lowpass_gausian = "./lowpass gausian.png"
    lowpass_butterworth = "./lowpass butterworth.png"
    highpass_laplacian = "./Highpass Laplacian.png"
    histogram_source = "./Histogram source.png"
    histogram_reference = "./Histogram reference.png"
    
    image_path = './image.png'  # Replace with the path to your image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    lowpass_gausian_img = cv2.imread(lowpass_gausian, cv2.IMREAD_GRAYSCALE)
    lowpass_butterworth_img = cv2.imread(lowpass_butterworth, cv2.IMREAD_GRAYSCALE)
    highpass_laplacian_img = cv2.imread(highpass_laplacian, cv2.IMREAD_GRAYSCALE)
    histogram_source_img = cv2.imread(histogram_source, cv2.IMREAD_GRAYSCALE)
    histogram_reference_img = cv2.imread(histogram_reference, cv2.IMREAD_GRAYSCALE)
    
    
#%%
    gaussian_filtered = apply_gaussian_filter(lowpass_gausian_img)
    butterworth_filtered = apply_butterworth_filter(lowpass_butterworth_img)
    laplacian_filtered = apply_laplacian_filter(highpass_laplacian_img)
    histogram_matched = histogram_matching(histogram_source_img, histogram_reference_img)  # Using the same image as template for example
#%%
    display_image('Original Image', img)
    display_image('Lowpass Gaussian Filter', gaussian_filtered)
    display_image('Lowpass Butterworth Filter', butterworth_filtered)
    display_image('Highpass Laplacian Filter', laplacian_filtered)
    display_image('Histogram Matching', histogram_matched)
#%%
    plt.show()
    
if __name__ == '__main__':
    main()
