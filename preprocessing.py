import math
import cv2  
import numpy as np

def read_jpg_image(filepath):
    img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Error: Cannot read image from {filepath}. Check if the file exists.")
        return None
    # BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb  # numpy array (H,W,3)

def generate_gaussian_kernel(kernel_size, sigma):
    half = kernel_size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    # 가우시안 공식 적용
    kernel = (1.0/(2.0*math.pi*(sigma**2))) * np.exp(-(x**2+y**2)/(2*(sigma**2)))
    kernel = kernel / kernel.sum()  # 정규화
    return kernel  # numpy array (kernel_size,kernel_size)

def apply_gaussian_filter(image, kernel):
    """
    image: numpy array (H,W,3), RGB
    kernel: numpy array (k,k)
    """
    height, width, _ = image.shape
    k = kernel.shape[0]
    half = k // 2

    # pad image for borders
    # 여기서는 단순 zero padding
    padded = np.pad(image, ((half,half),(half,half),(0,0)), mode='constant', constant_values=0)

    filtered = np.zeros_like(image, dtype=np.float64)  # float 계산 후 clip

    for y in range(height):
        for x in range(width):
            # Region of interest
            roi = padded[y:y+k, x:x+k, :]  # (k,k,3)
            # kernel broadcast: (k,k,1) vs (k,k,3)
            # element-wise multiply and sum
            # roi와 kernel을 각 채널별로 곱한 뒤 합산
            for c in range(3):
                filtered[y,x,c] = np.sum(roi[:,:,c]*kernel)

    # clip and convert to uint8
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    return filtered

def preprocess_images(images_paths, kernel_size=5, sigma=1.0):
    kernel = generate_gaussian_kernel(kernel_size, sigma)
    
    filtered_images = []
    for path in images_paths:
        img = read_jpg_image(path)  # numpy array
        if img is None:
            continue
        filtered_img = apply_gaussian_filter(img, kernel) # numpy array
        filtered_images.append(filtered_img)
    return filtered_images