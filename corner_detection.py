import math
import numpy as np

def generate_gaussian_kernel(kernel_size, sigma):
    half = kernel_size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    kernel = (1/(2*math.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))
    kernel /= kernel.sum()
    return kernel  # (kernel_size, kernel_size) numpy array

def apply_kernel(image, kernel, channel_data):
    # channel_data: numpy 2D array (H,W)
    # kernel: numpy 2D array (k,k)
    H, W = channel_data.shape
    k = kernel.shape[0]
    half = k // 2

    # zero padding
    padded = np.pad(channel_data, ((half,half),(half,half)), mode='constant', constant_values=0)

    filtered = np.zeros((H,W), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            roi = padded[y:y+k, x:x+k]
            filtered[y,x] = np.sum(roi * kernel)

    return filtered

def sobel_gradients(image):
    # image: numpy array (H,W,3)
    gx = np.array([[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]], dtype=np.float64)
    gy = np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float64)

    H, W, _ = image.shape

    # 패딩
    padded = np.pad(image, ((1,1),(1,1),(0,0)), mode='constant', constant_values=0)

    # 각 채널별 Gradient
    Ix_R = np.zeros((H,W), dtype=np.float64)
    Iy_R = np.zeros((H,W), dtype=np.float64)
    Ix_G = np.zeros((H,W), dtype=np.float64)
    Iy_G = np.zeros((H,W), dtype=np.float64)
    Ix_B = np.zeros((H,W), dtype=np.float64)
    Iy_B = np.zeros((H,W), dtype=np.float64)

    for y in range(H):
        for x in range(W):
            # 3x3 patch
            patch = padded[y:y+3, x:x+3, :]  # (3,3,3)
            # R,G,B 각각 gx,gy와 컨볼루션
            R_patch = patch[:,:,0]
            G_patch = patch[:,:,1]
            B_patch = patch[:,:,2]
            Ix_R[y,x] = np.sum(R_patch * gx)
            Iy_R[y,x] = np.sum(R_patch * gy)
            Ix_G[y,x] = np.sum(G_patch * gx)
            Iy_G[y,x] = np.sum(G_patch * gy)
            Ix_B[y,x] = np.sum(B_patch * gx)
            Iy_B[y,x] = np.sum(B_patch * gy)

    Ix = (Ix_R + Ix_G + Ix_B)/3.0
    Iy = (Iy_R + Iy_G + Iy_B)/3.0

    return Ix, Iy  # numpy arrays (H,W)

def harris_corner_detection(image, k=0.04, threshold=1000000, gaussian_kernel_size=5, gaussian_sigma=1.0):
    # image: numpy array (H,W,3)
    Ix, Iy = sobel_gradients(image)

    H, W, _ = image.shape

    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix * Iy

    gaussian_kernel = generate_gaussian_kernel(gaussian_kernel_size, gaussian_sigma)
    Ix2_sm = apply_kernel(image, gaussian_kernel, Ix2)
    Iy2_sm = apply_kernel(image, gaussian_kernel, Iy2)
    Ixy_sm = apply_kernel(image, gaussian_kernel, Ixy)

    # R_map
    R_map = (Ix2_sm * Iy2_sm - Ixy_sm**2) - k * ((Ix2_sm + Iy2_sm)**2)

    # threshold
    corners = []
    for y in range(H):
        for x in range(W):
            if R_map[y, x] > threshold:
                corners.append((x, y, R_map[y, x]))  # (x, y, R) 저장

    if len(corners) < 10:
        print("코너가 충분히 발견되지 않았습니다")

    return corners