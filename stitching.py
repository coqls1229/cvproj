import math, random
import numpy as np  # SVD
import random

def get_patch(image, x, y, patch_size):
    half = patch_size // 2
    height, width = image.shape[:2]
    patch = np.zeros((patch_size, patch_size), dtype=np.float64)

    # Grayscale 변환 함수 사용
    def compute_intensity(image):
        image_float = image.astype(np.float64)
        return np.mean(image_float, axis=2)

    intensity_image = compute_intensity(image)

    for py in range(-half, half + 1):
        for px in range(-half, half + 1):
            if 0 <= x + px < width and 0 <= y + py < height:
                patch[py + half, px + half] = intensity_image[y + py, x + px]
    return patch

def compute_mean(patch):
    return np.mean(patch)

def compute_ncc(patch1, patch2):
    mean1 = compute_mean(patch1)
    mean2 = compute_mean(patch2)
    p1 = patch1 - mean1
    p2 = patch2 - mean2
    numerator = np.sum(p1*p2)
    denom1 = np.sum(p1*p1)
    denom2 = np.sum(p2*p2)
    if denom1 == 0 or denom2 == 0:
        return 0.0
    return numerator / math.sqrt(denom1*denom2)

def match_points_ncc(image1, corners1, image2, corners2, patch_size=5, ncc_threshold=0.8):
    matches = []
    for (x1,y1) in corners1:
        patch1 = get_patch(image1, x1, y1, patch_size)
        best_ncc = -1.0
        best_match = None
        for (x2,y2) in corners2:
            patch2 = get_patch(image2, x2, y2, patch_size)
            ncc_val = compute_ncc(patch1, patch2)
            if ncc_val > best_ncc:
                best_ncc = ncc_val
                best_match = (x2, y2)
        
        if best_ncc >= ncc_threshold and best_match is not None:
            matches.append(((x1,y1), best_match))
    return matches

def compute_homography(pairs):
    A = []
    for ((x1,y1),(x2,y2)) in pairs:
        A.append([-x1, -y1, -1, 0,   0,   0, x1*x2, y1*x2, x2])
        A.append([0,   0,   0, -x1, -y1, -1, x1*y2, y1*y2, y2])

    A = np.array(A, dtype=np.float64)
    U,S,Vt = np.linalg.svd(A)
    h = Vt[-1,:]
    H = h.reshape((3,3))
    if H[2,2] != 0:
        H = H / H[2,2]
    return H

def transform_point(x, y, H):
    denom = H[2,0]*x + H[2,1]*y + H[2,2]
    if denom == 0:
        return x,y
    x_prime = (H[0,0]*x + H[0,1]*y + H[0,2]) / denom
    y_prime = (H[1,0]*x + H[1,1]*y + H[1,2]) / denom
    return x_prime, y_prime

def distance(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx*dx + dy*dy)

def ransac(matches, iterations=1000, inlier_threshold=5.0):
    if len(matches) < 4:
        return matches, None

    best_inliers = []
    best_H = None
    
    for _ in range(iterations):
        sample = random.sample(matches, 4)
        H = compute_homography(sample)
        inliers = []
        for ((x1,y1),(x2,y2)) in matches:
            x2_est, y2_est = transform_point(x1,y1,H)
            if distance(x2,y2,x2_est,y2_est) < inlier_threshold:
                inliers.append(((x1,y1),(x2,y2)))
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
    
    return best_inliers, best_H