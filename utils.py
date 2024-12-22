import cv2
import numpy as np
import math
import random
import os

def read_jpg_image(filepath):
    img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[Error] Cannot read image from {filepath}. Check if the file exists.")
        return None
    # img_bgr.shape: (height, width, 3), BGR 순서
    # RGB로 변환
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # img_rgb는 이미 numpy array
    return img_rgb

def save_image(filepath, image):
    if image is None:
        print(f"[Error] Cannot save {filepath}, image is None.")
        return
    if image.size == 0:
        print(f"[Error] Cannot save {filepath}, empty image.")
        return
    # image: numpy array (H,W,3) in RGB
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img_bgr)
    print(f"[Info] Saved image: {filepath}")

def draw_corners_on_image(image, corners):
    # image: numpy array (H,W,3) in RGB
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    # corners: [(x,y), ...]
    for (x,y) in corners:
        cv2.circle(img_bgr, (x,y), 3, (0,0,255), -1)
    return img_bgr  # BGR 반환. 필요시 저장 전에 별도 처리 가능(여기선 cv2.imwrite 사용하니 BGR OK)

def draw_matches(image1, image2, matches):
    # image1, image2: numpy array (H,W,3) in RGB
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    H = max(h1,h2)
    W = w1 + w2

    img_bgr = np.zeros((H,W,3), dtype=np.uint8)
    img_bgr[:h1, :w1] = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    img_bgr[:h2, w1:w1+w2] = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

    for ((x1,y1),(x2,y2)) in matches:
        pt1 = (x1,y1)
        pt2 = (x2+w1,y2)
        cv2.line(img_bgr, pt1, pt2, (0,255,0), 1)
        cv2.circle(img_bgr, pt1, 3, (0,0,255), -1)
        cv2.circle(img_bgr, pt2, 3, (0,0,255), -1)
    return img_bgr

def draw_inliers(image1, image2, inliers):
    return draw_matches(image1, image2, inliers)