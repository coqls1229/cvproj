import math
import numpy as np

def apply_homography(x, y, H):
    denom = H[2,0]*x + H[2,1]*y + H[2,2]
    if denom == 0:
        return x,y
    X = (H[0,0]*x + H[0,1]*y + H[0,2]) / denom
    Y = (H[1,0]*x + H[1,1]*y + H[1,2]) / denom
    return X, Y

def bilinear_interpolation(image, x, y):
    # image: numpy (H,W,3)
    H, W, _ = image.shape
    x0 = int(math.floor(x))
    x1 = x0 + 1
    y0 = int(math.floor(y))
    y1 = y0 + 1

    if x0 < 0 or x1 >= W or y0 < 0 or y1 >= H:
        return None

    dx = x - x0
    dy = y - y0

    p00 = image[y0, x0]
    p01 = image[y0, x1]
    p10 = image[y1, x0]
    p11 = image[y1, x1]

    top = (1-dx)*p00 + dx*p01
    bottom = (1-dx)*p10 + dx*p11
    val = (1-dy)*top + dy*bottom
    val = np.clip(val, 0, 255).astype(np.uint8)
    return val.tolist()

def warp_image(image, H, min_x, min_y, pano_width, pano_height):
    """
    NumPy 배열 기반으로 최적화된 이미지 워핑 함수
    """
    import math

    # 역호모그래피 계산
    H_inv = [[H[1][1] * H[2][2] - H[1][2] * H[2][1], H[0][2] * H[2][1] - H[0][1] * H[2][2], H[0][1] * H[1][2] - H[0][2] * H[1][1]],
             [H[1][2] * H[2][0] - H[1][0] * H[2][2], H[0][0] * H[2][2] - H[0][2] * H[2][0], H[0][2] * H[1][0] - H[0][0] * H[1][2]],
             [H[1][0] * H[2][1] - H[1][1] * H[2][0], H[0][1] * H[2][0] - H[0][0] * H[2][1], H[0][0] * H[1][1] - H[0][1] * H[1][0]]]

    # 정규화 (H_inv[2][2] == 1)
    det = H[0][0] * H[1][1] * H[2][2] + H[0][1] * H[1][2] * H[2][0] + H[0][2] * H[1][0] * H[2][1] - \
          H[0][2] * H[1][1] * H[2][0] - H[0][1] * H[1][0] * H[2][2] - H[0][0] * H[1][2] * H[2][1]
    for i in range(3):
        for j in range(3):
            H_inv[i][j] /= det

    # 결과 이미지를 NumPy 배열로 초기화
    result = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
    height, width, _ = image.shape

    # 워핑 수행
    for py in range(pano_height):
        for px in range(pano_width):
            global_x = px + min_x
            global_y = py + min_y

            # 역호모그래피로 원본 좌표 계산
            wx = H_inv[0][0] * global_x + H_inv[0][1] * global_y + H_inv[0][2]
            wy = H_inv[1][0] * global_x + H_inv[1][1] * global_y + H_inv[1][2]
            w = H_inv[2][0] * global_x + H_inv[2][1] * global_y + H_inv[2][2]
            if w != 0:
                wx /= w
                wy /= w

            # bilinear interpolation
            x0, y0 = int(math.floor(wx)), int(math.floor(wy))
            x1, y1 = x0 + 1, y0 + 1
            if 0 <= x0 < width and 0 <= y0 < height:
                dx, dy = wx - x0, wy - y0

                for c in range(3):  # R, G, B
                    f00 = image[y0, x0, c] if 0 <= x0 < width and 0 <= y0 < height else 0
                    f10 = image[y0, x1, c] if 0 <= x1 < width and 0 <= y0 < height else 0
                    f01 = image[y1, x0, c] if 0 <= x0 < width and 0 <= y1 < height else 0
                    f11 = image[y1, x1, c] if 0 <= x1 < width and 0 <= y1 < height else 0

                    result[py, px, c] = int(
                        (1 - dx) * (1 - dy) * f00 +
                        dx * (1 - dy) * f10 +
                        (1 - dx) * dy * f01 +
                        dx * dy * f11
                    )

    return result

def compute_panorama_bounds(images, homographies):
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf

    for i, img in enumerate(images):
        H, W, _ = img.shape
        corners = [(0,0),(W-1,0),(0,H-1),(W-1,H-1)]
        H_mat = homographies[i]
        for (cx,cy) in corners:
            X,Y = apply_homography(cx,cy,H_mat)
            if X < min_x: min_x = X
            if X > max_x: max_x = X
            if Y < min_y: min_y = Y
            if Y > max_y: max_y = Y

    return int(math.floor(min_x)), int(math.floor(min_y)), int(math.ceil(max_x)), int(math.ceil(max_y))

def feather_blend(existing_pixel, new_pixel):
    if existing_pixel is None:
        return new_pixel
    if new_pixel is None:
        return existing_pixel
    # both not None
    # existing_pixel, new_pixel: [R,G,B]
    R = (existing_pixel[0] + new_pixel[0])//2
    G = (existing_pixel[1] + new_pixel[1])//2
    B = (existing_pixel[2] + new_pixel[2])//2
    return [R,G,B]

def stitch_images(images, homographies):
    # images: list of numpy (H,W,3)
    min_x, min_y, max_x, max_y = compute_panorama_bounds(images, homographies)
    pano_width = max_x - min_x + 1
    pano_height = max_y - min_y + 1

    panorama = np.empty((pano_height, pano_width), dtype=object)
    panorama.fill(None)

    for i, img in enumerate(images):
        H_mat = homographies[i]
        H_inv = np.linalg.inv(H_mat)
        H_img, W_img, _ = img.shape

        for PY in range(pano_height):
            for PX in range(pano_width):
                globalX = PX + min_x
                globalY = PY + min_y
                srcX, srcY = apply_homography(globalX, globalY, H_inv)
                color = bilinear_interpolation(img, srcX, srcY)
                if color is not None:
                    pano_color = panorama[PY,PX]
                    new_color = feather_blend(pano_color, color)
                    panorama[PY,PX] = new_color

    # panorama 리스트를 numpy로 변환
    # None 부분을 0으로 처리
    H_pan, W_pan = panorama.shape
    pano_array = np.zeros((H_pan, W_pan, 3), dtype=np.uint8)
    for y in range(H_pan):
        for x in range(W_pan):
            if panorama[y,x] is None:
                pano_array[y,x] = [0,0,0]
            else:
                pano_array[y,x] = panorama[y,x]

    return pano_array  # 최종 파노라마 numpy array(RGB)