from preprocessing import *
from corner_detection import *
from stitching import *
from blending import *
from utils import *


def downsample_image(image, scale_factor=None, target_size=None, interpolation=cv2.INTER_AREA):
    if scale_factor is not None and target_size is None:
        # scale_factor에 따라 크기 계산
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
    elif target_size is not None:
        # target_size 사용
        new_width, new_height = target_size
    else:
        raise ValueError("Either scale_factor or target_size must be provided.")

    # OpenCV의 resize 함수로 리샘플링
    downsampled_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    return downsampled_image


if __name__ == "__main__":
    try:
        # 결과 저장 경로 설정
        output_dir = "output"
        step2_dir = os.path.join(output_dir, "step2_corners")
        step3_dir = os.path.join(output_dir, "step3_matches")
        step4_dir = os.path.join(output_dir, "step4_inliers")
        step5_dir = os.path.join(output_dir, "step5_warp")
        step6_dir = os.path.join(output_dir, "step6_panorama")

        # 1. 이미지 로드
        images_paths = ["img1.JPG", "img2.JPG"]  # 두 장의 이미지만 사용
        print("[Info] Loading images...")
        images = []
        for p in images_paths:
            img = read_jpg_image(p)
            if img is None:
                print(f"[Error] Failed to load {p}. Exiting.")
                exit(1)
            downsampled_img = downsample_image(img, scale_factor=0.3)  # 해상도 30%로 줄임
            images.append(downsampled_img)
        print("[Info] All images loaded successfully.")

        # 2. Harris Corner Detection
        corners_list = []
        print("[Info] Starting Harris Corner Detection...")
        for i, img in enumerate(images):
            corners = harris_corner_detection(img, k=0.04, threshold=2000000)
            corners = sorted(corners, key=lambda c: c[2], reverse=True)[:100]  # 상위 100개만 사용
            corners = [(x, y) for x, y, _ in corners]  # (x, y)만 남김
            corners_list.append(corners)
            corner_vis = draw_corners_on_image(img, corners)
            outfile = os.path.join(step2_dir, f"corners_{i+1}.jpg")
            cv2.imwrite(outfile, corner_vis)
            print(f"[Info] Detected corners in image {i+1}, saved to {outfile}")

        # 3. NCC 기반 매칭
        print("[Info] Starting NCC-based matching...")
        matches = match_points_ncc(
            images[0], corners_list[0],
            images[1], corners_list[1],
            patch_size=5, ncc_threshold=0.8
        )
        match_vis = draw_matches(images[0], images[1], matches)
        outfile = os.path.join(step3_dir, "matches_1_2.jpg")
        cv2.imwrite(outfile, match_vis)
        print(f"[Info] NCC matches between image 1 and 2 saved to {outfile}")

        # 4. RANSAC으로 inlier 추출 & Homography 계산
        print("[Info] Starting RANSAC...")
        inliers, H = ransac(matches, iterations=1000, inlier_threshold=5.0)
        if H is None:
            H = np.eye(3)
            print(f"[Warning] Homography for images not found, using identity.")
        inlier_vis = draw_inliers(images[0], images[1], inliers)
        outfile = os.path.join(step4_dir, "inliers_1_2.jpg")
        cv2.imwrite(outfile, inlier_vis)
        print(f"[Info] Inliers after RANSAC for images saved to {outfile}")

        # 5. Warping
        print("[Info] Warping second image for visualization...")
        min_x, min_y, max_x, max_y = compute_panorama_bounds(images, [np.eye(3), H])
        pano_width = max_x - min_x + 1
        pano_height = max_y - min_y + 1
        warp_test = warp_image(images[1], H, min_x, min_y, pano_width, pano_height)
        outfile = os.path.join(step5_dir, "warp_image2.jpg")
        save_image(outfile, warp_test)
        print("[Info] Warped second image saved.")

        # 6. 스티칭
        print("[Info] Starting stitching process...")
        panorama = stitch_images(images, [np.eye(3), H])
        outfile = os.path.join(step6_dir, "panorama.jpg")
        save_image(outfile, panorama)
        print(f"[Info] Panorama saved as {outfile}.")

        print("[Info] All steps completed successfully.")



    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")
        # traceback을 보고 싶다면 다음 주석 해제
        # import traceback
        # traceback.print_exc()