import numpy as np
import cv2 as cv
import glob
import os

# === Setup ===
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

objpoints = []  # 3D points
imgpoints = []  # 2D points

# Paths
image_dir = '/home/voice/Desktop/additional_chess'
output_base = '/home/voice/Desktop/outcalib'
images = glob.glob(os.path.join(image_dir, '*.jpg'))

# Create output base dir
os.makedirs(output_base, exist_ok=True)

image_counter = 1

# === Step 1: Process each image ===
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7, 7), None)
    corner_img = img.copy()

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(corner_img, (7, 7), corners2, ret)
        print(f"[✓] Corners found: {fname}")
    else:
        print(f"[✗] Corners not found: {fname}")

    # === Save original and corner image ===
    subdir = os.path.join(output_base, f"input_img{image_counter}")
    os.makedirs(subdir, exist_ok=True)

    cv.imwrite(os.path.join(subdir, 'input.jpg'), img)
    cv.imwrite(os.path.join(subdir, 'corners.jpg'), corner_img)

    image_counter += 1

cv.destroyAllWindows()

# === Step 2: Camera Calibration ===
if objpoints and imgpoints:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save calibration data
    np.savez(os.path.join(output_base, "calibration_data.npz"), camera_matrix=mtx, distortion_coefficients=dist)
    print(f"[💾] Calibration data saved to: {output_base}/calibration_data.npz")

    # === Step 3: Undistort and save ===
    image_counter = 1
    for fname in images:
        img = cv.imread(fname)
        h, w = img.shape[:2]

        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Method 1
        dst1 = cv.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w_roi, h_roi = roi
        dst1 = dst1[y:y+h_roi, x:x+w_roi]

        # Method 2
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst2 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        dst2 = dst2[y:y+h_roi, x:x+w_roi]

        subdir = os.path.join(output_base, f"input_img{image_counter}")
        cv.imwrite(os.path.join(subdir, 'undistort1.jpg'), dst1)
        cv.imwrite(os.path.join(subdir, 'undistort2.jpg'), dst2)

        print(f"[💾] Undistorted images saved to: {subdir}")
        image_counter += 1

    # === Step 4: Reprojection Error ===
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("Total reprojection error: {:.5f}".format(mean_error / len(objpoints)))
else:
    print("[!] Calibration skipped: Not enough valid detections.")
