import numpy as np
import cv2 as cv
import glob
import os
import shutil

def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, f"{x},{y}", (x, y), font, 1, (255, 0, 0), 2)
        cv.imshow('image', img)

    if event == cv.EVENT_RBUTTONDOWN:
        print(x, y)
        font = cv.FONT_HERSHEY_SIMPLEX
        b, g, r = img[y, x]
        cv.putText(img, f"{b},{g},{r}", (x, y), font, 1, (255, 255, 0), 2)
        cv.imshow('image', img)

#folders
bad_dir = "bad_images"
good_dir = "good_images"

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./images/*.jpg')

#clean up output directories
shutil.rmtree(good_dir, ignore_errors=True)
os.makedirs(good_dir)
shutil.rmtree(bad_dir, ignore_errors=True)
os.makedirs(bad_dir)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

        print("Chessboard corners found in image: ", fname)
        out_path = os.path.join(good_dir, os.path.basename(fname))
        cv.imwrite(out_path, img)

    else :
        print("Chessboard corners not found in image: ", fname)
        out_path = os.path.join(bad_dir, os.path.basename(fname))
        cv.imwrite(out_path, img)

    cv.destroyAllWindows()

manual_images = glob.glob('./bad_images/*.jpg')

for fname in manual_images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('image', img)
    cv.setMouseCallback('image', click_event)
    cv.waitKey(0)
    cv.destroyAllWindows()