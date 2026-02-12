import numpy as np
import cv2 as cv
import glob
import os
import shutil

pattern_size = (9,6)

top_left = (0,0)
top_right = (pattern_size[0] - 1, 0)
bottom_left = (0, pattern_size[1] - 1)
bottom_right = (pattern_size[0] - 1, pattern_size[1] - 1)

def select_corners(event, x, y, flags, param):
    """
    UI interface to select the corners in the bad images.
    It generates a point where the mouse clicks and it shows with text which corner has to be selected.
    Uses the parameters needed by the cv.setMouseCallback funciton.
    """
    corners = ['top-left', 'top-right', 'bottom-right', 'bottom-left']
    font = cv.FONT_HERSHEY_SIMPLEX

    if event == cv.EVENT_LBUTTONDOWN:
        if len(param['points']) < 4:
            param['points'].append((x,y))
            cv.circle(param['base'], (x,y), 5, (255,0,0), -1)
    
    elif event == cv.EVENT_RBUTTONDOWN:
        param['points'].clear()
        param['base'] = param['original'].copy()

    param['display'] = param['base'].copy()

    n = len(param['points'])
    if n < 4:
        msg = f"Click {corners[n]} corner"    
    else:
        msg = f"Done, press Enter to continue"
    
    cv.putText(param['display'], msg, (100,260), font, 1, (0, 0, 255), 2)
    cv.imshow('Image', param['display'])

def generate_board_points(cols, rows):
    """
    Function to define the board space.
    param: cols: number of columns in the grid
    param: rows: number of rows in the grid
    return: board_points: list containing the board poitns coordinates the given grid
    """
    board_tl = (0,0)
    board_tr = (cols - 1 , 0)
    board_br = (cols - 1, rows - 1)
    board_bl = (0, rows - 1)
    board_points = (board_tl, board_tr, board_br, board_bl)
    return np.array(board_points, dtype = np.float32)
   

def interpolate_corners(points, cols, rows):
    """
    Function that performs interpolation of image points.
    param: points: list of four points of coordinates that we want to interpolate
    param: cols: number of columns in the grid
    param: rows: number of rows in the grid
    return: img_grid: new grid containing all cols*rows set of points
    """
    if len(points) != 4:
        raise Exception("Four points should be selected.")
    img_points = np.array(points, dtype = np.float32)
    board_points = generate_board_points(cols, rows)
    H = cv.getPerspectiveTransform(board_points, img_points)
    all_points = []
    for y in range(0, rows):
        for x in range(0, cols):
            all_points.append((x,y))
    all_points = np.array(all_points, dtype = np.float32)
    img_grid = cv.perspectiveTransform(all_points.reshape(cols*rows, 1, 2), H)
    return img_grid


#folders
# bad_dir = "bad_images"
# good_dir = "good_images"

# # termination criteria
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*9,3), np.float32)
# objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.

# images = glob.glob('./images/*.jpg')

# #clean up output directories
# shutil.rmtree(good_dir, ignore_errors=True)
# os.makedirs(good_dir)
# shutil.rmtree(bad_dir, ignore_errors=True)
# os.makedirs(bad_dir)

# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, (9,6), None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)

#         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners2)

#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (9,6), corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(500)

#         print("Chessboard corners found in image: ", fname)
#         out_path = os.path.join(good_dir, os.path.basename(fname))
#         cv.imwrite(out_path, img)

#     else :
#         print("Chessboard corners not found in image: ", fname)
#         out_path = os.path.join(bad_dir, os.path.basename(fname))
#         cv.imwrite(out_path, img)

#     cv.destroyAllWindows()

manual_images = glob.glob('./bad_images/*.jpg')


for fname in manual_images:
    shutil.rmtree("new_images", ignore_errors=True)
    os.makedirs("new_images")
    img = cv.imread(fname)

    state = {
        "original": img.copy(),
        "base": img.copy(),
        "display": img.copy(),
        "interpolation": img.copy(),
        "points": [],
        "filename": fname
    }

    cv.imshow('Image', state['display'])
    cv.setMouseCallback("Image", select_corners, state)
    
    key = cv.waitKey(0)

    if key == 27:
        print("Exit")
        cv.destroyAllWindows()
        print(state['points'])
        break

    if key == ord('s'):
        print("Skipped image: ", fname)
        continue

    img_grid = interpolate_corners(state['points'], 9, 6)
    
    for pt in img_grid:
        x,y = pt[0]
        cv.circle(state['interpolation'], (int(x),int(y)), 5, (255,0,0), -1)
    cv.imshow('Image', state["interpolation"])
    new_path = os.path.join("new_images", os.path.basename(fname))
    cv.imwrite(new_path, state["interpolation"])

    cv.waitKey(0)
    cv.destroyAllWindows()
    print(state['points'])

    

