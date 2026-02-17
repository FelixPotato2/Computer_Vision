import numpy as np
import cv2 as cv
import glob
import os
import shutil

EDGE_SIZE = 25 # size of the square edge (in mm)
L = 75  # length of axes (TODO: for each square (?)) in mm
axis = np.float32([
    [0, 0, 0],   # origin
    [L, 0, 0],   # X
    [0, L, 0],   # Y
    [0, 0, -L]   # Z 
])

cube = np.float32([
    [0,0,0],       # bottom-front-left
    [L,0,0],       # bottom-front-right
    [L,L,0],       # bottom-back-right
    [0,L,0],       # bottom-back-left
    [0,0,-L],      # top-front-left
    [L,0,-L],      # top-front-right
    [L,L,-L],      # top-back-right
    [0,L,-L]       # top-back-left
])
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
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
        if len(param['2dpoints']) < 4:
            param['2dpoints'].append((x,y))
            cv.circle(param['base'], (x,y), 5, (255,0,0), -1)
    
    elif event == cv.EVENT_RBUTTONDOWN:
        param['2dpoints'].clear()
        param['base'] = param['original'].copy()

    param['display'] = param['base'].copy()

    n = len(param['2dpoints'])
    if n < 4:
        msg = f"Click {corners[n]} corner"    
    else:
        msg = f"Done! Press Enter to continue"
    
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

def create_object_points(cols, rows, square_size_mm = EDGE_SIZE):
    """
    Create the 3D world coordinates of the chessboard corners.
    """
    objp = []

    for y in range(rows):
        for x in range(cols):
            X = x * square_size_mm
            Y = y * square_size_mm
            Z = 0.0
            objp.append((X, Y, Z))

    return np.array(objp, dtype=np.float32)

def show_tvecs_rvecs(rvecs, tvecs):
    """
    Function to show rotation vectors and translation vectors
    """
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv.Rodrigues(rvec)
        extrinsic = np.hstack((R, tvec.reshape(3,1)))

        print(f"\nImage {i+1} extrinsic [R|t]:")
        print(extrinsic)
        
def color_face(img, face, color, dist_m, alpha = 0.5):
    """
    Function used to color a face of a convex polygon in a object
    param: img: image taken from img path
    param: face: np.array containing coordinates of face points
    param: color: color to use to color the face
    param: dist_m: distance from camera to top center (in meters)
    param: alpha: integer
    returns: img: copy of the used image with a convex polygon (cube) with the specified face colored
    """
    img = img.copy()
    overlay = img.copy()
    center = tuple(np.mean(face, axis = 0).astype(int))

    cv.fillConvexPoly(overlay, face, color)
    cv.circle(overlay, center, 3, (0,0,0), -1)

    # ---- Add distance text ----
    text = f"{dist_m:.2f} m"
    cv.putText(
        overlay,
        text,
        (center[0] + 10, center[1] +30),   # slight offset from center
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,0,255),
        2,
        cv.LINE_AA
    )

    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

def dynamic_color(img, face, dmin, dmax, rvec, tvec):
    """
    Function to dynamically assing color to the considered face of a convex polygon based 
    on the distance from the camera.
    param: img: 
    param: face:
    param dist:
    param dmin:
    param dmax: 
    param angle:
    returns: 
    """

    # Calculate the center and transform it to camera coordinates
    face_top_center = np.array([[L/2, L/2, -L]], dtype = np.float32)
    R, _ = cv.Rodrigues(rvec)

    print("Rvec: ", rvec)
    normal = R[:, 2]
    angle = np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))
    print("Angle: ", angle)

    face_top_center_cam = R @ face_top_center.T + tvec

    #Calculate the idistance form the camera to the top center point
    dist = float(np.linalg.norm(face_top_center_cam))
    dist_m = dist / 1000

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    t = (dist_m - dmin) / (dmax - dmin)
    t = float(np.clip(t, 0.0, 1.0))
    h = int(179 * (1 - np.clip(angle / 45.0, 0.0, 1.0)))
    s = 255
    v = int((1 - t) * 255)

    print(f"HSV: ({h}, {s}, {v})")
    
    #Convert color to BGR 
    hsv_color = np.uint8([[[h, s, v]]])
    bgr_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0, 0]
    bgr_color = tuple(int(c) for c in bgr_color)

    #Assign the color automatically
    return color_face(img, face, bgr_color, dist_m)

def draw_cube(img, cameraMatrix, distCoeffs, pattern_size, criteria,rvec=None, tvec=None, online=False):
    
    """
    Calibrates the camera for the given points and draws the 3D cube a the world origin
    
    param: objectPoints_run: 3d points from the calibration run
    param: imagePoints_run: 2d points from the calibration run
    param: img: the test image
    param: image_size: tuple containing size of the image we are considering
    param: criteria: ????
    param: flags: ????
    return: cameraMatrix: 3x3 calibration matrix
    return: distCoeffs: np array representing coefficient correcting lens distortion
    return: rvecs: 3x1 rotation vector representing the position of the object relative to the camera
    return tvecs: 3x1 position vectors representing the position of the object in the camera coordinate systems

    """

    if rvec is None or tvec is None:
        # fallback for still images
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, pattern_size)
        if not ret:
            return img
        objp = create_object_points(9,6,EDGE_SIZE)
        _, rvec, tvec = cv.solvePnP(objp, corners, cameraMatrix, distCoeffs)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Find the chess board corners for the test image
    ret, corners = cv.findChessboardCorners(gray, (pattern_size[0],pattern_size[1]), None)
    board_object_points = create_object_points(9, 6, EDGE_SIZE)

    #Estimate for test image tvec and rvec
    _, rvec, tvec = cv.solvePnP(board_object_points, corners, cameraMatrix1, distCoeffs1)

    # Project 3D axes to 2D image
    imgpts, _ = cv.projectPoints(cube, rvec, tvec, cameraMatrix1, distCoeffs1)
    pts = imgpts.reshape(-1,2).astype(int)
    
    origin_pt = tuple(pts[0]) #TODO: ???
    dist = float(np.linalg.norm(tvec))
    
    #Color top face
    top_face = np.array([pts[4], pts[5], pts[6], pts[7]], dtype=np.int32)
    img = dynamic_color(img, top_face, dmin=0.0, dmax=4.0, rvec=rvec, tvec=tvec)

    # Draw cube
    cv.line(img, tuple(pts[0]), tuple(pts[1]), (0,0,255), 2)
    cv.line(img, tuple(pts[1]), tuple(pts[2]), (0,0,255), 2)
    cv.line(img, tuple(pts[2]), tuple(pts[3]), (0,0,255), 2)
    cv.line(img, tuple(pts[3]), tuple(pts[0]), (0,0,255), 2)
    cv.line(img, tuple(pts[4]), tuple(pts[5]), (0,255,0), 2)
    cv.line(img, tuple(pts[5]), tuple(pts[6]), (0,255,0), 2)
    cv.line(img, tuple(pts[6]), tuple(pts[7]), (0,255,0), 2)
    cv.line(img, tuple(pts[7]), tuple(pts[4]), (0,255,0), 2)
    for i in range(4):
        cv.line(img, tuple(pts[i]), tuple(pts[i+4]), (255,0,0), 2)

    if online!= True:
        # Display image
        cv.imshow("Cube overlay", img)
        cv.waitKey(1)
        cv.destroyAllWindows()

    else:
        return img



# directories for the training images
images = glob.glob('./Images/*.jpg')
manual_images = glob.glob('./bad_images/*.jpg')
test_images = glob.glob('./test_images/*.jpg')
shutil.rmtree("new_images", ignore_errors=True)
os.makedirs("new_images")

# Arrays to store (3d) object points and (2d) image points from all the images
board_object_points = create_object_points(9, 6, EDGE_SIZE)
auto_objectPoints = []
auto_imagePoints = []
manual_objectPoints = []
manual_imagePoints = []

# create np arrays for object points (why do we need new object points?)
objp = np.zeros((pattern_size[1]*pattern_size[0],3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
objp[:,:2] *= EDGE_SIZE 

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (pattern_size[0],pattern_size[1]), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        auto_objectPoints.append(objp)
        #what is this for? do we really need it?
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # Reorder the corners to ensure top-left is origin 
        corners2 = corners2.reshape(pattern_size[1], pattern_size[0], 2)
        corners2 = np.flipud(np.fliplr(corners2))
        corners2 = corners2.reshape(-1,2)

        auto_imagePoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (pattern_size[0],pattern_size[1]), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

for fname in manual_images:
    
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    state = {
        "original": img.copy(),
        "base": img.copy(),
        "display": img.copy(),
        "interpolation": img.copy(),
        "2dpoints": [],
        "3dpoints": [],
        "filename": fname
    }

    cv.imshow('Image', state['display'])
    cv.setMouseCallback("Image", select_corners, state)
    
    key = cv.waitKey(0)

    if key == 27:
        print("Exit")
        cv.destroyAllWindows()
        print(state['2dpoints'])
        break

    if key == ord('s'):
        print("Skipped image: ", fname)
        continue

    src = np.array(state['2dpoints'], dtype=np.float32)
    W, H = 900, 600
    dst = np.float32([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]]) # the corners to be warped

    # warping for better corner estimation
    H_warp = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, H_warp, (W, H))

    # interpolation
    warped_grid = interpolate_corners(dst, pattern_size[0], pattern_size[1])
    warped_grid = warped_grid.reshape(-1, 1, 2).astype(np.float32)
    cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

    # inverse warping to go back to original image
    H_inv = cv.getPerspectiveTransform(dst, src)
    orig_pts = cv.perspectiveTransform(warped_grid, H_inv)

    img_points = orig_pts.reshape(-1, 2)
    manual_imagePoints.append(img_points)
    manual_objectPoints.append(board_object_points.copy())

    for pt in orig_pts:
        x, y = pt[0]
        cv.circle(state['interpolation'], (int(x), int(y)), 5, (255, 0, 0), -1)

    cv.imshow('Image', state["interpolation"])
    new_path = os.path.join("new_images", os.path.basename(fname))
    cv.imwrite(new_path, state["interpolation"])

    cv.waitKey(0)
    cv.destroyAllWindows()


"""
    Part 4: Three runs of calibration.

    We need to use cv.calibrateCamera(
    objectPoints,    # list of 3D points per image
    imagePoints,     # list of 2D points per image
    image_size,      # image width and height
    None,            # initial camera matrix (None → estimated)
    None,            # initial distortion coefficients
    flags=0,         # free center/origin point
    criteria=criteria
)

"""

# read image size 
image_size = (cv.imread(images[0]).shape[1], cv.imread(images[0]).shape[0]) 

#Run images
objectPoints_run1 = auto_objectPoints + manual_objectPoints
imagePoints_run1  = auto_imagePoints  + manual_imagePoints

ret, cameraMatrix1, distCoeffs1, rvecs1, tvecs1 = cv.calibrateCamera(objectPoints_run1, imagePoints_run1, image_size, None, None, flags=0, criteria=criteria)
print("############ RESULTS OF RUN 1 ############\n")
print("Intrinsic Parameters : Camera matrix K:\n", cameraMatrix1)
print("Extrinsic Parameters : [R|t] for each image:")
show_tvecs_rvecs(rvecs1, tvecs1)
print("##########################################\n")

# ---------------------------- Run 2 --------------------- 5 manual + 5 auto
objectPoints_run2 = auto_objectPoints[:5] + manual_objectPoints[:5]
imagePoints_run2  = auto_imagePoints[:5]  + manual_imagePoints[:5]
    
ret, cameraMatrix2, distCoeffs2, rvecs2, tvecs2 = cv.calibrateCamera(objectPoints_run2, imagePoints_run2, image_size, None, None, flags=0, criteria=criteria)
print("############ RESULTS OF RUN 2 ############\n")
print("Intrinsic Parameters : Camera matrix K:\n", cameraMatrix2)
print("Extrinsic Parameters : [R|t] for each image:")
show_tvecs_rvecs(rvecs2, tvecs2)
print("##########################################\n")

# ---------------------------- Run 3 ---------------------- 5 auto
objectPoints_run3 = auto_objectPoints[:5]
imagePoints_run3  = auto_imagePoints[:5]

ret, cameraMatrix3, distCoeffs3, rvecs3, tvecs3 = cv.calibrateCamera(objectPoints_run3, imagePoints_run3, image_size, None, None, flags=0, criteria=criteria)
print("############ RESULTS OF RUN 3 ############\n")
print("Instrinic Parameters : Camera matrix K:\n", cameraMatrix3)
print("Extrinsic parameters : [R|t] for each image:")
show_tvecs_rvecs(rvecs3, tvecs3)
print("##########################################\n")


"""
    Part 5: Project from 3d object to 2d pixel coordinates

    We must use -> cv.projectPoints(
    object_points_3D,
    rvec,
    tvec,
    K,
    distCoeffs
    )

"""


# Single test image
img_path = test_images[1]
#print(img_path)
img = cv.imread(img_path)


# Run 1
draw_cube(img.copy(), cameraMatrix1, distCoeffs1, pattern_size=pattern_size, criteria=criteria, online=False)

# Run 2
draw_cube(img.copy(), cameraMatrix2, distCoeffs2, pattern_size=pattern_size, criteria=criteria, online=False)

# Run 3
draw_cube(img.copy(), cameraMatrix3, distCoeffs3, pattern_size=pattern_size, criteria=criteria, online=False)


"""
    Choice task 1: real-time pose estimation
    
"""

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, pattern_size)

    if found and corners is not None and len(corners) == len(objp):
        corners = cv.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        frame = draw_cube(
            img=frame,
            cameraMatrix=cameraMatrix1,
            distCoeffs=distCoeffs1,
            rvec=None,      
            tvec=None,
            pattern_size=pattern_size,
            criteria=criteria,
            online=True
        )

    cv.imshow('Realtime pose', frame)

    if cv.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv.destroyAllWindows()
