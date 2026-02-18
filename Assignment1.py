import numpy as np
import cv2 as cv
import glob
import os
import shutil
import matplotlib.pyplot as plt

save_id = 1
EDGE_SIZE = 25 # size of the square edge (in mm)
L = 75  # length of axes in mm
AXES = np.float32([
    [0, 0, 0],   # origin
    [L*2, 0, 0],   # X
    [0, L*2, 0],   # Y
    [0, 0, -L*2]   # Z 
])

CUBE = np.float32([
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

def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

def order_points(src, pattern_size):
    """
    Function to order points in the correct clockwise order,
    based on the angle between each manually added corner points

    """
    src = np.asarray(src, np.float32)
    cx = np.mean(src[:,0])
    cy = np.mean(src[:,1])
    angles = np.arctan2(src[:, 1] - cy, src[:, 0] - cx)
    order = np.argsort(angles)
    pts = src[order]

    if signed_area(pts) > 0:
        pts = pts[::-1]

    best_i = None
    best_y = 1e10
    for i in range(4):
        p = pts[i]
        q = pts[(i+1) % 4]
        y_avg = 0.5 * (p[1] + q[1])
        if y_avg < best_y:
            best_y = y_avg
            best_i = i
    p_top0 = pts[best_i]
    p_top1 = pts[(best_i + 1) % 4]

    if p_top0[0] < p_top1[0]:
        top_left, top_right = p_top0, p_top1
        bottom_right = pts[(best_i + 2) % 4]
        bottom_left = pts[(best_i + 3) % 4]
    else:
        top_left, top_right = p_top1, p_top0
        bottom_right = pts[(best_i + 3) % 4]
        bottom_left = pts[(best_i + 2) % 4]
    
    ordered_points = np.array([top_left, top_right, bottom_right, bottom_left], np.float32)

    expected = (pattern_size[0] - 1) / (pattern_size[1] - 1)

    len_x = np.linalg.norm(ordered_points[1] - ordered_points[0])  # top right, top left
    len_y = np.linalg.norm(ordered_points[3] - ordered_points[0])  # bottom left, top left
    observed = len_x / (len_y + 1e-9)
    if observed < 1.0 and expected > 1.0:
        ordered_points = np.array([ordered_points[0], ordered_points[3], ordered_points[2], ordered_points[1]], np.float32)
    return ordered_points


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

    text = f"{dist_m:.2f} m"
    cv.putText(
        overlay,
        text,
        (center[0] + 10, center[1] +30),  
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2,
        cv.LINE_AA
    )

    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

def dynamic_color(img, face, dmin, dmax, rvec, tvec):
    """
    Function to dynamically assing color to the considered face of a convex polygon based 
    on the distance from the camera and the orientation.
    param: img: np.ndarray, input BGR image
    param: face: np.ndarray, Array of 2D image pixel coordinates defining the polygon face to color
    param dmin: float, representing the minimum distance (in meters) used for normalization/clipping for brightness
    param dmax: float, representing the maximum distance (in meters) used for normalization/clipping for brightness
    param rvec: np.ndarray, rotation vector
    param tvec: np.ndarray, translation vector
    returns: np.ndarray output BGR image
    """

    # Calculate the center and transform it to camera coordinates
    face_top_center = np.array([[L/2, L/2, -L]], dtype = np.float32)
    R, _ = cv.Rodrigues(rvec)

    normal = R[:, 2]
    angle = np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))

    face_top_center_cam = R @ face_top_center.T + tvec

    #Calculate the idistance form the camera to the top center point
    dist = float(np.linalg.norm(face_top_center_cam))
    dist_m = dist / 1000

    t = (dist_m - dmin) / (dmax - dmin)
    t = float(np.clip(t, 0.0, 1.0))
    h = int(179 * (1 - np.clip(angle / 45.0, 0.0, 1.0)))
    s = 255
    v = int((1 - t) * 255)
    
    #Convert color to BGR 
    hsv_color = np.uint8([[[h, s, v]]])
    bgr_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0, 0]
    bgr_color = tuple(int(c) for c in bgr_color)

    #Assign the color automatically
    return color_face(img, face, bgr_color, dist_m)

def draw_cube(img, cameraMatrix, distCoeffs, pattern_size ,rvec=None, tvec=None, online=False):
    
    """
    Calibrates the camera for the given points and draws the 3D cube a the world origin
    
    param: objectPoints_run: 3d points from the calibration run
    param: imagePoints_run: 2d points from the calibration run
    param: img: the test image
    param: pattern_size: tuple containing size of the image we are considering
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
        objp = create_object_points(pattern_size[0],pattern_size[1],EDGE_SIZE)
        _, rvec, tvec = cv.solvePnP(objp, corners, cameraMatrix, distCoeffs)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners for the test image
    ret, corners = cv.findChessboardCorners(gray, (pattern_size[0],pattern_size[1]), None)
    board_object_points = create_object_points(pattern_size[0], pattern_size[1], EDGE_SIZE)

    if ret:
        #Estimate for test image tvec and rvec
        _, rvec, tvec = cv.solvePnP(board_object_points, corners, cameraMatrix, distCoeffs)

        # Project 3D axes to 2D image
        imgpts, _ = cv.projectPoints(CUBE, rvec, tvec, cameraMatrix, distCoeffs)
        ax, _ = cv.projectPoints(AXES, rvec, tvec, cameraMatrix, distCoeffs)
        pts = imgpts.reshape(-1,2).astype(int)
        axpts = ax.reshape(-1,2).astype(int)

        #Color top face
        top_face = np.array([pts[4], pts[5], pts[6], pts[7]], dtype=np.int32)
        img = dynamic_color(img, top_face, dmin=0.0, dmax=4.0, rvec=rvec, tvec=tvec)

        # Draw axes
        cv.arrowedLine(img, tuple(axpts[0]), tuple(axpts[1]), (255, 0, 0), 2)
        cv.arrowedLine(img, tuple(axpts[0]), tuple(axpts[2]), (0, 255, 0), 2)
        cv.arrowedLine(img, tuple(axpts[0]), tuple(axpts[3]), (0, 0, 255), 2)

        # Draw cube
        cv.line(img, tuple(pts[0]), tuple(pts[1]), (0,0,0), 2)
        cv.line(img, tuple(pts[1]), tuple(pts[2]), (0,0,0), 2)
        cv.line(img, tuple(pts[2]), tuple(pts[3]), (0,0,0), 2)
        cv.line(img, tuple(pts[3]), tuple(pts[0]), (0,0,0), 2)
        cv.line(img, tuple(pts[4]), tuple(pts[5]), (0,0,0), 2)
        cv.line(img, tuple(pts[5]), tuple(pts[6]), (0,0,0), 2)
        cv.line(img, tuple(pts[6]), tuple(pts[7]), (0,0,0), 2)
        cv.line(img, tuple(pts[7]), tuple(pts[4]), (0,0,0), 2)
        for i in range(4):
            cv.line(img, tuple(pts[i]), tuple(pts[i+4]), (0,0,0), 2)

        if online!= True:
            # Display image
            global save_id
            cv.imshow("Cube overlay", img)
            cv.imwrite(f"./cube_runs/cube_overlay{save_id}.png", img)
            save_id += 1

            key = cv.waitKey(0)
            if key == 27:
                cv.destroyAllWindows()

        else:
            return img
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

objp = create_object_points(pattern_size[0], pattern_size[1],EDGE_SIZE)


for fname in images[:25]:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (pattern_size[0],pattern_size[1]), None)

    if ret == True:
        auto_objectPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

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

    if len(state['2dpoints']) != 4:
        print(f'This image was skipped : {fname}. Please select at least 4 points in the next one.')
        continue

    src = np.array(state['2dpoints'], dtype=np.float32)

    # --------------- Enforce order -----------------#

    src = np.array(state['2dpoints'], dtype = np.float32)
    ordered = order_points(src, pattern_size = pattern_size)
    
    ordered_ref = ordered.reshape(-1, 1, 2).astype(np.float32)
    cv.cornerSubPix(gray, ordered_ref, (11,11), (-1,-1), criteria)
    ordered_refined = ordered_ref.reshape(4, 2)

    #dst = np.float32([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]]) # the corners
    dst = np.float32([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ])
    W = (pattern_size[0] - 1) * 100
    H = (pattern_size[1] - 1) * 100
    
    #warping for better corner estimation
    H_warp = cv.getPerspectiveTransform(ordered_refined, dst)
    warped = cv.warpPerspective(img, H_warp, (W, H))
    gray_warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

    # interpolation
    warped_grid = interpolate_corners(dst, pattern_size[0], pattern_size[1])
    warped_grid = warped_grid.reshape(-1, 1, 2).astype(np.float32)

    # inverse warping to go back to original image
    H_inv = cv.getPerspectiveTransform(dst, ordered_refined)
    img_grid = cv.perspectiveTransform(warped_grid, H_inv)

    img_points = img_grid.reshape(-1, 2)
    manual_imagePoints.append(img_points)
    manual_objectPoints.append(board_object_points.copy())

    for pt in img_grid:
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
img = cv.imread(img_path)


# Run 1
draw_cube(img.copy(), cameraMatrix1, distCoeffs1, pattern_size=pattern_size, online=False)

# Run 2
draw_cube(img.copy(), cameraMatrix2, distCoeffs2, pattern_size=pattern_size, online=False)

# Run 3
draw_cube(img.copy(), cameraMatrix3, distCoeffs3, pattern_size=pattern_size, online=False)


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
            online=True
        )

    cv.imshow('Realtime pose', frame)

    if cv.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv.destroyAllWindows()


"""
    Choice task 6

"""
cam_pos1 = []
cam_orient = []
cam_R = []

for rvec, tvec in zip(rvecs1, tvecs1):
    R, _ = cv.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).reshape(3,)
    cam_pos1.append(cam_pos)
    cam_dir = (R.T @ np.array([0, 0, 1.0])).reshape(3)
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    cam_orient.append(cam_dir)
    cam_R.append(R)

#print(cam_pos1)

#plot the points in 3d plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
arrow_len = 4 * EDGE_SIZE

cam_pos1 = np.array(cam_pos1, dtype=float)
cam_orient = np.array(cam_orient, dtype=float)

# Flip Z axis
cam_pos1[:,2] *= -1
cam_orient[:,2] *= -1
ax.scatter(cam_pos1[:,0], cam_pos1[:,1], cam_pos1[:,2], s=10)

ax.quiver(cam_pos1[:,0], cam_pos1[:,1], cam_pos1[:,2],
          cam_orient[:,0], cam_orient[:,1], cam_orient[:,2],
          length=arrow_len, normalize=True)

qlo, qhi = 0.10, 0.90

xmin, ymin, zmin = np.quantile(cam_pos1, qlo, axis=0)
xmax, ymax, zmax = np.quantile(cam_pos1, qhi, axis=0)

pad = 0.15
dx, dy, dz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

ax.set_xlim(xmin - pad*dx, xmax + pad*dx)
ax.set_ylim(ymin - pad*dy, ymax + pad*dy)
ax.set_zlim(zmin - pad*dz, zmax + pad*dz)

ax.view_init(elev=25, azim=-60)

# chessboard plane z=0
board_w = (pattern_size[0] - 1) * EDGE_SIZE
board_h = (pattern_size[1] - 1) * EDGE_SIZE
xx, yy = np.meshgrid([0, board_w], [0, board_h])
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, alpha=0.1)

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

set_axes_equal(ax)

plt.show()