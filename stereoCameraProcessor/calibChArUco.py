import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os
import time

# This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}

# Try to import aruco (requires opencv-contrib)
try:
    import cv2.aruco as aruco
except Exception as e:
    print("ERROR: OpenCV ArUco module not found. Install OpenCV with contrib (aruco) support.")
    print("On Jetson: ensure python3-opencv includes aruco. On pip: pip3 install opencv-contrib-python")
    raise

# ---- Helpers for API compatibility ----


def has_attr(obj, name):
    try:
        return hasattr(obj, name)
    except Exception:
        return False


def charuco_board_create_compat(squares_x, squares_y, square_length, marker_length, dictionary):
    # 1) Old factory
    if has_attr(aruco, 'CharucoBoard_create'):
        return aruco.CharucoBoard_create(
            squaresX=int(squares_x),
            squaresY=int(squares_y),
            squareLength=float(square_length),
            markerLength=float(marker_length),
            dictionary=dictionary
        )
    # 2) Static method on class
    if has_attr(aruco, 'CharucoBoard') and has_attr(aruco.CharucoBoard, 'create'):
        return aruco.CharucoBoard.create(
            squaresX=int(squares_x),
            squaresY=int(squares_y),
            squareLength=float(square_length),
            markerLength=float(marker_length),
            dictionary=dictionary
        )
    # 3) Direct constructor (your build likely uses this path)
    try:
        return aruco.CharucoBoard((int(squares_x), int(squares_y)),
                                  float(square_length), float(marker_length),
                                  dictionary)
    except Exception:
        pass
    try:
        return aruco.CharucoBoard([int(squares_x), int(squares_y)],
                                  float(square_length), float(marker_length),
                                  dictionary)
    except Exception:
        pass
    raise RuntimeError(
        "cv2.aruco is present, but no CharucoBoard creation API works in this build.\n"
        "Please install OpenCV with contrib/aruco (ChArUco) fully enabled, or use a GridBoard fallback."
    )

# IMX219 modes (Argus):
# 0: 3280x2464 @21, 1: 3280x1848 @28, 2: 1920x1080 @30, 3: 1640x1232 @30, 4: 1280x720 @60


def get_sensor_mode_and_fps(width, height):
    mapping = {
        (3280, 2464): (0, 21),
        (3280, 1848): (1, 28),
        (1920, 1080): (2, 30),
        (1640, 1232): (3, 30),
        (1280, 720):  (4, 60),
    }
    return mapping.get((int(width), int(height)), (4, 60))


def gst_pipeline(sensor_id, width, height, fps=None, flip=0, sensor_mode=None):
    if sensor_mode is None or fps is None:
        mode_auto, fps_auto = get_sensor_mode_and_fps(width, height)
        sensor_mode = mode_auto if sensor_mode is None else sensor_mode
        fps = fps_auto if fps is None else fps

    return (
        f"nvarguscamerasrc sensor-id={int(sensor_id)} sensor-mode={int(sensor_mode)} "
        f"bufapi-version=1 ! "
        f"video/x-raw(memory:NVMM), width=(int){int(width)}, height=(int){int(height)}, "
        f"framerate=(fraction){int(fps)}/1, format=(string)NV12 ! "
        f"nvvidconv flip-method={int(flip)} ! "
        f"video/x-raw, format=(string)BGRx, width=(int){int(width)}, height=(int){int(height)} ! "
        f"videoconvert ! "
        f"appsink caps=video/x-raw,format=(string)BGR,width=(int){int(width)},height=(int){int(height)} "
        f"drop=true max-buffers=1 sync=false"
    )


def open_nvargus(sensor_id, width, height, fps=None, flip=0, sensor_mode=None):
    pipeline = gst_pipeline(sensor_id, width, height,
                            fps=fps, flip=flip, sensor_mode=sensor_mode)
    mode_auto, fps_auto = get_sensor_mode_and_fps(width, height)
    print(
        f"[INFO] Opening Argus sensor-id={sensor_id} mode={sensor_mode or mode_auto} {width}x{height}@{fps or fps_auto}")
    cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open nvargus camera sensor-id={sensor_id}")

    # Warmup frames
    ok = False
    for _ in range(15):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok = True
            break
        time.sleep(0.02)
    if not ok:
        cap.release()
        raise RuntimeError(
            f"Camera sensor-id={sensor_id} opened but no frames delivered.")
    return cap

# Utility: map string to aruco dictionary


def get_aruco_dict(name):
    if not isinstance(name, str):
        raise ValueError(
            "charuco_dict_name must be a string like 'DICT_5X5_1000'")
    key = name.strip().upper()
    if not hasattr(aruco, key):
        raise ValueError(
            f"Unknown ArUco dict name '{name}'. Check cv2.aruco docs.")
    return aruco.getPredefinedDictionary(getattr(aruco, key))


def make_charuco_board():
    # Read required settings
    squares_x = int(calibration_settings['charuco_squares_x'])  # columns (X)
    squares_y = int(calibration_settings['charuco_squares_y'])  # rows (Y)
    # real units (e.g., mm)
    square_length = float(calibration_settings['charuco_square_length'])
    marker_length = float(
        calibration_settings['charuco_marker_length'])  # same units
    dict_name = calibration_settings.get('charuco_dict_name', 'DICT_5X5_1000')

    dictionary = get_aruco_dict(dict_name)
    board = charuco_board_create_compat(
        squares_x, squares_y, square_length, marker_length, dictionary)
    return dictionary, board

def auto_guess_charuco_config(images_prefix, max_samples=5):
    """
    Try several 5x5 ArUco dictionaries and both (squares_x, squares_y) and (squares_y, squares_x)
    using a few saved images. Returns (dict_name, squares_x_best, squares_y_best) or None.
    """
    # Candidate dictionaries (add more if needed)
    dict_candidates = [
        "DICT_5X5_250",
        "DICT_5X5_1000",
        "DICT_5X5_100",
        "DICT_5X5_50",
    ]
    dicts = [d for d in dict_candidates if hasattr(aruco, d)]

    sx = int(calibration_settings["charuco_squares_x"])
    sy = int(calibration_settings["charuco_squares_y"])
    sq = float(calibration_settings["charuco_square_length"])
    ml = float(calibration_settings["charuco_marker_length"])

    # Load up to max_samples images
    names = sorted(glob.glob(images_prefix))[:max_samples]
    imgs = []
    for n in names:
        im = cv.imread(n, 1)
        if im is not None:
            imgs.append(im)
    if not imgs:
        print("[AUTO] No sample images found for auto-guessing.")
        return None

    results = []
    for dict_name in dicts:
        dictionary = get_aruco_dict(dict_name)
        for dims in [(sx, sy), (sy, sx)]:
            try:
                board = charuco_board_create_compat(dims[0], dims[1], sq, ml, dictionary)
            except Exception:
                continue

            total = 0
            for im in imgs:
                gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                cc, ci, _, _, _ = detect_charuco(gray, dictionary, board)
                total += 0 if ci is None else len(ci)

            results.append((total, dict_name, dims[0], dims[1]))

    if not results:
        print("[AUTO] Could not evaluate any ChArUco configurations.")
        return None

    results.sort(key=lambda x: x[0], reverse=True)
    total, dict_name, sx_best, sy_best = results[0]
    print(f"[AUTO] Best ChArUco config: dict={dict_name}, squares_x={sx_best}, squares_y={sy_best}, "
          f"total corners over samples={total}")
    return dict_name, sx_best, sy_best


def detect_charuco(gray, dictionary, board):
    # DetectorParameters() for your build
    params = aruco.DetectorParameters()

    # Friendlier settings for small markers at 720p
    for name, val in [
        ('adaptiveThreshWinSizeMin', 3),
        ('adaptiveThreshWinSizeMax', 23),
        ('adaptiveThreshWinSizeStep', 10),
        ('minMarkerPerimeterRate', 0.02),  # lower -> detect smaller markers
        ('polygonalApproxAccuracyRate', 0.03),
        ('cornerRefinementWinSize', 5),
        ('cornerRefinementMaxIterations', 30),
        ('cornerRefinementMinAccuracy', 0.01),
    ]:
        if hasattr(params, name):
            setattr(params, name, val)

    if has_attr(params, 'cornerRefinementMethod'):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    # Detect markers
    if has_attr(aruco, 'ArucoDetector'):
        detector = aruco.ArucoDetector(dictionary, params)
        marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, rejected = aruco.detectMarkers(
            gray, dictionary, parameters=params)

    # Optionally refine detections (helps in borderline cases)
    if has_attr(aruco, 'refineDetectedMarkers') and marker_ids is not None and len(marker_ids) > 0:
        try:
            marker_corners, marker_ids, rejected = aruco.refineDetectedMarkers(
                gray, board, marker_corners, marker_ids, rejected
            )
        except Exception:
            pass

    # Interpolate ChArUco corners; requires some markers
    charuco_corners, charuco_ids = None, None
    if marker_ids is not None and len(marker_ids) > 0:
        try:
            retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )
            if not retval:
                charuco_corners, charuco_ids = None, None
        except TypeError:
            out = aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board)
            if isinstance(out, tuple) and len(out) == 3:
                charuco_corners, charuco_ids, _ = out
            else:
                charuco_corners, charuco_ids = out[0], out[1]

    # Return whatever we got; acceptance will check only charuco count
    return charuco_corners, charuco_ids, marker_corners, marker_ids, rejected

# Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.


def DLT(P1, P2, point1, point2):
    A = [point1[1]*P1[2, :] - P1[1, :],
         P1[0, :] - point1[0]*P1[2, :],
         point2[1]*P2[2, :] - P2[1, :],
         P2[0, :] - point2[0]*P2[2, :]]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3]/Vh[3, 3]

# Open and load the calibration_settings.yaml file


def parse_calibration_settings_file(filename):
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    # Require ChArUco settings
    required = ['charuco_squares_x', 'charuco_squares_y',
                'charuco_square_length', 'charuco_marker_length']
    for k in required:
        if k not in calibration_settings:
            print(f"Missing '{k}' in calibration_settings.yaml (ChArUco).")
            quit()

# Open camera stream and save frames


def save_frames_single_camera(camera_name):
    if not os.path.exists('frames'):
        os.mkdir('frames')

    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings['mono_calibration_frames']
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']

    cap = open_nvargus(camera_device_id, width, height)

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if ret == False or frame is None:
            print("No video data received from camera. Exiting...")
            cap.release()
            quit()

        frame_small = cv.resize(
            frame, None, fx=1/float(view_resize), fy=1/float(view_resize))

        if not start:
            cv.putText(frame_small, "Press SPACEBAR to start collection frames",
                       (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        if start:
            cooldown -= 1
            cv.putText(frame_small, "Cooldown: " + str(cooldown),
                       (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame_small, "Num frames: " + str(saved_count),
                       (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            if cooldown <= 0:
                savename = os.path.join(
                    'frames', camera_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame_small', frame_small)
        k = cv.waitKey(1)

        if k == 27:
            cap.release()
            cv.destroyAllWindows()
            quit()

        if k == 32:
            start = True

        if saved_count == number_to_save:
            break

    cv.destroyAllWindows()
    cap.release()

# Calibrate single camera using ChArUco detections


def calibrate_camera_charuco(images_prefix):
# Load images once
    images_names = sorted(glob.glob(images_prefix))
    images = []
    for name in images_names:
        im = cv.imread(name, 1)
        if im is not None:
            images.append(im)

    if len(images) == 0:
        print("No images found for calibration at:", images_prefix)
        quit()

    # Optionally auto-guess the correct dictionary and board dims
    if calibration_settings.get("charuco_auto_dict", False):
        guess = auto_guess_charuco_config(images_prefix, max_samples=5)
        if guess is not None:
            dict_name, sx_best, sy_best = guess
            calibration_settings["charuco_dict_name"] = dict_name
            calibration_settings["charuco_squares_x"] = sx_best
            calibration_settings["charuco_squares_y"] = sy_best
        else:
            print("[AUTO] Could not guess ChArUco config; using YAML values.")

    # Build dictionary/board (after potential auto-guess)
    dictionary, board = make_charuco_board()

    width = images[0].shape[1]
    height = images[0].shape[0]

    # Collect charuco corners/ids per image
    all_charuco_corners = []
    all_charuco_ids = []
    # NEW: Store original indices of accepted images
    accepted_image_indices = []

    min_corners = int(calibration_settings.get('charuco_min_corners', 6))
    used = 0
    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids, _ = detect_charuco(gray, dictionary, board)

        # Get the actual file name without the path
        current_file_name = os.path.basename(images_names[i]) # <-- NEW

        disp = frame.copy()
        # Draw markers in yellow (info only)
        cnt_m = 0 if marker_ids is None else int(len(marker_ids))
        if cnt_m > 0:
            aruco.drawDetectedMarkers(disp, marker_corners, marker_ids, borderColor=(0, 255, 255))
        # Draw ChArUco corners in blue
        cnt_c = 0 if charuco_ids is None else int(len(charuco_ids))
        if cnt_c > 0:
            aruco.drawDetectedCornersCharuco(disp, charuco_corners, charuco_ids, (255, 0, 0))

        # Update the message to display the file name
        msg1 = f"[{current_file_name}] | Corners: {cnt_c} (min {min_corners}) | markers: {cnt_m}" # <-- CHANGED
        msg2 = "Space/Enter=accept if corners >= min | 'a'=force accept | 's'=skip"
        cv.putText(disp, msg1, (25, 35), cv.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 1)
        cv.putText(disp, msg2, (25, 65), cv.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 1)
        cv.imshow('charuco_detect', disp)
        k = cv.waitKey(0) & 0xFF

        accepted = False
        if k == ord('s'):
            # Print file name when skipping
            print(f'skipping {current_file_name} (user)') # <-- CHANGED
            continue
        elif k == ord('a'):
            # Print file name when force accepting
            print(f'adding {current_file_name} (forced)') # <-- CHANGED
            if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
                accepted = True
            else:
                print('    but detection empty; cannot accept.')
                continue
        else:
            # Accept ONLY based on ChArUco corner count
            if charuco_ids is None or cnt_c < min_corners:
                # Print file name when skipping due to corner count
                print(f'skipping {current_file_name} (corners {cnt_c}/{min_corners})') # <-- CHANGED
                continue
            # Print file name when accepting automatically
            print(f'adding {current_file_name}') # <-- CHANGED
            accepted = True
        
        if accepted:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            accepted_image_indices.append(i) # NEW: Store the original index
            used += 1

    cv.destroyAllWindows()
    if used < 4:
        print("Not enough valid ChArUco detections. Try more images.")
        quit()

    # Calibrate from ChArUco
    flags = 0  # you can add cv.CALIB_RATIONAL_MODEL if needed
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    try:
        retval, cmtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=(width, height),
            cameraMatrix=None,
            distCoeffs=None,
            flags=flags,
            criteria=criteria
        )
    except AttributeError:
        # Fallback for builds exposing only the Extended variant
        retval, cmtx, dist, rvecs, tvecs, _, _, _, _ = aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=(width, height),
            cameraMatrix=None,
            distCoeffs=None,
            distCoeffs2=None, # Added for Extended function signature compatibility
            flags=flags,
            criteria=criteria
        )
        
    # START OF REPROJECTION ERROR CALCULATION
    all_errors = []
    for i in range(len(all_charuco_corners)):
        # Get the calibration parameters for this image
        img_points = all_charuco_corners[i]
        rvec = rvecs[i]
        tvec = tvecs[i]
        
        # Get the object points (3D world coordinates) for the detected IDs
        ids = all_charuco_ids[i].flatten()
        # CORRECTED LINE: Use the method getChessboardCorners()
        obj_points = board.getChessboardCorners()[ids, :].astype(np.float32)

        # Project the 3D world points back onto the 2D image plane
        img_points_reproj, _ = cv.projectPoints(
            obj_points, rvec, tvec, cmtx, dist
        )

        # Calculate the Root Mean Square (L2 norm) error per corner
        error = cv.norm(img_points, img_points_reproj, cv.NORM_L2) / len(img_points)
        
        # Use the stored accepted index to get the original image name
        original_idx = accepted_image_indices[i] 
        
        all_errors.append((error, images_names[original_idx]))

    # Sort the errors to find the worst images
    all_errors.sort(key=lambda x: x[0], reverse=True)

    # Print the top 5 worst images
    print("\n--- Reprojection Error Analysis ---")
    print(f"Overall RMSE (Intrinsic): {retval:.4f} pixels")
    print("\nTop 5 images with highest Reprojection Error (pixels/corner):")
    for error, image_name in all_errors:
        print(f"  {image_name}: {error:.3f}")

    # END OF REPROJECTION ERROR CALCULATION

    print('\ncamera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist


def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join(
        'camera_parameters', camera_name + '_intrinsics.dat')
    with open(out_filename, 'w') as outf:
        outf.write('intrinsic:\n')
        for l in camera_matrix:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')

        outf.write('distortion:\n')
        for en in distortion_coefs.flatten():
            outf.write(str(en) + ' ')
        outf.write('\n')

# Save frames for both cameras


def save_frames_two_cams(camera0_name, camera1_name):
    if not os.path.exists('frames_pair'):
        os.mkdir('frames_pair')

    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']
    number_to_save = calibration_settings['stereo_calibration_frames']

    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']

    cap0 = open_nvargus(calibration_settings[camera0_name], width, height)
    cap1 = open_nvargus(calibration_settings[camera1_name], width, height)

    cooldown = cooldown_time
    start = False
    saved_count = 0
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1 or frame0 is None or frame1 is None:
            print('Cameras not returning video data. Exiting...')
            cap0.release()
            cap1.release()
            quit()

        frame0_small = cv.resize(
            frame0, None, fx=1./float(view_resize), fy=1./float(view_resize))
        frame1_small = cv.resize(
            frame1, None, fx=1./float(view_resize), fy=1./float(view_resize))

        if not start:
            cv.putText(frame0_small, "Make sure both cameras can see the calibration pattern well",
                       (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.putText(frame0_small, "Press SPACEBAR to start collection frames",
                       (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        if start:
            cooldown -= 1
            cv.putText(frame0_small, "Cooldown: " + str(cooldown),
                       (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame0_small, "Num frames: " + str(saved_count),
                       (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            cv.putText(frame1_small, "Cooldown: " + str(cooldown),
                       (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame1_small, "Num frames: " + str(saved_count),
                       (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            if cooldown <= 0:
                savename = os.path.join(
                    'frames_pair', camera0_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame0)
                savename = os.path.join(
                    'frames_pair', camera1_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame1)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame0_small', frame0_small)
        cv.imshow('frame1_small', frame1_small)
        k = cv.waitKey(1)

        if k == 27:
            cap0.release()
            cap1.release()
            cv.destroyAllWindows()
            quit()

        if k == 32:
            start = True

        if saved_count == number_to_save:
            break

    cv.destroyAllWindows()
    cap0.release()
    cap1.release()

# Stereo calibrate using per-pair matched ChArUco corners

def stereo_calibrate_charuco(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    dictionary, board = make_charuco_board()

    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    if len(c0_images_names) == 0 or len(c1_images_names) == 0:
        print("No stereo frames found. Did you run Step3?")
        quit()

    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    for idx, (frame0, frame1) in enumerate(zip(c0_images, c1_images)):
        g0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        g1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        c_corners0, c_ids0, m_corners0, m_ids0, _ = detect_charuco(
            g0, dictionary, board)
        c_corners1, c_ids1, m_corners1, m_ids1, _ = detect_charuco(
            g1, dictionary, board)

        disp0 = frame0.copy()
        disp1 = frame1.copy()
        # yellow for markers (info)
        if m_ids0 is not None:
            aruco.drawDetectedMarkers(
                disp0, m_corners0, m_ids0, borderColor=(0, 255, 255))
        if m_ids1 is not None:
            aruco.drawDetectedMarkers(
                disp1, m_corners1, m_ids1, borderColor=(0, 255, 255))
        # blue for charuco corners
        if c_ids0 is not None and len(c_ids0) > 0:
            aruco.drawDetectedCornersCharuco(
                disp0, c_corners0, c_ids0, (255, 0, 0))
        if c_ids1 is not None and len(c_ids1) > 0:
            aruco.drawDetectedCornersCharuco(
                disp1, c_corners1, c_ids1, (255, 0, 0))

        cv.imshow('left_charuco', disp0)
        cv.imshow('right_charuco', disp1)
        k = cv.waitKey(1)
        if k & 0xFF == ord('s'):
            print(f"Skipping pair {idx}")
            continue

        if c_ids0 is None or c_ids1 is None:
            print(f"Pair {idx}: insufficient charuco detections, skipping")
            continue

        # Match by charuco corner IDs
        ids0 = c_ids0.flatten()
        ids1 = c_ids1.flatten()
        common = np.intersect1d(ids0, ids1)
        if len(common) < 6:
            print(f"Pair {idx}: too few common IDs ({len(common)}), skipping")
            continue

        print(f"Adding pair {idx}")
        
        # FIX: Replaced 'board.chessboardCorners' with the function call 'board.getChessboardCorners()' 
        # to fix the AttributeError.
        obj = board.getChessboardCorners()[common, :].astype(np.float32)

        # Left/right image points in the same order
        ptsL = []
        ptsR = []
        for cid in common:
            i0 = np.where(ids0 == cid)[0][0]
            i1 = np.where(ids1 == cid)[0][0]
            ptsL.append(c_corners0[i0][0])  # (x,y)
            ptsR.append(c_corners1[i1][0])

        objpoints.append(obj)
        imgpoints_left.append(np.array(ptsL, dtype=np.float32))
        imgpoints_right.append(np.array(ptsR, dtype=np.float32))

    cv.destroyAllWindows()

    if len(objpoints) < 3:
        print("Not enough valid stereo pairs. Capture more or improve detections.")
        quit()

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    flags = cv.CALIB_FIX_INTRINSIC

    # CM1 and CM2 will return the same fixed K0 and K1 since CALIB_FIX_INTRINSIC is set.
    ret, CM1, dist_out0, CM2, dist_out1, R, T, E, F = cv.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpoints_left,
        imagePoints2=imgpoints_right,
        cameraMatrix1=mtx0,
        distCoeffs1=dist0,
        cameraMatrix2=mtx1,
        distCoeffs2=dist1,
        imageSize=(width, height),
        criteria=criteria,
        flags=flags
    )
    print('rmse (ChArUco stereo):', ret)
    return ret, R, T

# Homogeneous matrix helpers


def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1
    return P


def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]
    return P


def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift=50.):
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    coordinate_points = np.array([[0., 0., 0.],
                                  [1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]])
    z_shift = np.array([0., 0., _zshift]).reshape((1, 3))
    draw_axes_points = 5 * coordinate_points + z_shift

    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera0.append(uv)
        uv = P1 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera1.append(uv)
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0 = open_nvargus(calibration_settings[camera0_name], width, height)
    cap1 = open_nvargus(calibration_settings[camera1_name], width, height)

    try:
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            if not ret0 or not ret1 or frame0 is None or frame1 is None:
                print('Video stream not returning frame data')
                break

            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            origin = tuple(pixel_points_camera0[0].astype(np.int32))
            for col, _p in zip(colors, pixel_points_camera0[1:]):
                _p = tuple(_p.astype(np.int32))
                cv.line(frame0, origin, _p, col, 2)
            origin = tuple(pixel_points_camera1[0].astype(np.int32))
            for col, _p in zip(colors, pixel_points_camera1[1:]):
                _p = tuple(_p.astype(np.int32))
                cv.line(frame1, origin, _p, col, 2)

            cv.imshow('frame0', frame0)
            cv.imshow('frame1', frame1)
            k = cv.waitKey(1)
            if k == 27:
                break
    finally:
        cap0.release()
        cap1.release()
        cv.destroyAllWindows()


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix=''):
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    camera0_rot_trans_filename = os.path.join(
        'camera_parameters', prefix + 'camera0_rot_trans.dat')
    with open(camera0_rot_trans_filename, 'w') as outf:
        outf.write('R:\n')
        for l in R0:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')
        outf.write('T:\n')
        for l in T0:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')

    camera1_rot_trans_filename = os.path.join(
        'camera_parameters', prefix + 'camera1_rot_trans.dat')
    with open(camera1_rot_trans_filename, 'w') as outf:
        outf.write('R:\n')
        for l in R1:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')
        outf.write('T:\n')
        for l in T1:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')
    return R0, T0, R1, T1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Call with settings filename: "python3 calib.py calibration_settings.yaml"')
        quit()

    parse_calibration_settings_file(sys.argv[1])

    """Step1. Save calibration frames for single cameras"""
    save_frames_single_camera('camera0')
    save_frames_single_camera('camera1')

    """Step2. Obtain intrinsic parameters with ChArUco"""
    images_prefix = os.path.join('frames', 'camera0*')
    cmtx0, dist0 = calibrate_camera_charuco(images_prefix)
    save_camera_intrinsics(cmtx0, dist0, 'camera0')

    images_prefix = os.path.join('frames', 'camera1*')
    cmtx1, dist1 = calibrate_camera_charuco(images_prefix)
    save_camera_intrinsics(cmtx1, dist1, 'camera1')

    """Step3. Save calibration frames for both cameras simultaneously"""
    save_frames_two_cams('camera0', 'camera1')

    """Step4. Stereo extrinsics using ChArUco correspondences"""
    frames_prefix_c0 = os.path.join('frames_pair', 'camera0*')
    frames_prefix_c1 = os.path.join('frames_pair', 'camera1*')
    R, T = stereo_calibrate_charuco(
        cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)

    """Step5. Save calibration where camera0 defines world origin"""
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))
    save_extrinsic_calibration_parameters(R0, T0, R, T)

    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R, T]
    check_calibration('camera0', camera0_data, 'camera1',
                      camera1_data, _zshift=60.)
