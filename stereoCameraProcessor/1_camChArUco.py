#!/usr/bin/env python3
import os
import sys
import glob
import time
import cv2 as cv
import numpy as np
from os.path import join, basename

# Import your existing module
import calibChArUco as base

def ensure_bgr(img):
    if img is None or img.size == 0:
        return img
    # Gray -> BGR
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # BGRA -> BGR
    if img.ndim == 3 and img.shape[2] == 4:
        return cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    return img

def draw_markers_safe(img, corners, ids, color=(0,255,255)):
    img = ensure_bgr(img)
    if img is None or img.size == 0:
        return img
    if ids is None or len(ids) == 0 or corners is None or len(corners) == 0:
        return img
    try:
        # use aruco from your base module if that's how you imported it
        import calibChArUco as base
        base.aruco.drawDetectedMarkers(img, corners, ids, borderColor=color)
    except cv.error as e:
        print(f"[WARN] drawDetectedMarkers failed: {e}")
    return img

def draw_charuco_safe(img, corners, ids, color=(255,0,0)):
    img = ensure_bgr(img)
    if img is None or img.size == 0:
        return img
    if ids is None or len(ids) == 0 or corners is None or len(corners) == 0:
        return img
    try:
        import calibChArUco as base
        base.aruco.drawDetectedCornersCharuco(img, corners, ids, color)
    except cv.error as e:
        print(f"[WARN] drawDetectedCornersCharuco failed: {e}")
    return img

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def clear_frames_for_camera(camera_name):
    # NOTE: This function is defined but not called in the main capture loop
    # to support the user's request of keeping old images.
    ensure_dir('frames')
    pat = os.path.join('frames', f'{camera_name}_*.png')
    old = glob.glob(pat)
    for p in old:
        try:
            os.remove(p)
        except Exception:
            pass
    if old:
        print(f"[INFO] Cleared {len(old)} previous frames for {camera_name}")

def next_index_for_camera(camera_name):
    """
    Finds the highest index of existing files (e.g., camera0_N.png) and returns 
    the next available index (N+1) to prevent overwriting.
    """
    # Find all existing files matching the pattern
    pat = os.path.join('frames', f'{camera_name}_*.png')
    existing = glob.glob(pat)
    
    max_index = -1
    
    for full_path in existing:
        # Get just the filename (e.g., 'camera0_19.png')
        filename = basename(full_path)
        
        try:
            # Expected format: cameraX_N.png
            # 1. Split by underscore: ['cameraX', 'N.png']
            parts = filename.split('_')
            if len(parts) < 2:
                continue
            index_part_ext = parts[-1]
            
            # 2. Split by dot: ['N', 'png']
            index_str = index_part_ext.split('.')[0]
            
            # 3. Convert to integer
            current_index = int(index_str)
            
            if current_index > max_index:
                max_index = current_index
        except (ValueError, IndexError):
            # Ignore files that don't match the expected naming scheme
            continue
            
    # If max_index is -1 (no files found), start at 0. Otherwise, start at max_index + 1.
    return max_index + 1

def detect_counts(frame, dictionary, board):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Reuse your existing detection (returns charuco_corners, charuco_ids, marker_corners, marker_ids, rejected)
    cc, ci, mc, mi, _ = base.detect_charuco(gray, dictionary, board)
    cnt_c = 0 if ci is None else int(len(ci))
    cnt_m = 0 if mi is None else int(len(mi))
    return cnt_c, cnt_m, cc, ci, mc, mi

def capture_checked_for_camera(camera_name, dictionary, board):
    cs = base.calibration_settings
    width  = int(cs['frame_width'])
    height = int(cs['frame_height'])
    goal   = int(cs['mono_calibration_frames'])
    view_resize = float(cs.get('view_resize', 1))
    min_corners = int(cs.get('charuco_min_corners', 6))

    # We do NOT call clear_frames_for_camera(camera_name) here, 
    # as requested, to keep existing calibration images.

    sensor_id = cs[camera_name]
    cap = base.open_nvargus(sensor_id, width, height, flip=2)
    saved = 0
    # Start the index from the next available number
    idx = next_index_for_camera(camera_name)

    win = f'preview_{camera_name}'
    cv.namedWindow(win, cv.WINDOW_NORMAL)

    try:
        while saved < goal:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERR] Camera returned no frame.")
                time.sleep(0.05)
                continue

            # Show preview with status overlay
            small = cv.resize(frame, None, fx=1.0/view_resize, fy=1.0/view_resize)
            cv.putText(small, f"{camera_name} | saved {saved}/{goal}", (20, 40),
                        cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv.putText(small, "SPACE=capture | ESC=quit", (20, 80),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv.imshow(win, small)
            key = cv.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("[INFO] User aborted.")
                return saved

            if key == 32:  # SPACE -> capture and validate
                # Freeze a copy for review
                frame_cap = frame.copy()
                cnt_c, cnt_m, cc, ci, mc, mi = detect_counts(frame_cap, dictionary, board)

                review = ensure_bgr(frame_cap.copy())
                review = draw_markers_safe(review, mc, mi, color=(0,255,255))  # yellow markers (info)
                review = draw_charuco_safe(review, cc, ci, color=(255,0,0))    # blue ChArUco corners

                status = "ACCEPTED" if cnt_c >= min_corners else "REJECTED"
                color = (0,255,0) if status == "ACCEPTED" else (0,0,255)
                msg1 = f"{status} | corners: {cnt_c} (min {min_corners}) | markers: {cnt_m}"
                cv.putText(review, msg1, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv.putText(review, "Press any key to continue", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                # Show full-size review
                cv.imshow(win, review)
                cv.waitKey(0)

                if cnt_c >= min_corners:
                    # Save with the next available index
                    out_path = os.path.join('frames', f'{camera_name}_{idx}.png')
                    cv.imwrite(out_path, frame_cap)
                    saved += 1
                    idx += 1
                    print(f"[OK] Saved {out_path} (corners={cnt_c}, markers={cnt_m}) [{saved}/{goal}]")
                else:
                    print(f"[REJECT] corners={cnt_c} < min_corners={min_corners}. Not saved.")

    finally:
        cap.release()
        cv.destroyWindow(win)

    return saved

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 capture_charuco_checked.py calibration_settings.yaml")
        sys.exit(1)

    settings_path = sys.argv[1]
    base.parse_calibration_settings_file(settings_path)
    
    # Ensure the frames directory exists before starting
    ensure_dir('frames')

    # Create board/dictionary once
    dictionary, board = base.make_charuco_board()

    # Capture for camera0 then camera1
    print("\n--- Starting Camera 0 Capture ---")
    total0 = capture_checked_for_camera('camera0', dictionary, board)
    print(f"[INFO] camera0 done, saved {total0} images.")
    
    print("\n--- Starting Camera 1 Capture ---")
    total1 = capture_checked_for_camera('camera1', dictionary, board)
    print(f"[INFO] camera1 done, saved {total1} images.")

    print("\n[DONE] You can now run your calibration step on frames/.")

if __name__ == '__main__':
    main()