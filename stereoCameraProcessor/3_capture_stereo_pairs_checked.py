#!/usr/bin/env python3
import os
import sys
import glob
import time
import cv2 as cv
import numpy as np
import calibChArUco as base

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def clear_pairs():
    ensure_dir('frames_pair')
    n0 = 0
    for p in glob.glob(os.path.join('frames_pair', 'camera0_*.png')):
        try: os.remove(p); n0 += 1
        except: pass
    n1 = 0
    for p in glob.glob(os.path.join('frames_pair', 'camera1_*.png')):
        try: os.remove(p); n1 += 1
        except: pass
    if n0 or n1:
        print(f"[INFO] Cleared previous pairs: camera0={n0}, camera1={n1}")

def next_pair_index():
    existing = sorted(glob.glob(os.path.join('frames_pair', 'camera0_*.png')))
    return len(existing)

def ensure_bgr(img):
    if img is None or img.size == 0:
        return img
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
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
        base.aruco.drawDetectedCornersCharuco(img, corners, ids, color)
    except cv.error as e:
        print(f"[WARN] drawDetectedCornersCharuco failed: {e}")
    return img

def detect_counts(frame, dictionary, board):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cc, ci, mc, mi, _ = base.detect_charuco(gray, dictionary, board)
    cnt_c = 0 if ci is None else int(len(ci))
    cnt_m = 0 if mi is None else int(len(mi))
    return cnt_c, cnt_m, cc, ci, mc, mi

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 capture_stereo_pairs_checked.py calibration_settings.yaml")
        sys.exit(1)

    settings_path = sys.argv[1]
    base.parse_calibration_settings_file(settings_path)

    # Build dictionary/board once
    dictionary, board = base.make_charuco_board()

    cs = base.calibration_settings
    width  = int(cs['frame_width'])
    height = int(cs['frame_height'])
    goal_pairs = int(cs['stereo_calibration_frames'])
    view_resize = float(cs.get('view_resize', 1))
    min_corners = int(cs.get('charuco_min_corners', 6))

    clear_pairs()

    # Open cameras
    # Assuming base.open_nvargus is correctly defined in calibChArUco
    try:
        cap0 = base.open_nvargus(cs['camera0'], width, height, flip=2)
        cap1 = base.open_nvargus(cs['camera1'], width, height, flip=2)
    except Exception as e:
        print(f"[FATAL] Could not open cameras: {e}")
        sys.exit(1)


    saved = 0
    idx = next_pair_index()

    win0, win1 = 'preview_cam0', 'preview_cam1'
    cv.namedWindow(win0, cv.WINDOW_NORMAL)
    cv.namedWindow(win1, cv.WINDOW_NORMAL)

    try:
        while saved < goal_pairs:
            r0, f0 = cap0.read()
            r1, f1 = cap1.read()
            if not r0 or not r1 or f0 is None or f1 is None:
                print("[ERR] One of the cameras returned no frame.")
                time.sleep(0.05)
                continue

            # Display live feed
            s0 = cv.resize(f0, None, fx=1.0/view_resize, fy=1.0/view_resize)
            s1 = cv.resize(f1, None, fx=1.0/view_resize, fy=1.0/view_resize)
            cv.putText(s0, f"cam0 | saved {saved}/{goal_pairs}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv.putText(s1, "SPACE=capture | ESC=quit", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
            cv.imshow(win0, s0)
            cv.imshow(win1, s1)

            k = cv.waitKey(1) & 0xFF
            if k == 27:
                print("[INFO] Aborted by user.")
                break

            if k == 32:  # SPACE -> capture, validate, and enter review mode
                f0c = f0.copy()
                f1c = f1.copy()

                # --- VALIDATION STEP ---
                cnt_c0, cnt_m0, cc0, ci0, mc0, mi0 = detect_counts(f0c, dictionary, board)
                cnt_c1, cnt_m1, cc1, ci1, mc1, mi1 = detect_counts(f1c, dictionary, board)

                ok0 = cnt_c0 >= min_corners
                ok1 = cnt_c1 >= min_corners
                st0 = "GOOD (S=Save)" if ok0 else "BAD (D=Discard)"
                st1 = "GOOD (S=Save)" if ok1 else "BAD (D=Discard)"

                # Build review overlays
                r0v = draw_markers_safe(f0c.copy(), mc0, mi0, (0,255,255))
                r0v = draw_charuco_safe(r0v, cc0, ci0, (255,0,0))
                r1v = draw_markers_safe(f1c.copy(), mc1, mi1, (0,255,255))
                r1v = draw_charuco_safe(r1v, cc1, ci1, (255,0,0))

                # Add status text
                cv.putText(r0v, f"cam0 STATUS: {st0} corners={cnt_c0}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if ok0 else (0,0,255), 2)
                cv.putText(r1v, f"cam1 STATUS: {st1} corners={cnt_c1}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if ok1 else (0,0,255), 2)
                
                # Instruction text (yellow)
                review_text = "Press 'S' to Save (if GOOD) or 'D' to Discard"
                cv.putText(r0v, review_text, (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv.putText(r1v, review_text, (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                cv.imshow(win0, r0v)
                cv.imshow(win1, r1v)
                
                # --- REVIEW LOOP ---
                while True:
                    review_k = cv.waitKey(10) & 0xFF
                    
                    if review_k == ord('s') or review_k == ord('S'):
                        if ok0 and ok1:
                            p0 = os.path.join('frames_pair', f'camera0_{idx}.png')
                            p1 = os.path.join('frames_pair', f'camera1_{idx}.png')
                            cv.imwrite(p0, f0c)
                            cv.imwrite(p1, f1c)
                            saved += 1
                            idx += 1
                            print(f"[OK] Saved pair #{saved}: {p0}, {p1}")
                        else:
                            print(f"[REJECT] Save failed. Not enough corners detected (min {min_corners})")
                        break # Exit review loop
                        
                    elif review_k == ord('d') or review_k == ord('D'):
                        print(f"[DISCARDED] Pair was discarded.")
                        break # Exit review loop
                        
                    elif review_k == 27: # ESC key in review mode quits the whole program
                        print("[INFO] Aborted during review by user.")
                        cap0.release()
                        cap1.release()
                        cv.destroyAllWindows()
                        sys.exit(0)

    finally:
        cap0.release()
        cap1.release()
        cv.destroyAllWindows()

    print(f"[DONE] Saved {saved} stereo pairs in frames_pair/")

if __name__ == '__main__':
    main()