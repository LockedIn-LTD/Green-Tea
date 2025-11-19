#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
from os.path import join
import calibChArUco as base # This import now accesses the functions above

def load_intrinsics_dat(path):
    """
    Read camera_parameters/*_intrinsics.dat written by save_camera_intrinsics.
    Returns camera_matrix (3x3) and dist (1xN).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mat = []
    dist = []
    mode = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.lower().startswith('intrinsic'):
                mode = 'K'; continue
            if line.lower().startswith('distortion'):
                mode = 'D'; continue
            vals = [v for v in line.split() if v not in ['\n', '\r']]
            if not vals: continue
            if mode == 'K':
                mat.append([float(v) for v in vals])
            elif mode == 'D':
                dist.extend([float(v) for v in vals])
    K = np.array(mat, dtype=np.float64)
    if K.shape != (3,3):
        raise ValueError(f"Bad intrinsic matrix shape in {path}: {K.shape}")
    # OpenCV expects distortion coefficients as (1, N) or (N, 1) or (1, N)
    dist = np.array(dist, dtype=np.float64).reshape(1, -1) 
    return K, dist

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 4_process_stereo_charuco.py calibration_settings.yaml")
        sys.exit(1)

    settings = sys.argv[1]
    base.parse_calibration_settings_file(settings) # Loads config (board size, etc.)

    # --- 1. Load Intrinsics ---
    try:
        K0, d0 = load_intrinsics_dat(join('camera_parameters', 'camera0_intrinsics.dat'))
        K1, d1 = load_intrinsics_dat(join('camera_parameters', 'camera1_intrinsics.dat'))
    except FileNotFoundError as e:
        print(f"[FATAL ERROR] Intrinsic file not found: {e}")
        print("Please ensure you have successfully run the monocular calibration step first.")
        sys.exit(1)
        
    print("\n--- 1. Loaded Camera 0 Intrinsics (K0) ---")
    print(K0)
    print("\n--- 2. Loaded Camera 1 Intrinsics (K1) ---")
    print(K1)

    # --- 3. Compute Stereo Extrinsics ---
    print("\n--- 3. Starting Stereo Calibration... ---")
    
    # This call now executes the full point detection and cv2.stereoCalibrate logic
    try:
        ret, R, T = base.stereo_calibrate_charuco(
            K0, d0, K1, d1,
            join('frames_pair', 'camera0*'),
            join('frames_pair', 'camera1*')
        )
    except Exception as e:
        print(f"[FATAL ERROR] Stereo Calibration Failed: {e}")
        sys.exit(1)
    
    # --- 4. Report Results ---
    
    # The return value (ret) from OpenCV's stereoCalibrate is the final RMS error
    print("\n==============================================")
    print(f"[FINAL RESULT] Reprojection Error (RMSE): {ret:.4f} pixels")
    print("==============================================")
    
    if ret > 1.0:
        print("[WARNING] RMSE is high (> 1.0). Consider capturing more frames or filtering existing ones.")
    else:
        print("[SUCCESS] RMSE is excellent. Calibration is complete!")

    print("\n--- 5. Calculated Stereo Extrinsics (R & T) ---")
    print("R (Rotation Matrix from Camera 0 to Camera 1):")
    print(R)
    print("\nT (Translation Vector from Camera 0 to Camera 1, in meters):")
    print(T.flatten())
    
    # --- 6. Save Extrinsics ---
    # Save with camera0 as world origin (R0, T0 = identity)
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0.,0.,0.]).reshape(3,1)
    base.save_extrinsic_calibration_parameters(R0, T0, R, T)
    print("\n[DONE] Extrinsics saved to camera_parameters/camera0_rot_trans.dat and camera1_rot_trans.dat")

if __name__ == '__main__':
    main()