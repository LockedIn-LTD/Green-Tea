#!/usr/bin/env python3
import os
import sys
from os.path import join
import calibChArUco as base

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 process_intrinsics_charuco.py calibration_settings.yaml")
        sys.exit(1)

    settings = sys.argv[1]
    base.parse_calibration_settings_file(settings)

    # Optional: let the script auto-guess the dictionary and board dims from a few images.
    # Enable via YAML: charuco_auto_dict: true
    # You already added auto_guess_charuco_config in calibChArUco.py.

    # Camera0 intrinsics
    cmtx0, dist0 = base.calibrate_camera_charuco(join('frames', 'camera0*'))
    base.save_camera_intrinsics(cmtx0, dist0, 'camera0')

    # Camera1 intrinsics
    cmtx1, dist1 = base.calibrate_camera_charuco(join('frames', 'camera1*'))
    base.save_camera_intrinsics(cmtx1, dist1, 'camera1')

    print("[DONE] Intrinsics saved to camera_parameters/camera0_intrinsics.dat and camera1_intrinsics.dat")

if __name__ == '__main__':
    main()