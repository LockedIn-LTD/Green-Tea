import cv2 as cv
import numpy as np
import sys
import os

def undistort_test(intrinsics_path, image_path):
    """
    Loads intrinsic parameters, undistorts the test image, and saves a 
    comparison file with labels to a specific path.

    Args:
        intrinsics_path (str): Path to the .dat file containing the camera matrix and dist coeffs.
        image_path (str): Path to the image file to test.
    """
    if not os.path.exists(intrinsics_path):
        print(f"Error: Intrinsics file not found at {intrinsics_path}")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # --- 1. Load Intrinsic Parameters (Robust line-by-line reading) ---
    camera_matrix = np.zeros((3, 3), dtype=np.float64)
    dist_coeffs = None
    
    try:
        with open(intrinsics_path, 'r') as f:
            lines = f.readlines()
        
        # Line 0: "intrinsic:" (Skip)
        
        # Lines 1-3: Camera Matrix
        for i in range(3):
            line_index = i + 1 
            row = np.fromstring(lines[line_index].strip(), sep=' ')
            if len(row) != 3:
                raise ValueError(f"Camera matrix row {i+1} did not have 3 values.")
            camera_matrix[i] = row
            
        # Line 4: "distortion:" (Skip)
        
        # Line 5: Distortion Coefficients
        dist_coeffs_row = np.fromstring(lines[5].strip(), sep=' ')
        if len(dist_coeffs_row) != 5:
            raise ValueError(f"Distortion coefficients row did not have 5 values.")
        dist_coeffs = dist_coeffs_row.reshape((1, 5))
            
        print("\n--- Loaded Intrinsic Parameters ---")
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)

    except Exception as e:
        print(f"Error loading intrinsic data: {e}")
        return

    # --- 2. Load and Undistort Image ---
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    h, w = img.shape[:2]

    # Get the optimal new camera matrix and RoI
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # Undistort using remap
    mapx, mapy = cv.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5
    )
    undistorted_img = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # --- 3. Add Labels and Combine Images (Vertical Stack) ---
    
    # Define text properties
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    
    # Create copies to draw text on
    labeled_original = img.copy()
    labeled_undistorted = undistorted_img.copy()
    
    # Label Original (Distorted) - Yellow text
    cv.putText(labeled_original, "ORIGINAL (DISTORTED)", (50, 80), font, font_scale, (0, 255, 255), thickness + 4, cv.LINE_AA)
    cv.putText(labeled_original, "ORIGINAL (DISTORTED)", (50, 80), font, font_scale, (255, 0, 0), thickness, cv.LINE_AA)
    
    # Label Undistorted (Fixed) - Green text
    cv.putText(labeled_undistorted, "UNDISTORTED (FIXED)", (50, 80), font, font_scale, (0, 255, 255), thickness + 4, cv.LINE_AA)
    cv.putText(labeled_undistorted, "UNDISTORTED (FIXED)", (50, 80), font, font_scale, (0, 255, 0), thickness, cv.LINE_AA)

    # Vertically stack the labeled images
    comparison_img = np.vstack((labeled_original, labeled_undistorted))

    # --- 4. Save Results to a File ---
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_comparison_fixed.png"
    output_path = os.path.join(os.path.dirname(image_path), output_filename)
    
    cv.imwrite(output_path, comparison_img)
    
    print(f"\n[OUTPUT SAVED] Comparison image saved to: {output_path}")
    print("Please check this file to verify that straight lines, especially near the edges, are now straight.")
    
    # We explicitly do not use cv.imshow() here since the user cannot see the window.


if __name__ == '__main__':
    # Check for correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python3 undistort.py <path_to_intrinsics.dat> <path_to_test_image.png>")
        print("Example: python3 undistort.py camera_parameters/camera0_intrinsics.dat frames/camera0_0.png")
        sys.exit(1)

    intrinsics_path = sys.argv[1]
    image_path = sys.argv[2]
    
    undistort_test(intrinsics_path, image_path)