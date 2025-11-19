#!/usr/bin/env python3
import os
import glob
import re
import sys

def extract_numerical_index(filepath):
    """
    Extracts the numerical index from a filename like 'cameraX_N.png'.
    Returns -1 if no valid number is found.
    """
    # Get just the filename (e.g., 'camera0_19.png')
    filename = os.path.basename(filepath)
    
    # Use a regex pattern to find the number between '_' and '.png'
    # This is more robust than splitting, handling cases like camera0_2_extra.png if needed, 
    # but relies on the last number being the index.
    # We will stick to splitting for simplicity based on the current naming scheme.
    
    parts = filename.split('_')
    if len(parts) > 1:
        index_part_ext = parts[-1] # e.g., '19.png'
        try:
            # Get the number before the extension
            index_str = index_part_ext.split('.')[0]
            return int(index_str)
        except ValueError:
            pass # Not a number
    
    return -1

def reorder_frames_for_camera(camera_name, frames_dir='frames'):
    """
    Finds all existing image files for a given camera name, sorts them numerically,
    and renames them sequentially from 0 up to N.
    """
    print(f"\n--- Processing {camera_name} files ---")
    
    # 1. Find all files matching the pattern
    search_pattern = os.path.join(frames_dir, f'{camera_name}_*.png')
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print(f"No files found for {camera_name} in {frames_dir}.")
        return

    # 2. Extract index and create a sortable list of (filepath, index)
    sortable_files = []
    for filepath in all_files:
        index = extract_numerical_index(filepath)
        if index != -1:
            sortable_files.append((filepath, index))
        else:
            print(f"[WARN] Skipping file with unparsable index: {os.path.basename(filepath)}")
            
    # 3. Sort the list numerically based on the extracted index
    # We sort by the integer index (item[1])
    sortable_files.sort(key=lambda item: item[1])

    print(f"Found {len(sortable_files)} files to reorder (highest existing index: {sortable_files[-1][1] if sortable_files else 'N/A'}).")

    # 4. Rename files sequentially
    renamed_count = 0
    for new_index, (old_filepath, old_index) in enumerate(sortable_files):
        # Construct the new sequential name
        new_filename = f'{camera_name}_{new_index}.png'
        new_filepath = os.path.join(frames_dir, new_filename)
        
        # Only rename if the name is different (to avoid unnecessary disk writes)
        if old_filepath != new_filepath:
            try:
                os.rename(old_filepath, new_filepath)
                print(f"Renamed {os.path.basename(old_filepath)} -> {new_filename}")
                renamed_count += 1
            except OSError as e:
                print(f"[ERROR] Could not rename {os.path.basename(old_filepath)}: {e}")
        else:
            # File is already correctly named (e.g., camera0_0.png)
            pass

    print(f"Finished. Total files for {camera_name}: {len(sortable_files)}. Renamed: {renamed_count}.")


def main():
    # Ensure the frames directory exists
    if not os.path.isdir('frames'):
        print("[ERROR] 'frames/' directory not found. Please run capture script first.")
        sys.exit(1)
        
    # Process both cameras
    reorder_frames_for_camera('camera0')
    reorder_frames_for_camera('camera1')

    print("\n[DONE] All camera frames have been reindexed sequentially starting from _0.")
    print("You can now run the calibration script again.")

if __name__ == '__main__':
    main()