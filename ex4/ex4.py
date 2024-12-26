"""
Exercise 4 - Stero Mosaicing

1. Aligning consecutive frames using Lucas-Kanade method & creating the
transformation matrix for each transition.
2. Stabilizing rotations & translations.
3. Motion composition
4. Creating a mosaic image
5. Setting the convergence point
"""

import mediapy
import numpy as np
from scipy.signal import convolve

BOAT_INPUT_VIDEO_PATH = "inputs\\boat.mp4"

def calculate_homography_matrix(frame1, frame2):
    """
    Calculate the homography matrix for the given frames.
    """
    # Compute derivatives



    # Compute the LK matrix

    # Iterating until convergence
    pass

def read_video(path: str):
    return mediapy.read_video(path)

def main():
    video = read_video(BOAT_INPUT_VIDEO_PATH)
    
    # Iterate over consecutive pairs of frames
    for idx, frame in enumerate(video[:-1]):
        frame1 = video[idx]
        frame2 = video[idx+1]

        # Calculate the homography matrix
        homography_matrix = calculate_homography_matrix(frame1, frame2)

        # Apply the homography matrix to the second frame
        # to align it with the first frame
        pass

if __name__ == '__main__':
    main()