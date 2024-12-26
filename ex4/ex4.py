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
import cv2
import numpy as np
import matplotlib.pyplot as plt

BOAT_INPUT_VIDEO_PATH = "inputs\\boat.mp4"

def calculate_transformation(frame1, frame2, max_features=100):
    """
    Calculate the transformation matrix between two frames
    using the SIFT point correspondences
    :param max_features: maximum number of features to detect (number of keypoints)
                         only the best features are used
    """
    sift = cv2.SIFT_create(nfeatures=max_features)
    f1_kp, f1_desc = sift.detectAndCompute(frame1, None)
    f2_kp, f2_desc = sift.detectAndCompute(frame2, None)

    feature_matcher = cv2.BFMatcher()
    matches = feature_matcher.knnMatch(f1_desc, f2_desc, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    f1_points = np.float32([f1_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    f2_points = np.float32([f2_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography_matrix, mask = cv2.findHomography(f1_points, f2_points, cv2.RANSAC, 5.0)

    # Matches visualization
    # img = cv2.drawMatches(frame1, f1_kp, frame2, f2_kp, good_matches, None,
    #                       matchColor=(0, 255, 0), singlePointColor=None, flags=2, matchesMask=mask.ravel().tolist())
    # plt.imshow(img)
    # plt.show()

    return homography_matrix 

def read_video(path: str):
    return mediapy.read_video(path)

def debug_create_still_video(image):
    """
    Creating a still video from an single image, consisting of 100 frames
    """
    return np.array([image for _ in range(100)])

def main():
    # video = debug_create_still_video(mediapy.read_image("gus-fring.png"))
    video = read_video(BOAT_INPUT_VIDEO_PATH)
    
    # Iterate over consecutive pairs of frames
    for idx, frame in enumerate(video[:-1]):
        frame1 = video[idx]
        frame2 = video[idx+1]

        # Calculate the homography matrix
        homography_matrix = calculate_transformation(frame1, frame2, max_features=200)

        # Apply the homography matrix to the second frame
        # to align it with the first frame
        pass

if __name__ == '__main__':
    main()