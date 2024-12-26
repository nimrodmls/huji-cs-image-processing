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

    homography_matrix, inliers = cv2.estimateAffine2D(f1_points, f2_points, method=cv2.RANSAC)
    # homography_matrix, mask = cv2.findHomography(f1_points, f2_points, cv2.RANSAC, 5.0)

    # Matches visualization
    # img = cv2.drawMatches(frame1, f1_kp, frame2, f2_kp, good_matches, None,
    #                       matchColor=(0, 255, 0), singlePointColor=None, flags=2, matchesMask=inliers.ravel().tolist())
    # plt.imshow(img)
    # plt.show()

    return homography_matrix 

def stabilize_transformations(transformations):
    """
    Stabilize the transformations by removing rotation and y-translation
    (only applicable on videos with horizontal motion)
    """
    stabilized_transformations = []
    accumulated_transform = np.eye(3)
    
    for transform in transformations:
        transform = np.vstack([transform, [0, 0, 1]])
        accumulated_transform = np.dot(accumulated_transform, transform)
        
        # Removing both rotation and y-translation
        accumulated_transform[0, 1] = 0
        accumulated_transform[1, 0] = 0
        stabilized_transformations.append(accumulated_transform[:2])
    
    return stabilized_transformations

def read_video(path: str):
    return mediapy.read_video(path)

def debug_create_still_video(image):
    """
    Creating a still video from an single image, consisting of 100 frames
    """
    return np.array([image for _ in range(100)])

def debug_create_still_tilted_video(image):
    """
    Creating a still video from an single image, consisting of 100 frames
    introducing a 90 degree rotation between each frame
    """
    video = []

    # padding the image to square
    new_image = np.zeros((max(image.shape), max(image.shape), 3), dtype=image.dtype)
    new_image[:image.shape[0], :image.shape[1]] = image
    for _ in range(100):
        video.append(new_image)
        new_image = cv2.rotate(new_image, cv2.ROTATE_90_CLOCKWISE)
    return np.array(video)

def main():
    # video = debug_create_still_video(mediapy.read_image("gus-fring.png"))
    video = debug_create_still_tilted_video(mediapy.read_image("gus-fring.png"))
    mediapy.write_video("gus-fring.mp4", video)
    # video = read_video(BOAT_INPUT_VIDEO_PATH)
    
    # Iterate over consecutive pairs of frames
    transformations = []
    for idx, frame in enumerate(video[:-1]):
        frame1 = video[idx]
        frame2 = video[idx+1]

        # Calculate the homography matrix
        homography_matrix = calculate_transformation(frame1, frame2, max_features=200)
        transformations.append(homography_matrix)

    # Stabilize the transformations
    transformations = stabilize_transformations(homography_matrix)

    

if __name__ == '__main__':
    main()