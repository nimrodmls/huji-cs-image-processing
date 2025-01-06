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
        accumulated_transform = np.dot(accumulated_transform, np.linalg.inv(transform))
        
        # Removing both rotation and y-translation
        accumulated_transform[0, 1] = 0
        accumulated_transform[1, 0] = 0
        accumulated_transform[1, 2] = 0

        stabilized_transformations.append(accumulated_transform[:2])
    
    return stabilized_transformations

def warp_frame(frame, transformation_matrix):
    """
    Warp the frame using the transformation matrix
    """
    return cv2.warpAffine(
        frame, transformation_matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)


def read_video(path: str):
    return mediapy.read_video(path)

def debug_create_still_video(image):
    """
    Creating a still video from an single image, consisting of 100 frames
    """
    return np.array([image for _ in range(100)])

def debug_create_translated_video(image):
    """
    Creating a still video from an single image, consisting of 100 frames
    introducing a translation between each frame
    """
    video = []
    translation = 0
    alt = False
    for i in range(100):
        new_image = np.zeros_like(image)
        if alt:
            translation -= 1
        else:
            translation += 1
        if translation == 4:
            alt = True
        elif translation == 1:
            alt = False
        new_image[translation:image.shape[0]+translation, :image.shape[1]] = image[:image.shape[0]-translation, :]
        video.append(new_image)
    return np.array(video)

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
    # video = debug_create_still_tilted_video(mediapy.read_image("gus-fring.png"))
    # video = debug_create_translated_video(mediapy.read_image("gus-fring.png"))
    # mediapy.write_video("gus-fring.mp4", video)
    video = read_video(BOAT_INPUT_VIDEO_PATH)
    
    # Iterate over consecutive pairs of frames
    transformations = []
    for idx, frame in enumerate(video[:-1]):
        frame1 = video[idx]
        frame2 = video[idx+1]

        # Calculate the homography matrix
        homography_matrix = calculate_transformation(frame1, frame2, max_features=500)
        transformations.append(homography_matrix)

    # Stabilize the transformations
    transformations = stabilize_transformations(transformations)
    # new_frames = stabilize_video(video, transformations)
    # mediapy.write_video("gus-fring-stabilized.mp4", new_frames)

    # Applying the last transformation to the corners of a frame, to get the size of the mosaic
    frame = video[0]
    h, w = frame.shape[:2]
    corner = transformations[-1] @ [w, h, 1]
    mosaic_width = int(corner[0])
    mosaic_height = int(corner[1])

    # Create the mosaic by warping all the frames
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    prev_corners = [w, h, 1]
    current_offset_x = 0
    for i, (frame, transformation) in enumerate(zip(video[1:], transformations)):
        warped_frame = cv2.warpAffine(frame, transformation, (mosaic_width, mosaic_height))
        new_corners = transformation @ [w, h, 1]
        diff = new_corners - prev_corners
        # Now we take only a strip in the width of the difference
        # mosaic = cv2.add(mosaic, np.where(np.logical_and(warped_frame, prev_warped_frame), warped_frame, 0))
        mosaic[:, current_offset_x:current_offset_x + int(diff[0])] = warped_frame[:, current_offset_x:current_offset_x + int(diff[0])]
        current_offset_x += int(diff[0])
        # mosaic[warped_frame != 0] = warped_frame[warped_frame != 0]
        prev_corners = new_corners

    # Write the mosaic to an image
    mediapy.write_image(f"boat-mosaic.jpg", mosaic)

    # Warp all the frames using the stabilized transformations
    # warped_frames = [warp_frame(frame, transformation) for frame, transformation in zip(video[1:], transformations)]

    # Write the warped frames to a video
    # mediapy.write_video("boat-stabilized.mp4", warped_frames)

    

if __name__ == '__main__':
    main()