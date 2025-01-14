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

    # We calculate the transformation from the second image to the first image
    # (since we want to warp the second image to the coordinate system of the first image)
    homography_matrix, inliers = cv2.estimateAffinePartial2D(f2_points, f1_points, method=cv2.RANSAC, ransacReprojThreshold=3)
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
    min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
    stabilized_transformations = [np.array([[1, 0, 0], [0, 1, 0]])] # Initializing with identity
    accumulated_transform = np.eye(3)
    x_transforms = []
    
    # Accumulating the transformations
    for transform in transformations:
        transform = np.vstack([transform, [0, 0, 1]]) # Necessary to match the dimensions for mat-mul
        accumulated_transform = np.dot(accumulated_transform, transform)
        
        # Removing the rotations
        accumulated_transform[0, 1] = 0
        accumulated_transform[1, 0] = 0

        # Finding the canvas dimensions, according to the transformations
        min_x = min(min_x, accumulated_transform[0, 2])
        max_x = max(max_x, accumulated_transform[0, 2])
        min_y = min(min_y, accumulated_transform[1, 2])
        max_y = max(max_y, accumulated_transform[1, 2])

        # The x-translations are saved aside (x_transforms), to allow warping 
        # the frames into their correct position on the panorama
        x_transforms.append(transform[0, 2])

        stabilized_transformations.append(accumulated_transform[:2])

    # Stabilizing the transformations
    for transform in stabilized_transformations:

        # Removing x-translations (since we use the inverse, we accumulate the transformations
        # with respect to the first frame, hence we transform all frames to the coordinate system
        # of the first frame).
        transform[0, 2] = 0

        # Adjusting the y-translations to the minimum y-translation
        transform[1, 2] -= min_y

    return stabilized_transformations, x_transforms, (min_x, min_y, max_x, max_y)

def generate_panorama_from_strips(frames, strip_x, transform_diffs, canvas_dim):
    """
    Generating a panorama from the given frames, by taking strips from each frame
    on the specified strip_x coordinate, with respect to the transformation differences
    """
    # Initializing the panorama canvas (adding third dimension for RGB channels)
    panorama_canvas = np.zeros((*canvas_dim, 3), dtype=np.uint8)

    curr_ptr = 0
    for i, (frame, transform_diff) in enumerate(zip(frames, transform_diffs)):
        if transform_diff < 0:
            continue
        panorama_canvas[:, curr_ptr:curr_ptr + transform_diff] = \
            frame[:, strip_x:strip_x + transform_diff]
        curr_ptr += transform_diff

    return panorama_canvas

def warp_frame(frame, transformation_matrix, height, width):
    """
    Warp the frame using the transformation matrix
    """
    return cv2.warpAffine(
        frame, transformation_matrix.astype(np.float32), (width, height), flags=cv2.INTER_LINEAR)

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
        homography_matrix = calculate_transformation(frame1, frame2, max_features=2000)
        transformations.append(homography_matrix)

    # Stabilize the transformations
    transformations, x_transforms, dims = stabilize_transformations(transformations)

    min_x, min_y, max_x, max_y = dims
    x_diff = int(max_x - min_x)
    y_diff = int(max_y - min_y)

    # Calculate the panorama dimensions
    canvas_width = video[0].shape[1] + x_diff
    canvas_height = video[0].shape[0] + y_diff

    # Wraping all the frames in the video, with respect to the first frame of the video
    # (this process will result in warped_frames being populated with all the frames in
    # the video, transformed to the coordinate system of the first frame)
    warped_frames = [warp_frame(frame, transformation, canvas_height, video[0].shape[1])
                      for frame, transformation in zip(video, transformations)]
    
    # Rounding the x-axis transformations & converting to integers, to allow
    # proper slicing of strips into the panorama
    x_transforms = np.round(x_transforms).astype(int)

    pano = generate_panorama_from_strips(
        warped_frames, 0, x_transforms, (canvas_height, canvas_width))
    
    mediapy.write_image("pano.jpg", pano)
    # new_frames = stabilize_video(video, transformations)
    # mediapy.write_video("gus-fring-stabilized.mp4", new_frames)

    # Applying the last transformation to the corners of a frame, to get the size of the mosaic
    # frame = video[0]
    # h, w = frame.shape[:2]
    # corner = transformations[-1] @ [w, h, 1]
    # mosaic_width = int(corner[0])
    # mosaic_height = int(corner[1])

    # Create the mosaic by warping all the frames
    # warped = []
    # mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    # for i, (frame, transformation) in enumerate(zip(video, transformations)):
        # warped_frame = cv2.warpAffine(frame, transformation, (mosaic_width, mosaic_height))
        # warped_frame = warp_frame(frame, transformation)
        # warped.append(warped_frame)
        # mosaic[warped_frame != 0] = warped_frame[warped_frame != 0]
        #cv2.add(mosaic, warped_frame, mask=(warped_frame != 0).all(axis=2))
        # mosaic = cv2.add(mosaic, warped_frame)

    # Write the mosaic to an image
    # mediapy.write_image(f"boat-mosaic.jpg", mosaic)

    # Warp all the frames using the stabilized transformations
    # warped_frames = [warp_frame(frame, transformation) for frame, transformation in zip(video[1:], transformations)]

    # Write the warped frames to a video
    # mediapy.write_video("boat-stabilized.mp4", warped)

    

if __name__ == '__main__':
    main()