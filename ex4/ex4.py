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

def generate_panorama_from_strips(frames, start_x, strip_x, transform_diffs, canvas_dim):
    """
    Generating a panorama from the given frames, by taking strips from each frame
    on the specified strip_x coordinate, with respect to the transformation differences
    """
    # Initializing the panorama canvas (adding third dimension for RGB channels)
    panorama_canvas = np.zeros((*canvas_dim, 3), dtype=np.uint8)

    curr_ptr = start_x
    for i, (frame, transform_diff) in enumerate(zip(frames, transform_diffs)):
        
        # Ignoring negative translations
        if transform_diff < 0:
            continue

        if curr_ptr + transform_diff >= canvas_dim[1]:
            break

        if transform_diff == 0:
            continue

        # Stitching the strip from the current frame to the panorama canvas
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

def main():
    is_right_to_left = False
    video = read_video(BOAT_INPUT_VIDEO_PATH)

    # If the video is going from right to left, we need to reverse the frames
    # to allow the algorithm to work properly
    if is_right_to_left:
        video = video[::-1]
    
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
    mediapy.write_video('boat_warped.mp4', warped_frames)
    
    # Rounding the x-axis transformations & converting to integers, to allow
    # proper slicing of strips into the panorama
    x_transforms = np.round(x_transforms).astype(int)

    panoramas_cnt = 20
    strip_pts = np.linspace(0, video[0].shape[1] - max(x_transforms), panoramas_cnt, dtype=int)
    start_x = 0
    panoramas = []

    with mediapy.VideoWriter('boat_stereo.mp4', shape=(canvas_height, canvas_width), fps=10) as writer:
        # Calculating and adding the forward panoramas
        for strip_x in strip_pts:
            pano = generate_panorama_from_strips(
                warped_frames, start_x, strip_x, x_transforms, (canvas_height, canvas_width))
            start_x += max(x_transforms)
            panoramas.append(pano)
            writer.add_image(pano)
        # Adding the backward panoramas
        panoramas.reverse()
        for pano in panoramas[1:]:
            writer.add_image(pano)

if __name__ == '__main__':
    main()