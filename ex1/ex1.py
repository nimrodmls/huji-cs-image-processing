import numpy as np
import mediapy
import matplotlib.pyplot as plt

def read_video(path: str):
    return mediapy.read_video(path, output_format='gray')

def get_frame_histogram(frame):
    """
    """
    return np.histogram(frame, bins=256, range=(0, 256))[0]

def get_frame_cdf(frame):
    """
    """
    return np.cumsum(get_frame_histogram(frame))

def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    video = read_video(video_path)

    max_frames = None
    max_val = 0

    frame1_hist = get_frame_cdf(video[0])
    for idx, frame in enumerate(video[1:]):

        frame2_hist = get_frame_cdf(frame)
        diff = np.abs((frame1_hist - frame2_hist)).sum()
        if diff > max_val:
            max_val = diff
            max_frames = (idx, idx+1)

        frame1_hist = frame2_hist

    return max_frames
