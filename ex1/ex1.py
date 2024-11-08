import numpy as np
import mediapy
import PIL
import matplotlib.pyplot as plt

def read_video(path: str):
    return mediapy.read_video(path, output_format='gray')

def equalize_histogram(frame):
    hist, bins = np.histogram(frame, bins=range(0, 255))
    cdf = hist.cumsum()
    c_m = (cdf != 0).argmax()
    cdf = (255 * cdf * c_m) / (cdf[-1] - c_m)
    frame_equalized = np.interp(frame, bins[:-1], cdf)
    return frame_equalized

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
    max_frames_hist = None

    frame1_hist, _ = np.histogram(video[0], bins=range(0, 255))
    for idx, frame in enumerate(video[1:]):
        frame2_hist, _ = np.histogram(frame, bins=range(0, 255))

        diff = np.abs((frame1_hist - frame2_hist)).sum()
        if diff > max_val:
            max_val = diff
            max_frames = (idx, idx+1)
            max_frames_hist = (frame1_hist, frame2_hist)

        frame1_hist = frame2_hist

    video_rgb = mediapy.read_video(video_path)
    PIL.Image.fromarray(video_rgb[max_frames[0]]).save("frame1.png")
    PIL.Image.fromarray(video_rgb[max_frames[1]]).save("frame2.png")

    plt.hist(max_frames_hist[0], label='Frame 1', bins=range(0, 255))
    plt.savefig("frame1_hist.png")
    plt.clf()

    plt.hist(max_frames_hist[1], label='Frame 2', bins=range(0, 255))
    plt.savefig("frame2_hist.png")
    
if __name__ == '__main__':
    main('videos/video4_category2.mp4', 1)