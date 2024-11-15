import numpy as np
import mediapy
import PIL
import matplotlib.pyplot as plt

def read_video(path: str):
    return mediapy.read_video(path, output_format='gray')

def equalize_histogram(frame):
    hist, bins = np.histogram(frame, bins=256, range=(0, 256))
    cdf = hist.cumsum()
    c_m = cdf[(cdf != 0).argmax()]
    cdf = np.round((255 * (cdf - c_m)) / (cdf[-1] - c_m))
    frame_equalized = np.interp(frame, bins[:-1], cdf)
    return frame_equalized.astype('uint8')

def match_histogram(origin_hist, target_hist, target_frame):
    origin_cdf = origin_hist.cumsum()
    target_cdf = target_hist.cumsum()

    t = (np.array([origin_cdf, ]*256) - np.array([target_cdf, ]*256).T).T
    t = np.where(t > 0, t, np.inf).argmin(axis=1)
    frame_equalized = np.interp(target_frame, list(range(0,256)), t)
    return frame_equalized.astype('uint8')

def split_frame(frame, factor):
    sub_height = frame.shape[0] / factor
    sub_width = frame.shape[1] / factor
    sub_frames = []
    for i in range(factor):
        sub_frames.append(frame[int(i*sub_height):int((i+1)*sub_height), int(i*sub_width):int((i+1)*sub_width)])
    return sub_frames

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
    votes = 0
    max_frames_hist = None

    frame1_hist = get_frame_cdf(video[0])
    for idx, frame in enumerate(video[1:]):

        frame2_hist = get_frame_cdf(frame)
        diff = np.abs((frame1_hist - frame2_hist)).sum()
        if diff > max_val:
            max_val = diff
            max_frames = (idx, idx+1)

        frame1_hist = frame2_hist

    import os
    PIL.Image.fromarray(video[max_frames[0]]).save(f"{os.path.basename(video_path)}_frame1.png")
    PIL.Image.fromarray(video[max_frames[1]]).save(f"{os.path.basename(video_path)}_frame2.png")
    #PIL.Image.fromarray(match_histogram(max_frames_hist[0], max_frames_hist[1], video[max_frames[1]])).save("vid_match_frame2.png")

    print("Max frames: ", max_frames)

    # plt.bar(range(0, 256), max_frames_hist[0], label='Frame 1')
    # plt.savefig("hist_eq_frame1.png")
    # plt.clf()

    # plt.bar(range(0, 256), np.histogram(video[max_frames[0]], bins=256, range=(0, 256))[0], label='Frame 1')
    # plt.savefig("hist_frame1.png")
    # plt.clf()

    # plt.bar(range(0, 256), np.cumsum(np.histogram(video[max_frames[0]], bins=256, range=(0, 256))[0]), label='Frame 1')
    # plt.savefig("cdf_frame1.png")
    # plt.clf()

    # plt.bar(range(0, 256), max_frames_hist[1], label='Frame 2')
    # plt.savefig("hist_eq_frame2.png")
    # plt.clf()

    # plt.bar(range(0, 256), np.cumsum(max_frames_hist[1]), label='Frame 2')
    # plt.savefig("cdf_eq_frame2.png")
    # plt.clf()

    # plt.bar(range(0, 256), np.histogram(video[max_frames[1]], bins=256, range=(0, 256))[0], label='Frame 1')
    # plt.savefig("hist_frame2.png")
    # plt.clf()

    # plt.bar(range(0, 256), np.cumsum(np.histogram(video[max_frames[1]], bins=256, range=(0, 256))[0]), label='Frame 2')
    # plt.savefig("cdf_frame2.png")
    # plt.clf()
    
if __name__ == '__main__':
    main('videos/video1_category1.mp4', 1)
    main('videos/video2_category1.mp4', 1)
    main('videos/video3_category2.mp4', 1)
    main('videos/video4_category2.mp4', 1)