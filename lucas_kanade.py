import argparse
from pathlib import Path

import cv2
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                                The example file can be downloaded from: \
                                                https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
    parser.add_argument('--video', type=str, help='path to video file')
    return parser.parse_args()


def gen_video_writer(filename: str, size: tuple[int, int], fps: int = 24):
    return cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"MP4V"), fps, size)


if __name__ == "__main__":
    args = get_args()
    cap = cv2.VideoCapture(args.video)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file_path = output_dir / (Path(args.video).stem + "_lucas_kanade.mp4")
    video_size = ((int(cap.get(3)), int(cap.get(4))))
    fps = int(cap.get(5))
    print(video_size, fps)
    video_writer = gen_video_writer(str(output_file_path), video_size, fps)

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # params for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frames grabbed!")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        video_writer.write(img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    video_writer.release()
    cv2.destroyAllWindows()
