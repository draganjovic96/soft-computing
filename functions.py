import cv2
import sys


def video_to_frames(source):

    vidcap = cv2.VideoCapture(source)

    if not vidcap.isOpened():
        print('Cannot open video!')
        sys.exit()

    frames = []
    success, image = vidcap.read()
    success = True
    while success:
        success, image = vidcap.read()
        frames.append(image)

    return frames

