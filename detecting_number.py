from functions import detect_number_contour
from functions import video_to_frames

frames = video_to_frames('video-0.avi')

for frame in frames:
    detect_number_contour(frame)



