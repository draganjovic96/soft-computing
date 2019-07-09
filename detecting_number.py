import functions
import cv2
import keras
import numpy as np

frames = functions.video_to_frames('video-0.avi')

frame = cv2.imread('video0-frame0.jpg')

contours = functions.detect_number_regions(frame)

model = keras.models.load_model("keras_mnist.h5")

for contour in contours:
    print(contour)
    contour = contour / 255
    contour = contour.flatten()
    print(model.predict_classes(np.array([contour])))

