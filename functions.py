import cv2
import sys
import numpy as np
import math


def video_to_frames(source):
    video = cv2.VideoCapture(source)

    if not video.isOpened():
        print('Cannot open video!')
        sys.exit()

    frames = []
    success, image = video.read()
    # success = True
    while success:
        success, image = video.read()
        frames.append(image)

    video.release()
    return frames


def edges(image):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3))
    erode_image = cv2.erode(gray_scale, kernel, iterations=1)
    return cv2.Canny(erode_image, 75, 150)


def line(image, lower, upper):
    mask = cv2.inRange(image, lower, upper)
    line_image = cv2.bitwise_and(image, image, mask=mask)
    return cv2.HoughLinesP(edges(line_image), 1, np.pi / 180, 30, maxLineGap=30)


def detect_blue_and_green_line(start_image):
    blue_and_green = []

    # detecting blue line
    blue_lower = np.array([200, 0, 0], dtype="uint8")
    blue_upper = np.array([255, 100, 100], dtype="uint8")
    blue_lines = line(start_image, blue_lower, blue_upper)
    blue_and_green.append(equation_of_line(blue_lines))

    # detecting green line
    green_lower = np.array([0, 200, 0], dtype="uint8")
    green_upper = np.array([100, 255, 100], dtype="uint8")
    green_lines = line(start_image, green_lower, green_upper)

    blue_and_green.append(equation_of_line(green_lines))

    return blue_and_green


def equation_of_line(lines):
    k = 0
    n = 0

    # range of x, range of y = [x_min*k + n, x_max*k + n]
    x_min = sys.maxsize
    x_max = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 < x_min:
                x_min = x1
            if x2 > x_max:
                x_max = x2
            k_temp = (y2 - y1) / (x2 - x1)
            k += k_temp
            n_temp = y1 - k_temp * x1
            n += n_temp

    return k / len(lines), n / len(lines), x_min, x_max


def detect_number_regions(image):

    region_coordinates = []
    number_regions = []

    if type(image) is np.ndarray:

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        invert_thresh = 255 - thresh
        contours, hierarchy = cv2.findContours(invert_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 10 < w < 50 and 10 < h < 50:
                region = invert_thresh[y:y + h + 1, x:x + w + 1]
                region_coordinates.append([x, y, w, h])
                number_regions.append(cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST))

    return [number_regions, region_coordinates]


def distance(point, k, n):
    return abs(k * point[0] - point[1] + n) / math.sqrt(k * k + 1)


def region_center(region):
    x, y, w, h = region
    x_center = x + w / 2
    y_center = y + h / 2
    return [x_center, y_center]


def predict_number(model, region):
    region = region / 255
    region = region.flatten()
    return model.predict_classes(np.array([region]))

