import cv2
import keras
from functions import distance
from functions import video_to_frames
from functions import detect_blue_and_green_line
from functions import detect_number_regions
from functions import region_center
from functions import predict_number

model = keras.models.load_model("keras_mnist.h5")

for video_counter in range(0, 10):

    frames = video_to_frames('video-' + str(video_counter) + '.avi')
    frames_counter = 0
    blue_line = []
    green_line = []

    # x domain of blue line
    x_blue_min = 0
    x_blue_max = 0

    # x domain of green line
    x_green_min = 0
    x_green_max = 0

    sum = 0

    for frame in frames:

        if frames_counter == 0:
            blue_and_green_line = detect_blue_and_green_line(frame)

            blue_line = blue_and_green_line[0]
            x_blue_min = blue_line[2]
            x_blue_max = blue_line[3]

            green_line = blue_and_green_line[1]
            x_green_min = green_line[2]
            x_green_max = green_line[3]

        region_counter = 0
        regions, regions_coordinates = detect_number_regions(frame)
        for region in regions:
            center = region_center(regions_coordinates[region_counter])

            # check if number crossed blue line
            distance_from_blue = distance(center, blue_line[0], blue_line[1])
            if x_blue_min <= center[0] <= x_blue_max and distance_from_blue < 10 and center[1] > blue_line[0] * center[0] + blue_line[1]:
                sum += predict_number(model, region)

            # check if number crossed green line
            distance_from_green = distance(center, green_line[0], green_line[1])
            if x_green_min <= center[0] <= x_green_max and distance_from_green < 10 and center[1] > green_line[0] * center[0] + green_line[1]:
                sum -= predict_number(model, region)

            region_counter += 1
        frames_counter += 1

    print(sum)
