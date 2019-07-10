import cv2
import keras
from functions import distance
from functions import video_to_frames
from functions import detect_blue_and_green_line
from functions import detect_number_regions
from functions import region_center
from functions import predict_number

model = keras.models.load_model("keras_mnist.h5")

''' each region has 3 parameters
 first for coordinates
 second for crossing blue line
 third for crossing green line'''


class Box:
    def __init__(self, c_x, c_y, b_line, g_line):
        self.c_x = c_x
        self.c_y = c_y
        self.b_line = b_line
        self.g_line = g_line


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

    previous_frame = []

    for frame in frames:

        current_frame = []

        region_counter = 0
        regions, regions_coordinates = detect_number_regions(frame)

        if frames_counter == 0:
            blue_and_green_line = detect_blue_and_green_line(frame)

            blue_line = blue_and_green_line[0]
            x_blue_min = blue_line[2]
            x_blue_max = blue_line[3]

            green_line = blue_and_green_line[1]
            x_green_min = green_line[2]
            x_green_max = green_line[3]

        for region in regions:
            if frames_counter == 0:
                current_frame.append(Box(regions_coordinates[region_counter][0], regions_coordinates[region_counter][1], False, False))
            else:
                for previous_region in previous_frame:
                    if previous_region.c_x - 3 <= regions_coordinates[region_counter][0] <= previous_region.c_x + 3:
                        current_frame.append(Box(regions_coordinates[region_counter][0], regions_coordinates[region_counter][1], previous_region.b_line, previous_region.g_line))
                    else:
                        current_frame.append(Box(regions_coordinates[region_counter][0], regions_coordinates[region_counter][1], False, False))

            for current_region in current_frame:
                if current_region.c_x - 3 <= regions_coordinates[region_counter][0] <= current_region.c_x + 3:
                    center = region_center(regions_coordinates[region_counter])
                    if not current_region.b_line:
                        # check if number crossed blue line
                        distance_from_blue = distance(center, blue_line[0], blue_line[1])
                        if x_blue_min <= center[0] <= x_blue_max and distance_from_blue < 10 and center[1] > blue_line[0] * center[0] + blue_line[1]:
                            current_region.b_line = True
                            sum += predict_number(model, region)

                    if not current_region.g_line:
                        # check if number crossed green line
                        distance_from_green = distance(center, green_line[0], green_line[1])
                        if x_green_min <= center[0] <= x_green_max and distance_from_green < 10 and center[1] > green_line[0] * center[0] + green_line[1]:
                            sum -= predict_number(model, region)
                            current_region.g_line = True

            region_counter += 1
        frames_counter += 1
        previous_frame = current_frame

    print(sum)
