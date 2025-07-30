import os
import cv2


camera_names = ["front", "back", "left", "right"]

# --------------------------------------------------------------------
# (shift_width, shift_height): how far away the birdview looks outside
# of the calibration pattern in horizontal and vertical directions
shift_w = 70
shift_h = 100

board_w = 140
board_h = 120

# size of the gap between the calibration pattern and the car
# in horizontal and vertical directions
inn_shift_w = 20
inn_shift_h = 50

car_w = 120
car_h = 300

# total width/height of the stitched image
total_w = car_w + 2 * shift_w + 2 * inn_shift_w + 2 * board_w
total_h = car_h + 2 * shift_h + 2 * inn_shift_h + 2 * board_h

# four corners of the rectangular region occupied by the car
# top-left (x_left, y_top), bottom-right (x_right, y_bottom)
xl = shift_w + board_w + inn_shift_w
xr = total_w - xl
yt = shift_h + board_h + inn_shift_h
yb = total_h - yt
# --------------------------------------------------------------------

project_shapes = {
    "front": (total_w, yt),
    "back":  (total_w, yt),
    "left":  (total_h, xl),
    "right": (total_h, xl)
}

# pixel locations of the four points to be chosen.
# you must click these pixels in the same order when running
# the get_projection_map.py script

#########使用中心棋盘网格标定#############
# front_keypoint_x1 = shift_w + board_w + 10
# front_keypoint_x2 = total_w - front_keypoint_x1
# front_keypoint_y1 = shift_h
# front_keypoint_y2 = shift_h + board_h

# # left and right的横纵坐标是没有旋转前的坐标，与映射图中的x和y翻转了。
# left_keypoint_y1 = shift_w
# left_keypoint_y2 = shift_w + board_w
# left_keypoint_x1 = shift_h + board_h + 120
# left_keypoint_x2 = total_h - left_keypoint_x1

######## 使用角块校准##########
front_keypoint_x1 = shift_w + 90
front_keypoint_x2 = total_w - front_keypoint_x1
front_keypoint_y1 = shift_h + 30
front_keypoint_y2 = shift_h + 90

left_keypoint_y1 = shift_w + 30
left_keypoint_y2 = shift_w + 90
left_keypoint_x1 = shift_h + 90
left_keypoint_x2 = total_h - left_keypoint_x1

project_keypoints = {
    "front": [(front_keypoint_x1, front_keypoint_y1),
              (front_keypoint_x2, front_keypoint_y1),
              (front_keypoint_x1, front_keypoint_y2),
              (front_keypoint_x2, front_keypoint_y2)],

    "back":  [(front_keypoint_x1, front_keypoint_y1),
              (front_keypoint_x2, front_keypoint_y1),
              (front_keypoint_x1, front_keypoint_y2),
              (front_keypoint_x2, front_keypoint_y2)],

    "left":  [(left_keypoint_x1, left_keypoint_y1),
              (left_keypoint_x2, left_keypoint_y1),
              (left_keypoint_x1, left_keypoint_y2),
              (left_keypoint_x2, left_keypoint_y2)],

    "right": [(left_keypoint_x1, left_keypoint_y1),
              (left_keypoint_x2, left_keypoint_y1),
              (left_keypoint_x1, left_keypoint_y2),
              (left_keypoint_x2, left_keypoint_y2)],
}
print("Project keypoints:", project_keypoints)
car_image = cv2.imread(os.path.join(os.getcwd(), "images", "car.png"))
car_image = cv2.resize(car_image, (xr - xl, yb - yt))
