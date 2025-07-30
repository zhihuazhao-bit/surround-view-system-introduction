"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Manually select points to get the projection map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import argparse
import os
import numpy as np
import cv2
from surround_view import FisheyeCameraModel, PointSelector, display_image
import surround_view.param_settings as settings


def get_projection_map(camera_model, image):
    und_image = camera_model.undistort(image)
    name = camera_model.camera_name
    gui = PointSelector(und_image, title=name)
    dst_points = settings.project_keypoints[name]
    proj_image = camera_model.project(und_image)
    ret = display_image("Bird's View", proj_image)

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-camera", 
                        default="front",
                        # required=True,
                        choices=["front", "back", "left", "right"],
                        help="The camera view to be projected")
    parser.add_argument("-scale", nargs="+", default=None,
                        help="scale the undistorted image")
    parser.add_argument("-shift", nargs="+", default=None,
                        help="shift the undistorted image")
    args = parser.parse_args()

    if args.scale is not None:
        scale = [float(x) for x in args.scale]
    else:
        scale = (1.0, 1.0)

    if args.shift is not None:
        shift = [float(x) for x in args.shift]
    else:
        shift = (0, 0)

    camera_name = args.camera
    camera_file = os.path.join(os.getcwd(), "yaml", camera_name + ".yaml")
    image_file = os.path.join(os.getcwd(), "images", camera_name + ".jpg")
    image = cv2.imread(image_file)
    camera = FisheyeCameraModel(camera_file, camera_name)
    camera.set_scale_and_shift(scale, shift)
    und_image = camera.undistort(image)
    
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Undistorted Image", und_image)
    # cv2.imshow("Projected Image", camera.project(und_image))
    # cv2.waitKey(0)
    cap = cv2.VideoCapture(0)
    # 设置摄像头分辨率（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (640, 480))
        if not ret:
            print("Error: Could not read frame from video stream.")
            return
        und_image = camera.undistort(frame)
        proj_image = camera.project(und_image)
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Undistorted Image", und_image)
        cv2.imshow("Projected Image", proj_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            # Save the projection map
            output_file = os.path.join(os.getcwd(), "output", camera_name + "_projection_map.png")
            cv2.imwrite(output_file, proj_image)
            print(f"Projection map saved to {output_file}")


if __name__ == "__main__":
    main()
