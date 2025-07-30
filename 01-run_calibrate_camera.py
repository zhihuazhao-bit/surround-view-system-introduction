#file:/Users/zhaozhihua/code/surround-view-system-introduction-geo/01-run_calibrate_camera.py
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~
Fisheye Camera calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage:
    python calibrate_camera.py \
        -i 0 \
        -grid 9x6 \
        -out fisheye.yaml \
        -framestep 20 \
        --resolution 640x480
        --fisheye
"""
import argparse
import os
import numpy as np
import cv2
from surround_view import CaptureThread, MultiBufferManager
import surround_view.utils as utils


# we will save the camera param file to this directory
TARGET_DIR = os.path.join(os.getcwd(), "yaml")

# default param file
DEFAULT_PARAM_FILE = os.path.join(TARGET_DIR, "camera_params.yaml")


def main():
    parser = argparse.ArgumentParser()

    # input video stream
    parser.add_argument("-i", "--input", type=int, default=0,
                        help="input camera device")

    # chessboard pattern size
    parser.add_argument("-grid", "--grid", default="6x8",
                        help="size of the calibrate grid pattern")

    parser.add_argument("-r", "--resolution", default="960x640",
                        help="resolution of the camera image")

    parser.add_argument("-framestep", type=int, default=20,
                        help="use every nth frame in the video")
    
    parser.add_argument("-o", "--output", default=DEFAULT_PARAM_FILE,
                        help="path to output yaml file")

    parser.add_argument("-fisheye", "--fisheye", 
                        default=True,
                        # action="store_true",
                        help="set true if this is a fisheye camera")

    parser.add_argument("-flip", "--flip", default=0, type=int,
                        help="flip method of the camera")

    parser.add_argument("--no_gst", 
                        default=True,
                        # action="store_true",
                        help="set true if not use gstreamer for the camera capture")

    args = parser.parse_args()

    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)

    text1 = "press c to calibrate"
    text2 = "press q to quit"
    text3 = "device: {}".format(args.input)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.6

    resolution_str = args.resolution.split("x")
    W = int(resolution_str[0])
    H = int(resolution_str[1])
    grid_size = tuple(int(x) for x in args.grid.split("x"))
    grid_points = np.zeros((1, np.prod(grid_size), 3), np.float32)
    grid_points[0, :, :2] = np.indices(grid_size).T.reshape(-1, 2)

    device = args.input
    cap_thread = CaptureThread(device_id=device,
                               flip_method=args.flip,
                               resolution=(W, H),
                               use_gst=not args.no_gst,
                               )
    buffer_manager = MultiBufferManager()
    buffer_manager.bind_thread(cap_thread, buffer_size=8)
    if cap_thread.connect_camera():
        cap_thread.start()
    else:
        print("cannot open device")
        return

    quit = False
    i = -1
    
    while not quit:
        # Initialize object and image points for each calibration attempt
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane
        
        # Collection phase
        collecting = True
        i = -1
        while collecting:
            i += 1
            img = buffer_manager.get_device(device).get().image
            if i % args.framestep != 0:
                continue

            print("searching for chessboard corners in frame " + str(i) + "...")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                grid_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FILTER_QUADS
            )
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
                print("OK")
                imgpoints.append(corners)
                objpoints.append(grid_points)
                cv2.drawChessboardCorners(img, grid_size, corners, found)

            # Update text with current number of collected points
            text4 = "Collected corners: {}".format(len(objpoints))
            
            cv2.putText(img, text1, (20, 70), font, fontscale, (255, 200, 0), 2)
            cv2.putText(img, text2, (20, 110), font, fontscale, (255, 200, 0), 2)
            cv2.putText(img, text3, (20, 30), font, fontscale, (255, 200, 0), 2)
            cv2.putText(img, text4, (20, 150), font, fontscale, (255, 200, 0), 2)
            cv2.imshow("corners", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                print("\nPerforming calibration...\n")
                N_OK = len(objpoints)
                if N_OK < 12:
                    print("Less than 12 corners (%d) detected, calibration failed" %(N_OK))
                    # Show failure message and continue collecting
                    cv2.putText(img, "Need more points!", (220, 240), font, 2, (0, 0, 255), 2)
                    cv2.imshow("corners", img)
                    cv2.waitKey(1000)  # Show message for 1 second
                else:
                    collecting = False
                    break

            elif key == ord("q"):
                collecting = False
                quit = True
                break

        if quit:
            break

        # Calibration phase
        if len(objpoints) >= 12:  # Only proceed if we have enough points
            N_OK = len(objpoints)
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
            
            try:
                if args.fisheye:
                    calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                                        cv2.fisheye.CALIB_CHECK_COND +
                                        cv2.fisheye.CALIB_FIX_SKEW)
                    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                        objpoints,
                        imgpoints,
                        (W, H),
                        K,
                        D,
                        rvecs,
                        tvecs,
                        calibration_flags,
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                    )
                else:
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                        objpoints,
                        imgpoints,
                        (W, H),
                        None,
                        None)
            except Exception as e:
                print("Calibration failed:", e)
                ret = False

            if ret:
                fs = cv2.FileStorage(args.output, cv2.FILE_STORAGE_WRITE)
                fs.write("resolution", np.int32([W, H]))
                fs.write("camera_matrix", K)
                fs.write("dist_coeffs", D)
                fs.release()
                print("successfully saved camera data")
                
                # Show success message
                img = buffer_manager.get_device(device).get().image
                cv2.putText(img, "Success!", (220, 240), font, 2, (0, 255, 0), 2)
                cv2.imshow("corners", img)
                cv2.waitKey(2000)  # Show success message for 2 seconds
                quit = True  # Exit after successful calibration
            else:
                # Show failure message and continue to collect more points
                img = buffer_manager.get_device(device).get().image
                cv2.putText(img, "Calibration failed!", (120, 240), font, 2, (0, 0, 255), 2)
                cv2.putText(img, "Collecting more points...", (100, 280), font, 1, (0, 255, 255), 2)
                cv2.imshow("corners", img)
                cv2.waitKey(2000)  # Show message for 2 seconds
                # Continue to the next iteration to collect more points
        else:
            # Not enough points to calibrate
            if not quit:
                img = buffer_manager.get_device(device).get().image
                cv2.putText(img, "Not enough points!", (120, 240), font, 2, (0, 0, 255), 2)
                cv2.putText(img, "Press 'q' to quit", (150, 280), font, 1, (0, 255, 255), 2)
                cv2.imshow("corners", img)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    quit = True

    # Cleanup
    cap_thread.stop()
    cap_thread.disconnect_camera()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()