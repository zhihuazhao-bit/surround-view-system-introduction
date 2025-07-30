import cv2
import numpy as np
import os
import argparse
from surround_view import FisheyeCameraModel

def create_parameter_adjustment_ui(camera_model, image, camera_name):
    """
    创建一个UI界面用于实时调整鱼眼相机的scale_xy和shift_xy参数
    """
    
    # 创建窗口
    window_name = f"{camera_name} - Parameter Adjustment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 初始化参数
    scale_x, scale_y = camera_model.scale_xy[0], camera_model.scale_xy[1]
    shift_x, shift_y = camera_model.shift_xy[0], camera_model.shift_xy[1]
    
    # 创建跟踪栏
    cv2.createTrackbar('Scale X (x100)', window_name, int(scale_x * 100), 200, lambda x: None)
    cv2.createTrackbar('Scale Y (x100)', window_name, int(scale_y * 100), 200, lambda x: None)
    cv2.createTrackbar('Shift X', window_name, shift_x + 100, 200, lambda x: None)
    cv2.createTrackbar('Shift Y', window_name, shift_y + 100, 200, lambda x: None)
    
    original_image = image.copy()
    
    while True:
        # 获取跟踪栏值
        scale_x = cv2.getTrackbarPos('Scale X (x100)', window_name) / 100.0
        scale_y = cv2.getTrackbarPos('Scale Y (x100)', window_name) / 100.0
        shift_x = cv2.getTrackbarPos('Shift X', window_name) - 100
        shift_y = cv2.getTrackbarPos('Shift Y', window_name) - 100
        
        # 更新相机参数
        camera_model.set_scale_and_shift((scale_x, scale_y), (shift_x, shift_y))
        
        # 应用矫正
        undistorted_image = camera_model.undistort(original_image)
        
        # 显示图像
        cv2.imshow(window_name, undistorted_image)
        
        # 显示当前参数
        info_text = f"Scale: ({scale_x:.2f}, {scale_y:.2f}) Shift: ({shift_x}, {shift_y})"
        print(f"\r{info_text}", end='', flush=True)
        
        # 等待按键
        key = cv2.waitKey(30) & 0xFF
        
        # 按 's' 保存参数
        if key == ord('s'):
            camera_model.save_data()
            print(f"\nParameters saved: Scale=({scale_x:.2f}, {scale_y:.2f}), Shift=({shift_x}, {shift_y})")
        
        # 按 'r' 重置参数
        elif key == ord('r'):
            scale_x, scale_y = 1.0, 1.0
            shift_x, shift_y = 0, 0
            cv2.setTrackbarPos('Scale X (x100)', window_name, int(scale_x * 100))
            cv2.setTrackbarPos('Scale Y (x100)', window_name, int(scale_y * 100))
            cv2.setTrackbarPos('Shift X', window_name, shift_x + 100)
            cv2.setTrackbarPos('Shift Y', window_name, shift_y + 100)
            print("\nParameters reset to default")
        
        # 按 'q' 退出
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return scale_x, scale_y, shift_x, shift_y

def main():
    parser = argparse.ArgumentParser(description="UI for adjusting fisheye camera parameters")
    parser.add_argument("-camera", 
                        default="front",
                        choices=["front", "back", "left", "right"],
                        help="The camera to adjust parameters for")
    
    args = parser.parse_args()
    
    # 设置文件路径
    camera_name = args.camera
    camera_file = os.path.join(os.getcwd(), "yaml", camera_name + ".yaml")
    image_file = os.path.join(os.getcwd(), "images", camera_name + ".jpg")
    
    # 检查文件是否存在
    if not os.path.exists(camera_file):
        print(f"Camera parameter file not found: {camera_file}")
        return
    
    if not os.path.exists(image_file):
        print(f"Image file not found: {image_file}")
        return
    
    # 加载图像和相机模型
    image = cv2.imread(image_file)
    if image is None:
        print(f"Failed to load image: {image_file}")
        return
    
    try:
        camera_model = FisheyeCameraModel(camera_file, camera_name)
    except Exception as e:
        print(f"Failed to load camera model: {e}")
        return
    
    print("Parameter Adjustment UI")
    print("Controls:")
    print("  - Trackbars: Adjust scale and shift parameters")
    print("  - 's' key: Save current parameters to file")
    print("  - 'r' key: Reset parameters to default")
    print("  - 'q' key: Quit")
    print("------------------------------------------------")
    
    # 启动UI调整
    final_scale_x, final_scale_y, final_shift_x, final_shift_y = create_parameter_adjustment_ui(
        camera_model, image, camera_name
    )
    
    print(f"\nFinal parameters:")
    print(f"  Scale XY: ({final_scale_x:.2f}, {final_scale_y:.2f})")
    print(f"  Shift XY: ({final_shift_x}, {final_shift_y})")

if __name__ == "__main__":
    main()