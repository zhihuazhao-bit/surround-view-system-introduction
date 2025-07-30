import cv2
import os
from datetime import datetime

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # 设置摄像头分辨率（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 创建保存图片的文件夹
    save_dir = "captured_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    photo_count = 0
    
    print("摄像头已启动！")
    print("按 'a' 键拍照")
    print("按 'q' 键退出")
    
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        
        if not ret:
            print("错误：无法读取摄像头数据")
            break
        
        # 在窗口中显示实时画面
        cv2.imshow('Camera - Press A to capture, Q to quit', frame)
        
        # 等待按键
        key = cv2.waitKey(1) & 0xFF
        
        # 按 'a' 键拍照
        if key == ord('a') or key == ord('A'):
            # 生成文件名（使用时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename = f"left.jpg"
            filename = f"{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)
            
            # 保存图片
            cv2.imwrite(filepath, frame)
            photo_count += 1
            
            print(f"照片已保存：{filepath}")
            
            # 显示拍照效果（屏幕闪烁）
            flash_frame = frame.copy()
            cv2.rectangle(flash_frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
            cv2.imshow('Camera - Press A to capture, Q to quit', flash_frame)
            cv2.waitKey(100)  # 显示白色闪烁效果100ms
        
        # 按 'q' 键退出
        elif key == ord('q') or key == ord('Q'):
            break
    
    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
    print(f"程序结束，共拍摄了 {photo_count} 张照片")

if __name__ == "__main__":
    main()