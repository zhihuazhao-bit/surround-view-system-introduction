import os
import numpy as np
import cv2
from PIL import Image
import surround_view.param_settings as settings
from surround_view import FisheyeCameraModel
from surround_view.birdview import FI, FII, FM, BIII, BIV, BM, LI, LIII, LM, RII, RIV, RM
from surround_view import utils


class StaticBirdView:
    def __init__(self):
        # 初始化鸟瞰图图像
        self.image = np.zeros((settings.total_h, settings.total_w, 3), np.uint8)
        self.weights = None
        self.masks = None
        self.car_image = settings.car_image
        self.frames = None
        
        # 从param_settings导入关键参数
        self.xl = settings.xl
        self.xr = settings.xr
        self.yt = settings.yt
        self.yb = settings.yb

    def load_camera_models(self, camera_files, camera_names):
        """加载相机模型"""
        self.camera_models = []
        for camera_file, camera_name in zip(camera_files, camera_names):
            camera_model = FisheyeCameraModel(camera_file, camera_name)
            self.camera_models.append(camera_model)

    def load_images_and_process(self, image_files):
        """加载图像并进行处理"""
        processed_images = []
        
        for image_file, camera_model in zip(image_files, self.camera_models):
            # 读取图像
            img = cv2.imread(image_file)
            if img is None:
                raise ValueError(f"Cannot load image: {image_file}")
            
            # 处理图像：畸变矫正 -> 投影变换 -> 翻转
            img = camera_model.undistort(img)
            img = camera_model.project(img)
            img = camera_model.flip(img)
            processed_images.append(img)
        
        self.frames = processed_images

    def load_weights_and_masks(self, weights_image, masks_image):
        """加载权重和掩码矩阵"""
        # 加载权重矩阵
        GMat = np.asarray(Image.open(weights_image).convert("RGBA"), dtype=np.float32) / 255.0
        self.weights = [np.stack((GMat[:, :, k], GMat[:, :, k], GMat[:, :, k]), axis=2)
                        for k in range(4)]

        # 加载掩码矩阵
        Mmat = np.asarray(Image.open(masks_image).convert("RGBA"), dtype=np.float32)
        Mmat = utils.convert_binary_to_bool(Mmat)
        self.masks = [Mmat[:, :, k] for k in range(4)]

    def merge(self, imA, imB, k):
        """融合两个图像"""
        G = self.weights[k]
        return (imA * G + imB * (1 - G)).astype(np.uint8)

    # 定义鸟瞰图各个区域的属性
    @property
    def FL(self):
        return self.image[:self.yt, :self.xl]

    @property
    def F(self):
        return self.image[:self.yt, self.xl:self.xr]

    @property
    def FR(self):
        return self.image[:self.yt, self.xr:]

    @property
    def BL(self):
        return self.image[self.yb:, :self.xl]

    @property
    def B(self):
        return self.image[self.yb:, self.xl:self.xr]

    @property
    def BR(self):
        return self.image[self.yb:, self.xr:]

    @property
    def L(self):
        return self.image[self.yt:self.yb, :self.xl]

    @property
    def R(self):
        return self.image[self.yt:self.yb, self.xr:]

    @property
    def C(self):
        return self.image[self.yt:self.yb, self.xl:self.xr]

    def stitch_all_parts(self):
        """拼接所有部分生成鸟瞰图"""
        front, back, left, right = self.frames
        
        # 复制中央区域
        np.copyto(self.F, FM(front))
        np.copyto(self.B, BM(back))
        np.copyto(self.L, LM(left))
        np.copyto(self.R, RM(right))
        
        # 融合并复制角落区域
        np.copyto(self.FL, self.merge(FI(front), LI(left), 0))
        np.copyto(self.FR, self.merge(FII(front), RII(right), 1))
        np.copyto(self.BL, self.merge(BIII(back), LIII(left), 2))
        np.copyto(self.BR, self.merge(BIV(back), RIV(right), 3))

    def copy_car_image(self):
        """复制车辆图像到中央"""
        np.copyto(self.C, self.car_image)

    def make_luminance_balance(self):
        """亮度平衡调整"""
        def tune(x):
            if x >= 1:
                return x * np.exp((1 - x) * 0.5)
            else:
                return x * np.exp((1 - x) * 0.8)

        front, back, left, right = self.frames
        m1, m2, m3, m4 = self.masks
        
        # 分离颜色通道
        Fb, Fg, Fr = cv2.split(front)
        Bb, Bg, Br = cv2.split(back)
        Lb, Lg, Lr = cv2.split(left)
        Rb, Rg, Rr = cv2.split(right)

        # 计算亮度比率
        a1 = utils.mean_luminance_ratio(RII(Rb), FII(Fb), m2)
        a2 = utils.mean_luminance_ratio(RII(Rg), FII(Fg), m2)
        a3 = utils.mean_luminance_ratio(RII(Rr), FII(Fr), m2)

        b1 = utils.mean_luminance_ratio(BIV(Bb), RIV(Rb), m4)
        b2 = utils.mean_luminance_ratio(BIV(Bg), RIV(Rg), m4)
        b3 = utils.mean_luminance_ratio(BIV(Br), RIV(Rr), m4)

        c1 = utils.mean_luminance_ratio(LIII(Lb), BIII(Bb), m3)
        c2 = utils.mean_luminance_ratio(LIII(Lg), BIII(Bg), m3)
        c3 = utils.mean_luminance_ratio(LIII(Lr), BIII(Br), m3)

        d1 = utils.mean_luminance_ratio(FI(Fb), LI(Lb), m1)
        d2 = utils.mean_luminance_ratio(FI(Fg), LI(Lg), m1)
        d3 = utils.mean_luminance_ratio(FI(Fr), LI(Lr), m1)

        # 计算调整参数
        t1 = (a1 * b1 * c1 * d1)**0.25
        t2 = (a2 * b2 * c2 * d2)**0.25
        t3 = (a3 * b3 * c3 * d3)**0.25

        x1 = t1 / (d1 / a1)**0.5
        x2 = t2 / (d2 / a2)**0.5
        x3 = t3 / (d3 / a3)**0.5

        x1 = tune(x1)
        x2 = tune(x2)
        x3 = tune(x3)

        Fb = utils.adjust_luminance(Fb, x1)
        Fg = utils.adjust_luminance(Fg, x2)
        Fr = utils.adjust_luminance(Fr, x3)

        y1 = t1 / (b1 / c1)**0.5
        y2 = t2 / (b2 / c2)**0.5
        y3 = t3 / (b3 / c3)**0.5

        y1 = tune(y1)
        y2 = tune(y2)
        y3 = tune(y3)

        Bb = utils.adjust_luminance(Bb, y1)
        Bg = utils.adjust_luminance(Bg, y2)
        Br = utils.adjust_luminance(Br, y3)

        z1 = t1 / (c1 / d1)**0.5
        z2 = t2 / (c2 / d2)**0.5
        z3 = t3 / (c3 / d3)**0.5

        z1 = tune(z1)
        z2 = tune(z2)
        z3 = tune(z3)

        Lb = utils.adjust_luminance(Lb, z1)
        Lg = utils.adjust_luminance(Lg, z2)
        Lr = utils.adjust_luminance(Lr, z3)

        w1 = t1 / (a1 / b1)**0.5
        w2 = t2 / (a2 / b2)**0.5
        w3 = t3 / (a3 / b3)**0.5

        w1 = tune(w1)
        w2 = tune(w2)
        w3 = tune(w3)

        Rb = utils.adjust_luminance(Rb, w1)
        Rg = utils.adjust_luminance(Rg, w2)
        Rr = utils.adjust_luminance(Rr, w3)

        # 合并调整后的通道
        self.frames = [cv2.merge((Fb, Fg, Fr)),
                       cv2.merge((Bb, Bg, Br)),
                       cv2.merge((Lb, Lg, Lr)),
                       cv2.merge((Rb, Rg, Rr))]
        return self

    def make_white_balance(self):
        """白平衡调整"""
        self.image = utils.make_white_balance(self.image)

    def generate_birdview(self):
        """生成鸟瞰图的主函数"""
        # 执行处理流程
        self.make_luminance_balance().stitch_all_parts()
        self.make_white_balance()
        self.copy_car_image()
        return self.image


def main():
    # 设置路径
    current_dir = os.getcwd()
    yaml_dir = os.path.join(current_dir, "yaml")
    image_dir = os.path.join(current_dir, "images")
    
    # 相机名称和文件
    camera_names = settings.camera_names  # ['front', 'back', 'left', 'right']
    camera_files = [os.path.join(yaml_dir, name + ".yaml") for name in camera_names]
    image_files = [os.path.join(image_dir, name + ".jpg") for name in camera_names]
    
    # 权重和掩码文件
    weights_file = os.path.join(current_dir, "weights.png")
    masks_file = os.path.join(current_dir, "masks.png")
    
    # 检查文件是否存在
    for file in image_files + camera_files:
        if not os.path.exists(file):
            print(f"Missing file: {file}")
            return
    
    if not os.path.exists(weights_file) or not os.path.exists(masks_file):
        print("Weights or masks file not found. Please run run_get_weight_matrices.py first.")
        return

    try:
        # 创建静态鸟瞰图对象
        birdview = StaticBirdView()
        
        # 加载相机模型
        birdview.load_camera_models(camera_files, camera_names)
        
        # 加载并处理图像
        birdview.load_images_and_process(image_files)
        
        # 加载权重和掩码
        birdview.load_weights_and_masks(weights_file, masks_file)
        
        # 生成鸟瞰图
        result_image = birdview.generate_birdview()
        
        # 显示结果
        cv2.imshow("BirdView from Static Images", result_image)
        print("Press any key to save and exit...")
        cv2.waitKey(0)
        
        # 保存结果
        cv2.imwrite("static_birdview_result.jpg", result_image)
        print("Result saved as static_birdview_result.jpg")
        
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()