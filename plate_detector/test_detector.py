import cv2
import numpy as np
from pathlib import Path
from detector import CLPDetector  
from utility import clp_det_model
from datetime import datetime
import argparse

def main(input_dir):
    # 加载模型路径
    model_path = clp_det_model
    detector = CLPDetector(model_path)

    # 获取输入目录中的所有图像文件
    image_paths = list(Path(input_dir).glob('*.*'))  # 获取目录下所有文件
    if not image_paths:
        print(f"No images found in the directory: {input_dir}")
        return

    # 创建保存检测车牌的文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(f'runs/detected_plates_{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)

    # 遍历所有图像文件并进行车牌检测
    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read the image: {image_path}")
            continue

        # 进行车牌检测
        detections = detector(image)

        # 打印检测结果
        print(f"Detected plates in {image_path.name}:")
        for det in detections:
            print(det)

        # 保存每个检测出的车牌区域为单独的图片
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det[:4])

            # 提取检测到的车牌区域
            plate_image = image[y1:y2, x1:x2]

            # 检查提取的图像是否有效
            if plate_image.size == 0:
                print(f"Skipped saving plate {idx} due to invalid size.")
                continue

            # 构建保存路径并保存图像
            plate_image_path = save_dir / f'{image_path.stem}_plate_{idx}.jpg'
            cv2.imwrite(str(plate_image_path), plate_image)

            print(f"Saved detected plate {idx} from {image_path.name} to {plate_image_path}")

if __name__ == "__main__":
    # 使用argparse获取输入目录
    parser = argparse.ArgumentParser(description='Car Plate Detection')
    parser.add_argument('input_dir', type=str, help='Directory containing images to process')
    args = parser.parse_args()

    main(args.input_dir)



