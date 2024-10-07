import cv2
import numpy as np
from pathlib import Path
from detector import CLPDetector
from utility import clp_det_model
from datetime import datetime
import argparse


def parse_ccpd_filename(filename):
    try:
        # 去除文件扩展名（.jpg）
        filename = filename.split('.')[0]
        # 用 '-' 分割文件名，获得 7 个字段
        fields = filename.split('-')
        # 检查是否有 7 个字段，且第三个字段包含正确的边界框格式
        if len(fields) != 7 or '&' not in fields[2] or '_' not in fields[2]:
            raise ValueError("文件名不是 CCPD 标准格式。")
        # 第三个字段是边界框的坐标，包含左上和右下的两个顶点
        bbox_coords = fields[2]
        # 用 '_' 分割，得到左上角 (x1, y1) 和右下角 (x2, y2) 的坐标
        top_left, bottom_right = bbox_coords.split('_')
        # 进一步将坐标解析为整数
        x1, y1 = map(int, top_left.split('&'))
        x2, y2 = map(int, bottom_right.split('&'))
        return x1, y1, x2, y2
    except:
        # 捕获任何异常并输出提示信息
        print("文件名不是 CCPD 标准格式。")
        return None


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def main(input_dir):
    model_path = clp_det_model
    detector = CLPDetector(model_path)
    image_paths = list(Path(input_dir).glob('*.*'))
    if not image_paths:
        print(f"No images found in the directory: {input_dir}")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(f'runs/detected_plates_{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    correct_detections = 0 #准确率
    total_iou = 0
    iou_threshold = 0.7   # iou阈值

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read the image: {image_path}")
            continue

        detections = detector(image)

        try:
            x1, y1, x2, y2 = parse_ccpd_filename(image_path.stem)
        except ValueError as e:
            print(e)
            continue

        print(f"Detected plates in {image_path.name}:")
        # 检测结果包括 [x1, y1, x2, y2, score, lmk1_x, lmk1_y, lmk2_x, lmk2_y, lmk3_x, lmk3_y, lmk4_x, lmk4_y],
        # 边界框坐标：车牌的左上角和右下角的坐标（x1, y1, x2, y2）。置信度：检测到车牌的置信度分数。关键点坐标：车牌的关键点坐标（通常是车牌的四个角点）。
        #for det in detections:
        #   print(det)
        detected = False
        for det in detections:
            dx1, dy1, dx2, dy2 = map(int, det[:4])
            print(f"Predicted: ({dx1},{dy1}),({dx2},{dy2})")
            print(f"target   : ({x1},{y1}),({x2},{y2})")
            # 交并比（IoU）是两个边界框重叠区域与它们并集区域的比值。
            # IoU 的值在 0 到 1 之间，值越大表示两个边界框越接近。
            iou = calculate_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
            print(f"iou:{iou}")
            total_iou += iou
            if iou >= iou_threshold:
                detected = True
                break

        if detected:
            correct_detections += 1

        total_images += 1

        for idx, det in enumerate(detections):
            dx1, dy1, dx2, dy2 = map(int, det[:4])
            plate_image = image[dy1:dy2, dx1:dx2]
            if plate_image.size == 0:
                print(f"Skipped saving plate {idx} due to invalid size.")
                continue
            plate_image_path = save_dir / f'{image_path.stem}_plate_{idx}.jpg'
            cv2.imwrite(str(plate_image_path), plate_image)
            #print(f"Saved detected plate {idx} from {image_path.name} to {plate_image_path}")

    detection_rate = correct_detections / total_images if total_images > 0 else 0
    localization_accuracy = total_iou / total_images if total_images > 0 else 0
    print(f"Detection rate: {detection_rate:.2f}")
    print(f"Localization accuracy (average IoU): {localization_accuracy:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Car Plate Detection')
    parser.add_argument('input_dir', type=str, help='Directory containing images to process')
    args = parser.parse_args()
    main(args.input_dir)