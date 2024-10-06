import sys
import os

sys.path.append(os.getcwd())
from STN.model.STN import STNet
import numpy as np
import argparse
import torch
import time
import cv2


def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.squeeze(0).cpu()
    inp = inp.detach().numpy().transpose((1, 2, 0))
    inp = 127.5 + inp / 0.0078125
    inp = inp.astype('uint8')

    return inp


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载网络
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('STN/weights/STN_Model_LJK_CA_XZH.pth', map_location=lambda storage, loc: storage))
    STN.eval()

    print("空间变换网络搭建完成")

    # 指定输入图像文件夹
    input_folder = 'data/testSTN/'
    output_folder = 'data/testSTN_results/'

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有图片
    for img_name in os.listdir(input_folder):
        if img_name.endswith(('.jpg', '.png', '.jpeg','.JPG')):  # 根据需要修改支持的文件类型
            image_path = os.path.join(input_folder, img_name)

            # 读取图像
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)

            # 确保图像成功读取
            if image is None:
                print(f"无法读取图像文件: {image_path}")
                continue

            # 预处理图像
            im = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
            im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
            data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94])

            # 使用 STN 进行变换
            with torch.no_grad():
                transfer = STN(data)

            # 将转换后的图像转换为 OpenCV 格式
            transformed_img = convert_image(transfer)

            output_path = os.path.join(output_folder, f"{img_name}")
            cv2.imencode('.jpg', transformed_img)[1].tofile(output_path)
            # 展示
            #cv2.imshow('Simulated Rain Effect', transformed_img)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
