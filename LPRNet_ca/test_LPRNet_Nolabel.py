import torch
import time
import cv2
import os
import sys
from data.load_data_nolable import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image
from model.LPRNet import build_lprnet
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

# Import STN
from STN.model.STN import STNet  # Make sure the STN module is in the correct path

sys.path.append(os.getcwd())

# 识别传入路径的车牌图像，并把识别出的车牌号作为文件名保存。 python .\LPRNet_ca\test_LPRNet_Nolabel.py --test_img_dirs 'test_set'
def get_parser():
    parser = argparse.ArgumentParser(description='parameters to test net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')

    # 获取当前脚本文件的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 将相对路径转换为绝对路径
    default_test_img_dirs = os.path.join(script_dir, "data/test")
    default_pretrained_model = os.path.join(script_dir, "weights/LPRNet_model_Init.pth")

    parser.add_argument('--test_img_dirs', default=default_test_img_dirs, help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=1, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')  # Set default to False
    parser.add_argument('--pretrained_model', default=default_pretrained_model, help='pretrained base model')

    args = parser.parse_args()

    return args


def collate_fn(batch):
    imgs = []
    filenames = []
    for sample in batch:
        img, filename = sample
        imgs.append(torch.from_numpy(img))
        filenames.append(filename)
    return torch.stack(imgs, 0), filenames


def convert_image(inp):
    inp = inp.squeeze(0).cpu()
    inp = inp.detach().numpy().transpose((1, 2, 0))
    inp = 127.5 + inp / 0.0078125
    inp = inp.astype('uint8')
    return inp


def test():
    args = get_parser()

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    STN = STNet()
    STN.to(device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    STN.load_state_dict(torch.load(os.path.join(script_dir, 'STN/weights/STN_Model_LJK_CA_XZH.pth'), map_location=lambda storage, loc: storage))
    STN.eval()
    print("空间变换网络搭建完成")

    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    test_img_dirs = args.test_img_dirs if args.test_img_dirs else "./data/test"
    test_img_dirs = os.path.expanduser(test_img_dirs)

    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args, STN, device)
    finally:
        cv2.destroyAllWindows()




def Greedy_Decode_Eval(Net, datasets, args, STN, device):
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn))

    # 创建保存检测车牌的文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(f'runs/rec_plates_{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(epoch_size):
        images, filenames = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        images = STN(images)

        prebs = Net(images)
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)

        for filename, label in zip(filenames, preb_labels):
            pred_label = ''.join([CHARS[i] for i in label])
            new_filename = save_dir / f"{pred_label}.jpg"
            original_image = cv2.imread(filename)
            cv2.imencode('.jpg', original_image)[1].tofile(str(new_filename))
            print(f"Saved {new_filename}")

if __name__ == "__main__":
    test()


