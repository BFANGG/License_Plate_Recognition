from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os
import sys
sys.path.append(os.getcwd())

from STN.model.STN import STNet



def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
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
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default=default_pretrained_model, help='pretrained base model')

    args = parser.parse_args()

    return args


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.squeeze(0).cpu()
    inp = inp.detach().numpy().transpose((1, 2, 0))
    inp = 127.5 + inp / 0.0078125
    inp = inp.astype('uint8')

    return inp

def test():
    args = get_parser()

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")  # Updated device logic
    lprnet.to(device)
    print("Successful to build network!")

    ## 搭建空间变换网络
    STN = STNet()
    STN.to(device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    STN.load_state_dict(torch.load(os.path.join(script_dir, 'STN/weights/STN_Model_LJK_CA_XZH.pth'),
                                   map_location=lambda storage, loc: storage))
    STN.eval()

    print("空间变换网络搭建完成")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=device))  # Ensure the model is loaded onto the correct device
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    #test_img_dirs = os.path.expanduser(args.test_img_dirs)
    #test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    # 使用传入的测试集路径，如果没有则使用默认值 python test_LPRNet.py --test_img_dirs ./custom_test_dir
    test_img_dirs = args.test_img_dirs if args.test_img_dirs else "./data/test"
    test_img_dirs = os.path.expanduser(test_img_dirs)  # 展开用户路径

    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args, STN, device)
    finally:
        cv2.destroyAllWindows()

def Greedy_Decode_Eval(Net, datasets, args, STN, device):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    all_preds = []  # 用于保存所有的预测标签
    all_targets = []  # 用于保存所有的真实标签

    for i in range(epoch_size):
        # 加载训练数据
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        images = STN(images)

        # 前向传播
        prebs = Net(images)
        # 贪婪解码
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
            for c in preb_label:  # 去掉重复标签和空白标签
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)

        # 将预测结果与真实标签加入到列表中
        all_preds.extend(preb_labels)
        all_targets.extend(targets)

        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    # 计算准确率
    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))

    # 打印所有的预测标签和真实标签
    print("\n[Info] Predicted and Target Labels:")
    for pred, target in zip(all_preds, all_targets):
        pred_label = ''.join([CHARS[i] for i in pred])
        target_label = ''.join([CHARS[int(i)] for i in target])
        print(f"Predicted: {pred_label}, Target: {target_label}")

    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))



if __name__ == "__main__":
    test()
