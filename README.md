# License_Plate_Recognition  chinese plates generate in complex env
## 车牌检测
参数是含有需要进行车牌检测的图像集（必须是ccpd标准标注数据集）路径， 将检测到的车牌区域图像输出到当前目录的runs\detected_plates
，并会把预测和真实标签车牌框(车牌左上，右下点坐标)打印到控制台，可以计算交并率iou和识别准确率(交并率大于阈值iou_threshold的视为识别成功)
- python .\plate_detector\test_detector.py your_input_images_path
-  e.g. python .\plate_detector\test_detector.py test

## 车牌识别
参数是车牌图像集路径， 必须是带标签(文件名为车牌号，或者是ccpd标准标注数据集)的数据集，可以计算预测准确率
，并会把预测和真实标签打印到控制台
  - python .\LPRNet_ca\test_LPRNet.py --test_img_dirs  your_input_plates_path
  - e.g.  python .\LPRNet_ca\test_LPRNet.py --test_img_dirs .\runs\detected_plates_
  - e.g.   python .\LPRNet_ca\test_LPRNet.py --test_img_dirs  .\LPRNet_ca\data\test\

## 普通车牌生成，参数是输出的目录 
  - -T 或 --text：可选参数，用于指定目标车牌号码。
  - -r 或 --rand：可选参数，用于指示是否生成随机车牌。如果包含此参数，则生成随机车牌。
  - -t 或 --type：可选参数，用于指定车牌类型。默认值为 blue。
  -  python .\PlateGen\generator.py .\runs\fake_plates\ -r  （随机生成车牌）
## 复杂车牌生成，参数是需要进行复杂情况模拟的所有车牌路径
  -  python .\PlateGen\complex_plate_generator.py .\runs\fake_plates\




