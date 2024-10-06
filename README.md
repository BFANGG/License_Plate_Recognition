# License_Plate_Recognition  chinese plates generate in complex env
# 车牌检测
参数是含有需要进行车牌检测的图像集路径， 将检测到的车牌区域图像输出到当前目录的runs\detected_plates
- python .\plate_detector\test_detector.py your_input_images_path
-  e.g. python .\plate_detector\test_detector.py test

# 车牌识别
1. 参数是车牌图像集路径， 如果是带标签的图片(输入图片名为车牌号)， 可以计算预测正确率
  - python .\LPRNet_ca\test_LPRNet.py --test_img_dirs .\test_lprnet_withlabel\
  - e.g.  python .\LPRNet_ca\test_LPRNet.py --test_img_dirs .\runs\detected_plates_20241006_201021\

2. 参数是车牌图像集路径， 如果是不带标签的图片，将识别后的车牌号作为输入图片名称，并把输入图片保存到当前目录的runs\rec_plates
  - python .\LPRNet_ca\test_LPRNet_Nolabel.py --test_img_dirs .\runs\detected_plates_20241005_165651\ 
# 普通车牌生成，参数是输出的目录 
  - -T 或 --text：可选参数，用于指定目标车牌号码。
  - -r 或 --rand：可选参数，用于指示是否生成随机车牌。如果包含此参数，则生成随机车牌。
  - -t 或 --type：可选参数，用于指定车牌类型。默认值为 blue。
  -  python .\PlateGen\generator.py .\runs\fake_plates\ -r  （随机生成车牌）
# 复杂车牌生成，参数是需要进行复杂情况模拟的所有车牌路径
  -  python .\PlateGen\complex_plate_generator.py .\runs\fake_plates\




