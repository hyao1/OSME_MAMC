
"""
读取images.txt文件,获得每个图像的标签
读取train_test_split.txt文件,获取每个图像的train, test标签.其中1为训练,0为测试.
"""

import os
import shutil

import time

time_start = time.time()

# 文件路径
path_images = 'CUB_200_2011/CUB_200_2011/images.txt'
path_split = 'CUB_200_2011/CUB_200_2011/train_test_split.txt'
trian_save_path = 'CUB_200_2011/CUB_200_2011/dataset/train/'
test_save_path = 'CUB_200_2011/CUB_200_2011/dataset/test/'

# 读取images.txt文件
images = []
with open(path_images, 'r') as f:
    for line in f:
        images.append(list(line.strip('\n').split(',')))

# 读取train_test_split.txt文件
split = []
with open(path_split, 'r') as f_:
    for line in f_:
        split.append(list(line.strip('\n').split(',')))

# 划分
num = len(images)  # 图像的总个数
for k in range(num):
    file_name = images[k][0].split(' ')[1].split('/')[0]
    aaa = int(split[k][0][-1])
    if int(split[k][0][-1]) == 1:  # 划分到训练集
        # 判断文件夹是否存在
        if os.path.isdir(trian_save_path + file_name):
            shutil.copy('CUB_200_2011/CUB_200_2011/images/' + images[k][0].split(' ')[1],
                        trian_save_path + file_name + '/' + images[k][0].split(' ')[1].split('/')[1])
        else:
            os.makedirs(trian_save_path + file_name)
            shutil.copy('CUB_200_2011/CUB_200_2011/images/' + images[k][0].split(' ')[1],
                        trian_save_path + file_name + '/' + images[k][0].split(' ')[1].split('/')[1])
        print('%s处理完毕!' % images[k][0].split(' ')[1].split('/')[1])
    else:
        # 判断文件夹是否存在
        if os.path.isdir(test_save_path + file_name):
            aaaa = 'CUB_200_2011/CUB_200_2011/images/' + images[k][0].split(' ')[1]
            bbbb = test_save_path + file_name + '/' + images[k][0].split(' ')[1]
            shutil.copy('CUB_200_2011/CUB_200_2011/images/' + images[k][0].split(' ')[1],
                        test_save_path + file_name + '/' + images[k][0].split(' ')[1].split('/')[1])
        else:
            os.makedirs(test_save_path + file_name)
            shutil.copy('CUB_200_2011/CUB_200_2011/images/' + images[k][0].split(' ')[1],
                        test_save_path + file_name + '/' + images[k][0].split(' ')[1].split('/')[1])
        print('%s处理完毕!' % images[k][0].split(' ')[1].split('/')[1])

time_end = time.time()
print('CUB200训练集和测试集划分完毕, 耗时%s!!' % (time_end - time_start))