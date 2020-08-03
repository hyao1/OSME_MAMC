
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
#np.set_printoptions(threshold=np.inf)
from keras.applications.resnet50 import preprocess_input
import random
import copy


# this function reads .txt files and convert the image address to data which can be used in the #network.

class CUB2011():
    def __init__(self, root, class_num, train=True, validation=False, test=False):
        # get image address
        self.train = train
        self.test = test
        self.validation = validation
        ImageAddress = []
        ImageLabel = []
        self.class_num = class_num

        # 读取测试集和训练集图片及标签
        classes = []
        with open(root + '/classes.txt') as f:
            for r in f:
                classes.append(r.split()[1])

        if train:
            for n in range(class_num):

                for label in os.listdir(root + '/train/' + classes[n]):
                    ImageAddress.append(root + '/train/' + classes[n] + '/' + label)
                    ImageLabel.append(n)

        else:
            for n in range(class_num):
                for label in os.listdir(root + '/test/' + classes[n]):
                    ImageAddress.append(root + '/test/' + classes[n] + '/' + label)
                    ImageLabel.append(n)

        images_num = len(ImageAddress)
        print('%d个类, 总共%d个样本' % (class_num, len(ImageLabel)))
        # train\validation\test dataset
        self.ImageAddress = ImageAddress
        self.ImageLabel = ImageLabel
        self.changeImageAddress = copy.deepcopy(ImageAddress)
        self.changeImageLabel = copy.deepcopy(ImageLabel)
        '''
        # 图像变换，数据增强
        if transforms is None:
            normalize = T.Normalize([0.456], [0.225])
            if self.test or self.validation:
                self.transforms = T.Compose([
                    T.Resize(600),
                    T.CenterCrop(448),
                    T.ToTensor(),
                    normalize])
            else:
                self.transforms = T.Compose([
                    T.Resize(600),
                    T.RandomResizedCrop(448),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize])
    '''

    # get sample num of each class
    def get_each_class_num(self):
        each_class_num = []
        sum_class_num = []
        '''
        if len(self.changeImageLabel) == 0 or len(self.changeImageLabel) == 1:
            self.changeImageAddress = None
            self.changeImageLabel = None
            self.changeImageAddress = copy.deepcopy(self.ImageAddress)
            self.changeImageLabel = copy.deepcopy(self.ImageLabel)
        '''
        # print(len(self.changeImageAddress))
        for index in range(self.class_num):
            each_class_num.append(self.changeImageLabel.count(index))
            if index == 0:
                sum_class_num.append(each_class_num[0])
            else:
                sum_class_num.append(sum_class_num[index - 1] + each_class_num[index])
        # print(each_class_num)
        # print(sum_class_num)
        return each_class_num, sum_class_num

    # get class num
    def get_class_num(self, each_class_num, num, num_samples):
        enough = []
        for c in range(len(each_class_num)):
            if each_class_num[c] >= num_samples:
                enough.append(c+1)
        return random.sample(enough, num)

    # get samples from each class
    def get_samples_class(self, class_num_index, num, sum_class_num, each_class_num):
        samples = set()
        if each_class_num[class_num_index - 1] < num:
            num = each_class_num[class_num_index - 1]
        while len(samples) < num:
            if class_num_index == 1:
                samples.add(random.randint(0, sum_class_num[0] - 1))
            else:
                samples.add(random.randint(sum_class_num[class_num_index - 2], sum_class_num[class_num_index - 1] - 1))
        return list(samples)

    def transforms(self, image_input):
        img = image_input / np.float64(255.0)
        # img = (img - np.mean(img))/np.var(img)
        if self.train:
            # 裁剪
            img = cv2.resize(img, (600, 600))
            x = np.random.randint(0, 150)
            y = np.random.randint(0, 150)

            img = img[y:y + 448, x:x + 448]

            # 翻转
            if np.random.randint(0, 3) == 1:
                img = cv2.flip(img, np.random.randint(-1, 2))

            '''
            # 平移
            if np.random.randint(0, 3) == 1:
                mat = np.float32([[1, 0, np.random.randint(0, 50)], [0, 1, np.random.randint(0, 50)]])
                img = cv2.warpAffine(src=img, M=mat, dsize=img.shape[0:2])
            '''
            # 旋转
            if np.random.randint(0, 3) == 1:
                angle = np.random.randint(0, 20)
                matRotate = cv2.getRotationMatrix2D((img.shape[0]*0.5, img.shape[1] * 0.5), angle, 1)
                img = cv2.warpAffine(img, matRotate, img.shape[0:2])

        else:

            #img = (img - np.mean(img)) / np.var(img, ddof=1)
            img = cv2.resize(img, (600, 600))
            img = img[76:76 + 448, 76:76 + 448]

            img = cv2.resize(img, (448, 448))
        return img

    def get_samples(self, num_class, num_samples):
        # 获取每一类里的样本数量和总的样本数量

        # get sample num of each class
        while 1:
            each_class_num, sum_class_num = self.get_each_class_num()
            flag = 0
            for num in each_class_num:
                if num >= num_samples:
                    flag = flag+1
            if flag < num_class:
                self.changeImageAddress = None
                self.changeImageLabel = None
                self.changeImageAddress = copy.deepcopy(self.ImageAddress)
                self.changeImageLabel = copy.deepcopy(self.ImageLabel)
                continue
            # get class num
            class_num = self.get_class_num(each_class_num, num_class, num_samples)
            # get samples from each class
            image_path = []
            image_label = []
            all_samples_index = []
            for class_num_index in class_num:

                samples_index = self.get_samples_class(class_num_index, num_samples, sum_class_num, each_class_num)
                all_samples_index.append(samples_index)

                for index in samples_index:

                    image_path.append(self.changeImageAddress[index])
                    image_label.append(int(self.changeImageLabel[index]))

            del_index = []
            for class_num_index in range(len(class_num)):
                for index in all_samples_index[class_num_index]:
                    del_index.append(index)
            for index in sorted(del_index, reverse=True):
                del self.changeImageAddress[index]
                del self.changeImageLabel[index]

            image_data = []

            for path in image_path:
                data = cv2.imread(path)

                data = self.transforms(data)
                image_data.append(data)
            '''
            state = np.random.get_state()
            np.random.shuffle(image_data)
            np.random.set_state(state)
            np.random.shuffle(image_label)
            '''
            l = keras.utils.to_categorical(image_label, num_classes=self.class_num, dtype='float64')
            yield np.array(np.float64(image_data)), [l, l]

    def get_samples_my(self, num_samples):
        # return the data of the image

        while 1:
            image_data = []
            image_label = []
            if len(self.changeImageLabel) < num_samples:
                self.changeImageAddress = None
                self.changeImageLabel = None
                self.changeImageAddress = copy.deepcopy(self.ImageAddress)
                self.changeImageLabel = copy.deepcopy(self.ImageLabel)

            for i in range(num_samples):
                index = np.random.randint(0, len(self.changeImageLabel))
                data = cv2.imread(self.changeImageAddress[index])
                data = self.transforms(data)
                image_data.append(data)
                image_label.append(int(self.changeImageLabel[index]))
                del self.changeImageAddress[index]
                del self.changeImageLabel[index]
            l = keras.utils.to_categorical(image_label, num_classes=self.class_num)

            yield np.array(image_data), l


if __name__ == '__main__':
    root = 'CUB_200_2011/CUB_200_2011/dataset'
    a = CUB2011(root=root, class_num=200, train=False)

    for i, l in a.get_samples(2, 5):
        print(type(l[0][0][0]))
        cv2.imshow('1', i[0])
        cv2.imshow('2', i[1])
        cv2.waitKey()



