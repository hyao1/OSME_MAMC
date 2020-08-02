from OSME import osme
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset_new import CUB2011
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
import datetime
import os
import matplotlib

matplotlib.use('Agg')
import cv2
import numpy as np

'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1 # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )
'''
classes_path = 'classes.txt'
train_path = 'dataset/train'
test_path = 'dataset/train'
img_size = 600  # 448
BATCH_SIZE = 4
classes = []
num_classes = 20
test_nb = 40#5794
train_nb = 40#5994

n_epoch = 10  # 10次

with open(classes_path) as f:
    for r in f:
        data = r.split()
        classes.append(data[1])

# 图像预处理
train_data_gen = ImageDataGenerator(rescale=1.0 / 255,
                                    zoom_range=[0.7, 1.0],
                                    rotation_range=30,
                                    horizontal_flip=True,
                                    vertical_flip=False, )
test_data_gen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_data_gen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    shuffle=False)
test_generator = test_data_gen.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    shuffle=False)



for i, l in test_generator:
    true = np.array([np.argmax(one_hot) for one_hot in l])
    print(true)
    cv2.imshow('l', i[0])
    cv2.waitKey()

'''
def my_accuracy(y_true, y_pred):
    return tf.shape(y_true)[0]

osme_model = osme(num_classes)
opt = SGD(lr=0.001, momentum=1.6, decay=0.0005)
osme_model.compile(optimizer=opt, loss=[loss_function], metrics=['accuracy'])
osme_model.summary()


# 将在每个epoch后保存模型到model_osme_resnet50_gamma.best_loss_.hdf5
check_pointer = ModelCheckpoint(filepath='model_osme_resnet50_gamma.best_loss_.hdf5', verbose=1, save_best_only=True)

# 测试集损失不再提升时，减少学习率
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0000001)

history = osme_model.fit_generator(train_generator,
                                   steps_per_epoch=train_nb/BATCH_SIZE,
                                   epochs=n_epoch,
                                   validation_data=test_generator,
                                   validation_steps=64,
                                   verbose=1,
                                   callbacks=[reduce_lr, check_pointer])

osme_model.save("model.h5")

# 绘图
now = datetime.datetime.now()
plt.plot(history.history['acc'], color='red')
plt.plot(history.history['val_acc'], color='blue')
plt.title('model_without_MAMC accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("acc_se_ResNet50_with_OSME{0:%d%m}-{0:%H%M%S}.png".format(now))


plt.figure()
# loss
plt.plot(history.history['loss'], color='black')
plt.plot(history.history['val_loss'], color='yellow')
plt.title('model_without_MAMC loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loss_se_ResNET50_with_OSME{0:%d%m}-{0:%H%M%S}.png".format(now))


osme_model.evaluate_generator(generator=test_generator, steps=10, verbose=1)
'''
