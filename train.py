from OSME import osme
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset_new import CUB2011
import keras.backend.tensorflow_backend as KTF
from MAMC import mamc_loss, softmax_loss
import os
import datetime
import matplotlib
from keras import backend as K
matplotlib.use('Agg')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1 # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)
root = 'dataset'
img_size = 448  # 448
BATCH_SIZE = 10
classes = []
num_classes = 200
test_nb = 5794
train_nb = 5994

n_epoch = 60  # 60次


train_generator = CUB2011(root=root, class_num=num_classes, train=True).get_samples(2, BATCH_SIZE//2)
test_generator = CUB2011(root=root, class_num=num_classes, train=False).get_samples(2, BATCH_SIZE//2)


osme_model = osme(num_classes)
opt = SGD(lr=0.0001, momentum=0.9, decay=0.0005)

osme_model.compile(optimizer=opt,
                   loss=[mamc_loss, softmax_loss],
                   loss_weights=[0.5, 1.0],
                   metrics=['accuracy'])
osme_model.summary()


# 学习率衰减策略
def scheduler(epoch):
    # 每隔15个epoch，学习率减小为原来的1/10
    lr = K.get_value(osme_model.optimizer.lr)
    if epoch % 15 == 0 and epoch != 0:
        K.set_value(osme_model.optimizer.lr, lr * 0.1)
        print("\nlr changed to {}\n".format(lr * 0.1))
    else:
        print("\nlr is {}\n".format(lr))
    return K.get_value(osme_model.optimizer.lr)


reduce_lr = LearningRateScheduler(scheduler)

osme_model = osme(num_classes)
opt = SGD(lr=0.001, momentum=0.9, decay=0.0005)
osme_model.compile(optimizer=opt, loss=[mamc_loss, softmax_loss], metrics=['accuracy'])


# 将在每个epoch后保存模型到model_osme_resnet50_gamma.best_loss_.hdf5
check_pointer = ModelCheckpoint(filepath='model_osme_resnet50_gamma.best_loss_.hdf5', verbose=1, save_best_only=True)
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

