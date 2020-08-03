from keras.utils.vis_utils import plot_model
from keras.layers import Input, Dense, Flatten, Multiply, concatenate
from keras.layers import Input, Dense, Flatten, Multiply, concatenate, Lambda
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras_applications import resnet
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
import tensorflow as tf
import keras

img_size = 448
num_classes = 200
def osme_block(in_block, ch, ratio=16, name=None):
    #  压缩
    z = GlobalAveragePooling2D(a = 2)(in_block)  # 1
    x = Dense(ch // ratio, activation='relu')(z)  # 2
    x = Dense(ch, activation='sigmoid', name=name)(x)  # 3

    # 激励
    return Multiply()([in_block, x])  # 4


def osme(num_classes=200):  # OSME模型
    input_tensor = Input(shape=(img_size, img_size, 3))  # 输入层

    # 基础网络，不保留全连接层，加载预训练权重
    base_model = ResNet50(weights=None, include_top=False,  input_shape=(img_size, img_size, 3))

    base_model_out = base_model(input_tensor)

    s_1 = osme_block(base_model_out, base_model.output_shape[3])
    s_2 = osme_block(base_model_out, base_model.output_shape[3])

    # 矩阵平滑为向量
    fc1 = Flatten()(s_1)
    fc2 = Flatten()(s_2)

    # 全连接
    fc1 = Dense(1024, kernel_initializer='he_normal', name='fc1')(fc1)
    fc2 = Dense(1024, kernel_initializer='he_normal', name='fc2')(fc2)

    fc = concatenate([fc1, fc2])

    prediction = Dense(num_classes, activation='softmax', name='prediction')(fc)
    model = Model(inputs=input_tensor, outputs=prediction)
    return model


if __name__ == "__main__":
    model = osme()
    plot_model(model, to_file="model1.png", show_shapes=True)
