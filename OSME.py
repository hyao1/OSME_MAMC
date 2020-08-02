from keras.utils.vis_utils import plot_model
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

    # base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    base_model = ResNet50(weights=None, include_top=False,  input_shape=(img_size, img_size, 3))
    #base_model = VGG19(weights="imagenet", include_top=False, input_tensor=input_tensor)
    # base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=input_tensor)
    '''
    base_model = resnet.ResNet101(weights=None,
                                  include_top=False,
                                  input_tensor=input_tensor,
                                  backend=keras.backend,
                                  layers=keras.layers,
                                  models=keras.models,
                                  utils=keras.utils)
    # base_model.trainable = False
    '''
    base_model_out = base_model(input_tensor)
    s_1 = osme_block(base_model_out, base_model.output_shape[3])
    '''
    s_1 = osme_block(base_model_out, base_model.output_shape[3])
    s_2 = osme_block(base_model_out, base_model.output_shape[3])

    # 矩阵平滑为向量
    fc1 = Flatten()(s_1)
    fc2 = Flatten()(s_2)

    # 全连接
    fc1 = Dense(1024, kernel_initializer='he_normal', name='fc1')(fc1)
    fc2 = Dense(1024, kernel_initializer='he_normal', name='fc2')(fc2)

    fc = concatenate([fc1, fc2])
    '''
    '''
    vgg19后面加的网络
    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    '''

    #prediction = Dense(num_classes, activation='softmax', name='prediction')(x)
    model = Model(inputs=input_tensor, outputs=base_model.output)
    # 模型
    return model


def with_top(num_classes=200):
    input_tensor = Input(shape=(img_size, img_size, 3))  # 输入层
    base_model = ResNet50(weights="imagenet", include_top=True, input_tensor=input_tensor)

    model = Model(inputs=input_tensor, outputs=base_model.output)
    return model


if __name__ == "__main__":
    model1 = osme()
    model2 = with_top()
    # model.summary()
    plot_model(model1, to_file="model1.png", show_shapes=True)
    plot_model(model2, to_file="model2.png", show_shapes=True)
