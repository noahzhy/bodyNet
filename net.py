import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *

# from keras.applications.mobilenet_v3 import MobileNetV3Small
from model.mobilenet_v3_small import MobileNetV3_Small


def build_model(input_shape, n_class=10):
    inputs = Input(shape=input_shape)
    backbone = MobileNetV3_Small(
        input_shape,
        alpha=0.75,
        include_top=True
    )
    c1, c2, c3, c4, _ = backbone.build(inputs)
    shortcut = SeparableConv2D(16, kernel_size=3, strides=1, padding='SAME', activation='relu6')(c1)

    x = SeparableConv2D(64, kernel_size=3, strides=1, padding='SAME', activation='relu6')(c4)
    x = UpSampling2D((8,8), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([shortcut, x])
    x = SeparableConv2D(64, kernel_size=3, strides=1, padding='SAME', activation='relu6')(x)
    x = SeparableConv2D(n_class, kernel_size=3, strides=1, padding='SAME', activation='relu6')(x)
    x = UpSampling2D()(x)
    x = Activation('softmax')(x)
    model = Model(inputs, x)
    return model


if __name__ == '__main__':
    model = build_model((224,224,3))
    # model = MobileNetV3_Small((224, 224, 3), n_class=10, alpha=0.75, include_top=True).build()
    model.summary()
