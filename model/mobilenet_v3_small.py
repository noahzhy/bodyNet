"""MobileNet v3 small models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape
from keras.utils.vis_utils import plot_model

from model.mobilenet_base import *


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3_Small(MobileNetBase):
    def __init__(self, shape, alpha=1.0, include_top=True):
        """Init.
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if inculde classification layer.
        # Returns
            MobileNetv3 model.
        """
        super(MobileNetV3_Small, self).__init__(shape, alpha)
        self.include_top = include_top

    def depth(self, d):
        return _depth(d * self.alpha)

    def build(self, inputs):
        """build MobileNetV3 Small.
        # Arguments
            plot: Boolean, weather to plot model.
        # Returns
            model: Model, model.
        """
        # inputs = Input(shape=self.shape)
        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')
        c1 = x
        x = self._bottleneck(x, self.depth(16), (3, 3), e=16, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, self.depth(24), (3, 3), e=72, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, self.depth(24), (3, 3), e=88, s=1, squeeze=False, nl='RE')
        c2 = x
        x = self._bottleneck(x, self.depth(40), (5, 5), e=96, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, self.depth(40), (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, self.depth(40), (5, 5), e=240, s=1, squeeze=True, nl='HS')
        c3 = x
        x = self._bottleneck(x, self.depth(48), (5, 5), e=120, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, self.depth(48), (5, 5), e=144, s=1, squeeze=True, nl='HS')
        c4 = x
        x = self._bottleneck(x, self.depth(96), (5, 5), e=288, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, self.depth(96), (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, self.depth(96), (5, 5), e=576, s=1, squeeze=True, nl='HS')
        c5 = x

        out = [c1, c2, c3, c4, c5]

        # x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
        # x = GlobalAveragePooling2D()(x)
        # x = Reshape((1, 1, 576))(x)

        # x = Conv2D(1280, (1, 1), padding='same')(x)
        # x = self._return_activation(x, 'HS')

        # if self.include_top:
        #     x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
        #     x = Reshape((self.n_class,))(x)

        # model = Model(inputs, out)
        # return model
        return out
