from __future__ import print_function, division

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class SMDLS(ms.Model):
    def __init__(self):
        super(SMDLS, self).__init__()

        self.conv1 = nn.Conv2D(filters=150, kernel_size=(1, 1), strides=(1, 1), padding='same', name='conv1')
        self.act1 = nn.Activation('relu')

        self.conv2 = nn.Conv1D(filters=50, kernel_size=7, strides=1, padding='same', name='conv2')
        self.act2 = nn.Activation('relu')

        self.conv3 = nn.Conv1D(filters=50, kernel_size=3, strides=2, padding='valid', name='conv3')
        self.bn1 = nn.BatchNormalization(name='BN1')
        self.act3 = nn.Activation('relu')

        self.conv4 = nn.Conv1D(filters=50, kernel_size=3, strides=1, padding='same', name='conv4')
        self.bn2 = nn.BatchNormalization(name='BN2')
        self.act4 = nn.Activation('relu')

        self.conv5 = nn.Conv1D(filters=50, kernel_size=3, strides=1, padding='same', name='conv5')
        self.bn3 = nn.BatchNormalization(name='BN3')
        self.act5 = nn.Activation('relu')

        self.conv6 = nn.Conv1D(filters=50, kernel_size=1, strides=1, padding='valid', name='conv6')
        self.bn4 = nn.BatchNormalization(name='BN4')
        self.act6 = nn.Activation('relu')

        self.conv7 = nn.Conv1D(filters=1, kernel_size=1, strides=1, padding='same', name='conv7')
        self.act7 = nn.Activation('relu')

        self.flatten = nn.Flatten(name='flatten')
        self.fc = nn.Dense(units=1, activation='sigmoid', name='OUT')


    def call(self, inputs, *args, **kwargs):
        x1 = self.conv1(inputs)
        x1 = self.act1(x1)

        x1 = ops.reshape(x1, shape=(-1, 150, 1))

        x2 = self.conv2(x1)
        x2 = self.act2(x2)

        x3 = self.conv3(x2)
        x3 = self.bn1(x3)
        x3 = self.act3(x3)
        res1 = x3

        x4 = self.conv4(x3)
        x4 = self.bn2(x4)
        x4 = self.act4(x4)
        res2 = x4

        x5 = self.conv5(x4)
        x5 = self.bn3(x5)
        x5 = self.act5(x5)
        res3 = x5

        x6 = self.conv6(x5)
        x6 = self.bn4(x6)
        x6 = self.act6(x6)
        res4 = x6

        x7 = ops.add([res1, res2, res3, res4])

        x8 = self.conv7(x7)
        x8 = self.act7(x8)

        x9 = self.flatten(x8)
        out = self.fc(x9)

        return out

if __name__ == '__main__':

    data = ms.ones([1, 1, 1, 13])
    model = SMDLS()
    model.build(input_shape=(None, 1, 1, 13))
    out = model(data)
    print(out.shape)
    model.summary()