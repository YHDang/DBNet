from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import argparse
import numpy as np
import keras
import time
import os

# Set the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

parses = argparse.ArgumentParser()
parses.add_argument('--mode', default='train', help='Select running model (train, test or bls)')
args = parses.parse_args()

BATCH_SIZE = 128
EPOCH = 300
LR = 0.001
input_h, input_w = 28, 28
model_path = 'mnist_model/vgg_keras_model.h5'
data_path = '/home/dyh/Sources/dataset/MNIST_data/'
flag = args.mode


def build_vgg_model(input_shape):
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(inputs)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=LR), metrics=['accuracy'])

    return model


def train(model, x_train, y_train, x_test, y_test):
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,mode='auto', period=1)
    model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=EPOCH, batch_size=BATCH_SIZE,callbacks=[checkpoint])


def test(model, x_test, y_test):
    model.load_weights(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


def run_mian():
    mnist = input_data.read_data_sets(data_path, one_hot=True)

    x_train, x_test = mnist.train.images, mnist.test.images
    y_train, y_test = mnist.train.labels, mnist.test.labels

    x_train = x_train.reshape(-1, input_h, input_w, 1).astype('float32')
    x_test = x_test.reshape(-1, input_h, input_w, 1).astype('float32')

    if K.image_data_format() == 'channel_first':
        x_train = np.reshape(x_train, [x_train.shape[0], 1, input_h, input_w])
        x_test = np.reshape(x_test, [x_test.shape[0], 1, input_h, input_w])
        input_shape = (1, input_h, input_w)
    else:
        x_train = np.reshape(x_train, [x_train.shape[0], input_h, input_w, 1])
        x_test = np.reshape(x_test, [x_test.shape[0], input_h, input_w, 1])
        input_shape = (input_h, input_w, 1)
    print(input_shape)
    _model = build_vgg_model(input_shape)

    if flag == 'train':
        train(_model, x_train, y_train, x_test, y_test)
    elif flag == 'test':
        test_start = time.time()
        test(_model, x_test, y_test)
        test_end = time.time()
        print('Testing time is: ', test_end - test_start)


if __name__ == '__main__':
    run_mian()

