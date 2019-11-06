from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import keras
import numpy as np
import argparse
from keras.layers import Input, Dense, Dropout, Flatten, add
from keras.layers import Conv2D, Activation, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.models import Model
from keras.utils import plot_model
import keras.datasets.mnist as mnist
from keras.models import load_model
from BLS import *
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

parses = argparse.ArgumentParser()
parses.add_argument('--mode', '-m', default='test', help='Select Running mode: train, test, bls or dbnet')
args = parses.parse_args()

batch_size = 128
num_class = 10
epochs = 50
model_path = 'mnist_model/modeldir.mnist'
input_h, input_w = 28, 28
flag = args.mode


def res_block(x, channels, i):
    if i is 1:
        strides = (1, 1)
        x_add = x
    else:
        strides = (2, 2)
        x_add = Conv2D(channels, kernel_size=(3, 3), activation='relu', padding='same', strides=strides)(x)
    x = Conv2D(channels, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(channels, kernel_size=(3, 3), padding='same', strides=strides)(x)
    x = add([x, x_add])
    Activation(K.relu)(x)
    return x


def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # Conv1 28*28 -> 28*28*16
    x = Conv2D(16, kernel_size=(7, 7), activation='relu', input_shape=input_shape, padding='same')(inputs)

    # Conv2 28*28*16 -> 14*14*16
    for i in range(2):
        x = res_block(x, 16, i)

    # Conv3 14*14*16 -> 7*7*32
    for i in range(2):
        x = res_block(x, 32, i)

    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(num_class, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=x)
    plot_model(model, to_file='resnet.png')
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    return model


def train(model, x_train, y_train, x_test, y_test):
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                                 mode='auto', period=1)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[checkpoint])


def test(model, x_test, y_test):
    model.load_weights(model_path)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


def bls_res_train(model, x_train, y_train, N1, N2, N3, c, s):
    # features = model.predict(x_train, y_train, verbose=0)
    model.load_weights(model_path)
    feature_layer = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
    feature_layer.save('mnist_model/feature_layer_model.h5')

    # for bls_epoch in range(BLS_Epoch):
    OutputOfFeatureMappingLayer = []
    time_start = time.time()
    for i in range(N2):
        # random.seed(i)
        FeatureOfEachWindow = feature_layer.predict(x_train)  # 生成每个窗口的特征
        OutputOfFeatureMappingLayer.append(FeatureOfEachWindow)
        print(FeatureOfEachWindow.shape)
    N1 = OutputOfFeatureMappingLayer[0].shape[1]
    OutputOfFeatureMappingLayer = np.array(OutputOfFeatureMappingLayer)
    OutputOfFeatureMappingLayer = np.reshape(OutputOfFeatureMappingLayer,
                                             newshape=[-1, OutputOfFeatureMappingLayer[0].shape[1] * N2])
    # print(N1)
    print(OutputOfFeatureMappingLayer.shape)
    # 生成强化层
    # 以下为映射层输出加偏置（强化层输入）

    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    # 生成强化层权重
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(InputOfEnhanceLayerWithBias.shape[1], N3)) - 1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(InputOfEnhanceLayerWithBias.shape[1], N3).T - 1).T
    print(weightOfEnhanceLayer.shape)
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)

    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)

    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = np.dot(pinvOfInput, y_train)  # 全局违逆
    time_end = time.time()  # 训练完成

    trainTime = time_end - time_start

    # 训练输出
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, y_train)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')

    return weightOfEnhanceLayer, parameterOfShrink, OutputWeight


def bls_res_test(x_test, y_test, N2, weightOfEnhanceLayer, parameterOfShrink, OutputWeight):
    # 测试过程
    OutputOfFeatureMappingLayerTest = []
    feature_layer = load_model('mnist_model/feature_layer_model.h5')
    feature_layer.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    time_start = time.time()  # 测试计时开始
    #  映射层
    for i in range(N2):
        outputOfEachWindowTest = feature_layer.predict(x_test)
        OutputOfFeatureMappingLayerTest.append(outputOfEachWindowTest)
    #  强化层
    OutputOfFeatureMappingLayerTest = np.array(OutputOfFeatureMappingLayerTest)
    OutputOfFeatureMappingLayerTest = np.reshape(OutputOfFeatureMappingLayerTest,
                                                 newshape=[-1,
                                                           OutputOfFeatureMappingLayerTest[0].shape[1] * N2])
    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)
    #  强化层输出
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)
    #  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    #  最终测试输出
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()  # 训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, y_test)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    return testAcc, testTime


def BLS_Resnet(model, model_path, x_train, train_y, x_test, test_y, s, c, N1, N2, N3):
    #    u = 0
    model.load_weights(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    feature_layer = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
    feature_layer.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_x = feature_layer.predict(x_train)

    L = 0
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])
    Beta1OfEachWindow = []

    train_acc_all = np.zeros([1, L + 1])
    test_acc = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    time_start = time.time()  # 计时开始
    for i in range(N2):
        random.seed(i)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1;  # 生成每个窗口的权重系数，最后一行为偏差
        #        WeightOfEachWindow([],[],i) = weightOfEachWindow; #存储每个窗口的权重系数
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)  # 生成每个窗口的特征
        FeatureOfEachWindowAfterPreprocess = FeatureOfEachWindow
        # 通过稀疏化计算映射层每个窗口内的最终权重
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        print(betaOfEachWindow.shape)
        # 存储每个窗口的系数化权重
        Beta1OfEachWindow.append(betaOfEachWindow)
        # 每个窗口的输出 T1
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        #        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
        # distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        # minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        # outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        del outputOfEachWindow
        del FeatureOfEachWindow
        del weightOfEachWindow

        # 生成强化层
    # 以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    # 生成强化层权重
    if N1 * N2 >= N3:
        random.seed(67797325)
        #        dim = N1*N2+1
        #        temp_matric = stats.ortho_group(dim)
        #        weightOfEnhanceLayer = temp_matric[:,0:N3]
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    #    print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))

    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)

    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = np.dot(pinvOfInput, train_y)  # 全局违逆
    time_end = time.time()  # 训练完成
    trainTime = time_end - time_start

    # 训练输出
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')
    train_acc_all[0][0] = trainAcc
    train_time[0][0] = trainTime


    # 测试过程
    time_start = time.time()  # 测试计时开始
    test_x = feature_layer.predict(x_test)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])

    #  映射层
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = outputOfEachWindowTest
    #  强化层
    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)
    #  强化层输出
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)
    #  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    #  最终测试输出
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()  # 训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')

    test_acc[0][0] = testAcc
    test_time[0][0] = testTime

    return testAcc, testTime, trainAcc, trainTime


def main(_):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Parameters of BLS
    N1 = 10  # # of nodes belong to each window
    N2 = 1  # # of windows -------Feature mapping laye
    L = 5  # # of incremental steps
    s = 0.8  # shrink coefficient
    c = 2 ** -30  # Regularization coefficient
    BLS_Epoch = 10

    if K.image_data_format() == 'channel_first':
        x_train = np.reshape(x_train, [x_train.shape[0], 1, input_h, input_w])
        x_test = np.reshape(x_test, [x_test.shape[0], 1, input_h, input_w])
        input_shape = (1, input_h, input_w)
    else:
        x_train = np.reshape(x_train, [x_train.shape[0], input_h, input_w, 1])
        x_test = np.reshape(x_test, [x_test.shape[0], input_h, input_w, 1])
        input_shape = (input_h, input_w, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalization
    x_train /= 255
    x_test /= 255

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_class)
    y_test = keras.utils.to_categorical(y_test, num_class)

    _model = build_model(input_shape)

    if flag == 'train':
        train_time_start = time.time()
        train(_model, x_train, y_train, x_test, y_test)
        train_time_end = time.time()
        print('Trainging Time is: ', train_time_end-train_time_start)
    elif flag == 'test':
        test_time_start = time.time()
        test(_model, x_test, y_test)
        test_time_end = time.time()
        print('Testing Time is: ', test_time_end-test_time_start)
    elif flag == 'bls':
        n3 = 100
        m = 100
        L = 150
        Test_Acc = []
        Test_Time = []
        enhance_num = []
        for i in range(L):
            print("The Number of Enhancement Nodes is: ", n3)
            weightOfEnhanceLayer, parameterOfShrink, OutputWeight = bls_res_train(_model, x_train, y_train, N1, N2, n3, c, s)
            test_acc, test_time = bls_res_test(x_test, y_test, N2, weightOfEnhanceLayer, parameterOfShrink, OutputWeight)
            Test_Acc.append(test_acc)
            Test_Time.append(test_time)
            enhance_num.append(n3)
            n3 = n3 + m
        np.savetxt('result/resnet/enhance_num.txt', enhance_num)
        np.savetxt('result/resnet/dbnet_test_acc.txt', Test_Acc)
        np.savetxt('result/resnet/dbnet_test_time.txt', Test_Time)

    elif flag == 'dbnet':
        n3 = 100
        m = 100
        L = 150
        Test_Acc = []
        Test_Time = []
        enhance_num = []

        for i in range(L):
            print("The Number of Enhancement Nodes is: ", n3)
            test_acc, test_time, train_acc, train_time = BLS_Resnet(_model, model_path, x_train, y_train, x_test, y_test, s, c, 10, 10, n3)
            # test_acc, test_time, train_acc, train_time = BLS_Desnet(_model, model_path, x_train, y_train, x_test,
            #                                                         y_test, s, c, 10, 10, 5500)
            Test_Acc.append(test_acc)
            Test_Time.append(test_time)
            enhance_num.append(n3)
            n3 = n3 + m
        np.savetxt('result/resnet/enhance_num.txt', enhance_num)
        np.savetxt('result/resnet/resnet_bls_test_acc_1.txt', Test_Acc)
        np.savetxt('result/resnet/resnet_bls_test_time_1.txt', Test_Time)
if __name__ == '__main__':
    tf.app.run()





