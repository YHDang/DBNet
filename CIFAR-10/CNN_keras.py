from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import LeakyReLU, BatchNormalization, Dropout
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from keras.models import Model
from save_paralle_model import ParallelModelCheckpoint
import argparse
import numpy as np
from BLS import *
import keras
import time
import os


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISBILE_DEVICES'] = '0, 1'

parses = argparse.ArgumentParser()
parses.add_argument('--mode', default='train', help='Select the running model: train, test or bls')
args = parses.parse_args()

EPOCH = 300
BATCH_SIZE = 128
LR = 0.001
input_h, input_w = 32, 32
flag = args.mode
model_path = 'cifar-10-model/cnn_keras_model.h5'

N1 = 10  # # of nodes belong to each window
N2 = 1  # # of windows -------Feature mapping layer
N3 = 550  # # of enhancement nodes -----Enhance layer
L = 5  # # of incremental steps
M = 50  # # of adding enhance nodes
s = 0.8  # shrink coefficient
c = 2**-30  # Regularization coefficient
BLS_Epoch = 10

train_acc = np.zeros([1, L + 1])
train_time = np.zeros([1, L + 1])
test_acc = np.zeros([1, L + 1])
test_time = np.zeros([1, L + 1])


def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # Conv_block_1
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv_block_2
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv_block_3
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten
    x = Flatten()(x)
    x = Dropout(0.3)(x)
#    x = Dense(2500, activation='relu')(x)
#    x = Dropout(0.3)(x)
    x = Dense(1500, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
   # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train(model, x_train, y_train):
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpoint = ParallelModelCheckpoint(model, filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                                 mode='auto', period=1)
    history =parallel_model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=EPOCH, batch_size=BATCH_SIZE * 2, shuffle=True,
                        callbacks=[checkpoint])

    return history


def test(model, x_test, y_test):
    model.load_weights(model_path)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = model.evaluate(x_test, y_test)

    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


def bls_vgg_train(model, train_x, train_y):
    # 生成映射层
    model.load_weights(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    feature_layer = Model(inputs=model.input, outputs=model.get_layer('dropout_5').output)
    feature_layer.save('cifar-10-model/vgg_feature_layer_model.h5')

    distOfMaxAndMin = []
    minOfEachWindow = []

    time_start = time.time()  # 计时开始

    OutputOfFeatureMappingLayer = []
    for i in range(N2):
        FeatureOfEachWindow = feature_layer.predict(train_x)
       # print('Max of FeatureOfEachWindow: ', np.max(FeatureOfEachWindow))
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        print('Max of FeatureOfEachWindow: ', np.max(FeatureOfEachWindowAfterPreprocess))
        # 通过稀疏化计算映射层每个窗口内的最终权重
        outputOfEachWindow = FeatureOfEachWindowAfterPreprocess
       # distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
       # minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
       # OutputOfFeatureMappingLayer.append(outputOfEachWindow)
        OutputOfFeatureMappingLayer.append(FeatureOfEachWindow)
        del FeatureOfEachWindow
     
    OutputOfFeatureMappingLayer = np.array(OutputOfFeatureMappingLayer)
    print(OutputOfFeatureMappingLayer.shape)
    OutputOfFeatureMappingLayer = np.reshape(OutputOfFeatureMappingLayer, newshape=[-1, OutputOfFeatureMappingLayer[0].shape[1] * N2])
        # 生成强化层
    # 以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    print(InputOfEnhanceLayerWithBias.shape[1])
    # 生成强化层权重
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(InputOfEnhanceLayerWithBias.shape[1], N3) - 1)
        print(weightOfEnhanceLayer.shape)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(InputOfEnhanceLayerWithBias.shape[1], N3) - 1)
        print(weightOfEnhanceLayer.shape)
    print(weightOfEnhanceLayer.shape)
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    print('Max of InputOfOutputLayer is: ', np.max(InputOfOutputLayer))
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = pinvOfInput.dot(train_y)  # 全局违逆
    time_end = time.time()  # 训练完成
    trainTime = time_end - time_start

    # 训练输出
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime

    return weightOfEnhanceLayer, parameterOfShrink, OutputWeight, InputOfEnhanceLayerWithBias, InputOfOutputLayer, pinvOfInput


def bls_vgg_test(x_test, y_test, weightOfEnhanceLayer, parameterOfShrink, OutputWeight):
    # 测试过程
    OutputOfFeatureMappingLayerTest = []
    feature_layer = load_model('cifar-10-model/vgg_feature_layer_model.h5')
    feature_layer.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    time_start = time.time()  # 测试计时开始
    #  映射层
    for i in range(N2):
        outputOfEachWindowTest = feature_layer.predict(x_test)
        OutputOfFeatureMappingLayerTest.append(outputOfEachWindowTest)

    OutputOfFeatureMappingLayerTest = np.array(OutputOfFeatureMappingLayerTest)
    OutputOfFeatureMappingLayerTest = np.reshape(OutputOfFeatureMappingLayerTest,
                                                 newshape=[-1,
                                                           OutputOfFeatureMappingLayerTest[0].shape[1] * N2])
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
    testAcc = show_accuracy(OutputOfTest, y_test)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime

    return InputOfEnhanceLayerWithBiasTest, InputOfOutputLayerTest


def vgg_bls_enhance(train_y, test_y, InputOfEnhanceLayerWithBias, InputOfEnhanceLayerWithBiasTest, InputOfOutputLayer, pinvOfInput, InputOfOutputLayerTest):
    parameterOfShrinkAdd = []
    for e in list(range(L)):
        time_start = time.time()
        if N1 * N2 >= M:
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(InputOfEnhanceLayerWithBias.shape[1], M) - 1)
        else:
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(InputOfEnhanceLayerWithBias.shape[1], M) - 1)

        #        WeightOfEnhanceLayerAdd[e,:,:] = weightOfEnhanceLayerAdd
        #        weightOfEnhanceLayerAdd = weightOfEnhanceLayer[:,N3+e*M:N3+(e+1)*M]
        tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayerAdd)
        parameterOfShrinkAdd.append(s / np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd * parameterOfShrinkAdd[e])
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer, OutputOfEnhanceLayerAdd])

        D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T, D)).I.dot(np.dot(D.T, pinvOfInput))
        else:
            B = pinv(C, c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)), B])
        OutputWeightEnd = pinvOfInput.dot(train_y)
        InputOfOutputLayer = tempOfLastLayerInput
        Training_time = time.time() - time_start
        train_time[0][e + 1] = Training_time
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)
        TrainingAccuracy = show_accuracy(OutputOfTrain1, train_y)
        train_acc[0][e + 1] = TrainingAccuracy
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %')

        # 增量增加节点的 测试过程
        time_start = time.time()
        OutputOfEnhanceLayerAddTest = tansig(
            InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e]);
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)
        TestingAcc = show_accuracy(OutputOfTest1, test_y)

        Test_time = time.time() - time_start
        test_time[0][e + 1] = Test_time
        test_acc[0][e + 1] = TestingAcc
        print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %');
        print('Incremental Testing Time is: ', Test_time, 's')
    return test_acc, test_time, train_acc, train_time


def BLS_CNN(model, model_path, x_train, train_y, x_test, test_y, s, c, N1, N2, N3):
    #    u = 0
    model.load_weights(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    feature_layer = Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
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


def run_main():
    (x_train_img, y_train_label), (x_test_img, y_test_label) = cifar10.load_data()
    x_train_img = x_train_img.astype('float32') / 255
    x_test_img = x_test_img.astype('float32') / 255
    y_train_label = np_utils.to_categorical(y_train_label)
    y_test_label = np_utils.to_categorical(y_test_label)

    if K.image_data_format() == 'channel_first':
        x_train = np.reshape(x_train_img, [x_train_img.shape[0], 3, input_h, input_w])
        x_test = np.reshape(x_test_img, [x_test_img.shape[0], 3, input_h, input_w])
        input_shape = (3, input_h, input_w)
    else:
        x_train = np.reshape(x_train_img, [x_train_img.shape[0], input_h, input_w, 3])
        x_test = np.reshape(x_test_img, [x_test_img.shape[0], input_h, input_w, 3])
        input_shape = (input_h, input_w, 3)

    _model = build_model(input_shape)

    if flag == 'train':
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    #    os.environ['CUDA_VISBILE_DEVICES'] = '0, 1'
        history = train(_model, x_train, y_train_label)
    elif flag == 'test':
       # os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    #    os.environ['CUDA_VISBILE_DEVICES'] = '-1'
        test_start = time.time()
        test(_model, x_test, y_test_label)
        test_end = time.time()
        print('Testing time of CNN is: ', test_end - test_start)
    elif flag == 'bls':
     #   os.environ['CUDA_VISBILE_DEVICES'] = '-1'
        weightOfEnhanceLayer, parameterOfShrink, OutputWeight, InputOfEnhanceLayerWithBias,InputOfOutputLayer, pinvOfInput = bls_vgg_train(_model, x_train, y_train_label)
        InputOfEnhanceLayerWithBiasTest, InputOfOutputLayerTest = bls_vgg_test(x_test, y_test_label, weightOfEnhanceLayer, parameterOfShrink, OutputWeight)
        test_acc, test_time, train_acc, train_time = vgg_bls_enhance(y_train_label, y_test_label, InputOfEnhanceLayerWithBias, InputOfEnhanceLayerWithBiasTest, InputOfOutputLayer, pinvOfInput, InputOfOutputLayerTest)
        np.savetxt('result/cnn_test_acc.txt', test_acc)
        np.savetxt('result/cnn_test_time.txt', test_time)
    elif flag == 'dbnet':
        n3 = 100
        m = 100
        L = 150
        Test_Acc = []
        Test_Time = []
        enhance_num = []

        for i in range(L):
            test_acc, test_time, train_acc, train_time = BLS_CNN(_model, model_path, x_train, y_train_label, x_test,
                                                                    y_test_label, s, c, 10, 10, n3)
            Test_Acc.append(test_acc)
            Test_Time.append(test_time)
            enhance_num.append(n3)
            n3 = n3 + m
        np.savetxt('result/cnn/enhance_num.txt', enhance_num)
        np.savetxt('result/cnn/cnn_test_acc.txt', Test_Acc)
        np.savetxt('result/cnn/cnn_test_time.txt', Test_Time)

if __name__ == '__main__':
    run_main()
