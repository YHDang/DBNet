from tensorflow.examples.tutorials.mnist import input_data
import keras
from keras.models import Model, load_model, Input
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import os
import time
from BLS import *
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', help='Running model')
parser = parser.parse_args()

model_path = 'mnist_model/bp_mnist_model.h5'
epoch = 300
batch_size = 128
lr = 0.001
flag = parser.mode

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(units=256, input_dim=784, activation='relu', kernel_initializer='normal')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=64, input_dim=256, activation='relu', kernel_initializer='normal')(x)
    x = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(model, x_train, y_train, x_test, y_test):
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch,validation_data=(x_test, y_test),  verbose=2, callbacks=[checkpoint])


def test(model, x_test, y_test):
    model.load_weights(model_path)
    opt = optimizers.Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    score = model.evaluate(x_test, y_test, verbose=0)
    y_predict = model.predict(x_test)
    np.savetxt('result/bp/bp_predict.txt', y_predict)
    np.savetxt('result/bp/bp_ground_truth.txt', y_test)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


def bls_res_train(model, x_train, y_train, N1, N2, N3, c, s, BLS_Epoch):
    # features = model.predict(x_train, y_train, verbose=0)
    model.load_weights(model_path)
    feature_layer = Model(inputs=model.input, outputs=model.get_layer('dense_3').output)
    feature_layer.save('mnist_model/bp_removed_model.h5')

    for bls_epoch in range(BLS_Epoch):
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
    feature_layer = load_model('mnist_model/bp_removed_model.h5')
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


def BLS_BP(model, model_path, x_train, train_y, x_test, test_y, s, c, N1, N2, N3):
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
    np.savetxt('result/bp/bp_bls_predict.txt', OutputOfTest)
    time_end = time.time()  # 训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')

    return testAcc, testTime, trainAcc, trainTime


if __name__ == '__main__':
    mnist = input_data.read_data_sets('/home/dyh/Sources/dataset/MNIST_data/', one_hot=True)
    x_train = mnist.train.images.astype('float32')
    x_test = mnist.test.images.astype('float32')
    print(x_train.shape)
    y_train = mnist.train.labels
    y_test = mnist.test.labels

    N1 = 10  # # of nodes belong to each window
    N2 = 1  # # of windows -------Feature mapping layer
    N3 = 550  # # of enhancement nodes -----Enhance layer
    M1 = 50  # # of adding enhance nodes
    s = 0.8  # shrink coefficient
    c = 2 ** -30  # Regularization coefficient
    BLS_Epoch = 10
    
    x_train = x_train.reshape(-1, 1, 784, 1)
    x_test = x_test.reshape(-1, 1, 784, 1)
    print(x_train.shape)
    input_shape = (1, 784, 1)

    print(input_shape)
    _model = build_model(input_shape)

    if flag == 'train':
        train(_model, x_train, y_train, x_test, y_test)
    elif flag == 'test':
        bp_start = time.time()
        test(_model, x_test, y_test)
        bp_end = time.time()
        print('BP Testing Time is: ', bp_end - bp_start)
    elif flag == 'bls':
        weightOfEnhanceLayer, parameterOfShrink, OutputWeight = bls_res_train(_model, x_train, y_train, N1, N2, N3, c,
                                                                              s, BLS_Epoch)
        bls_res_test(x_test, y_test, N2, weightOfEnhanceLayer, parameterOfShrink, OutputWeight)
    elif flag == 'dbnet':
        n3 = 13800
        m = 100
        L = 50
        Test_Acc = []
        Test_Time = []
        test_acc, test_time, train_acc, train_time = BLS_BP(_model, model_path, x_train, y_train, x_test,
                                                            y_test, s, c, 10, 10, n3)
        # for i in range(L):
        #     print('The number of enhancement nodes: ', n3)
        #     test_acc, test_time, train_acc, train_time = BLS_BP(_model, model_path, x_train, y_train, x_test,
        #                                                             y_test, s, c, 10, 10, n3)
        #     Test_Acc.append(test_acc)
        #     Test_Time.append(test_time)
        #     n3 = n3 + m
        # np.savetxt('result/bp/bp_bls_test_acc3.txt', Test_Acc)
        # np.savetxt('result/bp/bp_bls_test_time3.txt', Test_Time)
