from BLS import *
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras import backend as K
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
np.random.seed(10)
parser = argparse.ArgumentParser(description='Select your running model')
parser.add_argument('--mode', type=str, default='test', help='Running Model')
args = parser.parse_args()


batch_size = 128
epoch = 200
lr = 0.001
input_h, input_w = 28, 28
model_path = 'mnist_model/cnn_keras_model.h5'
flag = args.mode.strip()
print(flag)
print(type(flag))

def buildModel(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def train(model, x_Train4D_normalize, y_Train, x_test, y_test):
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                                 mode='auto', period=1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x=x_Train4D_normalize, y=y_Train, epochs=epoch, batch_size=batch_size, verbose=2,
              callbacks=[checkpoint])


def test(model, x_test, y_test):
    model.load_weights(model_path)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


def bls_res_train(model, x_train, y_train, N1, N2, N3, c, s, BLS_Epoch):
    # features = model.predict(x_train, y_train, verbose=0)
    model.load_weights(model_path)
    feature_layer = Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
    feature_layer.save('mnist_model/keras_cnn_removed_model.h5')

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
    feature_layer = load_model('mnist_model/keras_cnn_removed_model.h5')
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
    time_end = time.time()  # 训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')

    return testAcc, testTime, trainAcc, trainTime

mnist = input_data.read_data_sets('/home/dyh/Sources/dataset/MNIST_data/', one_hot=True)
x_Train, y_Train = mnist.train.images, mnist.train.labels
x_Test, y_Test = mnist.test.images, mnist.test.labels

x_Train4D = x_Train.reshape(-1, 28, 28, 1).astype('float32')
x_Test4D = x_Test.reshape(-1, 28, 28, 1).astype('float32')


# Parameters of BLS
N1 = 10  # # of nodes belong to each window
N2 = 1  # # of windows -------Feature mapping layer
s = 0.8  # shrink coefficient
c = 2 ** -30  # Regularization coefficient
# BLS_Epoch = 10

if K.image_data_format() == 'channel_first':
    x_train = np.reshape(x_Train4D, [x_Train4D.shape[0], 1, input_h, input_w])
    x_test = np.reshape(x_Test, [x_Test.shape[0], 1, input_h, input_w])
    input_shape = (1, input_h, input_w)
else:
    x_train = np.reshape(x_Train4D, [x_Train4D.shape[0], input_h, input_w, 1])
    x_test = np.reshape(x_Test, [x_Test.shape[0], input_h, input_w, 1])
    input_shape = (input_h, input_w, 1)
print('input_shape: ', input_shape)
_model = buildModel(input_shape)

if flag == 'train':
    print('Start Training...')
    train(_model, x_train, y_Train, x_test, y_Test)
elif flag == 'test':
    print('Start Testing....')
    test_start = time.time()
    test(_model, x_test, y_Test)
    test_end = time.time()
    print('Testing time is: ', test_end - test_start)
elif flag == 'bls':
    print('Start BLS...')
    n3 = 100
    m = 100
    L = 150
    Test_Acc = []
    Test_Time = []
    enhance_num = []
    for i in range(L):
        print("The Number of Enhancement Nodes is: ", n3)
        weightOfEnhanceLayer, parameterOfShrink, OutputWeight = bls_res_train(_model, x_train, y_Train, N1, N2, n3, c, s)
        test_acc, test_time = bls_res_test(x_test, y_Test, N2, weightOfEnhanceLayer, parameterOfShrink, OutputWeight)
        Test_Acc.append(test_acc)
        Test_Time.append(test_time)
        enhance_num.append(n3)
        n3 = n3 + m
    np.savetxt('result/cnn/enhance_num.txt', enhance_num)
    np.savetxt('result/cnn/dbnet_test_acc.txt', Test_Acc)
    np.savetxt('result/cnn/dbnet_test_time.txt', Test_Time)

elif flag == 'dbnet':
    n3 = 100
    m = 100
    L = 150
    Test_Acc = []
    Test_Time = []
    enhance_num = []

    for i in range(L):
        print('The Number of Enhancement Nodes: ', n3)
        test_acc, test_time, train_acc, train_time = BLS_CNN(_model, model_path, x_train, y_Train, x_test,
                                                            y_Test, s, c, 10, 10, n3)
        Test_Acc.append(test_acc)
        Test_Time.append(test_time)
        enhance_num.append(n3)
        n3 = n3 + m

    np.savetxt('result/cnn/cnn_bls_test_acc_3.txt', Test_Acc)
    np.savetxt('result/cnn/cnn_bls_test_time_3.txt', Test_Time)
    np.savetxt('result/cnn/enhancement_node_num_3.txt', enhance_num)
