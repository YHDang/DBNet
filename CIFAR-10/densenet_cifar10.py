import keras
import math
import numpy as np
import os
import argparse
from keras.datasets import cifar10
from keras.utils import multi_gpu_model
from save_paralle_model import *
from BLS import *
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, AveragePooling2D, GlobalAveragePooling2D, Lambda, \
    concatenate
from keras.initializers import he_normal
from keras.layers.merge import Concatenate
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model, load_model
from keras import optimizers, regularizers
from keras import backend as K

if ('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
parses = argparse.ArgumentParser()
parses.add_argument('-m', '--mode', default='train', help='Select running mode: train, test or bls')
args = parses.parse_args()

# ------- Parameters of DenseNet-------------
growth_rate = 12
depth = 100
compression = 0.5
img_rows, img_cols = 32, 32
img_channels = 3
num_classes = 10
batch_size = 128  # 64 or 32 or other
epochs = 300
iterations = 391
weight_decay = 1e-4
model_path = 'cifar-10-model/densenet_cifar10_model.h5'
flag = args.mode
input_h, input_w = 32, 32

mean = [125.307, 122.95, 113.865]
std = [62.9932, 62.0887, 66.7048]

# --------Parameters of BLS----------------
N1 = 10  # # of nodes belong to each window
N2 = 1  # # of windows -------Feature mapping layer
N3 = 500  # # of enhancement nodes -----Enhance layer
L = 5  # # of incremental steps
M = 50  # # of adding enhance nodes
s = 0.8  # shrink coefficient
c = 2 ** -30  # Regularization coefficient
BLS_Epoch = 10

train_acc = np.zeros([1, L + 1])
train_time = np.zeros([1, L + 1])
test_acc = np.zeros([1, L + 1])
test_time = np.zeros([1, L + 1])


def scheduler(epoch):
    if epoch < 150:
        return 0.1
    if epoch < 225:
        return 0.01
    return 0.001


def densenet(img_input, classes_num):
    def conv(x, out_filters, k_size):
        return Conv2D(filters=out_filters,
                      kernel_size=k_size,
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=False)(x)

    def dense_layer(x):
        return Dense(units=classes_num,
                     activation='softmax',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def bottleneck(x):
        channels = growth_rate * 4
        x = bn_relu(x)
        x = conv(x, channels, (1, 1))
        x = bn_relu(x)
        x = conv(x, growth_rate, (3, 3))
        return x

    def single(x):
        x = bn_relu(x)
        x = conv(x, growth_rate, (3, 3))
        return x

    def transition(x, inchannels):
        outchannels = int(inchannels * compression)
        x = bn_relu(x)
        x = conv(x, outchannels, (1, 1))
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        return x, outchannels

    def dense_block(x, blocks, nchannels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat)
            concat = concatenate([x, concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels

    nblocks = (depth - 4) // 6
    nchannels = growth_rate * 2

    x = conv(img_input, nchannels, (3, 3))
    x, nchannels = dense_block(x, nblocks, nchannels)
    x, nchannels = transition(x, nchannels)
    x, nchannels = dense_block(x, nblocks, nchannels)
    x, nchannels = transition(x, nchannels)
    x, nchannels = dense_block(x, nblocks, nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x


def densenet_train(model, x_train, y_train, x_test, y_test):
    parallel_model = multi_gpu_model(model, gpus=2)
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    tb_cb = TensorBoard(log_dir='./result/densenet/', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    ckpt = ParallelModelCheckpoint(model, filepath=model_path, monitor='val_loss', save_best_only=True, save_weights_only=True,
                                   mode='auto', period=1)
    cbks = [change_lr, tb_cb, ckpt]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)

    datagen.fit(x_train)

    # start training
    parallel_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size * 2), steps_per_epoch=iterations,
                                 epochs=epochs, callbacks=cbks, validation_data=(x_test, y_test))


def densenet_test(model, x_test, y_test):
    model.load_weights(model_path)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    score = model.evaluate(x_test, y_test)

    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


def bls_dense_train(model, train_x, train_y):
    # 生成映射层
    model.load_weights(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    feature_layer = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d_1').output)
    feature_layer.save('cifar-10-model/densenet_feature_layer_model.h5')


    time_start = time.time()  # 计时开始

    OutputOfFeatureMappingLayer = []
    for i in range(N2):
        FeatureOfEachWindow = feature_layer.predict(train_x)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        # 通过稀疏化计算映射层每个窗口内的最终权重
        outputOfEachWindow = FeatureOfEachWindowAfterPreprocess
        OutputOfFeatureMappingLayer.append(outputOfEachWindow)
        del FeatureOfEachWindow
    N1 = OutputOfFeatureMappingLayer[0].shape[1]
    OutputOfFeatureMappingLayer = np.array(OutputOfFeatureMappingLayer)
    OutputOfFeatureMappingLayer = np.reshape(OutputOfFeatureMappingLayer,
                                             newshape=[-1, OutputOfFeatureMappingLayer[0].shape[1] * N2])
    # 生成强化层
    # 以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    # 生成强化层权重
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(InputOfEnhanceLayerWithBias.shape[1], N3) - 1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(InputOfEnhanceLayerWithBias.shape[1], N3).T - 1).T

    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
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

    return weightOfEnhanceLayer, parameterOfShrink, OutputWeight, InputOfEnhanceLayerWithBias, InputOfOutputLayer, pinvOfInput, N1


def bls_dense_test(x_test, y_test, weightOfEnhanceLayer, parameterOfShrink, OutputWeight):
    # 测试过程
    OutputOfFeatureMappingLayerTest = []
    feature_layer = load_model('cifar-10-model/densenet_feature_layer_model.h5')
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


def bls_dense_enhance(train_y, test_y, InputOfEnhanceLayerWithBias, InputOfEnhanceLayerWithBiasTest, InputOfOutputLayer,
                    pinvOfInput, InputOfOutputLayerTest, N1):
    parameterOfShrinkAdd = []
    for e in list(range(L)):
        time_start = time.time()
        if N1 * N2 >= M:
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(InputOfEnhanceLayerWithBias.shape[1], M) - 1)
        else:
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(InputOfEnhanceLayerWithBias.shape[1], M).T - 1).T

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
        print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %')

    return test_acc, test_time, train_acc, train_time


def BLS_Desnet(model, model_path, x_train, train_y, x_test, test_y, s, c, N1, N2, N3):
    #    u = 0
    model.load_weights(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    feature_layer = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d_1').output)
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
    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # - mean / std
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

    # build network
    img_input = Input(shape=(img_rows, img_cols, img_channels))
    output = densenet(img_input, num_classes)
    _model = Model(img_input, output)
    _model.summary()
    if flag == 'train':
        history = densenet_train(_model, x_train, y_train, x_test, y_test)
    elif flag == 'test':
        test_start = time.time()
        densenet_test(_model, x_test, y_test)
        test_end = time.time()
        print('Testing time of CNN is: ', test_end - test_start)
    elif flag == 'bls':
        weightOfEnhanceLayer, parameterOfShrink, OutputWeight, InputOfEnhanceLayerWithBias, InputOfOutputLayer, pinvOfInput, N1 = bls_dense_train(
            _model, x_train, y_train)
        InputOfEnhanceLayerWithBiasTest, InputOfOutputLayerTest = bls_dense_test(x_test, y_test,
                                                                               weightOfEnhanceLayer, parameterOfShrink,
                                                                               OutputWeight)
        test_acc, test_time, train_acc, train_time = bls_dense_enhance(y_train, y_test,
                                                                     InputOfEnhanceLayerWithBias,
                                                                     InputOfEnhanceLayerWithBiasTest,
                                                                     InputOfOutputLayer, pinvOfInput,
                                                                     InputOfOutputLayerTest, N1)
        np.savetxt('result/dense_test_acc.txt', test_acc)
        np.savetxt('result/dense_test_time.txt', test_time)

    elif flag == 'dbnet':
        n3 = 100
        m = 100
        L = 150
        Test_Acc = []
        Test_Time = []
        enhance_num = []

        for i in range(L):
            print('The number of enhancement nodes is: ', n3)
            test_acc, test_time, train_acc, train_time = BLS_Desnet(_model, model_path, x_train, y_train, x_test, y_test, s, c, 10, 10, n3)
            Test_Acc.append(test_acc)
            Test_Time.append(test_time)
            enhance_num.append(n3)
            n3 = n3 + m
        np.savetxt('result/densenet/enhance_num.txt', enhance_num)
        np.savetxt('result/densenet/dense_test_acc.txt', Test_Acc)
        np.savetxt('result/densenet/dense_test_time.txt', Test_Time)
if __name__ == '__main__':
    run_main()

