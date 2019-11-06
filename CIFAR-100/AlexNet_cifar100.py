import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import argparse
import keras
from keras.datasets import cifar100
from keras import backend as K
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, load_model
from keras.layers import concatenate, Dropout, Flatten
from keras.utils.training_utils import multi_gpu_model
from keras.utils import np_utils
from save_paralle_model import *
from BLS import *

from keras import optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint


parses = argparse.ArgumentParser()
parses.add_argument('--mode', default='train', help='Select running mode: train, test or bls')
args = parses.parse_args()

num_classes = 100
batch_size = 128  # 64 or 32 or other
epochs = 1000
iterations = 392
DROPOUT = 0.5  # keep 50%
CONCAT_AXIS = 3
weight_decay = 1e-4
DATA_FORMAT = 'channels_last'  # Theano:'channels_first' Tensorflow:'channels_last'
log_filepath = './result/log'
model_path = 'cifar-100-model/alexnet_cifar100_model.h5'
flag = args.mode
input_h, input_w = 32, 32

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


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 200:
        return 0.001
    return 0.0001


def alexnet(img_input, classes=100):
    x = Conv2D(96, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(
        img_input)  # valid

    x = Conv2D(256, (5, 5), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform')(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
#    x = Dense(1000, activation='relu')(x)
#    x = Dropout(0.5)(x) 
    out = Dense(classes, activation='softmax')(x)
    return out


def alexnet_train(model, x_train, y_train, x_test, y_test):
    parallel_model = multi_gpu_model(model, gpus=2)
    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    checkpoint = ParallelModelCheckpoint(model, filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True,
                                         save_weights_only=True,
                                         mode='auto', period=1)
    cbks = [change_lr, tb_cb, checkpoint]

    # set data augmentation
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)
    datagen.fit(x_train)

    # start training
    parallel_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size * 2),
                                 steps_per_epoch=iterations,
                                 epochs=epochs,
                                 callbacks=cbks,
                                 validation_data=(x_test, y_test))


def alexnet_test(model, x_test, y_test):
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
    feature_layer = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
    feature_layer.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    feature_layer.save('cifar-100-model/vgg_feature_layer_model.h5')

    time_start = time.time()  # 计时开始

    OutputOfFeatureMappingLayer = []
    for i in range(N2):
        FeatureOfEachWindow = feature_layer.predict(train_x)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        # 通过稀疏化计算映射层每个窗口内的最终权重
        outputOfEachWindow = FeatureOfEachWindowAfterPreprocess
        # distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        # minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
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


def bls_vgg_test(x_test, y_test, weightOfEnhanceLayer, parameterOfShrink, OutputWeight):
    # 测试过程
    OutputOfFeatureMappingLayerTest = []
    feature_layer = load_model('cifar-100-model/vgg_feature_layer_model.h5')
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


def vgg_bls_enhance(train_y, test_y, InputOfEnhanceLayerWithBias, InputOfEnhanceLayerWithBiasTest, InputOfOutputLayer,
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
        print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %');

    return test_acc, test_time, train_acc, train_time


def BLS_AlexNet(model, model_path, x_train, train_y, x_test, test_y, s, c, N1, N2, N3):
    #    u = 0
    model.load_weights(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    feature_layer = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
    feature_layer.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_x = feature_layer.predict(x_train)

    L = 0
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])
    Beta1OfEachWindow = []

    print('The shape of training data is: ', train_x.shape)
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
    (x_train_img, y_train_label), (x_test_img, y_test_label) = cifar100.load_data()
    x_train_img = x_train_img.astype('float32') / 255
    x_test_img = x_test_img.astype('float32') / 255
    y_train_label = np_utils.to_categorical(y_train_label)
    y_test_label = np_utils.to_categorical(y_test_label)
    print(y_train_label.shape)
    if K.image_data_format() == 'channel_first':
        x_train = np.reshape(x_train_img, [x_train_img.shape[0], 3, input_h, input_w])
        x_test = np.reshape(x_test_img, [x_test_img.shape[0], 3, input_h, input_w])
        input_shape = (3, input_h, input_w)
    else:
        x_train = np.reshape(x_train_img, [x_train_img.shape[0], input_h, input_w, 3])
        x_test = np.reshape(x_test_img, [x_test_img.shape[0], input_h, input_w, 3])
        input_shape = (input_h, input_w, 3)

    img_input = Input(shape=(32, 32, 3))
    output = alexnet(img_input)
    _model = Model(img_input, output)
    _model.summary()

    if flag == 'train':
        history = alexnet_train(_model, x_train, y_train_label, x_test, y_test_label)
    elif flag == 'test':
        test_start = time.time()
        alexnet_test(_model, x_test, y_test_label)
        test_end = time.time()
        print('Testing time of CNN is: ', test_end - test_start)
    elif flag == 'bls':
        weightOfEnhanceLayer, parameterOfShrink, OutputWeight, InputOfEnhanceLayerWithBias, InputOfOutputLayer, pinvOfInput, N1 = bls_vgg_train(
            _model, x_train, y_train_label)
        InputOfEnhanceLayerWithBiasTest, InputOfOutputLayerTest = bls_vgg_test(x_test, y_test_label,
                                                                               weightOfEnhanceLayer, parameterOfShrink,
                                                                               OutputWeight)
        test_acc, test_time, train_acc, train_time = vgg_bls_enhance(y_train_label, y_test_label,
                                                                     InputOfEnhanceLayerWithBias,
                                                                     InputOfEnhanceLayerWithBiasTest,
                                                                     InputOfOutputLayer, pinvOfInput,
                                                                     InputOfOutputLayerTest, N1)
        np.savetxt('result/alexnet/alexnet_test_acc.txt', test_acc)
        np.savetxt('result/alexnet/alexnet_test_time.txt', test_time)

    elif flag == 'dbnet':
        n3 = 100
        m = 100
        L = 150
        Test_Acc = []
        Test_Time = []
        enhance_num = []

        for i in range(L):
            print('The Number of Enhancement Nodes is: ', n3)
            test_acc, test_time, train_acc, train_time = BLS_AlexNet(_model, model_path, x_train, y_train_label, x_test, y_test_label, s, c, 10, 10, n3)
            Test_Acc.append(test_acc)
            Test_Time.append(test_time)
            enhance_num.append(n3)
            n3 = n3 + m
        np.savetxt('result/alexnet/enhance_num_3.txt', enhance_num)
        np.savetxt('result/alexnet/alexnet_test_acc_3.txt', Test_Acc)
        np.savetxt('result/alexnet/alexnet_test_time_3.txt', Test_Time)


if __name__ == '__main__':
    run_main()

