import keras
import os
import argparse
import numpy as np
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
from keras import optimizers, regularizers
from keras import backend as K
from save_paralle_model import *
from BLS import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

# set GPU memory
if ('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='NUMBER',
                    help='batch size(default: 128)')
parser.add_argument('-e', '--epochs', type=int, default=500, metavar='NUMBER',
                    help='epochs(default: 500)')
parser.add_argument('-n', '--stack_n', type=int, default=16, metavar='NUMBER',
                    help='stack number n, total layers = 6 * n + 2 (default: 5)')
parser.add_argument('-d', '--dataset', type=str, default="cifar100", metavar='STRING',
                    help='dataset. (default: cifar100)')
parser.add_argument('-m', '--mode', default='train', help='Select running mode:train, test or bls')

args = parser.parse_args()

stack_n = args.stack_n
layers = 6 * stack_n + 2
num_classes = 10
img_rows, img_cols = 32, 32
img_channels = 3
batch_size = args.batch_size
epochs = args.epochs
iterations = 50000 // batch_size + 1
weight_decay = 1e-4
flag = args.mode
model_path = 'cifar-100-model/resnet_' + str(stack_n) + '_cifar100_model.h5'

N1 = 10  # # of nodes belong to each window
N2 = 1  # # of windows -------Feature mapping layer
initial_N3 = 500  # # of enhancement nodes -----Enhance layer
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
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001


def residual_network(img_input, classes_num=10, stack_n=5):
    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x


def resnet_train(resnet, x_train, y_train, x_test, y_test):
    parallel_model = multi_gpu_model(resnet, gpus=2)

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    checkpoint = ParallelModelCheckpoint(resnet, model_path, monitor='val_loss', verbose=1, save_best_only=True,
                                         save_weights_only=True,
                                         mode='auto', period=1)
    # set callback
    cbks = [TensorBoard(log_dir='./resnet_{:d}_{}/'.format(layers, args.dataset), histogram_freq=0),
            LearningRateScheduler(scheduler), checkpoint]

#    parallel_model.fit(x=x_train, y=y_train, batch_size=batch_size * 2, epochs=epochs, verbose=1, callbacks=cbks, validation_split=0.2, shuffle=True)
    # dump checkpoint if you need.(add it to cbks)
    # ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)

    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)
    #
    datagen.fit(x_train)

    # start training
    parallel_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size * 2),
                                 steps_per_epoch=iterations,
                                 epochs=epochs,
                                 callbacks=cbks,
                                 validation_data=(x_test, y_test))


def resnet_test(model, x_test, y_test):
    model.load_weights(model_path)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = model.evaluate(x_test, y_test)

    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


def bls_resnet_train(model, train_x, train_y, N3):
    # 生成映射层
    model.load_weights(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    feature_layer = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d_1').output)
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


def bls_resnet_test(x_test, y_test, weightOfEnhanceLayer, parameterOfShrink, OutputWeight):
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

    return InputOfEnhanceLayerWithBiasTest, InputOfOutputLayerTest, testTime, testAcc*100


def bls_resnet_enhance(train_y, test_y, InputOfEnhanceLayerWithBias, InputOfEnhanceLayerWithBiasTest,
                       InputOfOutputLayer,
                       pinvOfInput, InputOfOutputLayerTest, OutputOfFeatureMappingLayer):
    parameterOfShrinkAdd = []
    N1 = OutputOfFeatureMappingLayer
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


def BLS_Resnet(model, model_path, x_train, train_y, x_test, test_y, s, c, N1, N2, N3):
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
    print("========================================")
    print("MODEL: Residual Network ({:2d} layers)".format(6 * stack_n + 2))
    print("BATCH SIZE: {:3d}".format(batch_size))
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))
    print("DATASET: {:}".format(args.dataset))

    print("== LOADING DATA... ==")
    # load data
    global num_classes
    if args.dataset == "cifar100":
        num_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("== DONE! ==\n== COLOR PREPROCESSING... ==")
    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)

    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    img_input = Input(shape=(img_rows, img_cols, img_channels))
    output = residual_network(img_input, num_classes, stack_n)
    resnet = Model(img_input, output)

    resnet.summary()

    enhance_node_number = []
    resnet_bls_test_acc = []
    resnet_bls_test_time = []

    if flag == 'train':
        hist = resnet_train(resnet, x_train, y_train, x_test, y_test)
    elif flag == 'test':
        test_start = time.time()
        resnet_test(resnet, x_test, y_test)
        test_end = time.time()
        print('ResNet Testing Time is: ', test_end - test_start, 's')

    elif flag == 'bls':
        N3 = initial_N3
        for enhance in range(BLS_Epoch):
            weightOfEnhanceLayer, parameterOfShrink, OutputWeight, InputOfEnhanceLayerWithBias, InputOfOutputLayer, pinvOfInput, OutputOfFeatureMappingLayer = bls_resnet_train(
                resnet, x_train, y_train, N3)
            InputOfEnhanceLayerWithBiasTest, InputOfOutputLayerTest, Testing_time, Testing_acc = bls_resnet_test(x_test, y_test,
                                                                               weightOfEnhanceLayer, parameterOfShrink,
                                                                               OutputWeight)
            test_acc, test_time, train_acc, train_time = bls_resnet_enhance(y_train, y_test,
                                                                     InputOfEnhanceLayerWithBias,
                                                                     InputOfEnhanceLayerWithBiasTest,
                                                                     InputOfOutputLayer, pinvOfInput,
                                                                     InputOfOutputLayerTest, OutputOfFeatureMappingLayer)
            enhance_node_number.append(N3)
            resnet_bls_test_acc.append(Testing_acc)
            resnet_bls_test_time.append(Testing_time)
            N3 += M
        np.savetxt('result/resnet/res_bls_' + str(stack_n) + '_test_acc.txt', resnet_bls_test_acc)
        np.savetxt('result/resnet/res_bls_' + str(stack_n) + '_test_time.txt', resnet_bls_test_time)
        np.savetxt('result/resnet/enhance_node_number.txt', enhance_node_number)
    elif flag == 'dbnet':
        n3 = 100
        m = 100
        L = 150
        Test_Acc = []
        Test_Time = []
        enhance_node = []
        for i in range(L):
            print('The Number of Enhancement Nodes is: ', n3)
            test_acc, test_time, train_acc, train_time = BLS_Resnet(resnet, model_path, x_train, y_train, x_test,
                                                                    y_test, s, c, 10, 10, n3)
            Test_Acc.append(test_acc)
            Test_Time.append(test_time)
            n3 = n3 + m
            enhance_node.append(n3)
        np.savetxt('result/resnet/resnet_test_acc_100.txt', Test_Acc)
        np.savetxt('result/resnet/resnet_test_time_100.txt', Test_Time)
        np.savetxt('result/resnet/enhance_node_100.txt', enhance_node)

if __name__ == '__main__':
    run_main()
