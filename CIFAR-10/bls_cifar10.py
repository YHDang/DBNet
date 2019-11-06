from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from BLS import *
import cv2


N1 = 10  # # of nodes belong to each window
N2 = 10  # # of windows -------Feature mapping layer
N3 = 100  # # of enhancement nodes -----Enhance layer
L = 150  # # of incremental steps
M = 100  # # of adding enhance nodes
s = 0.8  # shrink coefficient
c = 2 ** -30  # Regularization coefficient
# BLS_Epoch = 10


def run_main():
    (x_train_img, y_train_label), (x_test_img, y_test_label) = cifar10.load_data()
    x_train_img = x_train_img.astype('float32') / 255
    x_test_img = x_test_img.astype('float32') / 255
    y_train_label = np_utils.to_categorical(y_train_label)
    y_test_label = np_utils.to_categorical(y_test_label)

    print(x_train_img[0])
    train_n, w, h, c = x_train_img.shape
    test_n = x_test_img.shape[0]

    x_train = x_train_img.reshape([train_n, w * h * c])
    x_test = x_test_img.reshape([test_n, w * h * c])

    # BLS_AddEnhanceNodes(x_train, y_train_label, x_test, y_test_label, s, c, N1, N2, N3, L, M)
    Test_Acc = []
    Test_Time = []
    n3 = N3
    for i in range(L):
        print('Enhancement Nodes: ', n3)
        test_acc, test_time, train_acc, train_time = BLS(x_train, y_train_label, x_test, y_test_label, s, c, 10, 10, n3)
        Test_Acc.append(test_acc)
        Test_Time.append(test_time)
        n3 = n3 + M
        print('Loop ending')
    np.savetxt('result/bls/bls_test_acc.txt', Test_Acc)
    np.savetxt('result/bls/bls_test_time.txt', Test_Time)
    print('Finished!')

if __name__ == '__main__':
    run_main()
