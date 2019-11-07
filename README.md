# DBNet: A New Generalized Structure Efficient for Classification <br>
  Source code about the paper "DBNet: A New Generalized Structure Efficient for Classification " 

# Preparation
## Directories
  The structure of documents is as follows. Make sure the directories are created before you start working. 
  * |==> DBNet
    * |==> MNIST
      * |==> result
        * |==> bls
        * |==> bp
        * |==> cnn
        * |==> resnet
        * |==> log
      * |==> mnist_model
    * CIFAR-10
      * |==> result
        * |==> alexnet
        * |==> bls
        * |==> cnn
        * |==> densenet
        * |==> resnet
        * |==> vgg
        * |==> log
      * |==> cifar-10-model
    * CIFAR-100
      * |==> result
        * |==> alexnet
        * |==> densenet
        * |==> resnet
        * |==> vgg
        * |==> log
      * |==> cifar-100-model

## Download Models
  If you don't want to train your own model, you can download trained models from the link: <br>
  [Trained Models](https://pan.baidu.com/s/1bUq9eF8O7AAsA6w6PJodww) <br>
  Password: njdl
  
  You should place downloaded models including mnist_model.zip, CIFAR-10-model.zip, CIFAR-100-model.zip in the DBNet/MNIST/, DBNet/CIFAR-10/, DBNet/CIFAR-100/ respectively.
  
# Dependencies
  Make sure you have the following dependencies installed before proceeding:
  * python >= 3.6
  * tensorflow-gpu == 1.2.1
  * CUDA Version == 9.0
  * CUDNN == 7

# Quick Start
## Training on the MNIST Dataset
Under the MNIST/, run the following command.
 ```
    python BP_keras.py --mode train
    python CNN_keras.py --mode train
    python ResNet_keras.py --mode train
 ```  
## Testing on the MNIST Dataset
Under the MNIST/, run the following command.
  ```
    # Testing the deep structures
    python BP_keras.py --mode test
    python CNN_keras.py --mode test
    python ResNet_keras.py --mode test
    
    # Testing the DBNet
    python BP_keras.py --mode dbnet
    python CNN_keras.py --mode dbnet
    python ResNet_keras.py --mode dbnet
  ```
------------------------------

## Training on the CIFAR-10 Dataset
Under the CIFAR-10/, run the following command.
 ```
    python AlexNet_cifar10.py --mode train
    python densenet_cifar10.py --mode train
    python resnet_cifar10.py --mode train
    python vgg_cifar10.py --mode train
 ```  
## Testing on the CIFAR-10 Dataset
Under the CIFAR-10/, run the following command.
  ```
    # Testing the deep structures
    python AlexNet_cifar10.py --mode test
    python densenet_cifar10.py --mode test
    python resnet_cifar10.py --mode test
    python vgg_cifar10.py --mode test
    
    # Testing the DBNet
    python AlexNet_cifar10.py --mode dbnet
    python densenet_cifar10.py --mode dbnet
    python resnet_cifar10.py --mode dbnet
    python vgg_cifar10.py --mode dbnet
  ```
  ------------------------------

## Training on the CIFAR-100 Dataset
Under the CIFAR-100/, run the following command.
 ```
    python AlexNet_cifar100.py --mode train
    python densenet_cifar100.py --mode train
    python resnet_cifar100.py --mode train
    python vgg_cifar100.py --mode train
 ```  
## Testing on the CIFAR-100 Dataset
Under the CIFAR-10/, run the following command.
  ```
    # Testing the deep structures
    python AlexNet_cifar100.py --mode test
    python densenet_cifar100.py --mode test
    python resnet_cifar100.py --mode test
    python vgg_cifar100.py --mode test
    
    # Testing the DBNet
    python AlexNet_cifar100.py --mode dbnet
    python densenet_cifar100.py --mode dbnet
    python resnet_cifar100.py --mode dbnet
    python vgg_cifar100.py --mode dbnet
  ```
