# Awesome - Deep Vision Architecture 

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) ![](https://img.shields.io/badge/Citations&Stars%20Update-Mar%2012,%202022-blue.svg)

This repo provides an up-to-date list of progress made in deep learning vision architectures, which includes but not limited to paper (backbone design, loss deisgn, tricks etc), datasets, codebases, frameworks and etc. Please feel free to [open an issue](https://github.com/chenyaofo/awesome-vision-architecture/issues) to add new progress.


**Note**: The papers are grouped by published year. In each group, the papers are sorted by their citations. In addition, the paper with <ins>underline</ins> means a milestone in the field. The third-party code prefers `PyTorch`. The architectures searched by NAS are not included in this repo, please refer to my another repo [awesome-architecture-search](https://github.com/chenyaofo/awesome-architecture-search).

 - <a href="#Main Progress">Main Progress</a>
   - <a href="#2021 Venues">2021 Venues</a>
   - <a href="#2019 Venues">2019 Venues</a>
   - <a href="#2018 Venues">2018 Venues</a>
   - <a href="#2017 Venues">2017 Venues</a>
   - <a href="#2016 Venues">2016 Venues</a>
   - <a href="#2015 Venues">2015 Venues</a>
   - <a href="#2012 Venues">2012 Venues</a>
 - <a href="#Survey">Survey</a>
 - <a href="#Datasets">Datasets</a>

# <a name="Main Progress">Main Progress</a>

## <a name="2021 Venues">2021 Venues</a>

 - **RepVGG: Making VGG-style ConvNets Great Again** `Cited by 108` `CVPR` `2021` `Tsinghua University` `MEGVII Technology` `RepVGG` [`PDF`](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf) [`Official Code (Stars 2.3k)`](https://github.com/DingXiaoH/RepVGG)  ***TL;DR**: The authors propose a simple but powerful architecture named RepVGG, which has a multi-branch topology in the training and single-branch topology (VGG-like style) in the inference. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique.*



## <a name="2019 Venues">2019 Venues</a>

 - **Searching for MobileNetV3** `Cited by 1.8k` `ICCV` `2019` `Google AI` `Google Brain` `MobileNetV3` [`PDF`](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf) [`Official Code (Stars 73.0k)`](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) [`Third-party Code (Stars 11.1k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py) ***TL;DR**: This paper presents the next generation of MobileNets (MobileNetV3) based on a combination of complementary architecture search techniques as well as a novel architecture design.*



## <a name="2018 Venues">2018 Venues</a>

 - **Squeeze-and-Excitation Networks** `Cited by 11.5k` `CVPR` `2018` `Momenta` `University of Oxford` `SENet` [`PDF`](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf) [`Official Code (Stars 2.9k)`](https://github.com/hujie-frank/SENet)  ***TL;DR**: Based on the benefit of enhancing spatial encoding in prior works, the authors propose a novel architectural unit, which we term the “Squeezeand-Excitation” (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.*

 - <ins>**MobileNetV2: Inverted Residuals and Linear Bottlenecks**</ins> `Cited by 8.6k` `CVPR` `2018` `Google Inc.` `MobileNetV2` [`PDF`](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf) [`Official Code (Stars 73.0k)`](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) [`Third-party Code (Stars 11.1k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py) ***TL;DR**: Based on MobileNetV1, the authors devise a new mobile architecture, MobileNetV2, which is based on an inverted residual structure where the shortcut connections are between the thin bottleneck layers.*

 - **ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices** `Cited by 3.7k` `CVPR` `2018` `Megvii Inc (Face++)` `ShuffleNetV1` [`PDF`](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf)  [`Third-party Code (Stars 1.3k)`](https://github.com/megvii-model/ShuffleNet-Series) ***TL;DR**: The authors introduce an extremely computation-efficient CNN architecture named ShuffleNet, which utilizes two new operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining accuracy.*

 - **ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design** `Cited by 2.0k` `ECCV` `2018` `Megvii Inc (Face++)` `Tsinghua University` `ShuffleNetV2` [`PDF`](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.pdf)  [`Third-party Code (Stars 11.1k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py) ***TL;DR**: Prior architecture design is mostly guided by the indirect metric of computation complexity (i.e., FLOPs). In contrast, the authors proposes to use the direct metric (i.e., speed on the target platform) and derives several practical guidelines for efficient network (ShuffleNetV2) design from the empirical observations.*



## <a name="2017 Venues">2017 Venues</a>

 - <ins>**Densely Connected Convolutional Networks**</ins> `Cited by 23.3k` `CVPR` `2017` `Cornell University` `Tsinghua University` `Facebook AI Research` `DenseNet` [`PDF`](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) [`Official Code (Stars 4.4k)`](https://github.com/liuzhuang13/DenseNet) [`Third-party Code (Stars 11.1k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py) ***TL;DR**: The authors observe that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. Based on this, they introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion.*

 - **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications** `Cited by 12.3k` `arXiv` `2017` `Google Inc.` `MobileNetV1` [`PDF`](https://arxiv.org/pdf/1704.04861.pdf) [`Official Code (Stars 73.0k)`](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)  ***TL;DR**: The authors present a class of efficient models called MobileNets for mobile and embedded vision applications, which is a streamlined architecture with depthwise separable convolutions.*

 - **Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning** `Cited by 10.1k` `AAAI` `2017` `Google Inc.` `IneptionV4` [`PDF`](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14806/14311) [`Official Code (Stars 73.0k)`](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py)  ***TL;DR**: The authors propose IneptionV4 by combining Inception architectures with residual connections. Moreover, the authors seek to check if Inception can be more efficient with deeper and wider structure.*



## <a name="2016 Venues">2016 Venues</a>

 - <ins>**Deep Residual Learning for Image Recognition**</ins> `Cited by 109.7k` `CVPR` `2016` `Microsoft Research` `ResNet` [`PDF`](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) [`Official Code (Stars 5.9k)`](https://github.com/KaimingHe/deep-residual-networks) [`Third-party Code (Stars 11.1k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) ***TL;DR**: This paper presents a residual learning framework (ResNet) to ease the training of networks that are substantially deeper than those used previously, which reformulates the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.*

 - **Rethinking the Inception Architecture for Computer Vision** `Cited by 18.4k` `CVPR` `2016` `Google Inc.` `InceptionV3` [`PDF`](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf) [`Official Code (Stars 73.0k)`](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) [`Third-party Code (Stars 11.1k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py) ***TL;DR**: With version 1 and version 2 of Inception family, the authors want to explore ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization.*



## <a name="2015 Venues">2015 Venues</a>

 - <ins>**Very Deep Convolutional Networks for Large-Scale Image Recognition**</ins> `Cited by 74.8k` `ICLR` `2015` `Visual Geometry Group` `University of Oxford` `VGG` [`PDF`](https://arxiv.org/pdf/1409.1556.pdf)  [`Third-party Code (Stars 11.1k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py) ***TL;DR**: From the empirical results, the authors found that a network (VGG) with increasing depth and very small ( 3x3) convolution filters would lead to a significant performace improvement based on the prior-art configurations.*

 - <ins>**Going Deeper with Convolutions**</ins> `Cited by 38.0k` `CVPR` `2015` `Google Inc.` `GoogLeNet` `InceptionV1` [`PDF`](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) [`Official Code (Stars 73.0k)`](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py)  ***TL;DR**: The authors propose a deep convolutional neural network architecture codenamed Inception, which adopts multi-branch topology, leading to increasing of the depth and width of the network while keeping the computational budget constant.*

 - **Batch normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** `Cited by 35.1k` `ICML` `2015` `Google Inc.` `Batch Normalization` `InceptionV2` [`PDF`](http://proceedings.mlr.press/v37/ioffe15.pdf) [`Official Code (Stars 73.0k)`](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py)  ***TL;DR**: The authors propose **Batch Normalization**(BN) to alleviate the issue of **internal covariate shift**, which allows us to use much higher learning rates and be less careful about initialization. With the proposed BN, the authors devise a new architecture called InceptionV2.*



## <a name="2012 Venues">2012 Venues</a>

 - <ins>**ImageNet Classification with Deep Convolutional Neural Networks**</ins> `Cited by 104.8k` `NeurIPS` `2012` `University of Toronto` `AlexNet` [`PDF`](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)  [`Third-party Code (Stars 11.1k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py) ***TL;DR**: This is a pioneering work that exploits a deep convolutional neural network (AlexNet) for large image classification task (ImageNet), which achieves very impressing performance.*




# <a name="Survey">Survey</a>

 - **Transformers in Vision: A Survey** `Cited by 211` `arXiv` `2021` `University of Artificial Intelligence` `Transformers` `Survey` [`PDF`](https://arxiv.org/pdf/2101.01169.pdf)   ***TL;DR**: This survey aims to provide a comprehensive overview of the Transformer models in the computer vision discipline, which includes fundamental concepts of transformers, extensive applications of transformers in vision, the respective advantages and limitations of popular vision transformers and an analysis on open research directions && possible future works.*




# <a name="Datasets">Datasets</a>

 - **ImageNet** [`Download Link`](https://image-net.org/download.php) ***TL;DR**: ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. Currently, the most common used versions in academia are **ImageNet-1k** and **ImageNet-21k**. **1)** ImageNet-1k contains 1,281,167 training images, 50,000 validation images of 1000 object classes. **2)** ImageNet-21K, which is bigger and more diverse, consists of 14,197,122 images, each tagged in a single-label fashion by one of 21,841 possible classes. The dataset has no official train-validation split, and the classes are not well-balanced - some classes contain only 1-10 samples, while others contain thousands of samples. Lastly, it is recommended to download this dataset from [Academic Torrents](https://academictorrents.com/browse.php?search=ImageNet) instead of the official website.* `How to cite:` **Imagenet: A Large-scale Hierarchical Image Database** `Cited by 36.7k` `CVPR` `2009` `Princeton University` `ImageNet` [`PDF`](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf)
 - **CIFAR** [`Download Link`](https://www.cs.toronto.edu/~kriz/cifar.html) ***TL;DR**: There are two versions of CIFAR dataset: **1)** The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. **2)** The CIFAR-100 is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. Please refer to the [official website](https://www.cs.toronto.edu/~kriz/cifar.html) for more details.* `How to cite:` **Learning Multiple Layers of Features from Tiny Images** `Cited by 14.6k` `Tech Report` `2009` `Alex Krizhevsky` `CIFAR-10` `CIFAR-100` [`PDF`](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
