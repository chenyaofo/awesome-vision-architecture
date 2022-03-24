# Awesome - Deep Vision Architecture 

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) ![](https://img.shields.io/badge/Last%20Update-Mar%2024,%202022-blue.svg)

This repo provides an up-to-date list of progress made in deep learning vision architectures/image classification, which includes but not limited to paper (backbone design, loss deisgn, tricks etc), datasets, codebases, frameworks and etc. Please feel free to [open an issue](https://github.com/chenyaofo/awesome-vision-architecture/issues) to add new progress.


**Note**: The papers are grouped by published year. In each group, the papers are sorted by their citations. In addition, the paper with <ins>underline</ins> means a milestone in the field. The third-party code prefers `PyTorch`. The architectures searched by NAS are not included in this repo, please refer to my another repo [awesome-architecture-search](https://github.com/chenyaofo/awesome-architecture-search).

 - <a href="#Main Progress">Main Progress</a>
   - <a href="#2021 Venues">2021 Venues</a>
   - <a href="#2020 Venues">2020 Venues</a>
   - <a href="#2019 Venues">2019 Venues</a>
   - <a href="#2018 Venues">2018 Venues</a>
   - <a href="#2017 Venues">2017 Venues</a>
   - <a href="#2016 Venues">2016 Venues</a>
   - <a href="#2015 Venues">2015 Venues</a>
   - <a href="#2012 Venues">2012 Venues</a>
 - <a href="#Survey">Survey</a>
 - <a href="#Datasets">Datasets</a>
 - <a href="#Codebases">Codebases</a>
 - <a href="#Misc">Misc</a>

# <a name="Main Progress">Main Progress</a>

## <a name="2021 Venues">2021 Venues</a>

 - <ins>**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**</ins> `Cited by 3.0k` `ICLR` `2021` `Google Research, Brain Team` `Vision Transformer` `ViT` [`PDF`](https://openreview.net/pdf?id=YicbFdNTTy) [`Official Code (Stars 4.7k)`](https://github.com/google-research/vision_transformer)  ***TL;DR**: While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In this context, the authors seek to directly apply a pure transformer to sequences of image patches (called Vision Transformer), which performs very well on image classification tasks.*

 - **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** `Cited by 898` `ICCV` `2021` `Microsoft Research Asia` `Swin Transformer` [`PDF`](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) [`Official Code (Stars 7.5k)`](https://github.com/microsoft/Swin-Transformer)  ***TL;DR**: This paper presents a new vision Transformer, called Swin Transformer, whose representation is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection.*

 - **Training Data-efficient Image Transformers & Distillation through Attention** `Cited by 782` `ICML` `2021` `Facebook AI` `DeiT` [`PDF`](http://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf) [`Official Code (Stars 2.6k)`](https://github.com/facebookresearch/deit)  ***TL;DR**: In this work, the authors produce competitive convolution-free transformers trained on ImageNet only using a single computer in less than 3 days. They introduce a teacher-student strategy specific to transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention, typically from a convnet teacher.*

 - **Res2Net: A New Multi-Scale Backbone Architecture** `Cited by 755` `TPAMI` `2021` `Nankai University` `Res2Net` [`PDF`](https://arxiv.org/pdf/1904.01169.pdf) [`Official Code (Stars 856)`](https://github.com/Res2Net/Res2Net-PretrainedModels)  ***TL;DR**: The authors propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer.*

 - **Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet** `Cited by 309` `ICCV` `2021` `National University of Singapore` `T2T-ViT` [`PDF`](https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.pdf) [`Official Code (Stars 900)`](https://github.com/yitu-opensource/T2T-ViT)  ***TL;DR**: The authors propose a new Tokens-To-Token Vision Transformer (T2T-ViT), which incorporates 1) a layer-wise Tokens-to-Token (T2T) transformation to progressively structurize the image to tokens by recursively aggregating neighboring Tokens into one Token; 2) an efficient backbone with a deep-narrow structure for vision transformer motivated by CNN architecture design after empirical study.*

 - **CvT: Introducing Convolutions to Vision Transformers** `Cited by 212` `ICCV` `2021` `McGill University` `Microsoft Cloud + AI` `CvT` [`PDF`](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.pdf) [`Official Code (Stars 256)`](https://github.com/microsoft/CvT)  ***TL;DR**: The authors present in this paper a new architecture, named Convolutional vision Transformer (CvT), that improves Vision Transformer (ViT) in performance and efficiency by introducing convolutions into ViT to yield the best of both designs. This is accomplished through two primary modifications: a hierarchy of Transformers containing a new convolutional token embedding, and a convolutional Transformer block leveraging a convolutional projection.*

 - **RepVGG: Making VGG-style ConvNets Great Again** `Cited by 116` `CVPR` `2021` `Tsinghua University` `MEGVII Technology` `RepVGG` [`PDF`](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf) [`Official Code (Stars 2.3k)`](https://github.com/DingXiaoH/RepVGG)  ***TL;DR**: The authors propose a simple but powerful architecture named RepVGG, which has a multi-branch topology in the training and single-branch topology (VGG-like style) in the inference. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique.*

 - **DeepViT: Towards Deeper Vision Transformer** `Cited by 84` `arXiv` `2021` `National University of Singapore` `ByteDance US AI Lab` `DeepViT` [`PDF`](https://arxiv.org/pdf/2103.11886.pdf) [`Official Code (Stars 95)`](https://github.com/zhoudaquan/dvit_repo)  ***TL;DR**: The authors found the attention collapse issue: as the transformer goes deeper, the attention maps gradually become similar and even much the same after certain layers. Based on above observation, we propose a simple yet effective method, named Re-attention, to re-generate the attention maps to increase their diversity at different layers with negligible computation and memory cost.*

 - **LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference** `Cited by 28` `ICCV` `2021` `FAIR` `LeViT` [`PDF`](https://openaccess.thecvf.com/content/ICCV2021/papers/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.pdf) [`Official Code (Stars 412)`](https://github.com/facebookresearch/LeViT)  ***TL;DR**: The authors revisit principles from the extensive literature on convolutional neural networks to apply them to transformers, in particular activation maps with decreasing resolutions. They also introduce the attention bias, a new way to integrate positional information in vision transformers. As a result, LeVIT is proposed, which is a hybrid neural network for fast inference image classification.*



## <a name="2020 Venues">2020 Venues</a>

 - **Self-Training With Noisy Student Improves ImageNet Classification** `Cited by 1.0k` `CVPR` `2020` `Google Research` `NoisyStudent` [`PDF`](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf) [`Official Code (Stars 652)`](https://github.com/google-research/noisystudent)  ***TL;DR**: The authors present a simple self-training method that achieves 88.4% top-1 accuracy on ImageNet. To achieve this result, they first train an EfficientNet model on labeled ImageNet images and use it as a teacher to generate pseudo labels on 300M unlabeled images. They then train a larger EfficientNet as a student model on the combination of labeled and pseudo labeled images. We iterate this process by putting back the student as the teacher.*

 - **RandAugment: Practical automated data augmentation with a reduced search space** `Cited by 754` `CVPR` `2020` `Google Research, Brain Team` `RandAugment` `Data Augmentation` [`PDF`](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf)  [`Third-party Code (Stars 11.2k)`](https://github.com/pytorch/vision/blob/main/torchvision/transforms/autoaugment.py) ***TL;DR**: The authors propose a simplified search space for data augmentation that vastly reduces the computational expense of automated augmentation, and permits the removal of a separate proxy task. Despite the simplifications, our method achieves equal or better performance over previous automated augmentation strategies.*



## <a name="2019 Venues">2019 Venues</a>

 - **Searching for MobileNetV3** `Cited by 1.8k` `ICCV` `2019` `Google AI` `Google Brain` `MobileNetV3` [`PDF`](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf) [`Official Code (Stars 73.1k)`](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) [`Third-party Code (Stars 11.2k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py) ***TL;DR**: This paper presents the next generation of MobileNets (MobileNetV3) based on a combination of complementary architecture search techniques as well as a novel architecture design.*

 - **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features** `Cited by 1.3k` `ICCV` `2019` `Clova AI Research, NAVER Corp.` `CutMix` `Data Augmentation` [`PDF`](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf) [`Official Code (Stars 973)`](https://github.com/clovaai/CutMix-PyTorch)  ***TL;DR**: Prior works have proved to be effective for guiding the model to attend on less discriminative parts of objects (e.g. leg as opposed to head of a person). The authors therefore propose the CutMix augmentation strategy: patches are cut and pasted among training images where the ground truth labels are also mixed proportionally to the area of the patches.*

 - **AutoAugment: Learning Augmentation Policies from Data** `Cited by 908` `CVPR` `2019` `Google Brain` `AutoAugment` `Data Augmentation` [`PDF`](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf)  [`Third-party Code (Stars 11.2k)`](https://github.com/pytorch/vision/blob/main/torchvision/transforms/autoaugment.py) ***TL;DR**: Data augmentation is an effective technique for improving the accuracy of modern image classifiers. However, current data augmentation implementations are manually designed. In this paper, the authors describe a simple procedure called AutoAugment to automatically search for improved data augmentation policies.*

 - **Selective Kernel Networks** `Cited by 789` `CVPR` `2019` `Nanjing University of Science and Technology` `Momenta` `SKNet` [`PDF`](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Selective_Kernel_Networks_CVPR_2019_paper.pdf) [`Official Code (Stars 497)`](https://github.com/implus/SKNet)  ***TL;DR**: The authors propose a dynamic selection mechanism in CNNs that allows each neuron to adaptively adjust its receptive field size based on multiple scales of input information. A building block called Selective Kernel (SK) unit is designed, in which multiple branches with different kernel sizes are fused using softmax attention that is guided by the information in these branches.*



## <a name="2018 Venues">2018 Venues</a>

 - **Squeeze-and-Excitation Networks** `Cited by 11.6k` `CVPR` `2018` `Momenta` `University of Oxford` `SENet` [`PDF`](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf) [`Official Code (Stars 2.9k)`](https://github.com/hujie-frank/SENet)  ***TL;DR**: Based on the benefit of enhancing spatial encoding in prior works, the authors propose a novel architectural unit, which we term the “Squeezeand-Excitation” (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.*

 - <ins>**MobileNetV2: Inverted Residuals and Linear Bottlenecks**</ins> `Cited by 8.7k` `CVPR` `2018` `Google Inc.` `MobileNetV2` [`PDF`](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf) [`Official Code (Stars 73.1k)`](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) [`Third-party Code (Stars 11.2k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py) ***TL;DR**: Based on MobileNetV1, the authors devise a new mobile architecture, MobileNetV2, which is based on an inverted residual structure where the shortcut connections are between the thin bottleneck layers.*

 - **ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices** `Cited by 3.7k` `CVPR` `2018` `Megvii Inc (Face++)` `ShuffleNetV1` [`PDF`](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf)  [`Third-party Code (Stars 1.3k)`](https://github.com/megvii-model/ShuffleNet-Series) ***TL;DR**: The authors introduce an extremely computation-efficient CNN architecture named ShuffleNet, which utilizes two new operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining accuracy.*

 - **mixup: Beyond Empirical Risk Minimization** `Cited by 3.3k` `ICLR` `2018` `MIT` `FAIR` `MixUP` `Data Augmentation` [`PDF`](https://openreview.net/pdf?id=r1Ddp1-Rb) [`Official Code (Stars 902)`](https://github.com/facebookresearch/mixup-cifar10)  ***TL;DR**: The authors propose mixup, a simple learning principle/data augmentation, which trains a neural network on convex combinations of pairs of examples and their labels. By doing so, mixup regularizes the neural network to favor simple linear behavior in-between training examples.*

 - **ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design** `Cited by 2.1k` `ECCV` `2018` `Megvii Inc (Face++)` `Tsinghua University` `ShuffleNetV2` [`PDF`](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.pdf)  [`Third-party Code (Stars 11.2k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py) ***TL;DR**: Prior architecture design is mostly guided by the indirect metric of computation complexity (i.e., FLOPs). In contrast, the authors proposes to use the direct metric (i.e., speed on the target platform) and derives several practical guidelines for efficient network (ShuffleNetV2) design from the empirical observations.*



## <a name="2017 Venues">2017 Venues</a>

 - <ins>**Densely Connected Convolutional Networks**</ins> `Cited by 23.5k` `CVPR` `2017` `Cornell University` `Tsinghua University` `Facebook AI Research` `DenseNet` [`PDF`](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) [`Official Code (Stars 4.4k)`](https://github.com/liuzhuang13/DenseNet) [`Third-party Code (Stars 11.2k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py) ***TL;DR**: The authors observe that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. Based on this, they introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion.*

 - **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications** `Cited by 12.4k` `arXiv` `2017` `Google Inc.` `MobileNetV1` [`PDF`](https://arxiv.org/pdf/1704.04861.pdf) [`Official Code (Stars 73.1k)`](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)  ***TL;DR**: The authors present a class of efficient models called MobileNets for mobile and embedded vision applications, which is a streamlined architecture with depthwise separable convolutions.*

 - **Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning** `Cited by 10.2k` `AAAI` `2017` `Google Inc.` `IneptionV4` [`PDF`](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14806/14311) [`Official Code (Stars 73.1k)`](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py)  ***TL;DR**: The authors propose IneptionV4 by combining Inception architectures with residual connections. Moreover, the authors seek to check if Inception can be more efficient with deeper and wider structure.*

 - **Xception: Deep Learning with Depthwise Separable Convolutions** `Cited by 8.2k` `CVPR` `2017` `Google Inc.` `Xception` [`PDF`](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf)  [`Third-party Code (Stars 8.4k)`](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py) ***TL;DR**: The authors present an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution).*

 - **Aggregated Residual Transformations for Deep Neural Networks** `Cited by 6.4k` `CVPR` `2017` `UC San Diego` `Facebook AI Research` `ResNeXt` [`PDF`](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) [`Official Code (Stars 1.8k)`](https://github.com/facebookresearch/ResNeXt)  ***TL;DR**: This paper presents a simple, highly modularized network architecture for image classification, which is constructed by repeating a building block that aggregates a set of transformations with the same topology.*

 - **Improved Regularization of Convolutional Neural Networks with Cutout** `Cited by 1.6k` `arXiv` `2017` `University of Guelph` `Cutout` `Data Augmentation` [`PDF`](https://arxiv.org/pdf/1708.04552.pdf) [`Official Code (Stars 449)`](https://github.com/uoguelph-mlrg/Cutout)  ***TL;DR**: The authors show that the simple regularization technique of randomly masking out square regions of input during training, called cutout, can be used to improve the robustness and overall performance of convolutional neural networks.*



## <a name="2016 Venues">2016 Venues</a>

 - <ins>**Deep Residual Learning for Image Recognition**</ins> `Cited by 110.6k` `CVPR` `2016` `Microsoft Research` `ResNet` [`PDF`](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) [`Official Code (Stars 5.9k)`](https://github.com/KaimingHe/deep-residual-networks) [`Third-party Code (Stars 11.2k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) ***TL;DR**: This paper presents a residual learning framework (ResNet) to ease the training of networks that are substantially deeper than those used previously, which reformulates the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.*

 - **Rethinking the Inception Architecture for Computer Vision** `Cited by 18.6k` `CVPR` `2016` `Google Inc.` `InceptionV3` [`PDF`](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf) [`Official Code (Stars 73.1k)`](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) [`Third-party Code (Stars 11.2k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py) ***TL;DR**: With version 1 and version 2 of Inception family, the authors want to explore ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization.*

 - **SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters and <0.5MB Model Size** `Cited by 5.4k` `arXiv` `2016` `DeepScale` `SqueezeNet` [`PDF`](https://arxiv.org/pdf/1602.07360.pdf) [`Official Code (Stars 2.1k)`](https://github.com/forresti/SqueezeNet)  ***TL;DR**: This paper presents a small DNN architecture called SqueezeNet, which achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters.*



## <a name="2015 Venues">2015 Venues</a>

 - <ins>**Very Deep Convolutional Networks for Large-Scale Image Recognition**</ins> `Cited by 75.3k` `ICLR` `2015` `Visual Geometry Group` `University of Oxford` `VGG` [`PDF`](https://arxiv.org/pdf/1409.1556.pdf)  [`Third-party Code (Stars 11.2k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py) ***TL;DR**: From the empirical results, the authors found that a network (VGG) with increasing depth and very small ( 3x3) convolution filters would lead to a significant performace improvement based on the prior-art configurations.*

 - <ins>**Going Deeper with Convolutions**</ins> `Cited by 38.2k` `CVPR` `2015` `Google Inc.` `GoogLeNet` `InceptionV1` [`PDF`](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) [`Official Code (Stars 73.1k)`](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py)  ***TL;DR**: The authors propose a deep convolutional neural network architecture codenamed Inception, which adopts multi-branch topology, leading to increasing of the depth and width of the network while keeping the computational budget constant.*

 - **Batch normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** `Cited by 35.2k` `ICML` `2015` `Google Inc.` `Batch Normalization` `InceptionV2` [`PDF`](http://proceedings.mlr.press/v37/ioffe15.pdf) [`Official Code (Stars 73.1k)`](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py)  ***TL;DR**: The authors propose **Batch Normalization**(BN) to alleviate the issue of **internal covariate shift**, which allows us to use much higher learning rates and be less careful about initialization. With the proposed BN, the authors devise a new architecture called InceptionV2.*



## <a name="2012 Venues">2012 Venues</a>

 - <ins>**ImageNet Classification with Deep Convolutional Neural Networks**</ins> `Cited by 105.1k` `NeurIPS` `2012` `University of Toronto` `AlexNet` [`PDF`](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)  [`Third-party Code (Stars 11.2k)`](https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py) ***TL;DR**: This is a pioneering work that exploits a deep convolutional neural network (AlexNet) for large image classification task (ImageNet), which achieves very impressing performance.*




# <a name="Survey">Survey</a>

 - **Transformers in Vision: A Survey** `Cited by 224` `arXiv` `2021` `University of Artificial Intelligence` `Transformers` `Survey` [`PDF`](https://arxiv.org/pdf/2101.01169.pdf)   ***TL;DR**: This survey aims to provide a comprehensive overview of the Transformer models in the computer vision discipline, which includes fundamental concepts of transformers, extensive applications of transformers in vision, the respective advantages and limitations of popular vision transformers and an analysis on open research directions && possible future works.*




# <a name="Datasets">Datasets</a>

 - **ImageNet** [`Download Link`](https://image-net.org/download.php) ***TL;DR**: ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. Currently, the most common used versions in academia are **ImageNet-1k** and **ImageNet-21k**. **1)** ImageNet-1k contains 1,281,167 training images, 50,000 validation images of 1000 object classes. **2)** ImageNet-21K, which is bigger and more diverse, consists of 14,197,122 images, each tagged in a single-label fashion by one of 21,841 possible classes. The dataset has no official train-validation split, and the classes are not well-balanced - some classes contain only 1-10 samples, while others contain thousands of samples. Lastly, it is recommended to download this dataset from [Academic Torrents](https://academictorrents.com/browse.php?search=ImageNet) instead of the official website.* `How to cite:` **Imagenet: A Large-scale Hierarchical Image Database** `Cited by 36.9k` `CVPR` `2009` `Princeton University` `ImageNet` [`PDF`](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf)
 - **CIFAR** [`Download Link`](https://www.cs.toronto.edu/~kriz/cifar.html) ***TL;DR**: There are two versions of CIFAR dataset: **1)** The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. **2)** The CIFAR-100 is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. Please refer to the [official website](https://www.cs.toronto.edu/~kriz/cifar.html) for more details.* `How to cite:` **Learning Multiple Layers of Features from Tiny Images** `Cited by 14.7k` `Tech Report` `2009` `Alex Krizhevsky` `CIFAR-10` `CIFAR-100` [`PDF`](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
 - **Food-101** [`Download Link`](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) ***TL;DR**: It is a challenging data set of 101 food categories, with 101,000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.* `How to cite:` **Food-101–Mining Discriminative Components with Random Forests** `Cited by 868` `ECCV` `2014` `ETH Z¨urich` `F` `o` `o` `d` `-` `1` `0` `1` [`PDF`](https://link.springer.com/content/pdf/10.1007/978-3-319-10599-4_29.pdf)

# <a name="Codebases">Codebases</a>

 - [**rwightman/pytorch-image-models**](https://github.com/rwightman/pytorch-image-models) `Stars 17.1k` `timm` ***TL;DR**: A collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.*

# <a name="Misc">Misc</a>

 - [**Paper with Code**](https://paperswithcode.com/task/image-classification) `paperwithcode` `benchmark` `leaderboard` ***TL;DR**: This website provides a list of state-of-the-art papers and a leaderboard/benchmark of SoTA on varying datasets.*
