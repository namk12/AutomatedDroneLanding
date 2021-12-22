# DeepLearning-FinalProject
Drone Landing Site Detection using Semantic Segmentation

## Dataset
The dataset is freely available from http://dronedataset.icg.tugraz.at
Dataset obtained from - https://www.tugraz.at/index.php?id=22387
The dataset contains 400 training and 200 test set images of size 6000 x 4000px. The dataset depicts more than 20 houses from nadir view acquired at an altitude of 5 to 30 meters above ground.

## Introduction
Finding appropriate landing sites on land in dynamic environments can be used for unmanned aerial vehicles or drones. A large number of factors affect the suitability of potential landing sites from trees and buildings to someone walking their dog in the area. Thus both moving objects and more permanent ground changes can influence the safety of a landing zone. Deploying a trained machine learning model on a drone to recognise landing spots can reduce energy consumption by removing the need for an internet connection to send data back and forth to a hub therefore this paper proposes a UAV safe landing navigation pipeline that relies on computer vision methodology, able to be executed on the limited computational resources on-board a drone system. Aerial images are analyzed and segmented on 23 different types of obstacles like grass, gravel, dirt, to provide safe surface landing analysis data. Pre-trained Neural Networks (NNs) are mainly employed as the underlying building blocks, since deep learning has made a major impact on robotic perception by drastically improving the performance of relevant tasks, such as object detection or tracking, semantic image segmentation, etc.

## Goal
The goal of this work is to implement an image-based method to assess the appropriateness of drone landing sites proposed using the semantic segmentation model.
We aim to achieve following objectives with this project:
1. Model that is light weight
2. High accuracy and fast segmentation of surface image, possibly through a real time video.
3. Detection of safe landing surface.
4. Fast Inference Latency

## Instructions
1. Download the dataset from the [link](https://www.tugraz.at/index.php?id=22387) and place it in google drive location "/content/drive/MyDrive/DeepLearning/".
2. Run the notebook [Safe_Drone_Landing_Prediction_Semantic_Segmentation.ipynb](https://github.com/namk12/DeepLearning-FinalProject/blob/main/Safe_Drone_Landing_Prediction_Semantic_Segmentation.ipynb) for prediction.
3. For testing, the trained model "Unet-Mobilenet.pt" can be loaded (provided in the repo).

## References
1. [Understanding Semantic Segmentation with UNET](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)
2. [Review On Mobile Net v2](https://medium.datadriveninvestor.com/review-on-mobile-net-v2-ec5cb7946784)
3. [ResNeXt Explained, Part 1](https://medium.datadriveninvestor.com/resnext-explained-part-1-17a6b510fb0a)
4. [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)
5. [TorchVision Transforms: Image Preprocessing in PyTorch](https://sparrow.dev/torchvision-transforms/)
6. [Review: MobileNetV1 — Depthwise Separable Convolution (Light Weight Model)](https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-weight-model-a382df364b69)
7. [Albumenation](https://albumentations.ai/)
8. [Up-sampling with Transposed Convolution](https://naokishibuya.medium.com/up-sampling-with-transposed-convolution-9ae4f2df52d0)
9. [Modified U-Net Model with hyperparameter tuning](https://www.kaggle.com/weilinku/modified-u-net-model-with-hyperparameter-tuning#Overview)
