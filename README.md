# MRFIE
This is an implementation of Multi-view Representation Learning with Refined Fusion Information Exploration (MRFIE) in Pytorch.
## Requirements
  * Python=3.6.5  
  * Pytorch=1.6.0  
  * Torchvision=0.7.0
## Datasets
The model is trained on AWA/Caltech101/NUSWIDEOBJ/Reuters/CIFAR-10/Flowers-102 dataset, where each dataset are splited into two parts: 70% samples for training, 20% samples for validating, and the rest 10% for testing.  We utilize the classification accuracy to evaluate the performance of all the methods.
## Implementation

#Train/Test the model on Caltech20 dataset

`` python Train_HAO.py --dataset_dir=./mvdata/Caltech101-20 --data_name=Caltech20 --num_classes=20 --num_view=6 ``
