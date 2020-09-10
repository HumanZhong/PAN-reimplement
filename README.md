# PAN-reimplement
This is an unofficial pytorch re-implementation of "Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network"(19'ICCV,PAN/PSENet-v2) and some personal modifications.
* Thanks to the help of the official PSENet implementation[https://github.com/whai362/PSENet] and PAN.pytorch[https://github.com/WenmuZhou/PAN.pytorch]. This repo is mainly based on the PSENet repo and followed a similar framework.
* While PSENet is implemented in Python2, this repo uses Python3 to implement both PSENet and PAN. Compared to the PAN.pytorch repo, this repo fixes some bugs and achieves a higher f-score(but is still lower than the original paper by ~0.5%, sigh)
* **The author of PSENet and PAN has already released the official implementation of PAN[https://github.com/whai362/pan_pp.pytorch]. Since my re-implementation can not perfectly reproduce the results, I suggest that referring to the official repo might be a better choice.**

## Packages required
* pytorch 1.1.0 & torchvision
* Polygon3
* pyclipper
* pybind11(to compile the cpp version pse algorithm)

## How to train PSE
* train on icdar  
`CUDA_VISIBLE_DEVICES=0,1,2,3 python train_icdar15.py --backbone res18/res50 --lr YOUR.LR --resume YOUR.CHECKPOINT_FILE`
* train on ctw  
`CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ctw1500.py --backbone res18/res50 --lr YOUR.LR --resume YOUR.CHECKPOINT_FILE`

## How to train PAN
* train on icdar  
`CUDA_VISIBLE_DEVICES=0,1,2,3 python train_icdar15_PAN{_v2}.py --backbone res18/res50 --lr YOUR.LR --resume YOUR.CHECKPOINT_FILE`
* train on ctw  
`CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ctw1500_v2.py --backbone res18/res50 --lr YOUR.LR --resume YOUR.CHECKPOINT_FILE`
* Other training scripts are variants which have external extension like top/bottom boundary prediction. You can check the codes for detailed information

## Experimental Results
(TODO)

## Future work & Note
* This version is just a preliminary manuscript and has not been organized well. Sorry for organizing the codes in such a messy way, I'll update a cleaner version as soon as possible.
* The detailed performance of PAN/PSE (this repo's performance has a slightly drop compared to the original PAN paper)  
* ~~and a guide to the training process will also be uploaded soon. (done)~~
* Detailed information of my personal modifications(top/bottom boundary prediction) will also be released in the future.
* If this repo really helps you, please star it, thx!
