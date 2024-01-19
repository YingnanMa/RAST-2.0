# RAST-TOMM

This is the official PyTorch implementation of our paper: ["RAST: Restorable Arbitrary Style Transfer"](https://dl.acm.org/doi/abs/10.1145/3638770)(**TOMM 2023**)   


The objective of arbitrary style transfer is to apply a given artistic or photo-realistic style to a target image. Although current methods have shown some success in transferring style, arbitrary style transfer still has several issues, including content leakage. Embedding an artistic style can result in unintended changes to the image content. This paper proposes an iterative framework called Restorable Arbitrary Style Transfer (RAST) to effectively ensure content preservation and mitigate potential alterations to the content information. RAST can transmit both content and style information through multi-restorations and balance the content-style trade-off in stylized images using the image restoration accuracy. To ensure RAST’s effectiveness, we introduce two novel loss functions: multi-restoration loss and style difference loss. We also propose a new quantitative evaluation method to assess content preservation and style embedding performance. Experimental results show that RAST outperforms state-of-the-art methods in generating stylized images that preserve content and embed style accurately.

<div align=center>
<img src="https://github.com/xudongLi-Alex/RAST/blob/main/pic.png" width="1200" alt="Pipeline"/><br/>
</div>


## Requirements  
- python 3.8
- PyTorch 1.8.0
- CUDA 11.1


## Please take note of the following considerations
- We train RAST with three network architectures, AdaIN, IEAST, and SANet, each divided into combine and replace methods. If you want to training or testing, please proceed to the root directory of each method. The training and testing steps are outlined below:

## Model Testing
- Download [VGG pretrained](https://drive.google.com/file/d/1cI6ubAziMdOsSJZEvfofW-iCtnCmsONL/view?usp=share_link) model to ./model/ folder.
- Put content images to *./content/* folder.
- Put style images to *./style/* folder.
- Run the following command:
```
python eval.py --content_dir ./content/ --style_dir ./style/
```
## Model Training
- Create model,coco_train, wiki_train folder.
- Download [VGG pretrained](https://drive.google.com/file/d/1cI6ubAziMdOsSJZEvfofW-iCtnCmsONL/view?usp=share_link) model to *./model/* folder.
- Download COCO2014 dataset to *./coco_train/* folder
- Download Wiki dataset to *./wiki_train/* folder
- Run the following command:
```
python train.py --content_dir ./coco_train/ --style_dir ./wiki_train/
```
