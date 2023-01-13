# SSL4MedImg

This repo is a replication of [Semi-supervised Learning for Medical Image Segmentation (SSL4MIS)](https://github.com/HiLab-git/SSL4MIS/tree/master/code) that implements the model's application to three different datasets listed below. 

## data 


### Usage

1. Clone the repo:
```
git clone https://github.com/HiLab-git/SSL4MedImg.git
cd SSL4MedImg
```
2. Download the processed data and put the data in `../dataset/Dermatomyositis, ../dataset/Dermofit, ../dataset/; Download pretrained package to `code/pretrained_ckpt/` directory following its readme.

3. Train the model(change setting based on specific args
```
cd code
python train_train_cross_teaching_between_cnn_transformer_2D.py --
```

4. Test the model
```
python test_2D_fully.py
```
