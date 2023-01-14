# SSL4MedImg

This repo is a replication of [Semi-supervised Learning for Medical Image Segmentation (SSL4MIS)](https://github.com/HiLab-git/SSL4MIS/tree/master/code) that implements the model's application to three different datasets listed below. 

## data 
We currentl support three datasets
- Dermofit
- Dermatomyositis 
- ISIC-2017
    - Download the raw data from [ISIC Challenge Dataset](https://challenge.isic-archive.com/data/#2017) using `wget` and save the data into `../dataset/ISIC2017/original_data` folder. 

See [Data-Efficient Deep Learning model for Dermatomyositis and Dermfit](https://github.com/LuoyaoChen/DEDL_Semisupervised) repo in `Image_Preprocessing` folder for details in how to pre-process data for datasets. 

The folder structure should be as follows: 

```
root - 
    |- SSL4MedImg
        |- code
            |- ...
        |- model (auto-generated)
        |- README.md
        |- ....
    |- dataset
        |- Dermatomyositis
            |- original_data
                |- CD27_Panel_Component
                |- Labels
            |- tile_image
            |- tile_label
            |- interpolated_image
            |- interpolated_label
        |- Dermofit
            |- original_data
                |- Ak 
                |- ...
            |- tile_image
            |- tile_label
            |- interpolated_image
            |- interpolated_label
        |- ISIC2017
            |- original_data
                |- ISIC-2017_Training_Data
                |- ...
            |- tile_image
            |- tile_label
            |- resie_image
            |- resize_label
```

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
python train_train_cross_teaching_between_cnn_transformer_2D.py 
        --labeled_num=${label_num} 
        --exp=${save_path}
        --data_class=${data_class}
        --labeled_bs=${labeled_bs}
        --seed=${seed}
```

- `exp` is the saved directory path for saving model records
- `labeled_num` is the percentage (number) of labeled data (e.g. if want a 30% of training data, will be `--labeled_num 30p`, or if want a specific number of labeled training samples, should utilize the `patient_to_slice` dictionary to input)
- `labeled_bs` is the batch size, it is default to be 16. 
- `seed` indicates which random seed to be used in training. 
- `data_class` indicates which dataset to be used. 

4. Test the model
```
python test_2D_fully.py 
    --exp ${save_dir}
    --labeled_num ${label_num} 
    --one_or_two ${test_model} 
    --data_class ${data_class}
```
- `exp` is the saved directory path
- `labeled_num` is the percentage (number) of labeled data (e.g. if want a 30% of training data, will be `--labeled_num 30p`, or if want a specific number of labeled training samples, should utilize the `patient_to_slice` dictionary to input)
- `one_or_two` indicate whether to use model1 or model2 as the model target to test model performance 
- `data_class` indicates which dataset to be used. Note that the `data_class` and `labeled_num` should be exactly similiar to the tranining configurations, otherwise the test result isn't accurate. 
