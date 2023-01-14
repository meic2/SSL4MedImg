# SSL4MedImg

This repo is a replication of [Semi-supervised Learning for Medical Image Segmentation (SSL4MIS)](https://github.com/HiLab-git/SSL4MIS/tree/master/code) that implements the model's application to three different datasets listed below. 

## data 
We currently support three datasets listed below: 

### Dermofit
This is a paid dataset sourced by the University of Edinburgh, it contains 1,300 samples of high quality skin lesions. [1]

### Dermatomyositis 
This is a private dataset [Van Buren et al.] [2] of autoimmunity biopsies of 198 samples. This is a multi-label class classification. For train/validation/test splits we follow an 80/10/10 split.

### ISIC-2017
This is a collection of 2000 lesion images in JPEG format and 2000 corresponding superpixel masks in PNG format, with EXIF data stripped. For retrieval of data, please download the raw data (including train, validation and test sets) from [ISIC Challenge Dataset](https://challenge.isic-archive.com/data/#2017) using `wget` and save the data into `../dataset/ISIC2017/original_data` folder. 

### Preprocessing

See [Data-Efficient Deep Learning model for Dermatomyositis and Dermfit](https://github.com/LuoyaoChen/DEDL_Semisupervised) repository in `Image_Preprocessing` folder for details in pre-processing datasets and saving preprocessed datasets into destinated folder. 

The processed folder structure should be as follows: 

```
root - 
    |- SSL4MedImg
        |- code
            |- dataloaders
            |- configs
            |- augmentations
            |- networks
            |- pretrained_ckpt
            |- train_CT_between_cnn_transformer_2D.py
            |- val_2D.py
            |- test_2D.py
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
## Requirement 
Please use `environment.yml` file to install all required dependencies. 

## Usage
### Exploration of Files
- `model/` directory 
If follows the next section properly, `model/` directory should be automatically generated with `model/${save_dir}` subdirectory that contains all training records and saved trained model from the training. 

- `code/` directory 
The repository is structured in a way such that all necessary coding files are listed in `code/` directory. Within this directory, current parameter setting should already aligns with preprocessed data size (e.g., each interpolated image has a size of `224*224`). However, if parameter setting needs further update, please update `code/configs/`,  `code/networks/` and `code/config.py` correspondingly. 
    - `code/pretrained_ckpt` utilize the same Swin-Transformer pretrained model as SSL4MIS, please download the model following its readme correspondingly. 
    - `code/train_CT_between_cnn_transformer_2D.py` is the main training file for training semi-supervised model with different labeled ratio. Please utilize the file according to the below procedure. 
    - `code/val_2D.py` and `code/test_2D.py` implements the `dice, hd95, asd, iou` score for validation sets and test sets. Note that validation score is already included in the training process.

### Example Procedure
The overall routine of the training/testing procedure are as follows: 

1. Clone the repo:
```
git clone https://github.com/HiLab-git/SSL4MedImg.git
cd SSL4MedImg
```

2. Download, process, and put the data in `../dataset/Dermatomyositis, ../dataset/Dermofit, ../dataset/ISIC2017` folder; Download pretrained package to `code/pretrained_ckpt/` directory following its readme.

3. Train the model(change setting based on specific args)
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


## Reference 
- [1]:https://licensing.edinburgh-innovations.ed.ac.uk/product/dermofit-image-library
- [2]: https://www.sciencedirect.com/science/article/abs/pii/S0022175922000205
- 
