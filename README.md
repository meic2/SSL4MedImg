# SSL4MedImg

This repo is a replication of [Semi-supervised Learning for Medical Image Segmentation (SSL4MIS)](https://github.com/HiLab-git/SSL4MIS/tree/master/code) that implements the *Cross Teaching Between CNN and Transformer*'s application to three different datasets listed below. 

## Data 
We currently support three datasets listed below: 

#### Dermofit
This is a paid dataset sourced by the University of Edinburgh, and it contains 1,300 samples of high quality skin lesions. [[1]]

#### Dermatomyositis 
This is a private dataset [Van Buren et al.] [[2]] of autoimmunity biopsies of 198 samples. 

#### ISIC-2017
This is a collection of 2000 lesion images in JPEG format and 2000 corresponding superpixel masks in PNG format, with EXIF data stripped. For retrieval of data, please download the raw data (including train, validation and test sets) from [ISIC Challenge Dataset](https://challenge.isic-archive.com/data/#2017) using `wget` and save the data into `../dataset/ISIC2017/original_data` folder. 

## Preprocessing

See [Data-Efficient Deep Learning model for Dermatomyositis and Dermfit](https://github.com/LuoyaoChen/DEDL_Semisupervised) repository in `Image_Preprocessing` folder for details of pre-processing. Specifically, in previous experiments we utilized interpolation/tiling method to process all raw images/label sets into the same size. 

To make sure the training procedure to be as smoothly as possible, we advise to structure the `dataset/` folder as follows: 

```
root - 
    |- SSL4MedImg
        |- code
            |- train_CT_between_cnn_transformer_2D.py
            |- ...
        |- environment.yml
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
            |- interpolated_image
            |- interpolated_label
```
## Install 

1. Clone the repo:

```
git clone https://github.com/HiLab-git/SSL4MedImg.git
cd SSL4MedImg
```

2. Environmental installation：

    Please make sure your Singularity Overlays still have more than 13GB capacity.

    And run the code below:
```
conda env create -f environment.yml
```
    After running code, You will get a python environment called `ssl`, and all the required packages of this repo will be installed. 

3. Check points:

    Please download pretrained check points from https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY and put it int `code/pretrained_ckpt/` directory.

## Usage
### Exploration of Files
Here listed the usage for each directory/files in this repo. 

- `code/` directory 
    
    The repository is structured in a way such that all necessary coding files are listed in `code/` directory. Within this directory, current parameter setting should already aligns with preprocessed data size (e.g., each interpolated image has a size of `224*224`). However, if parameter setting needs a further update, please update `code/configs/`,  `code/networks/` and `code/config.py` correspondingly. 
    - `code/pretrained_ckpt` utilize the same Swin-Transformer pretrained model as SSL4MIS repository, please download the model following `code/pretrained_ckpt/README.md` correspondingly. 
    - `code/dataloaders/` contains the dataloader functions that loads the preprocessed data into a dataloader. 
    - `code/train_CT_between_cnn_transformer_2D.py` is the main training file for training semi-supervised model with different labeled ratio. Please utilize the file according to the below procedure. 
    - `code/val_2D.py` and `code/test_2D.py` implements the `dice, hd95, asd, iou` score for validation sets and test sets. Note that validation score is already included in the training process.

- `model/` directory (auto-generated after training)

    If properly run, `model/` directory should be automatically generated with `model/${save_dir}` subdirectory that contains all training records and saved trained model from the training. 

### Example Procedure
The overall routine of the training/testing procedure are as follows: 

1. Data:

    Please download, process, and put the data in `../dataset/Dermatomyositis`, `../dataset/Dermofit`, and  `../dataset/ISIC2017` folder. 

2. Train the model(change setting based on specific args)
```
cd code
python train_CT_between_cnn_transformer_2D.py.py 
        --exp=${save_path}
        --labeled_num=${label_num} 
        --data_class=${data_class}
        --labeled_bs=${labeled_bs}
        --seed=${seed}
```

- `exp` is the saved directory path for saving model records
- `labeled_num` is the percentage (number) of labeled data (e.g. if one want a 30% of training data, will be `--labeled_num 30p`, or if one want a specific number of labeled training samples, one should utilize the `patient_to_slice()` method to input desired number of labeled samples)
- `labeled_bs` is the batch size, it sets default to be 16. 
- `seed` indicates which random seed to be used in training. 
- `data_class` indicates which dataset to be used
    - `1`: interpolatd Dermofit dataset
    - `2`: tiled Dermatomyositis dataset
    - `3`: interpolatd Dermatomyositis dataset
    - `4`: interpolatd ISIC2017 dataset

4. Test the model
```
python test_2D.py 
    --exp ${save_dir}
    --labeled_num ${label_num} 
    --one_or_two ${test_model} 
    --data_class ${data_class}
```
- `exp` is the saved directory for training model 
- `labeled_num` is the percentage (number) of labeled data 
- `one_or_two` indicates which model will be used in testing the model performance 
- `data_class` indicates which dataset to be used. 

Note that the `data_class`,  `labeled_num`, `exp` args should be exactly the same to the tranining configurations, otherwise the test result isn't accurate.



[1]: https://licensing.edinburgh-innovations.ed.ac.uk/product/dermofit-image-library

[2]: https://www.sciencedirect.com/science/article/abs/pii/S0022175922000205
