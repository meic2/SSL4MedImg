# Github repo for CDS capstone project Fall 2022.

## File Path
```
root(netid file) - 
    |- DEDL_Semisupervised
    |- dataset
        |- Dermatomyositis
            |- original_data
                |- CD27_Panel_Component
                |- Labels
            |- tile_image
            |- tile_label
        |- Dermofit
            |- original_data
                |- Ak 
                |- ...
            |- tile_image
            |- tile_label
    |- DEDL_Saved_model
```

### Image_preprocessing
1. Jupyter version
2. Python file version: The files could be used for command line.

### sample_data
It has 7 sample data comes from Dermatomyositis -- CD27 Panel_Component and 2 sample data comes from Dermofit. \
The structure is same as **dataset**

### Segmentation_APP_dermofit
1. Jupyter notebook for running code
2. Jupyter notebook for visulize dataset
3. Jupyter notebook for check accurancy and IoI

### Segmentation_APP
1. Jupyter notebook for running code
2. python file could be run by HPC command line (dataloader, Seg_APP and test_model)

## Update
### 20220928
For slides:
* Use one result of three
* Do more official

For model:
* Unet to ResNet-18 
* Tiling: threshold for padding is minimal: generate 
* Try different Normalization

Task next week:
- [ ] Improve Dermatomyositis IoU to 0.4-0.5
- [ ] Present potential related work for any image tiling/Dermofit 
- [ ] Parallel: adapt the current dataset to new code repo


### 20220921
For slides:
* have title/summary for each slide
* Use jupyter notebook not jupyterLab

Task next week:
- [x] Familiar with [paper](https://openreview.net/forum?id=KUmlnqHrAbE) and [code](https://github.com/HiLab-git/SSL4MIS)

### 20220914
- [x] Setup HPC
- [x] Running the [code](https://github.com/pranavsinghps1/DEDL)