# Data-Efficient Deep Learning model for Dermatomyositis and Dermfit
[Jupyter notebook version](https://github.com/LuoyaoChen/DEDL_Semisupervised)

Reproduce the segmentation result from ["A Data-Efficient Deep Learning Framework for Segmentation and Classification of Histopathology Images"](https://github.com/pranavsinghps1/DEDL) with application to new dataset Dermofit. 

## Dataset description

### [Dermatomyositis](https://www.sciencedirect.com/science/article/pii/S0022175922000205)

The dataset we used contained 198 tiff image sets containing eight slide images per tiff image. These eight slide images had different protein staining DAPI, CXCR3, CD19, CXCR5, PD1, CD4, CD27 and autofluorescence. The DAPI stained slides were used for image segmentation. These are black and white images with sizes 1408 and 1876. 

### [Dermofit](https://homepages.inf.ed.ac.uk/rbf/DERMOFIT/)
The dataset contains 1300 png image sets containig 10 most commonly observed skin lesions listed below:

    Actinic Keratosis (AK): 45,                
    Basal Cell Carcinoma (BCC): 239,           
    Melanocytic Nevus / Mole (ML): 331,        
    Squamous Cell Carcinoma (SCC) sample 88,   
    Seborrhoeic Keratosis (SK): 257,           
    Intraepithelial carcinoma (IEC): 78,       
    Pyogenic Granuloma (PYO): 24,              
    Haemangioma (VASC): 96,                    
    Dermatofibroma (DF): 65,                   
    Melanoma (MEL): 76，                       
The images are normal RGB captured with a quality SLR camer under controlled (ring flash) indoor lighting; all images have corresponding 1-channel mask images that identify the skin lesion parts. 

## Dependencies 
  

Run the following commands to make sure all dependencies are installed. (Latest version recommended)
```
pip install pytorch_lightning
pip install torchcontrib
pip install torchmetrics
pip install -U git+https://github.com/qubvel/segmentation_models.pytorch
pip install timm
pip install tqdm
pip install einops
pip install argparse
```
  

NOTE: segmentation_models.pytorch installs an older version timm package for itself, which might break some part of the code. So, it is better to uninstall that and install the latest version of timm (version 0.5 and beyond).


We assume that the practitioner has a working installation (latest recommended) of Pytorch, Numpy, matplotlib and pandas.

## File Structure
```
root(netid file) - 
    |- DEDL_Semisupervised_code_version
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
    |- DEDL_Saved_model
```

### Image_preprocessing
- The first step of the code has two ways to preprocess the input image
    - interpolate image
    - tile image
- Python file version that can be used for command line.

### Segmentation_APP_dermofit
- the code for Dermofit dataset
- python file could be run by HPC command line (dataloader, Seg_APP and test_model)

### Segmentation_APP
- the code for Dermatomyositis dataset
- python file could be run by HPC command line (dataloader, Seg_APP and test_model)