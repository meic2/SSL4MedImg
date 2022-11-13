# Segmentation APP
The original [code](https://github.com/pranavsinghps1/DEDL) comes from the [paper](https://arxiv.org/abs/2207.06489)

1. Seg_APP.py is the main file

2. Seg_APP.SBATCH is the file running the HPC 
* please change the encoder_name, e.g. resnet18
* please change the mail-user in line 10
* please change the environment in line 16

3. Seg_APP.out is the output of Seg_APP.py

4. test_model.py could test the existent model

5. test_model.SBATCH is the file running the HPC 
* please change the model_name, e.g. Unet_resnet18_20221001_02.pt
* please change the mail-user in line 10
* please change the environment in line 16

6. test_model.out is the output of test_model.py