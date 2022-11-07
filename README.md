# SSL4MedImg

### Usage

Clone the repo:
```
git clone https://github.com/HiLab-git/SSL4MedImg.git
cd SSL4MedImg
```
Download the processed data and put the data in `../dataset/Dermatomyositis` and `../dataset/Dermofit`

Train the model(change setting based on specific args
```
cd code
python train_train_cross_teaching_between_cnn_transformer_2D.py
```

Test the model

```
python test_2D_fully.py
```
