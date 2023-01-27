# this is example to train Dermatomyositis_percentage_CEweight_CT_Between_CNN_Transformer_TilingOnly_10p_seed73
python train_CT_between_cnn_transformer_2D.py \
--root_path '../../dataset/Dermatomyositis' \
--exp 'Dermatomyositis_percentage_CEweight/CT_CNN_Transformer_TilingOnly_seed73' \
--labeled_num 10p \
--data_class 2 \
--batch_size 16 \
--labeled_bs 8 \
--seed 73