python test_2D_fully.py \
--root_path ../../dataset/Dermatomyositis \
--exp Dermatomyositis_percentage/CT_Between_CNN_Transformer_TilingOnly \
--saved_model_path /scratch/lc4866/SSL4MedImg/model/Dermatomyositis_percentage/CT_Between_CNN_Transformer_TilingOnly_99p/unet/unet_best_model2.pth \
--data_class 2 \
--labeled_num 99p \
--one_or_two 2 \
--patch_size [480,480]
