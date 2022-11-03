python test_2D_fully.py \
--root_path ../../dataset/Dermatomyositis \
--exp Dermatomyositis/CT_Between_CNN_Transformer_TilingOnly \
--saved_model_path /scratch/lc4866/SSL4MedImg/model/Dermatomyositis/unet_best_model2.pth \
--labeled_num 7 \
--one_or_two 2 \
--patch_size [480,480]
