python test_2D_fully.py \
--root_path ../../dataset/Dermato_interpolated \
--exp Dermatomyositis_percentage_withAE/CT_Between_CNN_Transformer_InterpolateOnly \
--saved_model_path /scratch/lc4866/SSL4MedImg/model/Dermatomyositis_percentage_withAE/CT_Between_CNN_Transformer_InterpolateOnly_30p/unet/unet_best_model2.pth \
--data_class 3 \
--labeled_num 30p \
--one_or_two 2 \
--patch_size [480,480]
