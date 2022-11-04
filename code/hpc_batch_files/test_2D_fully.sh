python test_2D_fully.py \
--root_path ../../dataset/Dermato_interpolated \
--exp Dermatomyositis_percentage/fullySupervised_CT_Between_CNN_Transformer_InterpolateOnly \
--saved_model_path /scratch/lc4866/SSL4MedImg/model/Dermatomyositis_percentage/fullySupervised_CT_Between_CNN_Transformer_InterpolateOnly_100p/unet/model2_iter_6000.pth \
--data_class 3 \
--labeled_num 100p \
--one_or_two 2 \
--patch_size [480,480]
