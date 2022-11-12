# ### fullySupervised_TilingOnly ####
# python train_cross_teaching_between_cnn_transformer_2D_fullySupervised.py \
# --root_path '../../dataset/Dermatomyositis' \
# --exp 'Dermatomyositis/fullySupervised_CT_Between_CNN_Transformer_TilingOnly' \
# --labeled_num 140 \
# --batch_size 16 \
# --labeled_bs 16/scratch/lc4866/SSL4MedImg/model/Dermatomyositis/fullySupervised_CT_Between_CNN_Transformer_TilingOnly_140

# #### fullySupervised_InterpolateOnly ####
# python train_cross_teaching_between_cnn_transformer_2D_fullySupervised.py \
# --root_path '../../dataset/Dermato_interpolated' \
# --exp 'Dermatomyositis_percentage/fullySupervised_CT_Between_CNN_Transformer_InterpolateOnly' \
# --labeled_num 140 \
# --batch_size 16 \
# --labeled_bs 16

## semi-supervised ####
# python train_cross_teaching_between_cnn_transformer_2D.py \
# --root_path '../../dataset/Dermatomyositis' \
# --exp 'Dermatomyositis_percentage/CT_Between_CNN_Transformer_TilingOnly' \
# --labeled_num 99p \
# --batch_size 16 \
# --labeled_bs 15

# ### with AE, InterpolateOnly, semi-supervised ####
# python train_CT_between_cnn_transformer_2D_withAE.py \
# --root_path '../../dataset/Dermato_interpolated' \
# --exp 'Dermatomyositis_percentage_withAE/CT_Between_CNN_Transformer_InterpolateOnly' \
# --labeled_num 10p \
# --batch_size 16 \
# --labeled_bs 8

### with AE, TilingOnly, semi-supervised ####
python train_CT_between_cnn_transformer_2D_withAE.py \
--root_path '../../dataset/Dermofit' \
--exp 'Dermofit_percentage_withAE/CT_Between_CNN_Transformer_InterpolateOnly' \
--labeled_num 10p \
--batch_size 16 \
--labeled_bs 8



