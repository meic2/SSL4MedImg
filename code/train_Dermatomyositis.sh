# ### fullySupervised_TilingOnly ####
# python train_cross_teaching_between_cnn_transformer_2D_fullySupervised.py \
# --root_path '../../dataset/Dermatomyositis' \
# --exp 'Dermatomyositis/fullySupervised_CT_Between_CNN_Transformer_TilingOnly' \
# --labeled_num 140 \
# --batch_size 16 \
# --labeled_bs 16/scratch/lc4866/SSL4MedImg/model/Dermatomyositis/fullySupervised_CT_Between_CNN_Transformer_TilingOnly_140

#### fullySupervised_InterpolateOnly ####
# python train_cross_teaching_between_cnn_transformer_2D_fullySupervised.py \
# --root_path '../../dataset/Dermatomyositis' \
# --exp 'Dermatomyositis/fullySupervised_CT_Between_CNN_Transformer_InterpolateOnly' \
# --labeled_num 140 \
# --batch_size 16 \
# --labeled_bs 16

### semi-supervised ####
python train_cross_teaching_between_cnn_transformer_2D.py \
--root_path '../../dataset/Dermatomyositis' \
--exp 'Dermatomyositis/CT_Between_CNN_Transformer_TilingOnly' \
--labeled_num 3 \
--batch_size 16 \
--labeled_bs 8

