import argparse
from dataclasses import dataclass
import os
import shutil
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
from networks.net_factory import net_factory
from dataloaders.dermofit_processing import build_dataset_ssl
from torch.utils.data import DataLoader
from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg
from medpy import metric
# from scipy.ndimage import zoom
import segmentation_models_pytorch as smp



parser = argparse.ArgumentParser()
parser.add_argument('--one_or_two', type=int, default=None,
                    help = 'test CNN (if input 1) or Transformer(if input 2)')
parser.add_argument('--data_class', type=int, default=None,
                    help = '1 for Dermofit, 2 for Dermatomyositis TilingOnly, 3 for Dermatomyositis interpolateOnly')
parser.add_argument('--root_path', type=str,
                    default='../../dataset/Dermatomyositis', help='Name of dataset')
parser.add_argument('--exp', type=str,
                    default='Dermatomyositis/CT_Between_CNN_Transformer_TilingOnly', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--saved_model_path', type=str,
                    default='Default', help='model you are testing')
parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size per gpu')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=str, default=7,
                    help='labeled data')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')

parser.add_argument(
        '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true',
                        help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')

FLAGS = parser.parse_args()
config = get_config(FLAGS)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if FLAGS.data_class ==1:
    DATA_PATH = '../../dataset/Dermofit/original_data/'
    TILE_IMAGE_PATH = '../../dataset/Dermofit_resize_noTiling/resize_image/'
    TILE_LABEL_PATH = '../../dataset/Dermofit_resize_noTiling/resize_label/'
elif FLAGS.data_class ==2:
    DATA_PATH = '../../dataset/Dermatomyositis/original_data/'
    TILE_IMAGE_PATH = '../../dataset/Dermatomyositis/tile_image/'
    TILE_LABEL_PATH = '../../dataset/Dermatomyositis/tile_label/'
elif FLAGS.data_class ==3:
    DATA_PATH = '../../dataset/Dermatomyositis/original_data/'
    TILE_IMAGE_PATH = '../../dataset/Dermatomyositis/InterpolateOnly_image/'
    TILE_LABEL_PATH = '../../dataset/Dermatomyositis/InterpolateOnly_label/'
elif FLAGS.data_class ==4:
    DATA_PATH = '../../dataset/ISIC2017/original_data/'
    TILE_IMAGE_PATH = '../../dataset/ISIC2017/resize_image/'
    TILE_LABEL_PATH = '../../dataset/ISIC2017/resize_label/'
def calculate_metric_iou(pred, label):
    pred = torch.tensor(pred)
    label = torch.tensor(label)
    tp, fp, fn, tn = smp.metrics.get_stats(pred, label.long(), mode='multilabel', threshold=0.5)
    # accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    return iou_score

def calculate_metric_percase(pred, gt):
    ## change both to np arr
    pred = pred.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if gt.sum()==0:
        return []
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        iou = calculate_metric_iou(pred, gt)

        return dice, hd95, asd, iou
    else:
        print(f"pred: {pred}\n gt: {gt}")
        return 0,0,0,0


def test_single_volume(sample, image_name, net, test_save_path, FLAGS):
    # image_name: str, eg: '121919_Myo231_[9554,43072]_component_data_0'
    image = sample['image']
    label = sample['label']
    # print(f"image.shape: {image.shape}, label.shape: {label.shape}") # image.shape: torch.Size([1, 480, 480]), label.shape: torch.Size([1, 480, 480])
    image = image.unsqueeze(0).to(device)
    label = label.squeeze(0).to(device)
    # prediction = np.zeros_like(label)
    # for ind in range(image.shape[0]):
        # slice = image[ind, :, :]
        # x, y = slice.shape[0], slice.shape[1]
        # slice = zoom(slice, (256 / x, 256 / y), order=0)
        # input = torch.from_numpy(slice).unsqueeze(
        #     0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        if FLAGS.model == "unet_urds":
            out_main, _, _, _ = net(image)
        else:
            out_main = net(image)
        out = torch.argmax(torch.softmax(
            out_main, dim=1), dim=1).squeeze(0)
        # out = out.cpu().detach().numpy()
        prediction = out
        # print(f"prediction.shape: {prediction.shape}") # prediction.shape: (480, 480)


    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    # second_metric = calculate_metric_percase(prediction == 2, label == 2)
    # third_metric = calculate_metric_percase(prediction == 3, label == 3)

    print(f"saving visualization to directory = {test_save_path}")
    np.save(f'{test_save_path}{image_name}_data.npy', image.cpu().detach().numpy())
    np.save(f'{test_save_path}{image_name}_mask.npy', label.cpu().detach().numpy())
    np.save(f'{test_save_path}{image_name}_pred.npy', prediction.cpu().detach().numpy())
    # img_itk = sitk.GetImageFromArray(image.cpu().detach().numpy().astype(np.float32))
    # img_itk.SetSpacing((1, 1, 10))
    # prd_itk = sitk.GetImageFromArray(prediction.cpu().detach().numpy().astype(np.float32))
    # prd_itk.SetSpacing((1, 1, 10))
    # lab_itk = sitk.GetImageFromArray(label.cpu().detach().numpy().astype(np.float32))
    # lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + image_name + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + image_name + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + image_name + "_gt.nii.gz")
    return first_metric#, second_metric, third_metric

def Inference(FLAGS):
    _, _, test_dataset, _, _, test_list = build_dataset_ssl(data_path=DATA_PATH, 
                                            tile_image_path=TILE_IMAGE_PATH,
                                            tile_label_path=TILE_LABEL_PATH, 
                                            dataclass=FLAGS.data_class, 
                                            output_size=FLAGS.patch_size)
    
    test_save_path = f"../model/{FLAGS.exp}_{FLAGS.labeled_num}/{FLAGS.model}_model{FLAGS.one_or_two}_predictions/"
    print(f"test is saved to path = {test_save_path}")
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    if FLAGS.one_or_two == 1:   ## CNN
        print('testing model1 (UNET)...')
        net = net_factory(FLAGS, config, net_type=FLAGS.model, in_chns=3 if (FLAGS.data_class == 1 or FLAGS.data_class == 4) else 1,
                        class_num=FLAGS.num_classes)
    elif FLAGS.one_or_two == 2: ## transformer
        print('testing model2 (SwinTransformer)...')
        net = ViT_seg(config, img_size=FLAGS.patch_size,
                     num_classes=FLAGS.num_classes).to(device)
    
    saved_model_path = f"../model/{FLAGS.exp}_{FLAGS.labeled_num}/{FLAGS.model}/{FLAGS.model}_best_model{FLAGS.one_or_two}.pth"
    if FLAGS.saved_model_path != 'Default':
        saved_model_path = FLAGS.saved_model_path
    print("init weight from {}".format(saved_model_path)) 
    net.load_state_dict(torch.load(saved_model_path))
    net.eval()

    first_total = 0.0
    # second_total = 0.0
    # third_total = 0.0
    for idx in tqdm(range(len(test_dataset))):
        # first_metric, second_metric, third_metric = test_single_volume(
        #     test_dataset[idx], net, test_save_path, FLAGS)
        # print(f"testing image name = {test_list[idx]}") # testing image name = ('121919_Myo231_[9554,43072]_component_data_0.npy', '121919_Myo231_[9554,43072]_component_mask_0.npy')
        first_metric= test_single_volume(
            test_dataset[idx], test_list[idx][0][:-4], net, test_save_path, FLAGS)
        if first_metric != []:
            first_total += np.asarray(first_metric)
        # second_total += np.asarray(second_metric)
        # third_total += np.asarray(third_metric)
    # avg_metric = [first_total / len(image_list), second_total /
    #               len(image_list), third_total / len(image_list)]
    avg_metric = [first_total / len(test_dataset)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    config = get_config(FLAGS)
    metric = Inference(FLAGS)
    print(f"dice, hd95, asd, iou: {metric}")
    # print((metric[0]+metric[1]+metric[2])/3)
