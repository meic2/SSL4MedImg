# -*- coding: utf-8 -*-
# Author: Xiangde Luo
# Date:   16 Dec. 2021
# Implementation for Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer.
# # Reference:
#   @article{luo2021ctbct,
#   title={Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer},
#   author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
#   journal={arXiv preprint arXiv:2112.04894},
#   year={2021}}
#   In the original paper, we don't use the validation set to select checkpoints and use the last iteration to inference for all methods.
#   In addition, we combine the validation set and test set to report the results.
#   We found that the random data split has some bias (the validation set is very tough and the test set is very easy).
#   Actually, this setting is also a fair comparison.
#   download pre-trained model to "code/pretrained_ckpt" folder, link:https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY

import argparse
import logging
import os
import random
import shutil
import sys
import time
from tkinter import S

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from dataloaders import utils
from dataloaders.dataset import TwoStreamBatchSampler #BaseDataSets, RandomGenerator,
from dataloaders.dermofit_processing import build_dataset_ssl
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, metrics, ramps
from val_2D import test_single_volume
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--data_class', type=int, default=None,
                    help = '1 for Dermofit, 2 for Dermatomyositis TilingOnly, 3 for Dermatomyositis interpolateOnly')
parser.add_argument('--exp', type=str,
                    default='Dermofit/Cross_Teaching_CNN_Transformer', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1234, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
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

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=str, default='30p',
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)
print(args)

if args.data_class == 1:
    DATA_PATH = '../../dataset/Dermofit/original_data/'
    TILE_IMAGE_PATH = '../../dataset/Dermofit_resize_noTiling/resize_image/'
    TILE_LABEL_PATH = '../../dataset/Dermofit_resize_noTiling/resize_label/'
if args.data_class == 2:
    DATA_PATH = '../../dataset/Dermatomyositis/original_data/'
    TILE_IMAGE_PATH = '../../dataset/Dermatomyositis/tile_image/'
    TILE_LABEL_PATH = '../../dataset/Dermatomyositis/tile_label/'

elif args.data_class == 3:
    DATA_PATH = '../../dataset/Dermatomyositis/original_data/'
    TILE_IMAGE_PATH = '../../dataset/Dermatomyositis/interpolateOnly_image/'
    TILE_LABEL_PATH = '../../dataset/Dermatomyositis/interpolateOnly_label/'

elif args.data_class == 4:
    DATA_PATH = '../../dataset/ISIC2017/original_data/'
    TILE_IMAGE_PATH = '../../dataset/ISIC2017/resize_image/'
    TILE_LABEL_PATH = '../../dataset/ISIC2017/resize_label/'


print(torch.cuda.is_available())
def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataclass, patiens_num, traindataset_len, UsePercentage_flag = False):
    # this method splits the dataset into labeled and unlabeled train dataset based on either percentage flags or customed number of labeled instance
    ref_dict = None
    ref_dict_percentage = {"10p": round(0.1*traindataset_len), "30p": round(0.3*traindataset_len), "50p": round(0.5*traindataset_len),
                        "70p": round(0.7*traindataset_len), "99p": traindataset_len-1, "100p": 1*traindataset_len} 
    if dataclass == 1: #Dermofit
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif dataclass == 2: # Dermatomyositits
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif dataclass == 3: # interpolated Dermatomyositits
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 121}
    elif dataclass == 4: # interpolated ISIC 2017
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 121}
    else:
        print("Error")
   
    if UsePercentage_flag:
        return ref_dict_percentage[str(patiens_num)]
    else:
        return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(args, config, net_type=args.model, in_chns=3 if (args.data_class==1 or args.data_class==4) else 1, 
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    print(config)
    model2 = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    model2.load_from(config)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(count_parameters(model1))
    print(count_parameters(model2))


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
    #     RandomGenerator(args.patch_size)
    # ]))
    # db_val = BaseDataSets(base_dir=args.root_path, split="val")
    db_train, db_val, db_test,  _, _, _ = build_dataset_ssl(DATA_PATH, TILE_IMAGE_PATH, TILE_LABEL_PATH, args.data_class, args.patch_size)

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.data_class, args.labeled_num, len(db_train), UsePercentage_flag=True)
    # print("Total silices is: {}, labeled slices is: {}".format(
    #     total_slices, labeled_slice))
    logging.info("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
    # trainloader = DataLoader(db_train,batch_size=args.batch_size,
    #                          num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.Adam(model1.parameters(), lr=3.6e-04, weight_decay=1e-05)
    optimizer2 = optim.Adam(model2.parameters(), lr=3.6e-04, weight_decay=1e-05)

    scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer1, T_max=50, eta_min=5e-6,last_epoch=-1)
    scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50, eta_min=5e-6,last_epoch=-1)

    ce_weights = losses.reverse_weight(losses.calculate_weights(TILE_LABEL_PATH))
    ce_weights=torch.tensor(ce_weights).type(torch.cuda.FloatTensor)
    print(f"CE Weights are {ce_weights}")
    ce_loss = CrossEntropyLoss(reduction='mean', weight=ce_weights)
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # print("set(volume_batch[0]): ", set(volume_batch[0]))
            # print(f"volume_batch.shape: {volume_batch.shape}, label_batch.shape: {label_batch.shape}")
            # print("max(volume_batch[0]): ", torch.max(volume_batch[0]), "min(volume_batch[0])", torch.min(volume_batch[0]))
            # print("set(label_batch[0]): ", set(label_batch[0]))
            # print("max(label_batch[0]): ", torch.max(label_batch[0]), "min(label_batch[0])", torch.min(label_batch[0]))

            #change dimension
            label_batch = label_batch.squeeze(1)
            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(
                iter_num // 150)
            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long())+ dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(
                outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(
                outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = dice_loss(
                outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(
                outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            scheduler1.step()
            scheduler2.step()

            iter_num = iter_num + 1

            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer1.param_groups:
            #     param_group['lr'] = lr_
            # for param_group in optimizer2.param_groups:
            #     param_group['lr'] = lr_

            # writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                iter_num, model1_loss.item(), model2_loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0: #default to be 200
                model1.eval()
                metric_list = 0.0

                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"].squeeze(1), model1, classes=num_classes, patch_size=args.patch_size)
                    if metric_i != []:
                        # if sample groundtruth label are empty, ignore this validation sample;
                        # print(f"metric_i: {metric_i}, metric_i.shape: {np.array(metric_i).shape}") 
                        # metric_i: [(0, 0, 0, 0)], metric_i.shape: (1, 4)
                        metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/model1_val_{}_asd'.format(class_i+1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/model1_val_{}_iou'.format(class_i+1), 
                                      metric_list[class_i, 3], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0] ## dice

                mean_hd951 = np.mean(metric_list, axis=0)[1] ## hd95
                mean_asd1 = np.mean(metric_list, axis=0)[2] ## asd
                mean_iou1 = np.mean([tup[3] for tup in metric_list if not np.isnan(tup[3])]) ## iou
                
                writer.add_scalar('info/model1_val_mean_dice',
                                  performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95',
                                  mean_hd951, iter_num)
                writer.add_scalar('info/model1_val_mean_asd',
                                  mean_asd1, iter_num)
                writer.add_scalar('info/model1_val_mean_iou',
                                  mean_iou1, iter_num)
                # change to mean_iou as val standards
                if mean_iou1 > best_performance1:
                    best_performance1 = mean_iou1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_iou_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f model1_mean_asd : %f model1_mean_iou : %f' % (iter_num, performance1, mean_hd951, mean_asd1, mean_iou1))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    # print(sampled_batch["image"].shape, sampled_batch["label"].shape)
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"].squeeze(1), model2, classes=num_classes, patch_size=args.patch_size)
                    if metric_i != []:
                        # if sample groundtruth label are empty, ignore this validation sample;
                        metric_list += np.array(metric_i)                
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/model2_val_{}_asd'.format(class_i+1), 
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/model2_val_{}_iou'.format(class_i+1), 
                                      metric_list[class_i, 3], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0] ## dice
                mean_hd952 = np.mean(metric_list, axis=0)[1] ## hd95
                mean_asd2 = np.mean(metric_list, axis=0)[2] ## asd
                mean_iou2 = np.mean([tup[3] for tup in metric_list if not np.isnan(tup[3])])
                
                writer.add_scalar('info/model2_val_mean_dice',
                                  performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95',
                                  mean_hd952, iter_num)
                writer.add_scalar('info/model2_val_mean_asd',
                                  mean_asd2, iter_num)
                writer.add_scalar('info/model2_val_mean_iou',
                                  mean_iou2, iter_num)

                if mean_iou2 > best_performance2:
                    best_performance2 = mean_iou2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_iou_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f model2_mean_asd : %f model2_mean_iou : %f' % (iter_num, performance2, mean_hd952, mean_asd2, mean_iou2))
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    print(f"snapshot_path: {snapshot_path}")
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__', '.out']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
