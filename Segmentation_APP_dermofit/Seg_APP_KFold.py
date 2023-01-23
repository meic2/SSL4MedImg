# Segmentation Using Autoencoder Post-Processing
import torch
import segmentation_models_pytorch as smp
import math
import numpy as np
import time
from datetime import timedelta
from torchvision import transforms
from torch import nn, optim
import random
import os
from datetime import datetime
import torch.nn.functional as F
from scipy.ndimage.interpolation import zoom

from autoencoder import AutoencoderReLu, AutoencoderGeLu
from dataloader import build_dataloader
from utili import save_metric
from test_model import test_model_func
from scipy import ndimage, misc

from torch.utils.tensorboard import SummaryWriter


import argparse
parser = argparse.ArgumentParser(description = "Model Parameter Setting")

parser.add_argument('-en', '--encoder', type=str,
                    default='resnet18', help='encoder_name')
parser.add_argument('-seed', type=int,
                    default=1234, help='torch manual seed')     
parser.add_argument('-gelu', '--gelu', action='store_true', 
                    help='auto encoder activation function')
parser.add_argument('-ae', '--ae', action='store_true', 
                    help='whether have Autoencoder')
parser.add_argument('-unet', '--unet', action='store_true', 
                    help='use Unet or Unet++')
parser.add_argument('-cuda', '--cuda', type=str,
                    default='cuda:0', help='choose gpu')
parser.add_argument('-r', '--ratio', type=int,
                    default=100, help='choose percentage of data being trained')    
parser.add_argument('-KthFold', type=int,
                    default=None, help='the Kth fold')
parser.add_argument('-CASSorDINO', type=str,
                    default='CASS', help='choose CASS or DINO pretrained resnet')
args = parser.parse_args()
os.environ['CUDA_LAUNCH_BLOCKING']='1'

'''
CONSTANTS
'''
data_path = '../../dataset/Dermofit/original_data/'
tile_image_path = '../../dataset/Dermofit_resize_noTiling/resize_image/'
tile_label_path = '../../dataset/Dermofit_resize_noTiling/resize_label/'
save_metric_path = './metric_save'
# save_model_directory='../../DEDL_Saved_model/'
save_model_directory = f"../Saved_model/{args.CASSorDINO}/fold_{args.KthFold}/"
    

class Resize(object):
    def __init__(self, output_size = [224,224]):
        self.output_size = output_size
    
    def __call__(self, image):
        """
        :param image: tensor image or label
        :return: zoomed image to size output_size
        """
        
        _, x, y = image.shape
        zoom_factor = 1, self.output_size[0]/ x, self.output_size[1]/ y
                
        image = zoom(image, zoom_factor, order=0)

        return torch.tensor(image)

    def __repr__(self):
        return self.__class__.__name__+'()'

class RandomGenerator(object):
    def __init__(self):
        self.ToPILImage = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        
        DATASET_IMAGE_MEAN = (0.485,0.456, 0.406)
        DATASET_IMAGE_STD = (0.229,0.224, 0.225)
        # self.Normalize = transforms.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD)
        self.Resize=Resize()

    def Rnorm(self, image=None):
        image[0] = image[0]/np.sqrt(image[0]*image[0] + image[1]*image[1] + image[2]*image[2] + 1e-11)
        return image
        
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.ToPILImage(image)
        label = self.ToPILImage(label)
        ''' random augmentation '''
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        ''' toTensor()'''
        image = self.ToTensor(image)   
        label = self.ToTensor(label)
        '''resize (zoom) input image'''

        image = self.Resize(image)
        label = self.Resize(label)  

        ''' Rnorm '''
        image = self.Rnorm(image)

        sample = {"image": image, "label": label}
        return sample

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4) # number of times to be rotated
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

"""
train
"""
def train_model(dataloader,
                device,
                model, 
                criterion, 
                optimizer, 
                autoencoder,
                criterion_ae,
                optimizer_ae,
                scheduler,
                model_name,
                ae_flag,
                num_epochs=20):
    writer = SummaryWriter(
        log_dir='/scratch/lc4866/DEDL_Semisupervised_code_version/Segmentation_APP_dermofit/tensorboard_runs')
    train_acc = []
    train_acc_mask = []
    train_acc_bg = []
    train_iou = []
    train_confusion_matrix = []

    val_acc = []
    val_acc_mask = []
    val_acc_bg = []
    val_iou = []
    val_confusion_matrix = []

    total_loss = 0
    prev = math.inf
    best_iou = 0
    counter = 0
    last_val = 0
    iter = 0
    train_val_time = timedelta(0)

    for epoch in range(num_epochs):
        start_time = datetime.now() 
        print('\n Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 70)

        tp_total = 0
        fp_total = 0
        fn_total = 0
        tn_total = 0

        for inputs, mask in dataloader['train']:
            #####################
            # Start train model #
            #####################
            model.train()
            autoencoder.train()
            # Get inputs and label
            inputs = inputs.to(device)
            mask = mask.squeeze().to(device)
            # input.shape: torch.Size([16, 1, 480, 480]), mask.shape: torch.Size([16, 480, 480])
            # print('mask.dtype:', mask.dtype)
            # mask.dtype: torch.float32
            
            # Get the output 
            outputs = model(inputs)
            # print("outputs.shape after model: ", outputs.shape) 
            # outputs.shape after model:  torch.Size([16, 2, 480, 480])
            # print('outputs.dtype: ', outputs.dtype)
            # outputs.dtype:  torch.float32
            outputs_ae = autoencoder(outputs)
            # print("outputs.shape after ae ", outputs_ae.shape) 
            # outputs.shape after ae  torch.Size([16, 2, 480, 480])
            # print('outputs.dtype after ae:', outputs_ae.dtype)
            # outputs.dtype after ae: torch.float32

            # Get the argmax of outputs
            pred = torch.argmax(outputs, dim=1)

            # Calculate loss and backpropagation
            loss1 = criterion(outputs, mask.long())
            # print('loss1.dtype:', loss1.dtype)
            # loss1.dtype: torch.float32
            if ae_flag:
                one_hot_label = F.one_hot(mask.to(torch.int64), num_classes=2)
                # print('one_hot_label', one_hot_label.shape) 
                # one_hot_label torch.Size([16, 480, 480, 2])
                # mask = one_hot_label.permute(0, 3, 1, 2)
                # print('mask reshape', mask.shape) 
                # mask reshape torch.Size([16, 2, 480, 480])
                # loss2 = criterion_ae(outputs_ae, mask)
                loss2 = criterion_ae(outputs_ae, one_hot_label.permute(0,3,1,2).float())
                # print('loss2.dtype:', loss2.dtype)
                # loss2.dtype: torch.float32

                loss = loss1 + loss2
            else:
                loss = loss1
            writer.add_scalar('loss',loss.item(), iter)
            iter +=1
            # print('loss.dtype:', loss.dtype)
            # loss.dtype: torch.float32
            # print("loss1.dtype, loss2.dtype: ", loss1.dtype, loss2.dtype)
            loss.backward()
            optimizer.step()
            optimizer_ae.step()
            optimizer_ae.zero_grad()
            optimizer.zero_grad()
            scheduler.step()
            # Calculate total loss
            total_loss += loss.item()
            print("loss: ", loss.item())

            # Get the metric
            tp, fp, fn, tn = smp.metrics.get_stats(pred, mask.long(), mode='multilabel', threshold=0.5)

            tp_total += torch.sum(tp)
            fp_total += torch.sum(fp)
            fn_total += torch.sum(fn)
            tn_total += torch.sum(tn)

        avg_loss = total_loss / len(dataloader['train'])
        # Get the metric
        tp_mask, fp_mask, fn_mask, tn_mask = smp.metrics.get_stats(pred==1, mask==1, mode='multilabel', threshold=0.5)
        mask_accuracy = smp.metrics.accuracy(tp_mask, fp_mask, fn_mask, tn_mask, reduction="macro")
        # print("Mask Accuracy:", mask_accuracy)

        tp_bg, fp_bg, fn_bg, tn_bg = smp.metrics.get_stats(pred==0, mask==0, mode='multilabel', threshold=0.5)
        bg_accuracy = smp.metrics.accuracy(tp_bg, fp_bg, fn_bg, tn_bg, reduction="macro")
        # print("Background Accuracy:", bg_accuracy)

        tp, fp, fn, tn = smp.metrics.get_stats(pred, mask.long(), mode='multilabel', threshold=0.5)
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        # train_iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        train_iou_score = smp.metrics.iou_score(tp_total, fp_total, fn_total, tn_total, reduction="micro")
        print("overall Acc:",accuracy)
        print('IoU:', train_iou_score)

        # Save metric
        train_acc_mask.append(mask_accuracy)
        train_acc_bg.append(bg_accuracy)
        train_acc.append(accuracy)
        train_confusion_matrix.append([tp, fp, fn, tn])
        train_iou.append(train_iou_score)

        if train_iou_score >= best_iou:
            # best_iou = train_iou_score

            ##################
            # Validate model #
            ##################
            with torch.no_grad():
                model.eval()
                autoencoder.eval()
                total_loss = 0
                print('Validation')
                print('-' * 15)

                tp_total_val = 0
                fp_total_val = 0
                fn_total_val = 0
                tn_total_val = 0

                for inputs, mask in dataloader['validation']:
                    # Get inputs and label
                    inputs = inputs.to(device)
                    mask = mask.squeeze().to(device)

                    # Get the output 
                    outputs = model(inputs)
                    # Get the argmax of outputs
                    pred = torch.argmax(outputs, dim=1)
                    # Calculate loss
                    loss = criterion(outputs, mask.long())
                    total_loss += loss.item()

                    tp, fp, fn, tn = smp.metrics.get_stats(pred, mask.long(), mode='multilabel', threshold=0.5)
                    tp_total_val += torch.sum(tp)
                    fp_total_val += torch.sum(fp)
                    fn_total_val += torch.sum(fn)
                    tn_total_val += torch.sum(tn)

                avg_loss = total_loss / len(dataloader['validation'])
            
                # # Get the metric
                tp_mask, fp_mask, fn_mask, tn_mask = smp.metrics.get_stats(pred==1, mask==1, mode='multilabel', threshold=0.5)
                mask_accuracy = smp.metrics.accuracy(tp_mask, fp_mask, fn_mask, tn_mask, reduction="macro")
                # print("Mask Accuracy:", mask_accuracy)

                tp_bg, fp_bg, fn_bg, tn_bg = smp.metrics.get_stats(pred==0, mask==0, mode='multilabel', threshold=0.5)
                bg_accuracy = smp.metrics.accuracy(tp_bg, fp_bg, fn_bg, tn_bg, reduction="macro")
                # print("Background Accuracy:", bg_accuracy)

                tp, fp, fn, tn = smp.metrics.get_stats(pred, mask.long(), mode='multilabel', threshold=0.5)
                # val_iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
                val_iou_score = smp.metrics.iou_score(tp_total_val, fp_total_val, fn_total_val, tn_total_val, reduction="micro")
                print("overall Acc:",accuracy)
                print("Validation IoU:",val_iou_score)
                
                # Save metric
                val_acc_mask.append(mask_accuracy)
                val_acc_bg.append(bg_accuracy)
                val_acc.append(accuracy)
                val_confusion_matrix.append([tp, fp, fn, tn])
                val_iou.append(val_iou_score)
                
                if val_iou_score >= last_val:
                    last_val = val_iou_score
                    date_time = time.strftime("%Y%m%d", time.localtime())
                    file_name = model_name + '_' + date_time + '.pt'
                    print("Saving Model!")
                    torch.save(model, save_model_directory+file_name)
                
            if avg_loss > prev:
                counter+=1
            else:
                counter = 0
                
            prev = avg_loss
            
            if counter > 5:
                print("YES!!!!!!")
            
            end_time = datetime.now() 
            dur_time = end_time - start_time
            print("Epoch time: ", dur_time)
            train_val_time += dur_time
        print('-' * 70)

    return train_acc, train_acc_mask, train_acc_bg, train_iou, val_acc, val_acc_mask, val_acc_bg, val_iou, train_confusion_matrix, val_confusion_matrix, file_name, train_val_time

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    print("current seed is "+ str(args.seed))

    transform_train = transforms.Compose([
        RandomGenerator(),
        ])

    transform_val=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(), Resize()])
    transform_test=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(), Resize()])

    dataloader = build_dataloader(
        data_path, tile_image_path, tile_label_path, 
        transform_train, transform_val, transform_test, args.ratio, args.KthFold)
    
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    print(device)

    encoder_name = args.encoder
    print("current ratio is "+str(args.ratio)+"%")
    print("the encoder name is "+ encoder_name)
    print("Use GELU: " + str(args.gelu))
    print("Use Unet: " + str(args.unet))
    print("Use AE: " + str(args.ae))
    print("CASS or DINO: ", args.CASSorDINO)
    if args.ae:    
        if args.gelu:
            if args.unet:
                model_name = 'Unet_' + encoder_name + '_dermofit'+ "_seed" + str(args.seed) + "_gelu"
            else:
                model_name = 'Unetpp_' + encoder_name + '_dermofit'+ "_seed" + str(args.seed) + "_gelu"
        else:
            if args.unet:
                model_name = 'Unet_' + encoder_name + '_dermofit'+ "_seed" + str(args.seed) + "_relu"
            else:
                model_name = 'Unetpp_' + encoder_name + '_dermofit'+ "_seed" + str(args.seed) + "_relu"
    else:
        if args.unet:
                model_name = 'Unet_' + encoder_name + '_dermofit'+ "_seed" + str(args.seed) + "_woAE"
        else:
            model_name = 'Unetpp_' + encoder_name + '_dermofit'+ "_seed" + str(args.seed) + "_woAE"
    
    if args.unet:
        model = smp.Unet(
            encoder_name=encoder_name, decoder_attention_type='scse', encoder_weights='imagenet',
            in_channels=3, classes=2, encoder_depth=3, decoder_channels=(256, 128, 64))
    else:
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name, decoder_attention_type='scse', encoder_weights='imagenet',
            in_channels=3, classes=2, encoder_depth=3, decoder_channels=(256, 128, 64))

    ## change model encoder to pretrained CASS/DINO resnet50.pt
    if args.CASSorDINO =="CASS":
        pretrained_pt = f"../KFold_pt/CASS/224/CASS with Swin/ResNet50/swin-base-r50-r50part-{args.KthFold}-dermofit-224-seg.pt"
    elif args.CASSorDINO == "DINO":
        pretrained_pt = f"../KFold_pt/DINO/224/r50-dino-{args.KthFold}-dermofit-224-seg.pt"
    # model.encoder = torch.load(pretrained_pt)
    model.encoder.load_state_dict(torch.load(pretrained_pt).state_dict(), strict=False)
    print(
        f"successfully load pretrained {args.CASSorDINO} model from dir = {pretrained_pt}!")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum', weight=torch.tensor([0.7180376788665532, 1.6465908143176757])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3.6e-04, weight_decay=1e-05)
    if args.gelu:
        autoencoder = AutoencoderGeLu()
        print("Autoencoder is gelu")
    else:
        autoencoder = AutoencoderReLu()
        print("Autoencoder is relu")
    
    autoencoder = autoencoder.to(device)
    criterion_ae = nn.MSELoss()
    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=1e-3)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=3.4e-04, last_epoch=-1)

    print('###############')
    print('# Train Model #')
    print('###############')
    train_acc, train_acc_mask, train_acc_bg, train_iou, val_acc, val_acc_mask, val_acc_bg, val_iou, train_confusion_matrix, val_confusion_matrix, file_name, train_val_time = train_model(
        dataloader=dataloader, device=device,
        model=model, criterion=criterion, optimizer=optimizer, 
        autoencoder=autoencoder, criterion_ae=criterion_ae, optimizer_ae=optimizer_ae,
        scheduler=scheduler, model_name=model_name, ae_flag=args.ae, num_epochs=50)
    print("Total training, validation time: ", train_val_time)

    print('###############')
    print('# Save Metric #')
    print('###############')
    save_metric(
        train_acc, train_acc_mask, train_acc_bg, train_iou, 
        val_acc, val_acc_mask, val_acc_bg, val_iou, 
        train_confusion_matrix, val_confusion_matrix,
        save_metric_path)
    
    print('##############')
    print('# Test Model #')
    print('##############')
    print("the encoder name is "+ encoder_name)
    print("Use GELU: " + str(args.gelu))
    print("Use Unet: " + str(args.unet))
    print("Use AE: " + str(args.ae))
    test_model_func(file_name, device, dataloader, file_directory=save_model_directory)