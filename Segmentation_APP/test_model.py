import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
import os
import random
import argparse

from dataloader import build_dataloader

# parser = argparse.ArgumentParser()
# parser.add_argument('--fn', type=str,
#                     default='Unet_resnet18_20221001_14.pt', help='file_name')
# args = parser.parse_args()

def test_model_func(file_name, device, dataloader, file_directory='../../DEDL_Saved_model/'):
    model = torch.load(file_directory + file_name)
    model = model.to(device)
    model.eval()

    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0

    with torch.no_grad():
        for inputs, mask in dataloader['test']:
            inputs = inputs.to(device)
            mask = mask.to(device)
            
            outputs = model(inputs)
            mask = mask.squeeze().to(device)
            pred = torch.argmax(outputs, dim=1)

            tp, fp, fn, tn = smp.metrics.get_stats(pred, mask.long(), mode='multilabel', threshold=0.5)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            tp_total += tp
            fp_total += fp
            fn_total += fn
            tn_total += tn
    print(file_name)
    print('Overall Scores:')
    
    iou_score = smp.metrics.iou_score(tp_total, fp_total, fn_total, tn_total, reduction="micro")
    accuracy = smp.metrics.accuracy(tp_total, fp_total, fn_total, tn_total, reduction="macro")
    print("Pixel Acc:",accuracy)
    print("IoU:",iou_score)

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING']='1'

    data_path = '../../dataset/Dermatomyositis/original_data/CD27_Panel_Component/'
    label_path = '../../dataset/Dermatomyositis/original_data/Labels/CD27_cell_labels/'
    mask_label_path = '../../dataset/Dermatomyositis/original_data/Labels/CD27_cell_labels/Mask_Labels/'
    tile_image_path = '../../dataset/Dermatomyositis/tile_image/'
    tile_label_path = '../../dataset/Dermatomyositis/tile_label/'
    save_metric_path = './metric_save'

    random_deg = random.random()
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(3),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
    transform_val=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    transform_test=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

    dataloader = build_dataloader(
        data_path, label_path, mask_label_path, 
        tile_image_path, tile_label_path, 
        transform_train, transform_val, transform_test)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print('##############')
    print('# Test Model #')
    print('##############')
    file_name = args.fn
    test_model(file_name, device, dataloader)