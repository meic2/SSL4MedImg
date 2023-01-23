import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import itertools
from torch.utils.data import DataLoader
from data_mapping import find_local_tile_image_label

train_csv = "/scratch/lc4866/DEDL_Semisupervised_code_version/KFold_pt/train.csv"
test_csv = "/scratch/lc4866/DEDL_Semisupervised_code_version/KFold_pt/test.csv"
mapping_txt = "/scratch/lc4866/DEDL_Semisupervised_code_version/KFold_pt/splits.txt"

class CustomDataset(Dataset):
    def __init__(self, file_list, tile_image_path, tile_label_path, transform = None, mode = 'train'):
        self.file_list = file_list
        self.tile_image_path = tile_image_path
        self.tile_label_path = tile_label_path
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        inputs = np.load(self.tile_image_path + (self.file_list[idx][0]), allow_pickle=True)
        mask_label = np.load(self.tile_label_path + (self.file_list[idx][1]), allow_pickle=True)
        mask_label = mask_label / 255
        
        inputs = torch.from_numpy(inputs)
        mask_label = torch.from_numpy(mask_label).unsqueeze(0)

        if self.transform and self.mode =='train':
            sample = self.transform({"image":inputs, 
                                    "label":mask_label})
            inputs, mask_label = sample['image'], sample['label']
        elif self.transform and self.mode!='train':
            inputs = self.transform(inputs)
            mask_label = self.transform(mask_label)
        return inputs, mask_label

def build_dataloader(data_path, tile_image_path, tile_label_path, 
                     transform_train, transform_val, transform_test, train_ratio=100, KthFold=0):

    train_lis = []
    validation_lis = []
    test_lis = []

    train_percent = 0.7
    validation_percent = 0.1
    ## remained 0.1 are test 

    for one_type in os.listdir(data_path):  # e.g. one_type =='AK
        if len(one_type) >= 4 and (one_type[-4:] == '.zip' or one_type[-4:] == '.txt'):
            continue
    
        type_all_instances = os.listdir(data_path + one_type + '/')
        type_total_count = len(type_all_instances)
        print(f"one_type = {one_type}, total count of instances = {type_total_count}")
        ## print(type_all_instances) ['B643', 'A75', 'P374', 'B666', 'D379',...]

        train_lis += [one_type + '_' + i for i in type_all_instances[:int(type_total_count*train_percent)]]
        validation_lis += [one_type + '_' + i for i in type_all_instances[int(type_total_count*train_percent): int(type_total_count*(train_percent+validation_percent))]]
        test_lis += [one_type + '_' + i for i in type_all_instances[int(type_total_count*(train_percent + validation_percent)):]]

    ## make sure the split has no overlap
    assert len(set(train_lis).intersection(set(validation_lis))) == 0
    assert len(set(train_lis).intersection(set(test_lis))) == 0
    assert len(set(validation_lis).intersection(set(test_lis))) == 0

    ## add suffix using the prefix
    all_images = sorted(list(os.listdir(tile_image_path)))
    all_labels = sorted(list(os.listdir(tile_label_path)))

        
    train_list = []
    validation_list = []
    test_list = []
    if KthFold is None:
        for image, label in zip(all_images, all_labels):
            if image[:-11] in train_lis:
                train_list.append([(image, label)])
            elif image[:-11] in validation_lis:
                validation_list.append([(image, label)])
            elif image[:-11] in test_lis:
                test_list.append([(image, label)])
    else: ## k-fold
        print("using K fold...")
        train_lis_image, train_lis_mask, val_lis_image, val_lis_mask = find_local_tile_image_label(train_csv,
                                                                                                   mapping_txt,
                                                                                                   KthFold=KthFold)
        test_lis_image = list((set(all_images).difference(set(train_lis_image))).difference(set(val_lis_image)))
        test_lis_mask = list((set(all_labels).difference(set(train_lis_mask))).difference(set(val_lis_mask)))
        for image, label in zip(train_lis_image, train_lis_mask):
            train_list.append([(image, label)])
        for image, label in zip(val_lis_image, val_lis_mask):
            validation_list.append([(image, label)])
        for image, label in zip(test_lis_image, test_lis_mask):
            test_list.append([(image, label)])

    train_list = list(itertools.chain(*train_list))
    print('origin train data: {}'.format(len(train_list)))
    
    train_list = train_list[:len(train_list)*train_ratio//100]
    validation_list = list(itertools.chain(*validation_list))
    test_list = list(itertools.chain(*test_list))

    print('\ntrain data: {}'.format(len(train_list)))
    print('validation data: {}'.format(len(validation_list)))
    print('test data: {}'.format(len(test_list)))

    dataloader = {}
    dataloader['train'] = DataLoader(CustomDataset(train_list, tile_image_path, tile_label_path, transform=transform_train, mode='train'), batch_size=16, shuffle=True, num_workers=3, drop_last=False)
    dataloader['validation'] = DataLoader(CustomDataset(validation_list, tile_image_path, tile_label_path, transform=transform_val, mode='validation'), batch_size=16, shuffle=False, num_workers=3, drop_last=False)
    dataloader['test'] = DataLoader(CustomDataset(test_list, tile_image_path, tile_label_path, transform=transform_test, mode='test'), batch_size=16, shuffle=False, num_workers=3, drop_last=True)

    return dataloader

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING']='1'

    data_path = '../../dataset/Dermofit/original_data/'
    tile_image_path = '../../dataset/Dermofit_resize_noTiling/resize_image/'
    tile_label_path = '../../dataset/Dermofit_resize_noTiling/resize_label/'
    save_metric_path = './metric_save'


    DATASET_IMAGE_MEAN = (0.485,0.456, 0.406)
    DATASET_IMAGE_STD = (0.229,0.224, 0.225)
    rotation = transforms.RandomRotation(3)
    transform_train_img = transforms.Compose([
        transforms.ToPILImage(),
        rotation,
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD),  ## this is unique for img
        ])
    transform_train_label = transforms.Compose([
        transforms.ToPILImage(),
        rotation,
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
    transform_val=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    transform_test=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

    dataloader = build_dataloader(
        data_path, tile_image_path, tile_label_path, 
        transform_train_img, transform_val, transform_test)

    for inputs, mask in dataloader['train']:
        print(inputs)
        break