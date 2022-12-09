import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import itertools
from torch.utils.data import DataLoader

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
                    transform_train, transform_val, transform_test, train_ratio= 100):

    train_origin_data='ISIC-2017_Training_Data/'
    train_origin_label='ISIC-2017_Training_Part1_GroundTruth/'
    val_origin_data='ISIC-2017_Validation_Data/'
    val_origin_label='ISIC-2017_Validation_Part1_GroundTruth/'
    test_origin_data='ISIC-2017_Test_v2_Data/'
    test_origin_label='ISIC-2017_Test_v2_Part1_GroundTruth/'

    train_percent = 0.8
    validation_percent = 0.1
    ## remained 0.1 are test 

    train = []
    validation = []
    test = []
    for path in os.listdir(data_path):
        if path[-3:] == 'zip':
            continue
        if path[-4:] == "Data": 
            for path2 in os.listdir(data_path+path):
                if path2[-3:] == 'txt':
                    continue
                if 'superpixel' in path2 or 'metadata' in path2:
                    continue
                if 'Training' in path:
                    train.append(path2[:-4])
                elif 'Validation' in path:
                    validation.append(path2[:-4])
                else:
                    test.append(path2[:-4])
    
    ## make sure the split has no overlap
    assert len(set(train).intersection(set(validation))) == 0
    assert len(set(train).intersection(set(test))) == 0
    assert len(set(validation).intersection(set(test))) == 0

    ## add suffix using the prefix
    all_images = sorted(list(os.listdir(tile_image_path)))
    all_labels = sorted(list(os.listdir(tile_label_path)))

    train_list = []
    validation_list = []
    test_list = []

    for image, label in zip(all_images, all_labels):
        if image[:-11] in train:
            train_list.append([(image, label)])
        elif image[:-11] in validation:
            validation_list.append([(image, label)])
        elif image[:-11] in test:
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

    data_path = '../../dataset/ISIC2017/original_data/'
    tile_image_path = '../../dataset/ISIC2017/resize_image/'
    tile_label_path = '../../dataset/ISIC2017/resize_label/'
    save_metric_path = './metric_save'


    DATASET_IMAGE_MEAN = (0.485,0.456, 0.406)
    DATASET_IMAGE_STD = (0.229,0.224, 0.225)
    rotation = transforms.RandomRotation(3)

    transform_val=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    transform_test=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

    dataloader = build_dataloader(
        data_path, tile_image_path, tile_label_path, 
        None, transform_val, transform_test)

    for inputs, mask in dataloader['train']:
        print(inputs)
        break