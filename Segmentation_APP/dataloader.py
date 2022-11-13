import numpy as np
import torch
from torch.utils.data import Dataset
import os
import itertools
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, file_list, tile_image_path, tile_label_path, transform=None, mode = 'train'):
        self.file_list = file_list
        self.transform = transform
        self.tile_image_path = tile_image_path
        self.tile_label_path = tile_label_path
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        inputs = np.load(self.tile_image_path + (self.file_list[idx][0]),allow_pickle=True)
        bdry = np.load(self.tile_label_path + (self.file_list[idx][1]),allow_pickle=True)
        mask_label = np.load(self.tile_label_path + (self.file_list[idx][1]),allow_pickle=True)
        mask_label = mask_label / 255
        
        inputs = torch.from_numpy(inputs).unsqueeze(0)
        bdry = torch.from_numpy(bdry).unsqueeze(0)
        mask_label = torch.from_numpy(mask_label).unsqueeze(0)
        label = torch.max(bdry * 2, mask_label).long().squeeze()

        if self.transform and self.mode =='train':
            sample = self.transform({"image":inputs, 
                                    "label":mask_label})
            inputs, mask_label = sample['image'], sample['label']
        elif self.transform and self.mode!='train':
            inputs = self.transform(inputs)
            mask_label = self.transform(mask_label)
        return inputs, mask_label

def build_dataloader(data_path, label_path, mask_label_path, 
                    tile_image_path, tile_label_path, 
                    transform_train, transform_val, transform_test):
    """
    To keep the results comparable we are using 
    the same splits as Van Buren et al (https://www.sciencedirect.com/science/article/pii/S0022175922000205), 
    and code at: https://github.com/tniewold/Artificial-Intelligence-and-Deep-Learning-to-Map-Immune-Cell-Types-in-Inflamed-Human-Tissue
    """

    data_file = os.listdir(data_path)
    label_file = os.listdir(label_path)
    mask_label_file = os.listdir(mask_label_path)

    selected_data = set([_[:-9] for _ in data_file]).intersection(set([_[:-11] for _ in label_file])).intersection(set([_[:-17] for _ in mask_label_file]))
    # print("Length of selected data: ", len(selected_data))

    train_list = [[(_ + '_data_' + str(idx) + '.npy',  _ + '_mask_' + str(idx) + '.npy') for idx in range(12)] 
                for _ in selected_data if _[:13] in ['121919_Myo089', '121919_Myo253', '121919_Myo368']]
    validation_list = [[(_ + '_data_' + str(idx) + '.npy', _ + '_mask_' + str(idx) + '.npy') for idx in range(12)] 
                    for _ in selected_data if _[:13] in ['121919_Myo208', '121919_Myo388']]
    test_list = [[(_ + '_data_' + str(idx) + '.npy', _ + '_mask_' + str(idx) + '.npy') for idx in range(12)] 
                for _ in selected_data if _[:13] in ['121919_Myo231', '121919_Myo511']]
                
    train_list = list(itertools.chain(*train_list))
    validation_list = list(itertools.chain(*validation_list))
    test_list = list(itertools.chain(*test_list))
    # print('train data: {}'.format(len(train_list)))
    # print('validation data: {}'.format(len(validation_list)))
    # print('test data: {}'.format(len(test_list)))

    dataloader = {}
    dataloader['train'] = DataLoader(CustomDataset(train_list, tile_image_path, tile_label_path, transform=transform_train, mode='train'), batch_size=16, shuffle=True, num_workers=3, drop_last=False)
    dataloader['validation'] = DataLoader(CustomDataset(validation_list, tile_image_path, tile_label_path, transform=transform_val, mode='validation'), batch_size=16, shuffle=False, num_workers=3, drop_last=False)
    dataloader['test'] = DataLoader(CustomDataset(test_list, tile_image_path, tile_label_path, transform=transform_test, mode='test'), batch_size=16, shuffle=False, num_workers=3, drop_last=True)

    return dataloader