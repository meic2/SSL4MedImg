import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import itertools
import random
from scipy import ndimage

class RandomGenerator(object):
    def __init__(self, isRnorm):
        self.ToPILImage = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        self.isRnorm = isRnorm
        
        DATASET_IMAGE_MEAN = (0.485,0.456, 0.406)
        DATASET_IMAGE_STD = (0.229,0.224, 0.225)
        self.Normalize = transforms.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD)
        
    def Rnorm(self, image=None):
        image[0] = image[0]/np.sqrt(image[0]*image[0] + image[1]*image[1] + image[2]*image[2] + 1e-11)
        return image
        
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.ToPILImage(image)
        label = self.ToPILImage(label)
        ''' random augmentation '''
        if random.random() > 0.5:
            image, label = self.random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = self.random_rotate(image, label)
        ''' toTensor()'''
        image = self.ToTensor(image)        
        ''' Rnorm '''
        if self.isRnorm:
            image = self.Rnorm(image)
        
        label = self.ToTensor(label)

        sample = {"image": image, "label": label}
        return sample

    def random_rot_flip(self, image, label=None):
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

    def random_rotate(self, image, label):
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label


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
            # inputs, mask_label = sample['image'], sample['label']
        # return sample
        elif self.transform and self.mode!='train':
            inputs = self.transform(inputs)
            mask_label = self.transform(mask_label)
            sample = {"image":inputs, "label":mask_label}
        return sample

def build_dataloader(data_path, tile_image_path, tile_label_path, 
                    transform_train, transform_val, transform_test, isDermorfit):

    train_lis = []
    validation_lis = []
    test_lis = []
    if isDermorfit:
        train_percent = 0.8
        validation_percent = 0.1
        ## remained 0.1 are test 
        
        for one_type in os.listdir(data_path):  # e.g. one_type =='AK
            if len(one_type) >= 4 and (one_type[-4:] == '.zip' or one_type[-4:] == '.txt'):
                continue
        
            type_all_instances = os.listdir(data_path + one_type + '/')
            type_total_count = len(type_all_instances)
            print(f"one_type = {one_type}, total count of instances = {type_total_count}")
            # print(type_all_instances) ['B643', 'A75', 'P374', 'B666', 'D379',...]

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

        for image, label in zip(all_images, all_labels):
            if image[:-11] in train_lis:
                train_list.append([(image, label)])
            elif image[:-11] in validation_lis:
                validation_list.append([(image, label)])
            elif image[:-11] in test_lis:
                test_list.append([(image, label)])
    else: # Dermatomyositis
        data_file = os.listdir(data_path+"CD27_Panel_Component/")
        label_file = os.listdir(data_path+"Labels/CD27_cell_labels/")
        mask_label_file = os.listdir(data_path+"Labels/CD27_cell_labels/Mask_Labels/")
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

    print('\ntrain data: {}'.format(len(train_list)))
    print('validation data: {}'.format(len(validation_list)))
    print('test data: {}'.format(len(test_list)))

    return CustomDataset(train_list, tile_image_path, tile_label_path, transform=transform_train, mode='train'), \
            CustomDataset(validation_list, tile_image_path, tile_label_path, transform=transform_val, mode='validation'), \
            CustomDataset(test_list, tile_image_path, tile_label_path, transform=transform_test, mode='test')

def build_dataloader_ssl(data_path, tile_image_path, tile_label_path, isDermorfit):
    transform_train = transforms.Compose([
        RandomGenerator(isDermorfit),
        ])

    transform_val=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    transform_test=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

    return build_dataloader(
        data_path, tile_image_path, tile_label_path, 

        transform_train, transform_val, transform_test, isDermorfit)

if __name__ == '__main__':

 ''' data files '''
 data_path = '../../../dataset/Dermofit/original_data/'
 tile_image_path = '../../../dataset/Dermofit_resize_noTiling/resize_image'
 tile_label_path = '../../../dataset/Dermofit_resize_noTiling/resize_label'
 
 train_list, validation_list, test_list = build_dataloader(data_path, tile_image_path, tile_label_path)
 print(train_list)