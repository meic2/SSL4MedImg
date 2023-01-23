import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
import os
import itertools
import random
from scipy import ndimage
import logging
from random import sample
from sklearn.model_selection import train_test_split
import cv2 as cv 

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
    def __init__(self, isRnorm, output_size = [224, 224]):
        self.ToPILImage = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        self.isRnorm = isRnorm
        self.output_size=output_size
        self.Resize=Resize(output_size)
        
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
        
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.uint8))
    
        ''' toTensor()'''
        image = self.ToTensor(image)  
        label = self.ToTensor(label)

        # image, label = self.resize(image, label) 
        image = self.Resize(image)
        label = self.Resize(label)       
          
        ''' Rnorm '''
        if self.isRnorm:
            image = self.Rnorm(image)


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
    
    # def resize(self, image, label):
    #     _, x, y = image.shape
    #     zoom_factor = 1, self.output_size[0]/ x, self.output_size[1]/ y
    #     image = zoom(image, zoom_factor, order=0)
    #     label = zoom(label, zoom_factor,  order=0)
    #     return image, label

def clustering(img_path, K=2, output_size=224):
    '''
    OpenCV clustering algo at RGB level and only return R classification results
    '''
    img = cv.imread(img_path)
    # print(img_path)
    # print(img.shape)
    float_img = np.float32(img)
    Z = np.reshape(float_img, (-1,3))
    # print(f'Z shape {Z.shape}')
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center=cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((float_img.shape))[:, :, 0]
    res2_torch = torch.from_numpy(res2)
    x, y, = res2_torch.shape
    zoom_factor =  output_size/ x, output_size/ y
    image = zoom(res2_torch, zoom_factor, order=0)

    return image 

class CustomDataset(Dataset):
    def __init__(self, file_list, tile_image_path, tile_label_path, transform = None, mode = 'train', original_path=None):
        self.file_list = file_list
        self.tile_image_path = tile_image_path
        self.tile_label_path = tile_label_path
        self.transform = transform
        self.mode = mode
        self.original_path=original_path

    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        inputs = np.load(self.tile_image_path + (self.file_list[idx][0]), allow_pickle=True)
        mask_label = np.load(self.tile_label_path + (self.file_list[idx][1]), allow_pickle=True)
        mask_label = mask_label / 255
        
        inputs = torch.from_numpy(inputs)
        mask_label = torch.from_numpy(mask_label).unsqueeze(0)
        # print(self.tile_image_path + (self.file_list[idx][0]))

        if self.transform and self.mode =='train':
            sample = self.transform({"image":inputs, 
                                    "label":mask_label,
                                    })
            #reverse to original data 
            p1, p2, _, _, = self.file_list[idx][0].split('_')
            raw_data_path = self.original_path + p1 + '/'+p2+'/'+p2+'.png'
            clustered_inputs = clustering(raw_data_path) /255
            sample['image_cluster'] = clustered_inputs
            # print(sample.keys())
            # inputs, mask_label = sample['image'], sample['label']
        # return sample
        elif self.transform and self.mode!='train':
            inputs = self.transform(inputs)
            mask_label = self.transform(mask_label)
            sample = {"image":inputs, "label":mask_label}
        return sample

def build_dataset(data_path, tile_image_path, tile_label_path, 
                    transform_train, transform_val, transform_test, dataclass):
    '''
    @variable:
        dataclass (int): Dermofit=1, tiled Dermatomyositis=2, interpolated Dermatomyositis=3
    '''

    train_lis = []
    validation_lis = []
    test_lis = []

    train_list = []
    validation_list = []
    test_list = []
    
    if dataclass == 1:
        train_percent = 0.7
        validation_percent = 0.1
        ## remained 0.2 are test 
        
        for one_type in os.listdir(data_path):  # e.g. one_type =='AK
            if len(one_type) >= 4 and (one_type[-4:] == '.zip' or one_type[-4:] == '.txt'):
                continue
        
            type_all_instances = os.listdir(data_path + one_type + '/')
            type_total_count = len(type_all_instances)
            logging.info(f"one_type = {one_type}, total count of instances = {type_total_count}")
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


        for image, label in zip(all_images, all_labels):
            if image[:-11] in train_lis:
                train_list.append([(image, label)])
            elif image[:-11] in validation_lis:
                validation_list.append([(image, label)])
            elif image[:-11] in test_lis:
                test_list.append([(image, label)])
    else: 
        # Dermatomyositis
        data_file = os.listdir(data_path+"CD27_Panel_Component/")
        label_file = os.listdir(data_path+"Labels/CD27_cell_labels/")
        mask_label_file = os.listdir(data_path+"Labels/CD27_cell_labels/Mask_Labels/")
        selected_data = list(set([_[:-9] for _ in data_file]).intersection(set([_[:-11] for _ in label_file])).intersection(set([_[:-17] for _ in mask_label_file]))) # 121919_Myo089_[7110,44031]_component
        # print("Length of selected data: ", len(selected_data))
        # print(f"selected_data: {selected_data}")
        X_train, X_test, y_train, _ = train_test_split(selected_data, [1]*len(selected_data), test_size=0.2, random_state=42)
        ## overwrite X_train once more
        X_train, X_val, _, _ = train_test_split(X_train, y_train, test_size=0.125, random_state=43)
        # print(f"X_train {X_train}, X_val: {X_val}")
        if dataclass==2:
            # train_list = [[(_ + '_data_' + str(idx) + '.npy',  _ + '_mask_' + str(idx) + '.npy') for idx in range(12)] 
            #             for _ in selected_data if _[:13] in ['121919_Myo089', '121919_Myo253', '121919_Myo368']]
            # validation_list = [[(_ + '_data_' + str(idx) + '.npy', _ + '_mask_' + str(idx) + '.npy') for idx in range(12)] 
            #                 for _ in selected_data if _[:13] in ['121919_Myo208', '121919_Myo388']]
            # test_list = [[(_ + '_data_' + str(idx) + '.npy', _ + '_mask_' + str(idx) + '.npy') for idx in range(12)] 
            #             for _ in selected_data if _[:13] in ['121919_Myo231', '121919_Myo511']]
            train_list = [[(_ + '_data_' + str(idx) + '.npy',  _ + '_mask_' + str(idx) + '.npy') for idx in range(12)] 
                        for _ in X_train]
            validation_list = [[(_ + '_data_' + str(idx) + '.npy', _ + '_mask_' + str(idx) + '.npy') for idx in range(12)] 
                            for _ in X_val]
            test_list = [[(_ + '_data_' + str(idx) + '.npy', _ + '_mask_' + str(idx) + '.npy') for idx in range(12)] 
                        for _ in X_test]
        
        if dataclass==3: #interpolation only contains one interpolated image per one raw image
            # train_list = [[(_ + '_data'  + '.npy',  _ + '_mask' + '.npy')]
            #             for _ in selected_data if _[:13] in ['121919_Myo089', '121919_Myo253', '121919_Myo368']]
            # validation_list = [[(_ + '_data' + '.npy', _ + '_mask' + '.npy')] 
            #                 for _ in selected_data if _[:13] in ['121919_Myo208', '121919_Myo388']]
            # test_list = [[(_ + '_data' + '.npy', _ + '_mask' + '.npy')] 
            #             for _ in selected_data if _[:13] in ['121919_Myo231', '121919_Myo511']]
            train_list = [[(_ + '_data'  + '.npy',  _ + '_mask' + '.npy')]
                        for _ in X_train]
            validation_list = [[(_ + '_data' + '.npy', _ + '_mask' + '.npy')] 
                            for _ in X_val]
            test_list = [[(_ + '_data' + '.npy', _ + '_mask' + '.npy')] 
                        for _ in X_test]

    train_list = list(itertools.chain(*train_list))
    validation_list = list(itertools.chain(*validation_list))
    test_list = list(itertools.chain(*test_list))
    print('\ntrain data: {}'.format(len(train_list)))
    print('validation data: {}'.format(len(validation_list)))
    print('test data: {}'.format(len(test_list)))

    return CustomDataset(train_list, tile_image_path, tile_label_path, transform=transform_train, mode='train', original_path=data_path), \
            CustomDataset(validation_list, tile_image_path, tile_label_path, transform=transform_val, mode='validation'), \
            CustomDataset(test_list, tile_image_path, tile_label_path, transform=transform_test, mode='test'), \
            train_list, \
            validation_list, \
            test_list
            

def build_dataset_ssl(data_path, tile_image_path, tile_label_path, dataclass, output_size):
    '''
    @variable:
        dataclass (int): Dermofit=1, tiled Dermatomyositis=2, interpolated Dermatomyositis=3
    '''
    transform_train = transforms.Compose([
        RandomGenerator(dataclass==1, output_size=output_size),
        ])

    transform_val=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(), Resize()])
    transform_test=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),Resize()])

    return build_dataset(
        data_path, tile_image_path, tile_label_path, 

        transform_train, transform_val, transform_test, dataclass)

if __name__ == '__main__':

 ''' data files '''
 DATA_PATH = '../dataset/Dermofit/original_data/'
 TILE_IMAGE_PATH = '../dataset/Dermofit_resize_noTiling/resize_image/'
 TILE_LABEL_PATH = '../dataset/Dermofit_resize_noTiling/resize_label/'
 
 db_train, db_val, db_test,  _, _, _ = build_dataset_ssl(DATA_PATH, TILE_IMAGE_PATH, TILE_LABEL_PATH, dataclass = 1, output_size=[224, 224])
 print(f"db_train[0]['image_cluster'].shape: {db_train[0]['image_cluster'].shape}")

 