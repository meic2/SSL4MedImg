import os
from array import array
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import itertools
import torchvision.transforms as T


def resize(file_dir, wanted_shape):
    '''
    Arguments
    ----
        all_img_lis: a list where each element is the absolute dir to image
    Opearations:
    ----
        save the bilinear interpolation image into save_image_dir
        
    Returns
    ----
        trabsformed image, shape(wanted_shape, wanted_shape, 3). 
    '''
    img = Image.open(file_dir)
    transform = T.Resize((wanted_shape, wanted_shape), interpolation = Image.Resampling.BILINEAR)
    return transform(img)

if __name__ == "__main__":
    resize_to_shape = 480
    step_size = (480, 480)

    # constants
    data_path = '/scratch/mc8895/dataset/ISIC2017/original_data/'
    save_data_path = '/scratch/mc8895/dataset/ISIC2017/resize_image/'
    save_label_path = '/scratch/mc8895/dataset/ISIC2017/resize_label/'

    train_origin_data='ISIC-2017_Training_Data/'
    train_origin_label='ISIC-2017_Training_Part1_GroundTruth/'
    val_origin_data='ISIC-2017_Validation_Data/'
    val_origin_label='ISIC-2017_Validation_Part1_GroundTruth/'
    test_origin_data='ISIC-2017_Test_v2_Data/'
    test_origin_label='ISIC-2017_Test_v2_Part1_GroundTruth/'

    # Train, val, test files
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

    train_path_to_data = [data_path+train_origin_data+img_path+'.jpg' for img_path in train]
    val_path_to_data = [data_path+val_origin_data+img_path+'.jpg' for img_path in validation]
    test_path_to_data = [data_path+test_origin_data+img_path+'.jpg' for img_path in test]

    train_path_to_mask = [data_path+train_origin_label+img_path+'_segmentation.png' for img_path in train]
    val_path_to_mask = [data_path+val_origin_label+img_path+'_segmentation.png' for img_path in validation]
    test_path_to_mask = [data_path+test_origin_label+img_path+'_segmentation.png' for img_path in test]

    path_to_data = train_path_to_data + val_path_to_data + test_path_to_data
    path_to_mask=train_path_to_mask+val_path_to_mask + test_path_to_mask

    print("image_path length: ", len(path_to_data))
    print(path_to_data[:1])
    print("mask_path length: ", len(path_to_mask))
    print(path_to_mask[:])


    #RESIZING ALL to 480 * 480 SIZE
    for img_idx in tqdm(range(len(path_to_data))):
        image_path = path_to_data[img_idx]
        class_name = path_to_data[img_idx].split('/')[-2]
        img_name = path_to_data[img_idx].split('/')[-1][:-4]
        print(f"using png = {class_name+'/'+img_name} ...")
        img = np.array(Image.open(image_path))

        '''image'''
        interpolated_img = np.array(resize(image_path, wanted_shape=resize_to_shape))
        # print("interpolateing, img.shape: ", interpolated_img.shape) # interpolateing, img.shape:  (480, 480, 3)
        img_3D = np.expand_dims(np.array([interpolated_img[:, :, 0], 
                                        interpolated_img[:, :, 1], 
                                        interpolated_img[:, :, 2]]), 
                                axis=0)

        for idx in range(img_3D.shape[0]):
            print(f"idx = {idx}, saving to name = {save_data_path + img_name}_data_{str(idx)}")
            np.save(save_data_path + img_name + '_data_' + str(idx), img_3D[idx])

        '''mask'''
        mask_path = path_to_mask[img_idx]
        mask = np.array(Image.open(mask_path))
    #     print("mask.shape: ", mask.shape) # mask.shape:  (338, 333)
        interpolate_mask = np.array(resize(mask_path, wanted_shape=resize_to_shape))
    #     print(interpolate_mask.shape) # (480, 480)
        aview = np.expand_dims(interpolate_mask,axis=0)
    #     print("aview.shape: ", aview.shape) # aview.shape:  (1, 480, 480)

        for idx in range(aview.shape[0]):
            print(f"idx = {idx}, save to dir = {save_label_path + img_name}_mask_{str(idx)}")
            np.save(save_label_path + img_name + '_mask_' + str(idx), aview[idx])
