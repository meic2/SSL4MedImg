import os
from array import array
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import torchvision.transforms as T

def listdir(path, image_path, mask_path): 
    """
    Get the path of image and mask image
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, image_path, mask_path)
        else:
            # Get the mask image file path
            if 'mask' in file_path:
                mask_path.append(file_path)
            else:
                # remove txt file and DS_store file
                if 'png' in file_path:
                    image_path.append(file_path)

def read_png(path: str) -> array:
    """
    read the png file to numpy array
    Param
    path: Path to the png file
    """
    img = Image.open(path)
    png_img_file = np.array(img)
    return png_img_file

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
    path = '../../dataset/Dermofit/original_data/'

    save_data_path = '../../dataset/Dermofit/interpolated_image/'
    save_label_path = '../../dataset/Dermofit/interpolated_label/'

    path_to_data = []
    path_to_mask = []
    listdir(path, path_to_data, path_to_mask)
    print("image_path length: ", len(path_to_data))
    print(path_to_data[:5])
    print("mask_path length: ", len(path_to_mask))
    print(path_to_mask[:5])

    print(path_to_data[0].split('/')[-2])

    for _ in tqdm(range(len(path_to_data))):
        image_path = path_to_data[_]
        class_name = path_to_data[_].split('/')[-3] ## AK
        img_name = path_to_data[_].split('/')[-2]  ## A41
        print(f"using png = {image_path}...")
        img = read_png(image_path)
        # print(img.shape) # (261, 397, 3)
        
        '''image'''
        print("interpolate...")
        interpolated_img = np.array(resize(image_path, wanted_shape=resize_to_shape))
        # print("interpolateing, img.shape: ", interpolated_img.shape) # interpolateing, img.shape:  (480, 480, 3)
        img_3D = np.expand_dims(np.array([interpolated_img[:, :, 0], 
                                          interpolated_img[:, :, 1], 
                                          interpolated_img[:, :, 2]]), 
                                axis=0)

        for idx in range(img_3D.shape[0]):
            print(f"idx = {idx}, saving to name = {save_data_path + class_name}_{img_name}_data_{str(idx)}")
            np.save(save_data_path + class_name + '_' + img_name + '_data_' + str(idx), img_3D[idx])

        '''mask'''
        mask_path = path_to_mask[_]
        mask = read_png(mask_path)
        # print("mask.shape: ", mask.shape) # mask.shape:  (338, 333)
        interpolate_mask = np.array(resize(mask_path, wanted_shape=resize_to_shape))
        # print(interpolate_mask.shape) # (480, 480)
        aview = np.expand_dims(interpolate_mask,axis=0)
        # print("aview.shape: ", aview.shape) # aview.shape:  (1, 480, 480)
        
        for idx in range(aview.shape[0]):
            print(f"idx = {idx}, save to dir = {save_label_path + class_name}_{img_name}_mask_{str(idx)}")
            np.save(save_label_path + class_name + '_' + img_name + '_mask_' + str(idx), aview[idx])
