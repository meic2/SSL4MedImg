from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import torchvision.transforms as T

'''
define function to read in the image.tif, label.tif
'''
def read_tiff(path: str):
    """
    read the tif file to numpy array

    Param
    path: Path to the multipage-tiff file
    """
    tif_img_file = Image.open(path)
    images = []
    for i in range(tif_img_file.n_frames):
        tif_img_file.seek(i)
        images.append(np.array(tif_img_file))
    return np.array(images)


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
    # For total data path
    data_path = '../../dataset/Dermatomyositis/original_data/CD27_Panel_Component/'
    label_path = '../../dataset/Dermatomyositis/original_data/Labels/CD27_cell_labels/'
    data_mask_path = '../../dataset/Dermatomyositis/original_data/Labels/CD27_cell_labels/Mask_Labels/'
    save_data_path = '../../dataset/Dermatomyositis/interpolated_image/'
    save_label_path = '../../dataset/Dermatomyositis/interpolated_label/'

    data_file = os.listdir(data_path)
    label_file = os.listdir(label_path)
    label_file.remove('Bounding_Rectangle')
    label_file.remove('Mask_Labels')
    selected_data = set([_[:-9] for _ in data_file]).intersection(set([_[:-11] for _ in label_file]))
    print("Selected data length: ", len(selected_data))

    '''
    interpolate image only: both as .tif -> store as .npy,
    '''
    resize_to_shape = 480

    for _ in tqdm(selected_data):
        print(f"using tiff = {_}...")
        path_to_data = data_path + _ + '_data.tif'
        interpolated_img = np.array(resize(path_to_data, wanted_shape=resize_to_shape))
        np.save(save_data_path + _ + '_data', interpolated_img)
        
        path_to_mask = data_mask_path + _ + '_data.tifMask.tif'
        interpolated_mask = np.array(resize(path_to_mask, wanted_shape=resize_to_shape))
        np.save(save_label_path  + _ + '_mask', interpolated_mask)