import os
from array import array
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

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

def window_nd(a, window, steps=None):
    '''
    cut the image to window size (would get more than 1 image)
    if image is not big enough, pad the image

    Param:
    a: originial image
    window: the size of cutting
    steps: the size of helping cutting
    '''
    ashp = a.shape
    pad = np.zeros((len(window),2)) # [2,2]
    for _ in range(pad.shape[0]):
        # pad[0, 1] = 480 - (480 * (480 // 480) + 1408 % 480)
        pad[_, 1] = window[_] - (steps[_] * (window[_] // steps[_]) + ashp[_] % steps[_])
        while pad[_, 1] < 0:
            pad[_, 1] += steps[_]
    pad = pad.astype(int)
    a = np.pad(a, pad)
    ashp = np.array(a.shape)
    wshp = np.array(window).reshape(-1)
    if steps:
        stp = np.array(steps).reshape(-1)
    else:
        stp = np.ones_like(ashp)
    astr = np.array(a.strides)
    assert np.all(np.r_[ashp.size == wshp.size, wshp.size == stp.size, wshp <= ashp])
    shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
    strides = tuple(astr * stp) + tuple(astr)
    as_strided = np.lib.stride_tricks.as_strided
    aview = as_strided(a, shape=shape, strides=strides)
    return aview

if __name__ == "__main__":
    # For total data
    path = '../../dataset/Dermofit/original_data/'
    save_data_path = '../../dataset/Dermofit/tile_image/'
    save_label_path = '../../dataset/Dermofit/tile_label/'

    # Please change the size of window and step
    window_size = (480, 480)
    step_size = (480, 480)

    path_to_data = []
    path_to_mask = []
    listdir(path, path_to_data, path_to_mask)
    print("image_path length: ", len(path_to_data))
    print(path_to_data[:5])
    print("mask_path length: ", len(path_to_mask))
    print(path_to_mask[:5])

    for img_idx in tqdm(range(len(path_to_data))):
        image_path = path_to_data[img_idx]
        class_name = path_to_data[img_idx].split('/')[-3]
        img_name = path_to_data[img_idx].split('/')[-2]
        print(f"using png = {image_path}...")
        img = read_png(image_path)
        # get a 3 Dimension array (RGB values), need to change to other mode
        aview_list = []
        for i in range(3):
            aview = window_nd(img[:, :, i], window_size, step_size) # number_partial_img, 1, 480, 480
            aview = aview.reshape(-1, window_size[0], window_size[1]) # 12, 480, 480
            aview_list.append(np.expand_dims(aview, axis=1)) # 12,1, 480, 480
            
        aview_3channel = np.concatenate(aview_list, axis = 1) # 12,3, 480, 480
        for idx in range(aview.shape[0]):
            print(f"idx = {idx}, saving to name = {save_data_path + class_name}_{img_name}_data_{str(idx)}")
            np.save(save_data_path + class_name + '_' + img_name + '_data_' + str(idx), aview_3channel[idx])

        mask_path = path_to_mask[img_idx]
        mask = read_png(mask_path)
        print(mask.shape)
        aview = window_nd(mask, window_size, step_size)
        aview = aview.reshape(-1, window_size[0], window_size[1])
        for idx in range(aview.shape[0]):
            print(f"idx = {idx}, save to dir = {save_label_path + class_name}_{img_name}_mask_{str(idx)}")
            np.save(save_label_path + class_name + '_' + img_name + '_mask_' + str(idx), aview[idx])