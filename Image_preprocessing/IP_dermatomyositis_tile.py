from array import array
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def read_tiff(path: str) -> array:
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
    # Please change the path
    data_path = '../../dataset/Dermatomyositis/original_data/CD27_Panel_Component/'
    label_path = '../../dataset/Dermatomyositis/original_data/Labels/CD27_cell_labels/'
    data_mask_path = '../../dataset/Dermatomyositis/original_data/Labels/CD27_cell_labels/Mask_Labels/'
    save_data_path = '../../dataset/Dermatomyositis/tile_image/'
    save_label_path = '../../dataset/Dermatomyositis/tile_label/'

    data_file = os.listdir(data_path)
    label_file = os.listdir(label_path)
    label_file.remove('Bounding_Rectangle')
    label_file.remove('Mask_Labels')
    selected_data = set([_[:-9] for _ in data_file]).intersection(set([_[:-11] for _ in label_file]))
    print("Selected data length: ", len(selected_data))

    # Please change the size of window and step
    window_size = (480, 480)
    step_size = (480, 480)

    for _ in tqdm(selected_data):
        print(f"using tiff = {_}...")
        path_to_data = data_path + _ + '_data.tif'
        img = read_tiff(path_to_data)
        aview = window_nd(img[0], window_size, step_size)
        aview = aview.reshape(-1, window_size[0], window_size[1])
        for idx in range(aview.shape[0]):
            print(f"idx = {idx}, saving to name = {save_data_path + _}_data_{str(idx)}")
            np.save(save_data_path + _ + '_data_' + str(idx), aview[idx])
        
        path_to_mask = data_mask_path + _ + '_data.tifMask.tif'
        mask = read_tiff(path_to_mask)
        aview = window_nd(mask[0], window_size, step_size)
        aview = aview.reshape(-1, window_size[0], window_size[1])
        for idx in range(aview.shape[0]):
            print(f"idx = {idx}, save to dir = {save_label_path + _}'_mask_'{str(idx)}")
            np.save(save_label_path + _ + '_mask_' + str(idx), aview[idx])
    
