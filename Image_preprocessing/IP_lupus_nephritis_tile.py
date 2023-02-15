from PIL import Image
from tifffile import TiffFile
from xml.etree import ElementTree
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

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

def create_circular_mask(h, w, center=None, radius=None):
    
    if center is None: # use the middle of the image
        center = (w//2, h//2)
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask.astype(int)

if __name__ == "__main__":
    data_file_path = "/scratch/ssc10020/IndependentStudy/dataset/Lupus_Nephritis/170-II-K4.qptiff"
    label_file_path = "/scratch/ssc10020/IndependentStudy/dataset/Lupus_Nephritis/170-II-K4_-CD25,CD45,CD31,CD44,MHCII,NKp46 2.csv"
    save_data_path = "/scratch/ssc10020/IndependentStudy/dataset/Lupus_Nephritis/tile_image/"
    if not os.path.exists(save_data_path):
        os.mkdir(save_data_path)
    
    save_label_path = '/scratch/ssc10020/IndependentStudy/dataset/Lupus_Nephritis/tile_label/'
    if not os.path.exists(save_data_path):
        os.mkdir(save_data_path)

    # Please change the size of window and step
    window_size = (480, 480)
    step_size = (480, 480)

    tif = TiffFile(data_file_path)
    
    # Choose channel Ch1cy2 containing DAPI stains
    img = tif.series[0].pages[1].asarray()  
    aview = window_nd(img, window_size, step_size)
    aview = aview.reshape(-1, window_size[0], window_size[1])
    print('Saving tiled images to ', save_data_path)
    for idx in tqdm(range(aview.shape[0])):
        np.save(save_data_path + data_file_path.split('/')[-1].split('.')[0] + '_data_' + str(idx), aview[idx])

    labels = pd.read_csv(label_file_path)
    label_masks = np.zeros((img.shape))  

    for idx in tqdm(range(labels.shape[0])):
        center = (round(labels.iloc[idx]['x']), round(labels.iloc[idx]['y']))
        radius = int(round(np.sqrt((labels.iloc[idx]['size'])/(np.pi))))
        x1, y1 = max(center[0]-radius, 0), max(center[1]-radius, 0)
        x2, y2 = min(center[0]+radius+1, img.shape[1]), min(center[1]+radius+1, img.shape[0])   

        temp_mask = create_circular_mask(y2-y1, x2-x1, radius=radius)
        label_masks[y1:y2, x1:x2]+=temp_mask
        del temp_mask
    
    aview = window_nd(label_masks, window_size, step_size)
    del label_masks
    aview = aview.reshape(-1, window_size[0], window_size[1])

    print('Saving tiled mask labels to ', save_data_path)
    for idx in tqdm(range(aview.shape[0])):
        aview[idx][aview[idx]>0] = 255
        np.save(save_label_path + data_file_path.split('/')[-1].split('.')[0] + '_mask_' + str(idx), aview[idx])


    # with TiffFile(path) as tif:
    #     for page in tif.series[0].pages:
    #         print(ElementTree.fromstring(page.description).find('Name').text)

    import pdb; pdb.set_trace()