import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from tqdm import tqdm
# import Tuple

def overlay(image: np.ndarray, 
            mask: np.ndarray,
            color: tuple[int, int, int] = (255, 0, 0),
            alpha: float = 0.5, 
            resize: tuple[int, int] = None # (1024, 1024)
            ) -> np.ndarray:

    '''
    Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image.
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.
    
    Returns:
        image_combined: The combined image.
    '''
    color = np.asarray(color).reshape(3,1,1)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    image = np.expand_dims(image, 0).repeat(3, axis=0)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv.resize(image_overlay.transpose(1, 2, 0), resize)
    
    image_combined = cv.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    return image_combined

if __name__ == "__main__":
    path_img = '/scratch/ssc10020/IndependentStudy/dataset/Lupus_Nephritis/tile_image/'
    path_mask = '/scratch/ssc10020/IndependentStudy/dataset/Lupus_Nephritis/tile_label/'
    save_masks_path = '/scratch/ssc10020/IndependentStudy/dataset/Lupus_Nephritis/segmentation_masks/'
    dataclass = '170-II-K4'
    if not os.path.exists(save_masks_path):
        os.mkdir(save_masks_path)

    imgs = os.listdir(path_img)
    masks = os.listdir(path_mask)

    for idx in tqdm(range(len(imgs))):
        im = np.load(path_img+dataclass+'_data_'+str(idx)+'.npy')
        mask = np.load(path_mask+dataclass+'_mask_'+str(idx)+'.npy')
        cut_overlay = overlay(im, mask, alpha=0.4)
        cut_overlay = np.moveaxis(cut_overlay, 0, -1)
        plt.imshow(cut_overlay)
        plt.savefig(save_masks_path+str(idx)+'.png')



