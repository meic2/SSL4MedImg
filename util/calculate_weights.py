# Author: Yahui Liu <yahui.liu@unitn.it>
# the original code: https://github.com/yhlleo/DeepSegmentor/blob/master/tools/calculate_weights.py
# Median Frequency Balancing: https://arxiv.org/pdf/1411.4734.pdf

import os
import glob
import numpy as np
import statistics

def get_weights(labels_dict):
    total_pixels = 0
    for lab in labels_dict:
        total_pixels += labels_dict[lab]
    for lab in labels_dict:
        labels_dict[lab] /= float(total_pixels)
    return labels_dict

def calculate_weights(im_path):
    assert os.path.isdir(im_path)
    img_list = glob.glob(os.path.join(im_path, '*.npy'))
    labels_dict = {}
    for im_path in img_list:
        im = np.load(im_path)
        im = np.around(im / 255)

        labels, counts = np.unique(im, return_counts=True)
        for lab, cnt in zip(labels, counts):
            if len(labels) == 2:
                # print(labels, counts)
                if lab not in labels_dict:
                    labels_dict[lab] = 0
                labels_dict[lab] += cnt
    return get_weights(labels_dict)

def reverse_weight(w):
    """
    Median Frequency Balancing: alpha_c = median_freq/freq(c).
    median_freq is the median of these frequencies 
    freq(c) is the number of pixles of class c divided by the total number of pixels in images where c is present
    """
    assert len(w) > 0, "Expected a non-empty weight dict."
    values = [w[k] for k in w]
    if len(w) == 1:
        value = 1.0
    else:
        # Median Frequency Balancing
        value = statistics.median(values)
        print('Value:', value)
    for k in w:
        w[k] = value/(w[k]+1e-10)
    return w

if __name__ == '__main__':
    # for Dermatomyositis
    label_path = '../../dataset/Dermatomyositis/tile_label/'
    # for Dermofit
    # label_path = '../../dataset/Dermofit/tile_label/'
    
    weights = calculate_weights(label_path)
    print(weights)
    print(reverse_weight(weights))