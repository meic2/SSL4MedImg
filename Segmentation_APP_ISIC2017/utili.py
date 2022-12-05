import os 
import glob
import numpy as np
import statistics

def save_metric(
    train_acc, train_acc_mask, train_acc_bg, train_iou, 
    val_acc, val_acc_mask, val_acc_bg, val_iou, 
    train_confusion_matrix, val_confusion_matrix,
    save_metric_path):

    train_acc_list = []
    train_acc_mask_list = []
    train_bg_accuracy_list = []
    train_iou_list = []
    val_acc_list = []
    val_acc_mask_list = []
    val_acc_bg_list = []
    val_iou_list = []
    train_confusion_matrix_list = []
    val_confusion_matrix_list = []
    temp_train_list = {}
    temp_val_list = {}

    for i in range(len(train_acc)):
        train_acc_list.append(train_acc[i].item())
        train_acc_mask_list.append(train_acc_mask[i].item())
        train_bg_accuracy_list.append(train_acc_bg[i].item())
        train_iou_list.append(train_iou[i].item())
        val_acc_list.append(val_acc[i].item())
        val_acc_mask_list.append(val_acc_mask[i].item())
        val_acc_bg_list.append(val_acc_bg[i].item())
        val_iou_list.append(val_iou[i].item())
        for j in range(4):
            if j == 0:
                temp_train_list["tp"] = train_confusion_matrix[i][j].cpu().numpy()
                temp_val_list["tp"] = val_confusion_matrix[i][j].cpu().numpy()
            if j == 1:
                temp_train_list["fp"] = train_confusion_matrix[i][j].cpu().numpy()
                temp_val_list["fp"] = val_confusion_matrix[i][j].cpu().numpy()
            if j == 2:
                temp_train_list["fn"] = train_confusion_matrix[i][j].cpu().numpy()
                temp_val_list["fn"] = val_confusion_matrix[i][j].cpu().numpy()
            if j == 3:
                temp_train_list["tn"] = train_confusion_matrix[i][j].cpu().numpy()
                temp_val_list["tn"] = val_confusion_matrix[i][j].cpu().numpy()
        train_confusion_matrix_list.append(temp_train_list)
        val_confusion_matrix_list.append(temp_val_list)

    np.save(save_metric_path + '/train_acc.npy', np.array(train_acc_list))
    np.save(save_metric_path + '/train_acc_mask.npy', np.array(train_acc_mask_list))
    np.save(save_metric_path + '/train_bg_accuracy.npy', np.array(train_bg_accuracy_list))
    np.save(save_metric_path + '/val_acc.npy', np.array(val_acc_list))
    np.save(save_metric_path + '/val_acc_mask.npy', np.array(val_acc_mask_list))
    np.save(save_metric_path + '/val_acc_bg.npy', np.array(val_acc_bg_list))
    np.save(save_metric_path + '/val_iou.npy', np.array(val_iou_list))
    np.save(save_metric_path + '/train_confusion_matrix.npy', train_confusion_matrix_list)
    np.save(save_metric_path + '/val_confusion_matrix_list.npy', val_confusion_matrix_list)

# Median Weight; Author: Yahui Liu <yahui.liu@unitn.it>
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
            if lab not in labels_dict:
                labels_dict[lab] = 0
            labels_dict[lab] += cnt
    return get_weights(labels_dict)

def reverse_weight(w, median_freq = False):
    """
    Median Frequency Balancing: alpha_c = median_freq/freq(c).
    median_freq is the median of these frequencies 
    freq(c) is the number of pixles of class c divided by the total number of pixels in images where c is present

    return a list of ordered weight
    """
    assert len(w) > 0, "Expected a non-empty weight dict."
    print(w)
    values = [w[k] for k in w]
    if median_freq == True:
        if len(w) == 1:
            value = 1.0
        # elif len(w) == 2:
        #     value = min(values)
        #     print('Value:', value
        else:
            # Median Frequency Balancing
            value = statistics.median(values)
            print('Value:', value)
        for k in w:
            w[k] = value/(w[k]+1e-10)
        return [w[key] for key in range(len(w))]
    else: #naively return reverse weight
        total_l = len(values)
        reversed = [values[total_l-i-1]/np.sum(values) for i in range(total_l)]
        return reversed