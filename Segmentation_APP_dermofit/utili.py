import numpy as np

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