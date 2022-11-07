import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import segmentation_models_pytorch as smp

def calculate_metric_iou(pred, label):
    pred = torch.tensor(pred)
    label = torch.tensor(label)
    tp, fp, fn, tn = smp.metrics.get_stats(pred, label.long(), mode='multilabel', threshold=0.5)
    # accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    return iou_score

def calculate_metric_percase(pred, gt):
    iou = calculate_metric_iou(pred, gt)
    bool_pred = pred == 1
    bool_gt = gt == 1
    bool_pred[bool_pred > 0] = 1
    bool_gt[bool_gt > 0] = 1
    if bool_gt.sum() == 0:
        return tuple()
    if pred.sum() > 0:
        dice = metric.binary.dc(bool_pred, bool_gt)
        hd95 = metric.binary.hd95(bool_pred, bool_gt) if bool_gt.sum()>0 else 0
        asd = metric.binary.asd(pred, gt)
        iou = calculate_metric_iou(pred, gt)
        return dice, hd95, asd, iou
    else:
        return 0, 0, 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    '''
    Note: if label is empty, return [] and ignore the validation metric.
    '''
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    # for ind in range(image.shape[0]):
    slice = image
    x, y = slice.shape[1], slice.shape[2]
    slice = zoom(slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(
            net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
    metric_list = []
    # for i in range(1, classes):
    res = calculate_metric_percase(prediction, label)
    if len(res) != 0:
        metric_list.append(res)
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
