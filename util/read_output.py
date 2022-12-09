from glob import glob
import numpy as np

def table(iou, time, file_name, num, time_num):
    prefix_idx = 2
    print(file_name.split('_'))
    seed_num = int(file_name.split('_')[3+prefix_idx][4:])
    resnet_num = int(file_name.split('_')[4+prefix_idx][6:])
    unet_num = file_name.split('_')[5+prefix_idx]
    if len(file_name.split('_')) == 8+prefix_idx:
        ae = file_name.split('_')[7+prefix_idx]
        activation = file_name.split('_')[6+prefix_idx]
    elif len(file_name.split('_')) == 7+prefix_idx:
        ae = file_name.split('_')[6+prefix_idx]
        activation = ''

    time_min = float(time_num[0]) * 60
    time_min += float(time_num[2:4])
    if float(time_num[5:]) >= 30:
        time_min += 0.5

    if resnet_num == 18:
        if seed_num == 73:
            i = 0
        elif seed_num == 211:
            i = 1
        elif seed_num ==1234:
            i = 2
    elif resnet_num == 34:
        if seed_num == 73:
            i = 3
        elif seed_num == 211:
            i = 4
        elif seed_num ==1234:
            i = 5
    elif resnet_num == 50:
        if seed_num == 73:
            i = 6
        elif seed_num == 211:
            i = 7
        elif seed_num ==1234:
            i = 8
    elif resnet_num == 101:
        if seed_num == 73:
            i = 9
        elif seed_num == 211:
            i = 10
        elif seed_num ==1234:
            i = 11

    if unet_num == 'unet':
        if ae == 'wAE':
            if activation == 'relu':
                j = 1
            elif activation == 'gelu':
                j = 2
        elif ae == 'woAE':
            j = 0
    elif unet_num == 'unetpp':
        if ae == 'wAE':
            if activation == 'relu':
                j = 4
            elif activation == 'gelu':
                j = 5
        elif ae == 'woAE':
            j = 3
    
    iou[i][j] = num
    time[i][j] = time_min
    return iou, time

if __name__ == "__main__":
    img_files = glob(r'../Segmentation_APP_ISIC2017/output_file/*.out')

    iou = np.zeros((12, 6))
    time = np.zeros((12, 6))

    for img_file in img_files:
        print(img_file)
        with open(img_file) as f:
            lines = f.readlines()
            last_line = lines[-1]
            time_line = lines[-15]
            file_name = img_file.split('.')[2]
            print(time_line[34:41])
            iou, time = table(iou, time, file_name, last_line[12:18], time_line[34:41])
                
    print(iou)
    np.savetxt("iou_table.csv", iou, delimiter=',', fmt='%.4f')
    print(time)
    np.savetxt("time_table.csv", time, delimiter=',', fmt='%.4f')