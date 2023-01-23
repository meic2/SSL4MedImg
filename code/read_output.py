from glob import glob
import numpy as np

#global variables
seeds = {'73': 0, '211':1, '1234': 2}
label_ratios = {'10p': 0, '30p': 1, '50p': 2, '70p':3, '99p': 4}

def table(iou, file_name, last_line, cur_data_class):
    name_lists = file_name.split('_')
    if name_lists[-1] == 'train':
        return

    test_model = int(name_lists[-1][-1])

    seed = name_lists[3]
    data_class = name_lists[1]
    label_ratio = name_lists[2]


    if cur_data_class!= int(data_class):
        return
    print(file_name)
    ious = last_line.split(',')
    print(ious)
    # time_min = float(time_num[0]) * 60
    # time_min += float(time_num[2:4])
    # if float(time_num[5:]) >= 30:
    #     time_min += 0.5

    for i in range(4):
        iou[seeds[seed]][i+4*(test_model-1)][label_ratios[label_ratio]] = float(ious[i].replace(" ", ""))
    # time[i][j] = time_min
    return iou #time

if __name__ == "__main__":
    img_files = glob(r'output_tests/*.out')
    # current data class 
    cur_data_class = 4
    iou = np.zeros((3, 8, 5))
    # time = np.zeros((3, 8, 5))

    # filename format: dataclass_2_70p_1234_test1.out
    for img_file in img_files:
        with open(img_file) as f:
            file_name = img_file.split('.')[0].split('/')[1]
            
            lines = f.readlines()
            last_line = lines[-1]
            # time_line = lines[-15]
            table(iou, file_name, last_line[30:-4], cur_data_class)
                
    print(iou)
    for i, s in enumerate(seeds.keys()):
        np.savetxt(f"iou_table_{s}_class_{cur_data_class}.csv", iou[i], delimiter=',', fmt='%.4f')

    
    # np.savetxt("time_table.csv", time, delimiter=',', fmt='%.4f')