import os
import numpy as np
import pandas as pd

train_csv = "/scratch/lc4866/DEDL_Semisupervised_code_version/KFold_pt/train.csv"
test_csv = "/scratch/lc4866/DEDL_Semisupervised_code_version/KFold_pt/test.csv"
mapping_txt = "/scratch/lc4866/DEDL_Semisupervised_code_version/KFold_pt/splits.txt"


def find_local_tile_image_label(train_csv, mapping_txt, 
                                KthFold, 
                                ):#base_tile_image_dir, base_tile_label_dir)
   '''
   find 
   
   return
   ----
   @train_lis: list of directorys containing images: [.../.../data1.npy,    .../.../data2.npy]
   @mask_lis: list of directorys containing masks: [.../.../mask1.npy,    .../.../mask2.npy,]

   '''
   train_df = pd.read_csv(train_csv, sep=',', header=0)
   train_df.columns = ['index', 'path', 'target']
   print(train_df['index'][0])
   mapping_file = open(mapping_txt, 'r')
   Lines = mapping_file.readlines()
   train_mat = []
   val_mat = []
   idx = 0
   mat= [] 
   for line in Lines:
      if not line:  # if end of file is reached, close
        print("end of file has been reached")
        mapping_file.close()
        break
      else:
         line = line.strip().strip('\n').strip('[').strip("]").split()
         if len(line) == 1: ## find the start of this fold  
            if line[0] !="test":
               val_mat.append(mat)
               print(f"append to val")
               mat = []
               idx += 1

            elif line[0] =="test":
               train_mat.append(mat)
               print(f"append to train")
               mat = []    
         else:
            mat+=[int(elem) for elem in line]
   val_mat = val_mat[1:]
   print("end")
   print(f"len(train_mat): {[len(i) for i in train_mat]}")
   print(f"len(val_mat): {[len(i) for i in val_mat]}")
   print(f"max index in train_mat = {max([max(i) for i in train_mat])}")
   print(f"max index in val_mat = {max([max(i) for i in val_mat])}")

   # print(train_mat[-1])
   # print(val_mat[-1])
   ## take a trunk of the dataframe 
   KthFold_train_df = train_df.loc[train_mat[KthFold]]
   KthFold_val_df = train_df.loc[val_mat[KthFold]]
   
   ## make itinto list of absolute path
   train_lis_image = [('_').join([row['target'], row['path'].split(
       '/')[-1], 'data', '0'])+".npy" for _, row in KthFold_train_df.iterrows()]
   train_lis_image = sorted(train_lis_image)
   train_lis_mask = [('_').join([row['target'], row['path'].split(
       '/')[-1], 'mask', '0']) + ".npy" for _, row in KthFold_train_df.iterrows()]
   train_lis_mask = sorted(train_lis_mask)


   val_lis_image = [('_').join([row['target'], row['path'].split('/')[-1], 'data', '0']) + ".npy" for _, row in KthFold_val_df.iterrows()]
   val_lis_image = sorted(val_lis_image)
   val_lis_mask = [('_').join([row['target'], row['path'].split(
       '/')[-1], 'mask', '0']) + ".npy" for _, row in KthFold_val_df.iterrows()]
   val_lis_mask = sorted(val_lis_mask)
   
   
   print(train_lis_image[:5])
   # ['AK_A99_data_0.npy', 'AK_B138_data_0.npy', 'AK_B265a_data_0.npy', 'AK_B481_data_0.npy', 'AK_B511b_data_0.npy']
   print(train_lis_mask[:5])
   # ['AK_A99_mask_0.npy', 'AK_B138_mask_0.npy', 'AK_B265a_mask_0.npy', 'AK_B481_mask_0.npy', 'AK_B511b_mask_0.npy']
   print(val_lis_image[:5])
   # ['AK_B265b_data_0.npy', 'AK_B618b_data_0.npy', 'AK_B622_data_0.npy', 'AK_B644a_data_0.npy', 'AK_D231_data_0.npy']
   print(val_lis_mask[:5])
   # ['AK_B265b_mask_0.npy', 'AK_B618b_mask_0.npy', 'AK_B622_mask_0.npy', 'AK_B644a_mask_0.npy', 'AK_D231_mask_0.npy']
   
   return train_lis_image, train_lis_mask, val_lis_image, val_lis_mask
         




if __name__ =="__main__":
  find_local_tile_image_label(train_csv,
                              mapping_txt,
                              0)
                              #base_tile_image_dir,
                              #base_tile_label_dir)

