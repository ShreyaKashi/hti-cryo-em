import os
import csv
import numpy as np




def store_label_image_dataset(dataset_root):
        for subdir, _, files in os.walk(dataset_root):
            for file in files:

                if file.endswith('selection.csv'):
                    gt = np.genfromtxt(os.path.join(subdir, file), delimiter=',')
                    gt = np.delete(gt, (0), axis=0) # remove header
                   
                    for (idx, gt_label_val) in gt:
                        zfill_num = len(str(gt.shape[0])) 
                        img_file_name = subdir.split('/')[-1] + '-classes-' + str(int(idx)+1).zfill(zfill_num) + '.png'
                        img_file_complete_path = os.path.join(subdir, img_file_name)


                        file = open('data.csv', 'a+', newline ='')
                        with file:    
                            write = csv.writer(file)
                            write.writerows([[img_file_complete_path, gt_label_val]])

        print('Done')