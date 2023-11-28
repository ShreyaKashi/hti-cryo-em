


import pandas as pd 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import os


class CryoClassification():

    def __init__(self, root) -> None:
        self.dataset_root = root
        self.gt_labels = np.array([], dtype=object).reshape(0,2)
        self.img_array = []
        
    def get_label_image_dataset(self):

        for subdir, _, files in os.walk(self.dataset_root):
            for file in files:

                if file.endswith('selection.csv'):
                    gt = np.genfromtxt(os.path.join(subdir, file), delimiter=',')
                    gt = np.delete(gt, (0), axis=0) # remove header
                    self.gt_labels = np.concatenate((self.gt_labels, gt), axis=0)

                    for (idx, _) in gt:
                        zfill_num = len(str(gt.shape[0])) 
                        img_file_name = subdir.split('/')[-1] + '-classes-' + str(int(idx)+1).zfill(zfill_num) + '.png'
                        img_arr = imread(os.path.join(subdir, img_file_name))
                        img_resized = resize(img_arr, (64,64)) 

                        img_resized = img_resized.flatten().tolist()
                        self.img_array.append(img_resized)

        
        assert len(self.img_array) == self.gt_labels.shape[0]

        return self.img_array, self.gt_labels       

    def viz_img(flattened_list):
        img = np.reshape(np.array(flattened_list), (64,64))
        plt.imshow(img)      

    def run_svm(self):
        x, y = self.get_label_image_dataset()
        x_train, x_test, y_train, y_test = train_test_split(x, list(y[:, -1]), test_size=0.20, random_state=77, stratify=y) 

        param_grid={'C':[0.1,1,10,100], 
                    'gamma':[0.0001,0.001,0.1,1], 
                    'kernel':['rbf','poly']} 
        
        svc = svm.SVC(probability=True) 
        model = GridSearchCV(svc,param_grid)

        model.fit(x_train[:100],y_train[:100])

        y_pred = model.predict(x_test) 
        accuracy = accuracy_score(y_pred, y_test) 
        print('Accuracy: ', accuracy)



if __name__ == '__main__':
    root = "/home/kashis/Desktop/HTI/png"
    cryo_obj = CryoClassification(root)
    cryo_obj.run_svm()

