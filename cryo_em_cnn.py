import csv
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
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import cv2

class CryroDataset():
    def __init__(self, data, transform=None):
        
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = imread(list(self.data.itertuples(index=False, name=None))[index][0])
        img = cv2.resize(img, (48,48))
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 0.0001)

        assert not np.isnan(img).any()

        # if np.isnan(img).any():
        #     print(img)

        # plt.imshow(img)
        # plt.show()
        
        if self.transform is not None:
            img = self.transform(img)

        y_label = torch.tensor(self.data.iloc[index][1])

        return img, y_label
    
    def viz(self):
        img = imread(self.data.iloc[0, 0])
        plt.imshow(img)


    
class CryoCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64*12*12, 256)
        self.batch_norm = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu5(x)
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




if __name__ == '__main__':
    root = "/home/kashis/Desktop/HTI"
    data_file = pd.read_csv(os.path.join(root, 'data.csv'))

    train_data, test_data = train_test_split(data_file, test_size=0.2, random_state=0, shuffle=True)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=0, shuffle=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5), (0.5))
    ])

    train_dataset = CryroDataset(data=train_data)
    val_dataset = CryroDataset(data=val_data)
    test_dataset = CryroDataset(data=test_data)

    # train_dataset.viz()

    num_epochs = 200
    learning_rate = 0.00001
    batch_size = 256
    shuffle = True
    pin_memory = True
    num_workers = 4

    train_loader = DataLoader(dataset=train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory)
    val_loader = DataLoader(dataset=val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    model = CryoCNN()

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

    loss_list = []

    loss_log = []
    acc_log = []
    val_acc_log = []
    val_loss_log = []
    best_val_acc = 0.0
    best_model_weights = None 

    for epoch in range(num_epochs):
        train_running_loss = 0
        train_running_acc = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):  
            images = images.to(device).to(torch.float32).to(device)
            labels = labels.to(device).to(torch.long).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum()

            train_running_loss += loss.item()
            train_running_acc += correct.item()
            loss_log.append(loss.item())
            acc_log.append(correct.item()/len(labels))

        train_running_loss /= i
        train_running_acc /= len(train_dataset)
            
        print('Training Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        loss_list.append(loss.item())

    # plt.plot(loss_list)
    # plt.show()


    for epoch in range(num_epochs):
        val_acc = 0
        val_loss = 0
        model.eval()
        for i, (images, labels) in enumerate(val_loader):  
            images = images.to(device).to(torch.float32).to(device)
            labels = labels.to(device).to(torch.long).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum()

            val_acc += correct.item()
            val_loss += loss.item()

        val_acc /= len(val_dataset)
        val_loss /= i

        val_acc_log.append(val_acc)
        val_loss_log.append(val_loss)
            
        print('Validation Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict()

    print(best_val_acc)        

    torch.save(best_model_weights, 'model_weights.pth')
