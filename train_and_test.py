import argparse

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pickle
import os
import numpy as np
from glob import glob
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import copy
import math


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.nn.functional.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return abs(1 - dice)

# data loader
class imDist2(Dataset):
    def __init__(self, imPkl, gtPkl, label):

        self.imPkl = imPkl
        # print(self.imPkl)
        self.gtPkl = gtPkl
        self.label = label
        # print("Here")

    def __len__(self):
        return len(self.imPkl)
    #
    def __getitem__(self, item):
        ip = self.imPkl[item]  #input image dir
        # print("here", ip)
        gt = self.gtPkl[item] # electrode array groundtruth dir
        dp = self.label[item] # fold or non-fold label
        file = open(ip, "rb")
        image_file = np.fromfile(file, dtype=np.int16)    # input image
        file.close()
        im_file = image_file.reshape(32, 32, 32)   # reshaping image to 3D
        image_patch = im_file[None, ...]

        file2 = open(gt, "rb")
        gt_file2 = np.fromfile(file2, dtype=np.int16)  # electrode array groundtruth
        file2.close()
        gt_file = gt_file2.reshape(32, 32, 32) # reshaping electrode array groundtruth to 3D
        gt_patch = gt_file[None, ...]
        im_gt2=np.concatenate([im_file, gt_file], -1)
        im_gt = im_gt2[None, ...]
        dist_patch = dp

        image_patch = image_patch.astype('float')
        imarray = (image_patch - np.min(image_patch)) / (np.max(image_patch) - np.min(image_patch)) # normalization
        imarray = imarray*250
        gtarray = (gt_patch - np.min(gt_patch)) / (np.max(gt_patch) - np.min(gt_patch)+0.001)
        ht_num=len(np.where(gtarray>0.5)[0])
        dmarray = dist_patch
        return  imarray, gtarray, dmarray, ht_num
def main(fileDir='None', batch_sz=13):

    # load training data
    # fileDir = r'D:\research\fold_detect_code\Final_Dataset255'
    imgDir = os.path.join(fileDir, 'training', 'dataset_all_aug_2')
    imgFiles_training = glob(os.path.join(imgDir, '*.im'))
    LabelDir_training = os.path.join(fileDir, 'label_2')
    GTDir = os.path.join(fileDir, 'training', 'EA_GroundTruth_2')
    img_files=os.listdir(imgDir)
    # GTFiles_training = glob(os.path.join(GTDir, '*.im'))
    GTFiles_training=img_files
    for i in range(len(img_files)):
        GTFiles_training[i]=os.path.join(GTDir, img_files[i])

    xlsx_file_train = os.path.join(fileDir, 'training', 'label_2', 'training_label_all_aug.xlsx')
    df = pd.read_excel(xlsx_file_train, header=None)

    Label_training=df.values[0:,1:]
    Label_training = Label_training.astype('int16')

    # load validation data
    imgDir = os.path.join(fileDir, 'validation', 'dataset')
    imgFiles_validation = glob(os.path.join(imgDir, '*.im'))
    LabelDir_validation = os.path.join(fileDir, 'validation', 'label')
    GTDir = os.path.join(fileDir, 'validation', 'EA_GroundTruth')
    # GTFiles_validation = glob(os.path.join(GTDir, '*.im'))
    img_files=os.listdir(imgDir)
    GTFiles_validation=img_files
    for i in range(len(img_files)):
        GTFiles_validation[i]=os.path.join(GTDir, img_files[i])
    xlsx_file_val = os.path.join(fileDir, 'validation', 'label', 'validation_label.xlsx')
    df = pd.read_excel(xlsx_file_val, header=None)

    Label_validation=df.values[0:,1:]
    Label_validation = Label_validation.astype('int16')

    # load testing data
    imgDir = os.path.join(fileDir, 'testing', 'test_data_5_29_2023')
    imgFiles_testing = glob(os.path.join(imgDir, '*.im'))
    LabelDir_testing = os.path.join(fileDir, 'testing', 'label')
    fileDir2 = r'D:\tip_fold_over\Final_Dataset255\testing'
    GTDir = os.path.join(fileDir, 'testing', 'EA_GroundTruth_5_29_2023')

    img_files=os.listdir(imgDir)

    GTFiles_testing=img_files
    for i in range(len(img_files)):
        GTFiles_testing[i]=os.path.join(GTDir, img_files[i])
    xlsx_file_test = os.path.join(fileDir, 'testing', 'label', 'test_label_5_29_2023.xlsx')
    df = pd.read_excel(xlsx_file_test, header=None)

    Label_testing=df.values[0:,1:]
    Label_testing = Label_testing.astype('int16')

    dset_training = imDist2(imgFiles_training, GTFiles_training, Label_training)
    x, y, z, z2 = next(iter(dset_training))

    dset_validation = imDist2(imgFiles_validation, GTFiles_validation, Label_validation)
    x, y, z, z2 = next(iter(dset_validation))

    dset_testing = imDist2(imgFiles_testing, GTFiles_testing, Label_testing)
    x, y, z, z2 = next(iter(dset_testing))

    # batch_sz=13  # batch size
    trainLoader2 = DataLoader(dset_training, batch_size=batch_sz, shuffle=True, num_workers=0)
    validLoader = DataLoader(dset_validation, batch_size=1, shuffle=False, num_workers=0)
    testLoader = DataLoader(dset_testing, batch_size=1, shuffle=False, num_workers=0)


    from model2 import Modified3DUNet  # Maltitasking 3D Unet model
    net = Modified3DUNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    loss2 = nn.BCELoss().to(device)

    loss1 = nn.BCELoss().to(device)

    optimizer = optim.Adam(params=net.parameters(), lr=4e-5)

    max_epoch = 10   # number of epochs
    losses_training = []
    losses_validation = []
    losses1_training = []
    losses1_validation = []
    losses2_training = []
    losses2_validation = []
    for epoch in range(max_epoch):
        running_loss_training = 0.0
        running_loss_validation = 0.0
        running_loss1_training = 0.0
        running_loss1_validation = 0.0
        running_loss2_training = 0.0
        running_loss2_validation = 0.0
        for i, sample in enumerate(trainLoader2):
            optimizer.zero_grad()
            img = sample[0].float()  # input image
            gt= sample[1].float()   # electrode array groundtruth
            real_img = img.to(device)
            gt_img = gt.to(device)
            pred_class, pred_seg, pred_ht_num = net(real_img)
            trg = sample[2].float().to(device)
            l1 = loss1(pred_seg,gt_img)
            l2 = loss2(pred_class, trg)

            trg_ht_num = sample[3].float()
            l3=torch.sum(10*torch.exp(pred_ht_num*(-1)))
            l = (l1+5*l2)
            # l=1
            print(epoch, i, sample[0].shape, l1.data, l2.data, l3.data, l.data)
            l.backward()
            optimizer.step()
            running_loss_training += l.data.cpu().numpy()
            running_loss1_training += l1.data.cpu().numpy()
            running_loss2_training += l2.data.cpu().numpy()
        # scheduler.step()
        mn=0
        for j, sample in enumerate(validLoader):
            optimizer.zero_grad()
            img = sample[0].float() # input image
            gt = sample[1].float()   # electrode array groundtruth
            real_img = img.to(device)
            gt_img = gt.to(device)
            pred_class, pred_seg1, pred_ht_num = net(real_img)
            trg = sample[2].float().to(device)
            pred_seg = torch.reshape(pred_seg1, gt_img.shape)
            l1 = (loss1(pred_seg,gt_img))
            l2 = loss2(pred_class, trg)

            trg_ht_num = sample[3].float()
            l3=torch.sum(10*torch.exp(pred_ht_num*(-1)))
            l = (l1 + 5*l2)
            print(epoch, j, sample[0].shape, l1.data, l2.data, l3.data, l.data)
            running_loss_validation += l.data.cpu().numpy()
            running_loss1_validation += l1.data.cpu().numpy()
            running_loss2_validation += l2.data.cpu().numpy()

        epoch_loss_trianing = running_loss_training / (i+1)
        epoch_loss_validation = running_loss_validation / (j+1)
        epoch_loss1_trianing = running_loss1_training / (i + 1)
        epoch_loss1_validation = running_loss1_validation / (j + 1)
        epoch_loss2_trianing = running_loss2_training / (i + 1)
        epoch_loss2_validation = running_loss2_validation / (j + 1)

        losses_training.append(epoch_loss_trianing)
        losses_validation.append(epoch_loss_validation)
        losses1_training.append(epoch_loss1_trianing)
        losses1_validation.append(epoch_loss1_validation)
        losses2_training.append(epoch_loss2_trianing)
        losses2_validation.append(epoch_loss2_validation)
        if epoch == 0:
            Lowest_loss=100
        elif epoch > 0:
            if losses_validation[epoch]<Lowest_loss:
                torch.save(net.state_dict(), 'net_trained')
                Lowest_loss=losses_validation[epoch].copy()
                Lowest_loss_epoch=copy.deepcopy(epoch)
    plt.subplot(3, 1, 1)
    plt.plot(losses_training, label = "Training Loss")
    plt.plot(losses_validation, label = "Validation Loss")
    plt.ylabel('Overall Loss')
    # plt.yscale("log")
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(losses1_training, label = "Training Loss")
    plt.plot(losses1_validation, label = "Validation Loss")
    plt.ylabel('Segmentation Loss')
    # plt.yscale("log")
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(losses2_training, label = "Training Loss")
    plt.plot(losses2_validation, label = "Validation Loss")
    plt.ylabel('Classification Loss')
    # plt.yscale("log")
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    print("training completed")

    net.load_state_dict(torch.load('net_trained'))
    H=0
    pred_actual1=[]
    pred_actual2=[]
    prediction_test=[]
    target_test=[]
    for epoch in range(1):
        for i, sample in enumerate(testLoader):
            optimizer.zero_grad()
            img = sample[0].float()   # input image
            gt = sample[1].float()    # electrode array groundtruth
            real_img = img.clone().detach().to(device)
            gt_img = gt.clone().detach().to(device)
            pred_class, pred_seg, pred_ht_num = net(real_img)
            if pred_class.cpu().detach().numpy()[0, 0] > 0.1:
                prediction=1;
            else:
                prediction=0;
            trg = sample[2][0,0].float().to(device)
            pred_actual1.append(pred_class.cpu().detach().numpy()[0, 0])
            pred_actual2.append(pred_class.cpu().detach().numpy()[0, 1])
            prediction_test.append(prediction)
            target_test.append(trg.cpu().numpy())

            pred_seg_ind = pred_seg[0,0,:,:,:].cpu().detach().clone()
            pred_seg_ind=pred_seg_ind*1000
            folder_res = os.path.join(fileDir, 'testing')
            folder_result = folder_res + '\\seg_results\\'
            file_name=str(i)+".im"
            file_path=folder_result+file_name
            newFile = open(file_path, "wb")
            t_ct = pred_seg_ind.numpy().astype('int16')
            immutable_bytes = bytes(t_ct)
            newFile.write(immutable_bytes)
            newFile.close()

    cf_matrix = confusion_matrix(np.array(target_test), np.array(prediction_test))
    print(cf_matrix)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    # ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    ## Display the visualization of the Confusion Matrix.
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fileDir", default=r"D:\research\fold_detect_code\Final_Dataset255", help="Directory path")
    parser.add_argument("--batch_sz", default=13)
    args = parser.parse_args()

    main(args.fileDir, args.batch_sz)
