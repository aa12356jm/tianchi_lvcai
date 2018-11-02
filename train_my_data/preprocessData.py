'''
此文件主要有三个作用
(1)将图像重新划分为训练集和验证集
(2)将训练集和验证集的图像文件路径和标签存储在csv文件中
(3)将测试集的图像文件路径存储在csv文件中

'''

import json
import pandas as pd
import os
import os.path as osp

#可以再加一部分,如何将整个图像文件夹划分为训练集,验证集
split_train_valid = False

def reg_Data(dataDir='./data'):
    pass



if split_train_valid:
    pass




#将标签类别转换为数字,因为pytorch不支持将list中的字符串转换为tensor
#在test.py中再转换回来
label_warp = {'Black-grass': 0,
              'Charlock': 1,
              'Cleavers': 2,
              'Common Chickweed': 3,
              'Common wheat': 4,
              'Fat Hen': 5,
              'Loose Silky-bent': 6,
              'Maize': 7,
              'Scentless Mayweed': 8,
              'Shepherds Purse': 9,
              'Small-flowered Cranesbill': 10,
              'Sugar beet': 11,
              }

#训练集,将训练集的所有图像路径和标签写入到一个文件中
img_path_train, label_train = [], []  #存储训练集每个图像文件的路径,存储训练集每个图像文件的标签
jsonTrain = "../data/train/"  #存储图像各个类别文件夹的路径
train_labels = os.listdir('../data/train')  #每个文件夹作为当前类别的标签
for i in train_labels:  #遍历每个类别中的图像
    img_train=os.listdir(os.path.join(jsonTrain, i)) #当前类别文件中的所有图像文件名
    for j in img_train: #遍历所有图像
        img_path_train.append(jsonTrain+"/"+i+"/"+j)  #存储每个图像文件的路径
        label_train.append(i)  #存储每个图像文件的标签

label_file_train = pd.DataFrame({'img_path': img_path_train, 'label': label_train})
label_file_train['label'] = label_file_train['label'].map(label_warp) #将文件夹名字转换为数字

label_file_train.to_csv('trainlabel.csv', index=False)  #保存到本地csv文件中
print("train end \n")


#验证集,同上
img_path_valid, label_valid = [], []
jsonValid = "../data/valid/"
valid_labels = os.listdir('../data/valid')
for i in valid_labels:
    img_valid = os.listdir(os.path.join(jsonValid, i))
    for j in img_valid:
        img_path_valid.append(jsonValid+"/"+i+"/"+j)
        label_valid.append(i)
label_file_valid = pd.DataFrame({'img_path': img_path_valid, 'label': label_valid})
label_file_valid['label'] = label_file_valid['label'].map(label_warp)
label_file_valid.to_csv('validlabel.csv', index=False)
print("valid end \n")


#加载测试集
test_data_path = '../data/test'
all_test_img = os.listdir(test_data_path) #获取路径下的所有图像文件
test_img_path = [] #存储所有测试图像的路径

#遍历图像
for img in all_test_img:
    if osp.splitext(img)[1] == '.png': #指定图像扩展名
        test_img_path.append(osp.join(test_data_path, img)) #保存图像路径

#保存所有测试图像的路径到csv文件中
test_file = pd.DataFrame({'img_path': test_img_path})
test_file.to_csv('test.csv', index=False)