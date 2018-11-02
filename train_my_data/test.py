'''
此文件的主要作用:
(1)加载训练好的模型,进行预测
(2)将预测结果按照指定格式写入到csv文件中

'''


import os
import time
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import model_v4

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# 获取当前文件名，用于创建模型及结果文件的目录
#file_name = os.path.basename(__file__).split('.')[0]
file_name = 'main_inception_v4'  #加载哪个文件训练的模型,这里是加载  main_inception_v4.py


#将数字标签转换为对应的真实标签,写入到预测结果中,和preprocessData.py中的label_warp对应.
#中间结果使用数字来表示对应标签的原因是pytorch不支持将含有字符串的list转换为tensor
label_warp = {"0": 'Black-grass',
              '1': 'Charlock',
              '2': 'Cleavers',
              '3': 'Common Chickweed',
              '4': 'Common wheat',
              '5': 'Fat Hen',
              '6': 'Loose Silky-bent',
              '7': 'Maize',
              '8': 'Scentless Mayweed',
              '9': 'Shepherds Purse',
              '10': 'Small-flowered Cranesbill',
              '11': 'Sugar beet',
              }


# 测试函数
def test(test_loader, model):
    csv_map = OrderedDict({'filename': [], 'probability': []}) #字典

    # switch to evaluate mode
    model.eval()#转换模型为测试模式,eval()时，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值

    #遍历所有测试图像,使用enumerate函数,可以得到i的值,为每次取的次数,依次加1
    #每次取的图像个数在test_loader函数中设置,图像存放在images,路径在这里filepath
    for i, (images, filepath) in enumerate(tqdm(test_loader)):
        # bs, ncrops, c, h, w = images.size()
        filepath = [os.path.basename(i) for i in filepath]#将所有的路径转换为list
        image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4将images转换为tensor

        with torch.no_grad():
            y_pred = model(image_var) #对image_var中的所有图像进行预测
            # 使用softmax函数将图片预测结果转换成类别概率
            smax = nn.Softmax(1)  #使用一维的softmax
            smax_out = smax(y_pred) #将各个类别的得分转换为概率值

        # 保存图片名称与预测概率
        csv_map['filename'].extend(filepath) #本次所有预测的图像文件名
        for output in smax_out: #将每个图像的预测结果,每张图像有多个预测结果
            prob = ';'.join([str(i) for i in output.data.tolist()]) #本张图像的所有预测结果
            csv_map['probability'].append(prob) #添加每张图像的预测结果

    result = pd.DataFrame(csv_map)
    result['probability'] = result['probability'].map(lambda x: [float(i) for i in x.split(';')])

    # 转换成提交样例中的格式
    sub_filename, sub_label = [], []
    for index, row in result.iterrows():
        sub_filename.append(row['filename'])
        pred_label = np.argmax(row['probability']) #得到预测的类别数
        sub_label.append(label_warp['%d' % pred_label])#将类别数转换为对应的string字符串,写入到文件中

    # 生成结果文件，保存在result文件夹中，可用于直接提交
    submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
    submission.to_csv('./result/%s/submission.csv' % file_name, header=None, index=False)
    return


# 默认使用PIL读图
def default_loader(path):
    # return Image.open(path)
    return Image.open(path).convert('RGB')


# 测试集图片读取
class TestDataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'])) #所有图像
        self.imgs = imgs
        self.transform = transform #图像变换函数
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index] #取出一张图像
        img = self.loader(filename) #加载图像
        if self.transform is not None:
            img = self.transform(img) #对图像进行转换
        return img, filename

    def __len__(self):
        return len(self.imgs) #得到图像数量


# 图片归一化，由于采用ImageNet预训练网络，因此这里直接采用ImageNet网络的参数
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_data_list = pd.read_csv('test.csv') #读取所有图像


batch_size = 24
workers = 0


model = model_v4.v4(num_classes=12) #网络模型
model = torch.nn.DataParallel(model).cuda()

# 测试集图片变换
test_data = TestDataset(test_data_list,
                            transform=transforms.Compose([
                                transforms.Resize((400, 400)),
                                transforms.CenterCrop(384),
                                transforms.ToTensor(),
                                normalize,
                            ]))

test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)


# 读取最佳模型，预测测试集，并生成可直接提交的结果文件
best_model = torch.load('./model/%s/model_best.pth.tar' % file_name) #加载模型权重文件
model.load_state_dict(best_model['state_dict'])
test(test_loader=test_loader, model=model)