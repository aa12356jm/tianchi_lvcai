'''
此文件的主要作用:
1,加载训练好的模型读取测试图像进行测试
2,选择不同的模型进行测试
待做:
3,将多个模型结果做融合
4,TTA 测试增强

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
#file_name = 'main_inception_v4'
file_name ='main_denseNet'


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
    csv_map = OrderedDict({'filename': [], 'probability': []})
    # switch to evaluate mode
    model.eval()
    for i, (images, filepath) in enumerate(tqdm(test_loader)):
        # bs, ncrops, c, h, w = images.size()
        filepath = [os.path.basename(i) for i in filepath]
        image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4

        with torch.no_grad():
            y_pred = model(image_var)
            # 使用softmax函数将图片预测结果转换成类别概率
            smax = nn.Softmax(1)
            smax_out = smax(y_pred)

        # 保存图片名称与预测概率
        csv_map['filename'].extend(filepath)
        for output in smax_out:
            prob = ';'.join([str(i) for i in output.data.tolist()])
            csv_map['probability'].append(prob)

    result = pd.DataFrame(csv_map)
    result['probability'] = result['probability'].map(lambda x: [float(i) for i in x.split(';')])

    # 转换成提交样例中的格式
    sub_filename, sub_label = [], []
    for index, row in result.iterrows():
        sub_filename.append(row['filename'])
        pred_label = np.argmax(row['probability'])
        sub_label.append(label_warp['%d' % pred_label])

    # 生成结果文件，保存在result文件夹中，可用于直接提交
    submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
    #submission['label'] = submission['label'].map(label_warp)
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
            imgs.append((row['img_path']))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)


resize_img = 400
center_crop = 384

model = model_v4.v4(num_classes=12)
if file_name =='main_denseNet':
    model = models.densenet161(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 12)

    resize_img = 256
    center_crop = 224

model = torch.nn.DataParallel(model).cuda()

batch_size = 24
workers = 0

test_data_list = pd.read_csv('test.csv')

# 图片归一化，由于采用ImageNet预训练网络，因此这里直接采用ImageNet网络的参数
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 测试集图片变换
test_data = TestDataset(test_data_list,
                            transform=transforms.Compose([
                                transforms.Resize((resize_img, resize_img)),
                                transforms.CenterCrop(center_crop),
                                transforms.ToTensor(),
                                normalize,
                            ]))
test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)


# 读取最佳模型，预测测试集，并生成可直接提交的结果文件
best_model = torch.load('./model/%s/model_best.pth.tar' % file_name)
model.load_state_dict(best_model['state_dict'])
test(test_loader=test_loader, model=model)