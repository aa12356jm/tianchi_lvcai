# -*- coding: utf-8 -*-
'''
Created on Thu Sep 20 16:16:39 2018
 
@ author: herbert-chen
和baseline相比，这里只是更换了网络模型为densenet，但是无法跑通，具体原因待查。。。
图像大小必须要改为224才可以跑通，不清楚为什么
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


def main():
    # 随机种子
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    random.seed(666)

    # 获取当前文件名，用于创建模型及结果文件的目录
    file_name = os.path.basename(__file__).split('.')[0]
    # 创建保存模型和结果的文件夹
    if not os.path.exists('./model/%s' % file_name):
        os.makedirs('./model/%s' % file_name)
    if not os.path.exists('./result/%s' % file_name):
        os.makedirs('./result/%s' % file_name)
    # 创建日志文件
    if not os.path.exists('./result/%s.txt' % file_name):
        with open('./result/%s.txt' % file_name, 'w') as acc_file:
            pass
    with open('./result/%s.txt' % file_name, 'a') as acc_file:
        acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), file_name))

    # 默认使用PIL读图
    def default_loader(path):
        # return Image.open(path)
        return Image.open(path).convert('RGB')

    # 训练集图片读取
    class TrainDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, label = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)

    # 验证集图片读取
    class ValDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, label = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)

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

    # 数据增强：在给定角度中随机进行旋转
    class FixedRotation(object):
        def __init__(self, angles):
            self.angles = angles

        def __call__(self, img):
            return fixed_rotate(img, self.angles)

    def fixed_rotate(img, angles):
        angles = list(angles)
        angles_num = len(angles)
        index = random.randint(0, angles_num - 1)
        return img.rotate(angles[index])

    # 训练函数
    def train(train_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading
            data_time.update(time.time() - end)
            image_var = torch.tensor(images).cuda(async=True)
            label = torch.tensor(target).cuda(async=True)

            # compute y_pred
            y_pred = model(image_var)
            loss = criterion(y_pred, label)

            # measure accuracy and record loss
            prec, PRED_COUNT = accuracy(y_pred.data, target, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            acc.update(prec, PRED_COUNT)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))

    # 验证函数
    def validate(val_loader, model, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (images, labels) in enumerate(val_loader):
            image_var = torch.tensor(images).cuda(async=True)
            target = torch.tensor(labels).cuda(async=True)

            # compute y_pred
            with torch.no_grad():
                y_pred = model(image_var)
                loss = criterion(y_pred, target)

            # measure accuracy and record loss
            prec, PRED_COUNT = accuracy(y_pred.data, labels, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            acc.update(prec, PRED_COUNT)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('TrainVal: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))

        print(' * Accuray {acc.avg:.3f}'.format(acc=acc), '(Previous Best Acc: %.3f)' % best_precision,
              ' * Loss {loss.avg:.3f}'.format(loss=losses), 'Previous Lowest Loss: %.3f)' % lowest_loss)
        return acc.avg, losses.avg

    # 测试函数
    def test(test_loader, model):
        csv_map = OrderedDict({'filename': [], 'probability': []})
        # switch to evaluate mode
        model.eval()
        for i, (images, filepath) in enumerate(tqdm(test_loader)):
            # bs, ncrops, c, h, w = images.size()
            filepath = [i.split('/')[-1] for i in filepath]
            image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4

            with torch.no_grad():
                y_pred = model(image_var)

                # get the index of the max log-probability
                smax = nn.Softmax(1)
                smax_out = smax(y_pred)
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
            if pred_label == 0:
                sub_label.append('norm')
            else:
                sub_label.append('defect%d' % pred_label)

        # 生成结果文件，保存在result文件夹中
        submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
        submission.to_csv('./result/%s/submission.csv' % file_name, header=None, index=False)
        return

    # 保存最新模型以及最优模型
    def save_checkpoint(state, is_best, is_lowest_loss, filename='./model/%s/checkpoint.pth.tar' % file_name):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, './model/%s/model_best.pth.tar' % file_name)
        if is_lowest_loss:
            shutil.copyfile(filename, './model/%s/lowest_loss.pth.tar' % file_name)

    # 用于计算精度和时间的变化
    class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    # 学习率衰减：lr = lr / lr_decay
    def adjust_learning_rate():
        nonlocal lr
        lr = lr / lr_decay
        return optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)

    # 计算top K准确率
    def accuracy(y_pred, y_actual, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        final_acc = 0
        maxk = max(topk)
        # for prob_threshold in np.arange(0, 1, 0.01):
        PRED_COUNT = y_actual.size(0)
        PRED_CORRECT_COUNT = 0
        prob, pred = y_pred.topk(maxk, 1, True, True)
        # prob = np.where(prob > prob_threshold, prob, 0)
        for j in range(pred.size(0)):
            if int(y_actual[j]) == int(pred[j]):
                PRED_CORRECT_COUNT += 1
        if PRED_COUNT == 0:
            final_acc = 0
        else:
            final_acc = PRED_CORRECT_COUNT / PRED_COUNT
        return final_acc * 100, PRED_COUNT
    
    # 程序主体

    # GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    batch_size = 8
    workers = 0

    # epoch数量，分stage进行，跑完一个stage后降低学习率进入下一个stage
    stage_epochs = [10, 20]
    # 初始学习率
    lr = 1e-4
    # 学习率衰减系数 (new_lr = lr / lr_decay)
    lr_decay = 5
    # 正则化系数
    weight_decay = 1e-4

    # 参数初始化
    stage = 0
    start_epoch = 0
    total_epochs = sum(stage_epochs)
    best_precision = 0
    lowest_loss = 100

    # 训练及验证时的打印频率，用于观察loss和acc的实时变化
    print_freq = 1
    # 验证集比例
    val_ratio = 0.12
    # 是否只验证，不训练
    evaluate = False
    # 是否从断点继续跑
    resume = False

    # 创建模型
    model_ft = models.densenet161(pretrained=True)
    #有时候我们加载了训练模型后，只想调节最后的几层，其他层不训练。其实不训练也就意味着不进行梯度计算，
    # PyTorch中提供的requires_grad使得对训练的控制变得非常简单。
    #for param in model_ft.parameters():
         #param.requires_grad = False

    num_ftrs = model_ft.classifier.in_features
    #model_ft.avg_pool = nn.AdaptiveAvgPool2d(1)
    model_ft.classifier = nn.Linear(num_ftrs, 12)
    # model_ft.classifier = nn.Sequential(
    #     nn.AdaptiveAvgPool2d(1),
    #     #nn.BatchNorm1d(num_ftrs),
    #     #nn.Dropout(0.5),
    #     nn.Linear(num_ftrs, 12),
    # )

    print(model_ft)

    model = torch.nn.DataParallel(model_ft).cuda()


    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch'] + 1
            best_precision = checkpoint['best_precision']
            lowest_loss = checkpoint['lowest_loss']
            stage = checkpoint['stage']
            lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            # 如果中断点恰好为转换stage的点，需要特殊处理
            if start_epoch in np.cumsum(stage_epochs)[:-1]:
                stage += 1
                optimizer = adjust_learning_rate()
                model.load_state_dict(torch.load('./model/%s/model_best.pth.tar' % file_name)['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    # 读取训练图片列表
    #all_data = pd.read_csv('data/label.csv')
    # 分离训练集和测试集，stratify参数用于分层抽样
    #train_data_list, val_data_list = train_test_split(all_data, test_size=val_ratio, random_state=666, stratify=all_data['label'])
    # 读取测试图片列表

    train_data_list = pd.read_csv('trainlabel.csv')
    val_data_list = pd.read_csv('validlabel.csv')

    test_data_list = pd.read_csv('test.csv')

    # 图片归一化，由于采用ImageNet预训练网络，因此这里直接采用ImageNet网络的参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 训练集图片变换
    train_data = TrainDataset(train_data_list,
                              transform=transforms.Compose([
                                  transforms.Resize((256, 256)),
                                  transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomGrayscale(),
                                  # transforms.RandomRotation(20),
                                  FixedRotation([0, 90, 180, 270]),
                                  transforms.RandomCrop(224),
                                  transforms.ToTensor(),
                                  normalize,
                              ]))

    # 验证集图片变换
    val_data = ValDataset(val_data_list,
                          transform=transforms.Compose([
                              transforms.Resize((256, 256)),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              normalize,
                          ]))

    # 测试集图片变换
    test_data = TestDataset(test_data_list,
                            transform=transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    # 生成图片迭代器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss().cuda()

    # 优化器，使用带amsgrad的Adam
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)


    if evaluate:
        validate(val_loader, model, criterion)
    else:
        for epoch in range(start_epoch, total_epochs):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            precision, avg_loss = validate(val_loader, model, criterion)

            # 在日志文件中记录每个epoch的精度和loss
            with open('./result/%s.txt' % file_name, 'a') as acc_file:
                acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, precision, avg_loss))

            # 记录最高精度与最低loss，保存最新模型与最佳模型
            is_best = precision > best_precision
            is_lowest_loss = avg_loss < lowest_loss
            best_precision = max(precision, best_precision)
            lowest_loss = min(avg_loss, lowest_loss)
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_precision': best_precision,
                'lowest_loss': lowest_loss,
                'stage': stage,
                'lr': lr,
            }
            save_checkpoint(state, is_best, is_lowest_loss)

            # 判断是否进行下一个stage
            if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:
                stage += 1
                optimizer = adjust_learning_rate()
                model.load_state_dict(torch.load('./model/%s/model_best.pth.tar' % file_name)['state_dict'])
                print('Step into next stage')
                with open('./result/%s.txt' % file_name, 'a') as acc_file:
                    acc_file.write('---------------Step into next stage----------------\n')

    # 记录线下最佳分数
    with open('./result/%s.txt' % file_name, 'a') as acc_file:
        acc_file.write('* best acc: %.8f  %s\n' % (best_precision, os.path.basename(__file__)))
    with open('./result/best_acc.txt', 'a') as acc_file:
        acc_file.write('%s  * best acc: %.8f  %s\n' % (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), best_precision, os.path.basename(__file__)))

    # 读取最佳模型，预测测试集
    best_model = torch.load('./model/%s/model_best.pth.tar' % file_name)
    model.load_state_dict(best_model['state_dict'])
    test(test_loader=test_loader, model=model)

    # 释放GPU缓存
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
