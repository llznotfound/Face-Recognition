import os
import numpy as np 
import cv2
import glob
import random
import copy
from datetime import datetime
from liblinearutil import *

datapath = './yaleBExtData/' ##存放数据的路径

p = 20 ##每个人照片训练样本个数7,13,20
print('p=', p)
def split_dataset(datapath, p):
    sample_folder_list = os.listdir(datapath) ##各个样本的文件夹
    sample_folder_list_dict = {} ##各个文件夹下的图片路径信息
    train_dir_dict = {} ##用作训练的样本路径集合
    test_dir_dict = {}##用作测试的样本路径集合

    for folder in sample_folder_list:
        sample_folder_list_dict[folder] = os.listdir(datapath+folder)
        train_dir_dict[folder] = random.sample(sample_folder_list_dict[folder], p)
        test_dir_dict[folder] = list(set(sample_folder_list_dict[folder])-set(train_dir_dict[folder]))

    train_label = []
    train_data = []

    test_label = []
    test_data = []

    for key in train_dir_dict:
        for path in train_dir_dict[key]:
            train_label.append(int(key[-2:]))
            img = cv2.imread(datapath+key+'/'+path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (42, 48))
            vec = np.zeros(42*48)
            for i in range(48):
                vec[i*42:(i+1)*42] = img[i, :]
            train_data.append(vec[:])
        
        for path in test_dir_dict[key]:
            test_label.append(int(key[-2:]))
            img = cv2.imread(datapath+key+'/'+path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (42, 48))
            vec = np.zeros(42*48)
            for i in range(48):
                vec[i*42:(i+1)*42] = img[i, :]
            test_data.append(vec[:])

    return train_data, train_label, test_data, test_label

def cv_c(train_label, train_data):
    ###交叉验证选择c
    best_accuracy = 0
    best_c = 0
    c_range = np.linspace(2**-5, 2**5, 50)
    for c in c_range:
        cmd = '-v 5' + ' -c ' + str(c)
        cv_accuracy = train(train_label, train_data, cmd)
        if cv_accuracy > best_accuracy:
            best_c = c
            best_accuracy = cv_accuracy
    return best_c

time_cost = []##时间记录
test_accuracy = []##准确度记录
best_cv = []##最优参数记录

for i in range(10):

    train_data, train_label, test_data, test_label = split_dataset(datapath, p)
    best_c = cv_c(train_label, train_data)

    cmd = '-c ' + str(best_c)
    model = train(train_label, train_data, cmd)

    t0 = datetime.now()
    t_lable, t_acc, t_val = predict(test_label, test_data, model)
    t1 = datetime.now()

    time = (t1-t0).total_seconds() / len(test_data)
    time_cost.append(time)
    test_accuracy.append(t_acc[0])
    best_cv.append(best_c)


print('time cost:', time_cost)
print('test accuracy:', test_accuracy)
print('best c is ', best_cv)


    
    




