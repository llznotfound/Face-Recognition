import numpy as np
import os
from sklearn.decomposition import sparse_encode
from sklearn.preprocessing import normalize
import cv2
import random
from datetime import datetime

datapath = './yaleBExtData/' ##存放数据的路径
sample_folder_list = os.listdir(datapath) ##各个样本的文件夹
sample_folder_list_dict = {} ##各个文件夹下的图片路径信息
train_dir_dict = {} ##用作训练的样本路径集合
test_dir_dict = {} ##用作测试的样本路径集合

p = 20 ##每个人照片训练样本个数7,13,20

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
        img = cv2.imread(datapath + key + '/' + path, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('original', img)
        img = cv2.resize(img, (42, 48))
        # cv2.imshow('downsample', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        vec = np.zeros(42 * 48)
        for i in range(48):
            vec[i * 42:(i + 1) * 42] = img[i, :]

        train_data.append(vec[:])

    for path in test_dir_dict[key]:
        test_label.append(int(key[-2:]))
        img = cv2.imread(datapath + key + '/' + path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (42, 48))

        vec = np.zeros(42 * 48)
        for i in range(48):
            vec[i * 42:(i + 1) * 42] = img[i, :]

        test_data.append(vec[:])

train_data_unit = normalize(train_data)
test_data_unit = normalize(test_data)
print(train_data_unit.shape)
print(test_data_unit.shape)

error = 1e-3

t0 = datetime.now()
coder = sparse_encode(X=test_data_unit, dictionary=train_data_unit, algorithm='omp', alpha=error, max_iter=1000)
print(coder.shape, type(coder))

res = np.zeros(38)
predict_label = []
for test_num in range(coder.shape[0]):
    for i in range(38):
        coder_test = coder[test_num]
        res_vec = test_data_unit[test_num] - coder_test[i*p:(i+1)*p]@train_data_unit[i*p:(i+1)*p]
        res[i] = np.linalg.norm(res_vec)
    min_index = np.argmin(res)
    predict_label.append(train_label[min_index*p])

accuracy = sum(np.array(predict_label)==np.array(test_label))/len(test_label)
t1 = datetime.now()
time = (t1-t0).total_seconds() / len(test_data)

print('the average prediction time is ', time, 'seconds')
print('the prediction results are', accuracy)
