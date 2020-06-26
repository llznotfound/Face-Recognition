import numpy as np
import os
import cv2
import random
from datetime import datetime
from sklearn import svm
from sklearn.decomposition import PCA

datapath = './yaleBExtData/'  ##存放数据的路径
sample_folder_list = os.listdir(datapath)  ##各个样本的文件夹
sample_folder_list_dict = {}  ##各个文件夹下的图片路径信息
train_dir_dict = {}  ##用作训练的样本路径集合
test_dir_dict = {}  ##用作测试的样本路径集合

p = 20  ##每个人照片训练样本个数7,13,20

for folder in sample_folder_list:
    sample_folder_list_dict[folder] = os.listdir(datapath + folder)
    train_dir_dict[folder] = random.sample(sample_folder_list_dict[folder], p)
    test_dir_dict[folder] = list(set(sample_folder_list_dict[folder]) - set(train_dir_dict[folder]))

train_label = []
train_data = []

test_label = []
test_data = []

hog = cv2.HOGDescriptor() #使用默认参数winSize(64,128), blockSize(16,16), blockStride(8,8), cellSize(8,8), nbins(9)

for key in train_dir_dict:
    for path in train_dir_dict[key]:
        train_label.append(int(key[-2:]))
        img = cv2.imread(datapath + key + '/' + path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 128))
        descriptor = hog.compute(img).reshape(-1)
        train_data.append(descriptor)

    for path in test_dir_dict[key]:
        test_label.append(int(key[-2:]))
        img = cv2.imread(datapath + key + '/' + path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 128))
        descriptor = hog.compute(img).reshape(-1)
        test_data.append(descriptor)
print(np.array(train_data).shape)
print(np.array(test_data).shape)

model = svm.SVC(C=1.0, kernel='linear', decision_function_shape='ovo')
model.fit(train_data, train_label)

t0 = datetime.now()
accuracy = model.score(test_data, test_label)
t1 = datetime.now()

print('the whole prediction time is ', (t1 - t0).total_seconds(), 'seconds')
print('the prediction results are', accuracy)


dimension = 200 # 设置pca降维后的维数，原始特征维数2016
pca = PCA(n_components=dimension)
pca_train_data = pca.fit_transform(train_data)
print(pca_train_data.shape, type(pca_train_data))
pca_test_data = pca.transform(test_data)
print(pca_test_data.shape)

model_pca = svm.SVC(C=1.0, kernel='linear', decision_function_shape='ovo')
model_pca.fit(pca_train_data, train_label)

t0 = datetime.now()
accuracy = model_pca.score(pca_test_data, test_label)
t1 = datetime.now()

print('the whole prediction time after pca is ', (t1 - t0).total_seconds(), 'seconds')
print('the prediction results after pca are', accuracy)
