# -*- coding:utf-8 -*-
import os
import numpy as np
from PIL import Image
from features import color_feature, shape_feature
from sklearn.neural_network import MLPClassifier
import time


TRAIN_DIR = '../图片/my-dataset/train/'
TEST_DIR = '../图片/my-dataset/test/'


def read_train_image_files():
    """获得所有的训练样本文件名及其所属标签"""
    train_img = []
    train_label= []

    labels = os.listdir(TRAIN_DIR)

    for label in labels:
        img_files = os.listdir(TRAIN_DIR+label+'/images/')
        train_img += img_files
        train_label += [label]*len(img_files)

    return np.array(train_img), np.array(train_label)


def read_test_image_files(labels):
    """获取测试集的测试样本文件名及其所属标签"""
    test_img = []
    test_label = []
    txt = open(TEST_DIR+'labels.txt')

    for line in txt.readlines(): 
        sline = line.split()
        img_file, label = sline[0], sline[1]
        test_img.append(img_file)
        test_label.append(label)

    return test_img, test_label


def feature_extract(image_file):
    """读取图片进行特征提取,返回一个一维向量"""
    img = Image.open(image_file)
    gray_img = img.convert("L")
    c_feat = color_feature(img)
    s_feat = shape_feature(gray_img)
    return np.append(c_feat, s_feat)
    # return c_feat


def get_train_set(train_img, train_label):
    """获得训练集"""
    X = []
    Y = train_label
    for i, img_file in enumerate(train_img):
        img_dir = TRAIN_DIR + train_label[i] +'/images/'
        vector = feature_extract(img_dir+img_file)
        X.append(vector)
        if (i+1) % 1000 == 0:
            print('已完成%d张图片' % (i+1))
    return np.array(X), np.array(Y)


def get_test_set(test_img, test_label):
    """获得测试集"""
    X = []
    Y = test_label
    for i, img_file in enumerate(test_img):
        img_dir = TEST_DIR +'/images/'
        vector = feature_extract(img_dir+img_file)
        X.append(vector)
        if (i+1) % 1000 == 0:
            print('已完成%d张图片' % (i+1))
    return np.array(X), np.array(Y)



def main():

    print('+------------------------------+')
    print('|   wlh tiny-imagenet 分类实验 |')
    print('+------------------------------+')
    # step1 : 读取所有的训练集和文件集 read file data
    print('step 1: 读取训练集和测试集文件及标签信息')
    train_img, train_label = read_train_image_files()
    test_img, test_label = read_test_image_files(train_label)
    print('训练集:共 %d 张' % len(train_img))
    print('测试集:共 %d 张' % len(test_img))
    print('step 1: done')

    # step2 : 特征提取 feature extract
    print('step 2: 开始特征提取')
    X, Y = get_train_set(train_img, train_label)
    print('训练集提取完毕')
    X1, Y1 = get_test_set(test_img, test_label)
    print('测试集提取完毕')
    print('step 2: done')

    # step3 : 训练 train
    print('step 3: 使用训练集训练分类器')
    t1 = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(100,50),max_iter=100)
    clf.fit(X,Y)
    t2 = time.time()
    print('训练结束,用时 %f s' % (t2-t1))
    print('step 3: done')

    # step4 : 分类 classify
    print('step 4: 得到分类结果')

    YT = clf.predict(X)
    print("训练集上的正确率: %.6f" % ((YT==Y).sum() / len(Y)))

    YY = clf.predict(X1)
    print("测试集上的正确率: %.6f" % ((YY==Y1).sum() / len(Y1)))
    print('step 4 done')

if __name__ == '__main__':
    main()
