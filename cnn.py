# -*- coding:utf-8 -*-
import os
import numpy as np
from PIL import Image

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import load_model
from skimage import img_as_float
import time


TRAIN_DIR = '../图片/my-dataset/train/'
TEST_DIR = '../图片/my-dataset/test/'

def build_model():
    input_shape = (32,32,3)
    num_classes = 20
    model = Sequential()

    # conv layer 1
    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
    model.add(Activation('relu'))
    # conv layer 2
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    # pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # conv layer 3
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    # pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # full connect layer 1
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate SGD optimizer
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def read_train_image_files():
    """获得所有的训练样本文件名及其所属标签"""
    train_img = []
    train_label= []

    labels = os.listdir(TRAIN_DIR)

    for i, label in enumerate(labels):
        img_files = os.listdir(TRAIN_DIR+label+'/images/')
        train_img += img_files
        train_label += [i]*len(img_files)

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
    """读取图片,进行预处理"""
    img = Image.open(image_file)
    # img = np.array(img)
    img = img.resize((32,32))

    img = img_as_float(np.array(img))

    if img.ndim != 3:
        """灰度图较少，为了保证维度一致，返回一个假的"""
        fake_img = np.zeros((img.shape[0], img.shape[1], 3))
        fake_img[:,:,0] = img[:,:]
        fake_img[:,:,1] = img[:,:]
        fake_img[:,:,2] = img[:,:]
        return fake_img
    else:
        return img


def get_train_set(train_img, train_label):
    """获得训练集"""
    labels = os.listdir(TRAIN_DIR)
    X = []
    Y = train_label
    for i, img_file in enumerate(train_img):
        img_dir = TRAIN_DIR + labels[train_label[i]] +'/images/'
        vector = feature_extract(img_dir+img_file)
        X.append(vector)
        if (i+1) % 1000 == 0:
            print('已完成%d张图片' % (i+1))
    return np.array(X), np.array(Y)


def get_test_set(test_img, test_label):
    """获得测试集"""
    labels = os.listdir(TRAIN_DIR)
    X = []
    Y = test_label
    for i, img_file in enumerate(test_img):
        img_dir = TEST_DIR +'/images/'
        vector = feature_extract(img_dir+img_file)
        X.append(vector)
        if (i+1) % 1000 == 0:
            print('已完成%d张图片' % (i+1))

    for i,y in enumerate(Y):
        Y[i] = labels.index(y)
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
    print(X.shape)
    print('训练集提取完毕')
    X1, Y1 = get_test_set(test_img, test_label)
    print('测试集提取完毕')
    print('step 2: done')

    # step3 : 训练 train
    print('step 3: 使用训练集训练分类器')
    t1 = time.time()
    model = build_model()
    Y = np_utils.to_categorical(Y, 20)
    Y1 = np_utils.to_categorical(Y1, 20)
    result = model.fit(X, Y, batch_size=100,epochs=20,shuffle=True,verbose=1,validation_split=0.2, validation_data=(X1,Y1))
    t2 = time.time()
    model.save_weights('my_model.h5')
    print('训练结束,用时 %f s' % (t2-t1))
    print('step 3: done')

    # step4 : 分类 classify
    print('step 4: 得到分类结果')
    score = model.evaluate(X1, Y1)
    print("\n验证集上的正确率:")
    print(score[1])
    print('step 4 done')
    # YT = clf.predict(X)
    # print("训练集上的正确率: %.6f" % ((YT==Y).sum() / len(Y)))

    # YY = clf.predict(X1)
    # print("测试集上的正确率: %.6f" % ((YY==Y1).sum() / len(Y1)))
    # print('step 4 done')

if __name__ == '__main__':
    main()
