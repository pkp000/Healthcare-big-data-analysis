# N a m e   :裴鲲鹏
# Student ID:202100172014
# Date&Time :2024/3/29 18:29



import os
import time
import shutil
import pathlib
import itertools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, layers
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization

import warnings
warnings.filterwarnings("ignore")


# 创建带有标签的数据路径
def define_paths(data_dir):
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        if pathlib.Path(foldpath).suffix != '':
            continue

        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)

            if pathlib.Path(foldpath).suffix == '':
                # 检查用不到的 masks文件夹
                if pathlib.Path(fpath).parts[-1] == 'masks' or pathlib.Path(fpath).parts[-1] == 'Masks' or \
                        pathlib.Path(fpath).parts[-1] == 'MASKS':
                    continue

                else:
                    o_file = os.listdir(fpath)
                    for f in o_file:
                        ipath = os.path.join(fpath, f)
                        filepaths.append(ipath)
                        labels.append(fold)

            else:
                filepaths.append(fpath)
                labels.append(fold)

    return filepaths, labels


# 将带有标签的数据路径转换成dataframe表格类型，之后用于输入模型
def define_df(files, classes):
    Fseries = pd.Series(files, name='filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis=1)


# 划分数据集 train、 valid、 和 test
def split_data(data_dir):
    # 划分train dataframe表格类型
    files, classes = define_paths(data_dir)
    df = define_df(files, classes)
    strat = df['labels']
    train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123, stratify=strat)

    # 划分valid、test dataframe表格类型
    strat = dummy_df['labels']
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=strat)

    return train_df, valid_df, test_df


def create_gens(train_df, valid_df, test_df, batch_size):
    """
    该函数接受训练、验证和测试数据（dataframe类型），并将它们放入图像数据生成器中，随后模型从图像数据生成器获取数据。
    图像数据生成器还能将图像转换为张量。
    """

    # 定义参数
    img_size = (224, 224)
    channels = 3  # either BGR or Grayscale
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels)

    ts_length = len(test_df)
    test_batch_size = max(
        sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
    test_steps = ts_length // test_batch_size

    # 将在图像数据生成器中用于数据增强
    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
    ts_gen = ImageDataGenerator(preprocessing_function=scalar)

    train_gen = tr_gen.flow_from_dataframe(
        train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
        color_mode=color, shuffle=True, batch_size=batch_size)

    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                           class_mode='categorical',
                                           color_mode=color, shuffle=True, batch_size=batch_size)

    # 可以自定义test_batch_size，并使shuffle= false
    test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                          class_mode='categorical',
                                          color_mode=color, shuffle=False, batch_size=test_batch_size)

    return train_gen, valid_gen, test_gen


def show_images(gen):
    """
    本函数使用数据生成器并显示图像样本。
    """

    g_dict = gen.class_indices  # 定义个字典变量
    classes = list(g_dict.keys())
    images, labels = next(gen)  # 从发数据生成器上取一批样本

    # 计算显示的样本数量
    length = len(labels)
    sample = min(length, 25)

    plt.figure(figsize=(20, 20))

    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255  # 将数据缩放到范围(0 - 255)
        plt.imshow(image)
        index = np.argmax(labels[i])  # 得到图片的索引
        class_name = classes[index]  # 得到图片的类别标签
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.show()


data_dir = './COVID-19_Radiography_Dataset'

try:
    # 划分数据
    train_df, valid_df, test_df = split_data(data_dir)

    batch_size = 16
    train_gen, valid_gen, test_gen = create_gens(train_df, valid_df, test_df, batch_size)

except:
    print('Invalid Input')

show_images(train_gen)

def create_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 设置模型结构
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

# 分为几类
class_count = len(list(train_gen.class_indices.keys()))

# 创建模型
model = create_model(img_shape, class_count)

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 训练模型
epochs = 10  # 假设训练10个epochs
model.fit(train_gen, epochs=epochs, validation_data=valid_gen)

# 在测试集上评估模型
test_loss, test_acc = model.evaluate(test_gen)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
