# N a m e   :裴鲲鹏
# Student ID:202100172014
# Date&Time :2024/3/30 9:35


import os
import pathlib
import itertools
import numpy as np
import pandas as pd
from torch import nn
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator




class Datasets:
    # 该部分与Version1.py的本质一样，故不再添加注释
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def define_paths(self):
        filepaths = []
        labels = []

        folds = os.listdir(self.data_dir)
        for fold in folds:
            foldpath = os.path.join(self.data_dir, fold)

            if pathlib.Path(foldpath).suffix != '':
                continue

            filelist = os.listdir(foldpath)
            for file in filelist:
                fpath = os.path.join(foldpath, file)


                if pathlib.Path(foldpath).suffix == '':

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

    def define_df(self, files, classes):
        Fseries = pd.Series(files, name='filepaths')
        Lseries = pd.Series(classes, name='labels')
        return pd.concat([Fseries, Lseries], axis=1)

    def split_data(self):

        files, classes = self.define_paths()
        df = self.define_df(files, classes)
        strat = df['labels']
        train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123, stratify=strat)
        strat = dummy_df['labels']
        valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=strat)

        return train_df, valid_df, test_df


class DataLoaders:
    def __init__(self, train_df, valid_df, test_df, batch_size):
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size

    def create_gens(self):
        img_size = (224, 224)
        channels = 3
        color = 'rgb'
        img_shape = (img_size[0], img_size[1], channels)

        def scalar(img):
            return img

        tr_gen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
        ts_gen = ImageDataGenerator(preprocessing_function=scalar)

        train_gen = tr_gen.flow_from_dataframe(
            self.train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
            color_mode=color, shuffle=True, batch_size=self.batch_size)

        valid_gen = ts_gen.flow_from_dataframe(self.valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                               class_mode='categorical',
                                               color_mode=color, shuffle=True, batch_size=self.batch_size)

        ts_length = len(self.test_df)
        test_batch_size = max(
            sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
        test_steps = ts_length // test_batch_size

        test_gen = ts_gen.flow_from_dataframe(self.test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                              class_mode='categorical',
                                              color_mode=color, shuffle=False, batch_size=test_batch_size)

        return train_gen, valid_gen, test_gen



class Model(nn.Module):
    def __init__(self, img_shape, num_classes):
        self.img_shape = img_shape
        self.num_classes = num_classes

    def create_model(self):
        base_model = keras.applications.VGG16(
            include_top=False,  # 不包含顶层的全连接层
            weights='imagenet',  # 在ImageNet数据集上预训练的权重
            input_shape=self.img_shape
        )
        # 冻结预训练模型的权重，不参与训练
        base_model.trainable = False

        model = keras.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


# 设置数据集路径，实例化datasets，切分数据
data_dir = './COVID-19_Radiography_Dataset'
datasets = Datasets(data_dir)
train_df, valid_df, test_df = datasets.split_data()

# 加载数据
batch_size = 32
data_loaders = DataLoaders(train_df, valid_df, test_df, batch_size)
train_gen, valid_gen, test_gen = data_loaders.create_gens()

# 定义输入图片形状和类别总数
img_shape = (224, 224, 3)  # Adjust the image shape if necessary
num_classes = len(train_df['labels'].unique())

# 创建模型实例
model = Model(img_shape, num_classes).create_model()
model.summary()

# 训练模型
epochs = 10  # 根据所需随时调整
history = model.fit(train_gen, validation_data=valid_gen, epochs=epochs)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_gen)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

# 绘制训练集和验证集上的准确率随Epoch的变化关系
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

