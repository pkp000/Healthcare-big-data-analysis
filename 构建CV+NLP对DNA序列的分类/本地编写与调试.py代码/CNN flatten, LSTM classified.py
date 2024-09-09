# N a m e   :裴鲲鹏
# Student ID:202100172014
# Date&Time :2024/3/24 9:45

import os
import keras
import torch
import warnings
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
warnings.filterwarnings("ignore")


# 读取数据
species_data = []
species_labels = []

data_folder = "DNA_sequence"
files = ["chimpanzee.txt", "dog.txt", "human.txt"]

for file in files:
    file_path = os.path.join(data_folder, file)
    with open(file_path, "r") as f:
        next(f)  # 跳过标题行
        for line in f:
            sequence = line.split()[0]
            species_data.append(sequence)
            species_labels.append(file.split(".")[0])

# 将标签编码成数字
label_mapping = {"chimpanzee": 0, "dog": 1, "human": 2}
species_labels_encoded = np.array([label_mapping[label] for label in species_labels])

# 将序列编码成二维数字矩阵
species_data_encoded = []
for sequence in species_data:
    encoded_sequence = [1 if char == 'A' else 2 if char == 'T' else 3 if char == 'C' else 4 for char in sequence]
    species_data_encoded.append(encoded_sequence)

# 确定最长序列的长度
max_length = max(len(sequence) for sequence in species_data_encoded)

# 将其他序列补零到最长序列长度
for i in range(len(species_data_encoded)):
    sequence = species_data_encoded[i]
    if len(sequence) < max_length:
        sequence.extend([0] * (max_length - len(sequence)))

# 创建二维矩阵
matrix = np.array(species_data_encoded)

# 创建CNN模型
cnn_model = tf.keras.Sequential()
cnn_model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(max_length, 1)))
cnn_model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
cnn_model.add(tf.keras.layers.Flatten())

# 打印CNN模型摘要信息
cnn_model.summary()

# 将矩阵视为图像，调整形状
image_data = np.expand_dims(matrix, axis=2)

# 提取CNN模型的特征向量
features = cnn_model.predict(image_data)

# 创建NLP模型（如LSTM）进行分类
nlp_model = tf.keras.Sequential()
nlp_model.add(tf.keras.layers.LSTM(units=64, input_shape=(features.shape[1],)))
nlp_model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

# 编译NLP模型
nlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练NLP模型
nlp_model.fit(features, species_labels_encoded, epochs=10, batch_size=32)

# 进行预测
predictions = nlp_model.predict(features)


