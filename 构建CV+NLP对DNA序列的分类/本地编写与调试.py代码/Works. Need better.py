# N a m e   :裴鲲鹏
# Student ID:202100172014
# Date&Time :2024/3/21 11:36

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import warnings

warnings.filterwarnings("ignore")

# 设置随机种子以保持结果可复现性
np.random.seed(42)
tf.random.set_seed(42)

# 定义数据文件夹路径和类别标签
data_folder = "./DNA_sequence"
labels = ["chimpanzee", "dog", "human"]

# 读取DNA序列数据
sequences = []
class_labels = []
for label in labels:
    file_path = os.path.join(data_folder, label + ".txt")
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:  # 跳过首行
            line = line.strip()
            sequence = line.split("\t")[0]  # 提取DNA序列部分
            sequences.append(sequence)
            class_labels.append(label)

# 将DNA序列转换为数值表示
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)

# 对序列进行填充，使其长度一致
max_sequence_length = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# 创建标签
labels = np.array([labels.index(label) for label in class_labels])

# 将数据集划分为训练集和测试集
num_samples = len(sequences)
indices = np.arange(num_samples)
np.random.shuffle(indices)
sequences = sequences[indices]
labels = labels[indices]
train_size = int(0.8 * num_samples)
train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# 将标签进行独热编码
num_classes = len(labels)
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# 构建模型
embedding_dim = 128
num_filters = 128
filter_sizes = [3, 4, 5]
dropout_rate = 0.5

model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=max_sequence_length))
model.add(Dropout(dropout_rate))

# 添加多个不同尺寸的卷积核
conv_blocks = []
for filter_size in filter_sizes:
    conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation="relu")(model.output)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)

model_output = tf.keras.layers.concatenate(conv_blocks, axis=-1)
model_output = Dropout(dropout_rate)(model_output)
model_output = Dense(128, activation="relu")(model_output)
model_output = Dense(num_classes, activation="softmax")(model_output)

model = tf.keras.Model(model.input, model_output)
# 编译模型
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 模型训练
batch_size = 32
epochs = 10

model.fit(train_sequences, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_sequences, test_labels))

# 模型评估
_, accuracy = model.evaluate(test_sequences, test_labels, batch_size=batch_size)
print("Test Accuracy:", accuracy)
