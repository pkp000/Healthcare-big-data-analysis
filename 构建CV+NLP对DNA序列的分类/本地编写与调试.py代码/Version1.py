# N a m e   :裴鲲鹏
# Student ID:202100172014
# Date&Time :2024/3/18 10:19

import os
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DataLoaderDNA:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.sequences = []
        self.class_labels = []

    def load_datasets(self, labels):
        for label in labels:
            file_path = os.path.join(self.data_folder, label + ".txt")
            with open(file_path, "r") as file:
                lines = file.readlines()

                # 跳过首行
                for line in lines[1:]:
                    line = line.strip()

                    # 只提取DNA序列部分
                    sequence = line.split("\t")[0]
                    self.sequences.append(sequence)
                    self.class_labels.append(label)

    def preprocess_sequences(self):
        # 使用Tokenizer对象对DNA序列进行处理，char_level=True表示对字符级别进行编码
        tokenizer = Tokenizer(char_level=True)

        # 用fit_on_texts方法将DNA序列数据拟合到Tokenizer对象上，以建立文本与索引之间的映射关系
        tokenizer.fit_on_texts(self.sequences)

        # 使用texts_to_sequences方法将DNA序列转换为序列索引的形式，将每个字符替换为对应的整数索引
        sequences = tokenizer.texts_to_sequences(self.sequences)

        # 计算最大的序列长度，以便在填充时使用
        max_sequence_length = max(len(seq) for seq in sequences)

        # 使用pad_sequences方法将所有序列填充到相同的长度，以保证序列的长度一致
        sequences = pad_sequences(sequences, maxlen=max_sequence_length)

        # 将类标签转换为数字编码,用labels.index(label)获取其在self.class_labels列表中的索引，并将其添加到class_labels数组中
        class_labels = np.array([labels.index(label) for label in self.class_labels])
        return sequences, class_labels

    def create_data_sets(self, test_size=0.2, random_state=42):
        # 调用上面的方法
        sequences, labels = self.preprocess_sequences()

        # 获取序列的样本数量，并使用np.arange生成相应的索引数组
        num_samples = len(sequences)
        indices = np.arange(num_samples)

        # 随机打乱索引数组，以打乱序列和标签的顺序
        np.random.shuffle(indices)
        sequences = sequences[indices]
        labels = labels[indices]

        train_size = int((1 - test_size) * num_samples)  # 根据给定的test_size参数和总样本数量，计算训练集的大小

        # 打乱后的序列和标签根据训练集大小的索引进行切片，得到训练集序列和标签，以及测试集序列和标签
        train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
        train_labels, test_labels = labels[:train_size], labels[train_size:]

        return train_sequences, train_labels, test_sequences, test_labels


class DNASequenceClassifier(nn.Module):
    # hidden_dim（隐藏层的维度）、n_layers（RNN的层数）
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(DNASequenceClassifier, self).__init__()

        # 初始化参数
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 定义深度学习网络各层
        # RNN层。hidden_dim：RNN隐藏状态的维度大小，也可以理解为RNN单元的数量；n_layers为RNN的层数，表示堆叠多少个RNN单元；
        # batch_first表示输入数据的形状中第一个维度是batch size。如果设置为True，输入数据的形状应为(batch_size, seq_len, input_size）
        # 其中seq_len表示序列长度；如果设置为False，输入数据的形状应为(seq_len, batch_size, input_size)。
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # nn.Linear作为全连接层，将RNN的输出维度映射到output_size
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # 通常数据会以 batch 的形式输入，所以.size(0)表第一个维度的长度
        batch_size = x.size(0)

        # 用后面写的方法初始化隐藏层
        hidden = self.init_hidden(batch_size)

        # 输入数据，求得输出和隐藏层
        out, hidden = self.rnn(x, hidden)

        # 将原始的输出张量 out 进行连续化处理，并将其形状重新调整为一个二维张量，其中第一个维度的大小自动调整以适应数据量，
        # 第二个维度的大小为 self.hidden_dim，利于后续的全连接层处理或其他操作
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # 用来初始化RNN的隐藏状态，生成一个全零的张量作为初始隐藏状态，隐藏状态的大小为(n_layers, batch_size, hidden_dim)
        # 其中n_layers为RNN的层数，batch_size为batch大小，hidden_dim为隐藏层的维度
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


# 定义数据文件夹路径和类别标签
data_folder = "./DNA_sequence"
labels = ["chimpanzee", "dog", "human"]

# 创建数据加载器
data_loader = DataLoaderDNA(data_folder)
data_loader.load_datasets(labels)
train_sequences, train_labels, test_sequences, test_labels = data_loader.create_data_sets()

input_seq = torch.tensor(train_sequences)
target_seq = torch.tensor(train_labels)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


dict_size = len(input_seq)

# 创建DNA序列分类器python
model = DNASequenceClassifier(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)



# 训练模型
batch_size = 32
epochs = 10
lr = 0.1

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# train_model(train_sequences, train_labels, test_sequences, test_labels, batch_size, epochs)

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()  # Clears existing gradients from previous epoch
    input_seq.to(device)
     output, hidden = model(input_seq)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward()  # Does backpropagation and calculates gradients
    optimizer.step()  # Updates the weights accordingly

    if epoch % 10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

# 评估模型
model.evaluate_model(test_sequences, test_labels, batch_size)

# 保存模型
model_path = "dna_sequence_classifier.h5"
model.save_model(model_path)
