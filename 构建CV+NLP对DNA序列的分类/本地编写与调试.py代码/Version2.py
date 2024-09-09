# N a m e   :裴鲲鹏
# Student ID:202100172014
# Date&Time :2024/3/23 11:44

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义序列嵌入类
class SeqEmbedding:
    def __init__(self):
        self.Dictionary = {}

    def __Sequence_to_Numeric(self, sequence):
        if sequence in self.Dictionary:
            return self.Dictionary[sequence]
        else:
            numeric = []
            for base in sequence:
                if base == 'A':
                    numeric.append(0)
                elif base == 'T':
                    numeric.append(1)
                elif base == 'C':
                    numeric.append(2)
                elif base == 'G':
                    numeric.append(3)
            self.Dictionary[sequence] = numeric
            return numeric

    def fit(self, sequences, window_size, stride_size):
        embeddings = []
        for sequence in sequences:
            length = len(sequence)
            for i in range(0, length - window_size + 1, stride_size):
                window = sequence[i : i + window_size]
                numeric = self.__Sequence_to_Numeric(window)
                embeddings.append(numeric)
        max_length = max(len(embedding) for embedding in embeddings)
        embeddings = [embedding + [0] * (max_length - len(embedding)) for embedding in embeddings]
        return np.array(embeddings)

# 加载DNA序列数据
chimpanzee_sequences = open('DNA_sequence/chimpanzee.txt').read().split('\n')[1::2]
dog_sequences = open('DNA_sequence/dog.txt').read().split('\n')[1::3]
human_sequences = open('DNA_sequence/human.txt').read().split('\n')[1::2]

# 构建训练集和标签
sequences = chimpanzee_sequences + dog_sequences + human_sequences
labels = [0] * len(chimpanzee_sequences) + [1] * len(dog_sequences) + [2] * len(human_sequences)

# 数据处理和划分
seq_embedding = SeqEmbedding()
embeddings = seq_embedding.fit(sequences, window_size=10, stride_size=5)
X = torch.tensor(embeddings, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)
num_classes = len(set(labels))

# 模型定义
class CVNLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CVNLPModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度顺序适应卷积层
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # 调整维度顺序适应LSTM层
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = self.fc(x)
        return x

# 定义超参数
input_size = 4  # 输入特征的维度（A、T、C、G）
hidden_size = 64  # LSTM隐藏层大小
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# 创建模型实例和优化器
model = CVNLPModel(input_size, hidden_size, num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 划分训练集和测试集
indices = list(range(len(X)))
np.random.shuffle(indices)
split = int(0.8 * len(X))
train_indices, test_indices = indices[:split], indices[split:]

# 确保索引不超出范围
train_indices = [index for index in train_indices if index < len(X)]
test_indices = [index for index in test_indices if index < len(X)]

# 重新计算划分后的长度
print("Length of train_indices:", len(train_indices))
print("Length of test_indices:", len(test_indices))

train_X, train_y = X[train_indices], y[train_indices]
test_X, test_y = X[test_indices], y[test_indices]

# 模型训练
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_X)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        train_acc = (outputs.argmax(dim=1) == train_y).float().mean()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc.item():.4f}")

# 模型测试
model.eval()
with torch.no_grad():
    outputs = model(test_X)
    test_acc = (outputs.argmax(dim=1) == test_y).float().mean()
    print(f"Test Accuracy: {test_acc.item():.4f}")
