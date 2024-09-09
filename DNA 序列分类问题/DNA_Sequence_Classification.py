# N a m e   :裴鲲鹏
# Student ID:202100172014
# Date&Time :2024/3/15 14:26

import os
import warnings
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
warnings.filterwarnings("ignore")


# 读取数据并预处理
species_data = []
species_labels = []

data_folder = "DNA_sequence"
files = ["chimpanzee.txt", "dog.txt", "human.txt"]

for file in files:
    file_path = os.path.join(data_folder, file)
    with open(file_path, "r") as f:
        # 跳过标题行,也就是首行
        next(f)
        for line in f:
            sequence = line.split()[0]
            species_data.append(sequence)
            species_labels.append(file.split(".")[0])

# 数据统计分析
sequence_lengths = [len(seq) for seq in species_data]
print("Sequence Lengths:")
print("Mean:", sum(sequence_lengths) / len(sequence_lengths))
print("Min:", min(sequence_lengths))
print("Max:", max(sequence_lengths))

# 绘制序列长度分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(sequence_lengths, bins=50)
plt.xlabel("Sequence Length")
plt.ylabel("Count")
plt.title("Distribution of Sequence Length")
plt.show()

# 统计每个物种的样本数量
species_counts = {}
for label in species_labels:
    species_counts[label] = species_counts.get(label, 0) + 1

# 可视化每个物种的样本数量
plt.figure(figsize=(10, 6))
plt.bar(species_counts.keys(), species_counts.values())
plt.xlabel("Species")
plt.ylabel("Sample Count")
plt.title("Number of DNA Sequence Samples per Species")
plt.show()


# 可视化每个物种的样本数量（饼状图）
plt.figure(figsize=(6, 6))
plt.pie(species_counts.values(), autopct="%.1f%%", labels=species_counts.keys())
plt.title("Distribution of DNA Sequence Samples per Species")
plt.show()


# 序列表征
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 7))
X = vectorizer.fit_transform(species_data)

# 机器学习分类
X_train, X_test, y_train, y_test = train_test_split(X, species_labels, test_size=0.2, random_state=42)

# 定义多个机器学习模型
models = [
    ("BalancedRandomForest", BalancedRandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ("SVM", SVC()),
    ("LogisticRegression", LogisticRegression()),
    ("MLPClassifier", MLPClassifier())]

# 交叉验证评估不同模型的分类性能
for model_name, model in models:
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{model_name} - Cross-Validation Scores: {scores}")
    print(f"{model_name} - Mean Cross-Validation Score: {scores.mean()}")

# 选择性能最好的模型进行训练和预测，假设第三个模型性能最好
best_model = models[0][1]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# 可视化混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 输出分类报告
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)
