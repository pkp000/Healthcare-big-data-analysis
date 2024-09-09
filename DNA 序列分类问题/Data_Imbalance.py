# N a m e   :裴鲲鹏
# Student ID:202100172014
# Date&Time :2024/3/16 15:58

import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer

# 读取列数据并预处理
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

# 创建数据框
df = pd.DataFrame({"Sequence": species_data, "Label": species_labels})

# 特征提取和标签准备，将每四个碱基序列作为一个词语，每次移动一个碱基
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 4))

# 提取特征向量
X = vectorizer.fit_transform(df["Sequence"])
y = df["Label"]


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据不平衡处理和分类
# 阈值移动
threshold = 0.1
class_counts = y_train.value_counts() / len(y_train)
y_train_threshold = y_train.isin(class_counts[class_counts >= threshold].index)
y_test_threshold = y_test.isin(class_counts[class_counts >= threshold].index)

X_train_threshold = X_train[y_train_threshold]
y_train_threshold = y_train[y_train_threshold]
X_test_threshold = X_test[y_test_threshold]
y_test_threshold = y_test[y_test_threshold]

# 数据不平衡处理和分类
# 欠采样
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# 过采样
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# 集成方法 - EasyEnsembleClassifier
eec = EasyEnsembleClassifier(n_estimators=10, random_state=42)
eec.fit(X_train, y_train)
y_pred_eec = eec.predict(X_test_threshold)

# 模型评估
print("Threshold Moving:")
print(classification_report(y_test_threshold, y_pred_eec))

# 随机欠采样分类器
rus_classifier = DecisionTreeClassifier()
rus_classifier.fit(X_resampled, y_resampled)
y_pred_rus = rus_classifier.predict(X_test_threshold)
print("Random Under-sampling:")
print(classification_report(y_test_threshold, y_pred_rus))

# 过采样分类器
smote_classifier = DecisionTreeClassifier()
smote_classifier.fit(X_smote, y_smote)
y_pred_smote = smote_classifier.predict(X_test)
print("SMOTE:")
print(classification_report(y_test, y_pred_smote))

# Easy Ensemble Classifier
y_pred_eec = eec.predict(X_test)
print("Easy Ensemble Classifier:")
print(classification_report(y_test, y_pred_eec))
