# N a m e   :裴鲲鹏
# Student ID:202100172014
# Date&Time :2024/3/27 18:43

import os
import cv2

imagePaths = []
IMG_SIZE = 224

X = []
y = []

for dirname, _, filenames in os.walk('./COVID-19_Radiography_Dataset'):
    for filename in filenames:
        if filename[-3:] == 'png':
            imagePaths.append(os.path.join(dirname, filename))

for img_path in imagePaths:
    label = img_path.split(os.path.sep)[-2]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

    X.append(img)
    y.append(label)
