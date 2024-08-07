import tensorflow as ts
from keras.api.models import Sequential
from keras.api.layers import MaxPool2D 
from keras.api.layers import MaxPooling2D
from keras.api.layers import Conv2D
from keras.api.layers import Flatten
from keras.api.layers import Dense
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
images = []
labels = []

path = r'dataset'
counter = 0
dir_list = os.listdir(path)
for i in dir_list:
  dir = os.path.join(path, i)
  file_list = os.listdir(dir) #r'data\training_data\A\10.png'
  for j in file_list: 
    files = os.path.join(dir, j)
    img = cv2.imread(files)
    img = cv2.resize(img, (128,128))
    img = np.array(img, dtype=np.float32)
    img = img/255
    images.append(img)
    labels.append(i)

X = np.array(images)
y = np.array(labels)
le = LabelEncoder()
y = le.fit_transform(y)
X_sh, y_sh = shuffle(X, y, random_state=42)
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=32, kernel_size=(3,3),  activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(3,3),  activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=36, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_sh, y_sh, batch_size=16, epochs=15)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
model.save('starsModel2.h5')
#sho(model)
