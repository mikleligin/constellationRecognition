import tensorflow as tf
from keras.api.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
import pickle
import os
class Recognizer:
    lables = []
    pic = 0
    model = None
    pkl = None
    def __init__(self, model, pkl) -> None:
        #.\models\animalsModel.h5
        self.model = tf.keras.models.load_model(model)
        self.pkl = pkl

    # Путь к папке с картинкой
    def show(self, picPath):
        path = picPath
        pic = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]) - 1
        if(pic<0):pic=0
        
        with open(self.pkl, 'rb') as f:
            le = pickle.load(f)
        test_images = []
        test_labels = []
        dir_list = os.listdir(path)
        
        for i in dir_list:
            dir = os.path.join(path, i)
            file_list = os.listdir(dir)
            for j in file_list:
                files = os.path.join(dir, j)
                img = cv2.imread(files)
                img = cv2.resize(img, (64,64))
                img = np.array(img, dtype=np.float32)
                img = img/255
                test_images.append(img)
                test_labels.append(i)
        X_test = np.array(test_images)
        preds = self.model.predict(X_test)
        predicted_labels = le.inverse_transform(np.argmax(preds, axis=1))
        return predicted_labels[self.pic]
Recognizer.show()