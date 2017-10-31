from keras.preprocessing.image import ImageDataGenerator
# import sequential model type
from keras.models import Sequential
# import CNN layers from Keras. Efficiently train on image data by these module:
from keras.layers import Conv2D, MaxPooling2D
# import "core" layers from Keras:
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import model_from_json
import cv2
import numpy as np
import json

pic_sample_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
list_of_animals = ["dog", "cat", "rat", "horse"]
data_file = "data.txt"


class JsonManager:

    def __init__(self, file_name):
        self.file_name = file_name

    def processDataFromFile(self):
        with open(self.file_name) as json_file:
            data = json.load(json_file)

        return data


class MachineLearningModel:

    def __init__(self, img_width, img_height, json_name, h5_name, list_of_animals, pic_sample_list,
                 pic_sample_dir_pre, pic_sample_dir_pos):
        self.img_width = img_width
        self.img_height = img_height
        self.json_name = json_name
        self.h5_name = h5_name
        self.list_of_animals = list_of_animals
        self.pic_sample_list = pic_sample_list
        self.pic_sample_dir_pre = pic_sample_dir_pre
        self.pic_sample_dir_pos = pic_sample_dir_pos

    def loadDataModel(self):
        json_file = open(self.json_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(self.h5_name)
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        for element in pic_sample_list:
            classes = loaded_model.predict_classes(self.imageOps(self.pic_sample_dir_pre + element + self.pic_sample_dir_pos))
            self.printAnimal(classes)

    def imageOps(self, pic_sample_dir):
        print("Loaded sample picture from directory " + pic_sample_dir + ":")

        img = cv2.imread(pic_sample_dir)
        img = cv2.resize(img, (self.img_width, self.img_height))
        return np.reshape(img, [1, self.img_width, self.img_height, 3])

    def printAnimal(self, prediction_result):
        if prediction_result[0] == 0:
            print("It's a " + list_of_animals[0])
        elif prediction_result[0] == 1:
            print("It's a " + list_of_animals[1])
        elif prediction_result[0] == 2:
            print("It's a " + list_of_animals[2])
        elif prediction_result[0] == 3:
            print("It's a " + list_of_animals[3])
        else:
            print("ERROR!")

JM = JsonManager(data_file)
data = JM.processDataFromFile()

for p in data['CNN_predicting_settings']:
    MLM = MachineLearningModel(p['image_width'],
                               p['image_height'],
                               p['model_json_name'],
                               p['model_h5_name'],
                               list_of_animals,
                               pic_sample_list,
                               p['pic_sample_dir_pre'],
                               p['pic_sample_dir_pos'])

MLM.loadDataModel()
