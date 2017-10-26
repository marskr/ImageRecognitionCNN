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

# dimensions of our images.
img_width, img_height = 150, 150
json_name, h5_name = "modelCNN.json", "modelCNN.h5"
pic_sample_dir_pre = "test_pictures/"
pic_sample_dir_pos = ".jpg"
pic_sample_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
list_of_animals = ["dog", "cat", "rat", "horse"]


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
        json_file = open(json_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(h5_name)
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        for element in pic_sample_list:
            classes = loaded_model.predict_classes(self.imageOps(pic_sample_dir_pre + element + pic_sample_dir_pos))
            self.printAnimal(classes)

    def imageOps(self, pic_sample_dir):
        print("Loaded sample picture from directory " + pic_sample_dir + ":")

        img = cv2.imread(pic_sample_dir)
        img = cv2.resize(img, (img_width, img_height))
        return np.reshape(img, [1, img_width, img_height, 3])

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

MLM = MachineLearningModel(img_width, img_height, json_name, h5_name, list_of_animals, pic_sample_list,
                           pic_sample_dir_pre, pic_sample_dir_pos)
MLM.loadDataModel()
