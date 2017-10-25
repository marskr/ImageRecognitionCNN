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
pic_sample_dir = "test_pictures/cat.7119.jpg"
list_of_animals = ["dogs", "cats"]


class MachineLearningModel:

    def __init__(self, img_width, img_height, json_name, h5_name, pic_sample_dir):
        self.img_width = img_width
        self.img_height = img_height
        self.json_name = json_name
        self.h5_name = h5_name
        self.pic_sample_dir = pic_sample_dir

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

        print(pic_sample_dir + ":")

        img = cv2.imread(pic_sample_dir)
        img = cv2.resize(img, (img_width, img_height))
        img = np.reshape(img, [1, img_width, img_height, 3])

        classes = loaded_model.predict_classes(img)
        return classes

    def printAnimal(self, prediction_result):
        if prediction_result[0] == 0:
            print("It's a dog!")
        elif prediction_result[0] == 1:
            print("It's a cat!")
        else:
            print("ERROR!")

MLM = MachineLearningModel(img_width, img_height, json_name, h5_name, pic_sample_dir)
MLM.printAnimal(MLM.loadDataModel())
