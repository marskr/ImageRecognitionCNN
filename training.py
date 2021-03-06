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

data_file = "data.txt"


class JsonManager:

    def __init__(self, file_name):
        self.file_name = file_name

    def processDataFromFile(self):
        with open(self.file_name) as json_file:
            data = json.load(json_file)

        return data


class MachineLearningModel:

    def __init__(self, img_width, img_height, json_name, h5_name, train_data_dir, validation_data_dir,
                       nb_train_samples, nb_validation_samples, epochs, batch_size):
        self.img_width = img_width
        self.img_height = img_height
        self.json_name = json_name
        self.h5_name = h5_name
        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.nb_train_samples = nb_train_samples
        self.nb_validation_samples = nb_validation_samples
        self.epochs = epochs
        self.batch_size = batch_size

    def trainDataModel(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (3, self.img_width, self.img_height)
        else:
            input_shape = (self.img_width, self.img_height, 3)

        # declaration of a sequential model format:
        model = Sequential()

        # declaration of the input layer:
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        # the shape of 1 sample (depth, width, height) of each digit image (in our case (3, 150, 150)
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='binary')

        model.fit_generator(
            train_generator,
            steps_per_epoch=self.nb_train_samples // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=self.nb_validation_samples // self.batch_size)

        model_json = model.to_json()
        with open(self.json_name, "w") as json_file:
            json_file.write(model_json)
        print("saved model in json format to: " + self.json_name)

        model.save_weights(self.h5_name)
        print("saved model in h5 format to: " + self.h5_name)

JM = JsonManager(data_file)
data = JM.processDataFromFile()

for p in data['CNN_training_settings']:
    MLM = MachineLearningModel(p['image_width'],
                               p['image_height'],
                               p['model_json_name'],
                               p['model_h5_name'],
                               p['train_data_direction'],
                               p['validation_data_direction'],
                               p['nb_train_samples'],
                               p['nb_validation_samples'],
                               p['model_epochs'],
                               p['model_batch_size'])

MLM.trainDataModel()
