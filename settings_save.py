import json

img_width, img_height = 150, 150
json_name, h5_name = "modelCNN.json", "modelCNN.h5"
pic_sample_dir_pre = "test_pictures/"
pic_sample_dir_pos = ".jpg"
pic_sample_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
list_of_animals = ["dog", "cat", "rat", "horse"]

train_data_dir = 'algo_learning_data/train'
validation_data_dir = 'algo_learning_data/validation'
nb_train_samples = 2000
nb_validation_samples = 1000
epochs = 50
batch_size = 16

data = {}
data['CNN_predicting_settings'] = []
data['CNN_predicting_settings'].append({
    'image_width': 150,
    'image_height': 150,
    'model_json_name': 'modelCNN.json',
    'model_h5_name': 'modelCNN.h5',
    'pic_sample_list': 'pic_sample_list',
    'list_of_animals': 'list_of_animals',
    'pic_sample_dir_pre': 'test_pictures/',
    'pic_sample_dir_pos': '.jpg'
})
data['CNN_training_settings'] = []
data['CNN_training_settings'].append({
    'image_width': 150,
    'image_height': 150,
    'model_json_name': 'modelCNN.json',
    'model_h5_name': 'modelCNN.h5',
    'model_epochs': 50,
    'model_batch_size': 16,
    'train_data_direction': 'algo_learning_data/train',
    'validation_data_direction': 'algo_learning_data/validation',
    'nb_train_samples': 2000,
    'nb_validation_samples': 1000
})

with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)

# how you turn a Python data structure into JSON
# json_str = json.dumps(data)

# how you turn a JSON-encoded string back into a Python data structure
# data = json.loads(json_str)



