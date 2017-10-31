The simple script which creates machine to recognize what type of animal we provided on the picture.
It uses machine learning - def. teaching computers how to learn from the data to make decisions or predictions.
The Keras python library was used, which includes Convolutional Neural Network.
Convolutional Neural Network - multi-layer neural network that assume the input data to be image.

LAYERS USED TO BUILD COVNETS:
INPUT -> hold the raw pixel values of the image i. e. [32x32x3] = width 32, height 32, three colour channels R, G, B;

CONV  -> layer compute the output of neuron that care connected to local regions in the input. Each computing a dot
product between their weights and a small region they are connected to the input value;

RELU  -> elementwise activation function. Leaves the volume unchanged;

POOL  -> perform a downsampling operation along spatial dimentions (width, height);

FC    -> (fully-connected) layer will compute the class scores, resulting in volume of size.


Some main result methods:

model.evaluate() - returns the loss value & metric value for the model in test mode,

model.metric_names() - give the display labels for scalar outputs,

model.predict() - generates output predictions for the input samples. Computation is done in batches.


Additional dictionary:

individual weights - represent the strength of connections between units. If the weight from unit A to unit B has 
greater magnitude (wielkoœæ), it means that A has greater influence over B (all else being equal).

weights - a set of filter coefficients, defining an image feature. For units in higher layers the inputs are not
from pixels anymore but from units in lower layers. Incoming weights are more like - preferred input patterns.

bias - a value that allows to shift the activation function to the left/right, which may be critical for succesful
learning.

http://cs231n.stanford.edu/
