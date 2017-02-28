# keras-squeezenet
SqueezeNet v1.1 Implementation using Keras Functional Framework 1.1

This [network model](https://github.com/rcmalli/keras-squeezenet/blob/master/images/SqueezeNet.png) has AlexNet accuracy with small footprint (5.1 MB)
Pretrained models are converted from original caffe network.

### Library Versions

- Keras v1.1
- Tensorflow v10
- Theano v0.8.2

### Example Usage

~~~python
from scipy import misc
import copy
import numpy as np
from squeezenet import get_squeezenet

model = get_squeezenet(1000, dim_ordering='th')
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.load_weights('../model/squeezenet_weights_th_dim_ordering_th_kernels.h5', by_name=True)

# read and prepare image input
from keras.applications.resnet50 import preprocess_input,decode_predictions
im = misc.imread('../images/cat.jpeg')
im = misc.imresize(im, (227, 227)).astype(np.float32)
im = np.expand_dims(im, axis=0)
im=preprocess_input(im)

res = model.predict(im)
~~~


### References

1) [Keras Framework](www.keras.io)

2) [SqueezeNet Official Github Repo](https://github.com/DeepScale/SqueezeNet)

3) [SqueezeNet Paper](http://arxiv.org/abs/1602.07360)

###Licence 

MIT License 
