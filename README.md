# Edgar Alan Turing
Text Generation with LSTM Neural Networks using [Keras](https://keras.io/) running on top of [Theano](http://deeplearning.net/software/theano/).

# Training
By using the Amazon Web Services infrastructure, we were able to access GPUs to speed up the training of our deep learning models.
A Community Amazon Machine Image was used (ami-125b2c72), which was made for the Stanford CS231n class.
The following are the specifications for the GPU Graphics Instance that was used:
```
g2.2xlarge:
  15 GiB memory,
  1 x NVIDIA GRID GPU (Kepler GK104),
  60 GB of local instance storage,
  64-bit platform
```
The models each trained for 50 epochs, which took between 4 and 9 hours on Amazon's GPUs, depending on the size of the text files.

The training loss for the models went as follows:
![alt text](https://github.com/evanotero/edgar-alan-turing/blob/master/photo.png "Training Loss by Epoch")


# Results
The different authors' models vary in performance, with the Edgar Allan Poe model performing poorly, and the JK Rowling model showing the most promise. The models show both a base-level understanding of the English language, reliably forming proper words and sentences, and a higher level understanding of character, plot, and rhythm, creating sequences of text featuring similar dialogue and rhythmic patterns to those of the authors they were modeled from. 

# Web Application
Available at ~~[edgaralanturing.com](edgaralanturing.com)~~.

_Created for HackHarvard 2016._
