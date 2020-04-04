# AI Continuous Learning Project

## Introduction:
Developing computer systems that are able to mimic the functionalities of the human brain has been a major task for the computer science community for a couple of decades, with the development of several machine learning models, such as deep neural network, convolutional neural network and so on, we are able to achieve high accuracy for tasks like classifying everyday objects and predicting continuous data. However, one of the problems that is keeping us from moving forward is that these machine learning models “forget” what they previously “learned” when given new training data, which causes the models to perform poorly on previous test data. This behavior is addressed as “catastrophic forgetting”, and the major goal of “continuous learning” models is to prevent such behavior from happening. 
	There are several well-known continuous learning strategies, in this project, we are going to examine two specific strategies: PathNet and Synaptic Intelligence. The data we will be using is a rotated MNIST dataset provided [here](https://github.com/facebookresearch/GradientEpisodicMemory/tree/master/data). The analysis of the dataset’s structure and performance of the two models is provided in both the notebooks and section below.
## Data:
The raw MNIST dataset contains 70,000 images of handwritten digits, which 60,000 of them are training data and 10,000 of them are testing data. Each image is represented by 784 (28 by 28) float points valued between 0 and 1, each float point is the grey scale value of each pixel in the image. We rotated all 70,000 images by 20 randomly generated degrees in the range of 0 to 180 to construct 20 rotated MNIST dataset where each set will be a task for our models.
## PathNet
### Principle Feature
### Result
### Conculsion
## Synaptic Intelligence
### Principle Feature
### Result
### Conclusion
## Reference
[1]. https://github.com/facebookresearch/GradientEpisodicMemory
