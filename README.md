# AI Continuous Learning Project

## Introduction:
Developing computer systems that are able to mimic the functionalities of the human brain has been a major task for the computer science community for a couple of decades, with the development of several machine learning models, such as deep neural network, convolutional neural network and so on, we are able to achieve high accuracy for tasks like classifying everyday objects and predicting continuous data. However, one of the problems that is keeping us from moving forward is that these machine learning models “forget” what they previously “learned” when given new training data, which causes the models to perform poorly on previous test data. This behavior is addressed as “catastrophic forgetting”, and the major goal of “continuous learning” models is to prevent such behavior from happening. 
	There are several well-known continuous learning strategies, in this project, we are going to examine two specific strategies: PathNet and Synaptic Intelligence. The data we will be using is a rotated MNIST dataset provided [here](https://github.com/facebookresearch/GradientEpisodicMemory/tree/master/data). The analysis of the dataset’s structure and performance of the two models is provided in both the notebooks and section below.
## Data:
The raw MNIST dataset contains 70,000 images of handwritten digits, which 60,000 of them are training data and 10,000 of them are testing data. Each image is represented by 784 (28 by 28) float points valued between 0 and 1, each float point is the grey scale value of each pixel in the image. We rotated all 70,000 images by 20 randomly generated degrees in the range of 0 to 180 to construct 20 rotated MNIST dataset where each set will be a task for our models.
## PathNet
### Principle Feature
PathNet is a strategy aimed at reusing the parameters of a large neural network without catastrophic forgetting. It falls within the architectural strategies for continous learning. Pathways (views) through the network are used to determine the subset of parameters used in the forward pass and updated in the backward backpropogation pass. A tournament selection of these pathways is used to determine the most important parameters, which are then frozen once training on a task is completed.

A PathNet is a modular deep neural network with *L* layers and *M* modules per layer. In our example, modules are linear or convolutional, followed by a ReLu transfer function. A pathway (called a genotype in our code) is a selection of *N* modules per layer, which are used for the forward and backward passes. The final fully connected layer is unique for each task. 

For each task, a *population* of pathways (genotypes) are generated, and for several *generations* a tournament is held to select the better pathway. Each *generation*, the two pathways are trained, and the pathway with the better fitness (in our code accuracy) is selected as the winner. The loser of the tournament has his genotype overwritten by the winner. The winner has his genotype mutated to change the active modules. After several of these *generations*, ideally the tournament will cause a convergence on a single best pathway. This pathway then has its parameters frozen, then for the next task a new *population* of pathways/genotypes are chosen and the tournament selection begins again.
### Result
### Conculsion
## Synaptic Intelligence
### Principle Feature
Synaptic Intelligence is a regularization strategy for continuous learning. It defines a regularization term which is called the surrogate loss:

![Imgur](https://i.imgur.com/I4JS0ay.png)

This surrogate loss has an Omega term, which is the per parameter regularization strength defined as:

![Imgur](https://i.imgur.com/yMr0q3p.png)

little omega is the parameter specific contribution to the change in total loss.
### Result
### Conclusion
## Reference
[1]. https://github.com/facebookresearch/GradientEpisodicMemory
