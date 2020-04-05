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

After the path selection and training phase, ideally, the network should have one optimal path for each task. During the testing phase, the network will first determine which task does the test dataset belong to, then the network will make the prediction using the path that is reserved for that specific task. Therefore, we can expect that the network will produce the same test accuracy for the same test data every time as long as they were fed into the correct path. 

This [video](https://youtu.be/7fHN5zA7R3o) is provided by DeepMind, the network in the video was designed to perform transfer learning from Pong to Asterix. From the video, we can see a visual representation of path selection and modules being frozen after each task. The first 6 seconds of the video, the network performs the path selection, and at the 7th second of the video, we can see that the network finished training the first task, an optimal path was found and modules that construct this path were frozen. Then the network started the same process for the next task. Below is a screenshot of the video showing the path that was selected for the first task.

![Imgur](https://i.imgur.com/N1KIwg0.png)

### Result
Navigate to the PathNet notebook for the full result, we will take a look at the test result after training the first task and after the last task. 
```
Training Task 0 started...
Evaluating on task test sets
Test Accuracy on Task Set 0: 0.977400004863739
Test Accuracy on Task Set 1: 0.9394999742507935
Test Accuracy on Task Set 2: 0.9243999719619751
Test Accuracy on Task Set 3: 0.6528000235557556
Test Accuracy on Task Set 4: 0.5504000186920166
Test Accuracy on Task Set 5: 0.39660000801086426
Test Accuracy on Task Set 6: 0.30720001459121704
Test Accuracy on Task Set 7: 0.23399999737739563
Test Accuracy on Task Set 8: 0.18469999730587006
Test Accuracy on Task Set 9: 0.14669999480247498
Test Accuracy on Task Set 10: 0.14390000700950623
Test Accuracy on Task Set 11: 0.13379999995231628
Test Accuracy on Task Set 12: 0.13420000672340393
Test Accuracy on Task Set 13: 0.1535000056028366
Test Accuracy on Task Set 14: 0.19439999759197235
Test Accuracy on Task Set 15: 0.2085999995470047
Test Accuracy on Task Set 16: 0.2728999853134155
Test Accuracy on Task Set 17: 0.27410000562667847
Test Accuracy on Task Set 18: 0.29980000853538513
Test Accuracy on Task Set 19: 0.2955999970436096
Average Test Accuracy: 0.37122500091791155
Task 0 done.
```
```
Training Task 19 started...
Evaluating on task test sets
Test Accuracy on Task Set 0: 0.977400004863739
Test Accuracy on Task Set 1: 0.9718000292778015
Test Accuracy on Task Set 2: 0.9775999784469604
Test Accuracy on Task Set 3: 0.9761000275611877
Test Accuracy on Task Set 4: 0.9609000086784363
Test Accuracy on Task Set 5: 0.9782000184059143
Test Accuracy on Task Set 6: 0.9785000085830688
Test Accuracy on Task Set 7: 0.9761000275611877
Test Accuracy on Task Set 8: 0.9753000140190125
Test Accuracy on Task Set 9: 0.9754999876022339
Test Accuracy on Task Set 10: 0.9800000190734863
Test Accuracy on Task Set 11: 0.9761999845504761
Test Accuracy on Task Set 12: 0.9703999757766724
Test Accuracy on Task Set 13: 0.9767000079154968
Test Accuracy on Task Set 14: 0.965499997138977
Test Accuracy on Task Set 15: 0.9732000231742859
Test Accuracy on Task Set 16: 0.9731000065803528
Test Accuracy on Task Set 17: 0.9581000208854675
Test Accuracy on Task Set 18: 0.974399983882904
Test Accuracy on Task Set 19: 0.9801999926567078
Average Test Accuracy: 0.9737600058317184
Task 19 done.
```
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
[1]. https://github.com/facebookresearch/GradientEpisodicMemory <br>
[2]. https://youtu.be/7fHN5zA7R3o
