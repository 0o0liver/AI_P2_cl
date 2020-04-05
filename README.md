# AI Continuous Learning Project
## Team 5 Members
| Name       | NetID  |
|------------|--------|
| David Chu  | dfc296 |
| Chuhan Jin | cj1436 |
| Binghan Li | bl1890 |
## Introduction:
Developing computer systems that are able to mimic the functionalities of the human brain has been a major task for the computer science community for a couple of decades, with the development of several machine learning models, such as deep neural network, convolutional neural network and so on, we are able to achieve high accuracy for tasks like classifying everyday objects and predicting continuous data. However, one of the problems that is keeping us from moving forward is that these machine learning models “forget” what they previously “learned” when given new training data, which causes the models to perform poorly on previous test data. This behavior is addressed as “catastrophic forgetting”, and the major goal of “continuous learning” models is to prevent such behavior from happening. 
	There are several well-known continuous learning strategies, in this project, we are going to examine two specific strategies: PathNet and Synaptic Intelligence. The data we will be using is a rotated MNIST dataset provided [here](https://github.com/facebookresearch/GradientEpisodicMemory/tree/master/data). The analysis of the dataset’s structure and performance of the two models is provided in both the notebooks and section below.
## Data:
The raw MNIST dataset contains 70,000 images of handwritten digits, which 60,000 of them are training data and 10,000 of them are testing data. Each image is represented by 784 (28 by 28) float points valued between 0 and 1, each float point is the grey scale value of each pixel in the image. We rotated all 70,000 images by 20 randomly generated degrees in the range of 0 to 180 to construct 20 rotated MNIST dataset where each set will be a task for our models.

## Metrics:
We will use metrics as defined in the Facebook GEM paper:

![Imgur](https://i.imgur.com/ojiE2a7.png)

## PathNet
### Principle Feature
PathNet is a strategy aimed at reusing the parameters of a large neural network without catastrophic forgetting. It falls within the architectural strategies for continous learning. Pathways (views) through the network are used to determine the subset of parameters used in the forward pass and updated in the backward backpropogation pass. A tournament selection of these pathways is used to determine the most important parameters, which are then frozen once training on a task is completed.

A PathNet is a modular deep neural network with *L* layers and *M* modules per layer. In our example, modules are linear or convolutional, followed by a ReLu transfer function. A pathway (called a genotype in our code) is a selection of *N* modules per layer, which are used for the forward and backward passes. The final fully connected layer is unique for each task. 

For each task, a *population* of pathways (genotypes) are generated, and for several *generations* a tournament is held to select the better pathway. Each *generation*, the two pathways are trained, and the pathway with the better fitness (in our code accuracy) is selected as the winner. The loser of the tournament has his genotype overwritten by the winner. The winner has his genotype mutated to change the active modules. After several of these *generations*, ideally the tournament will cause a convergence on a single best pathway. This pathway then has its parameters frozen, then for the next task a new *population* of pathways/genotypes are chosen and the tournament selection begins again.

After the path selection and training phase, ideally, the network should have one optimal path for each task. During the testing phase, the network will first determine which task does the test dataset belong to, then the network will make the prediction using the path that is reserved for that specific task. Therefore, we can expect that the network will produce the same test accuracy for the same test data every time as long as they were fed into the correct path. 

This [video](https://youtu.be/7fHN5zA7R3o) is provided by DeepMind, the network in the video was designed to perform transfer learning from Pong to Asterix. From the video, we can see a visual representation of path selection and modules being frozen after each task. The first 6 seconds of the video, the network performs the path selection, and at the 7th second of the video, we can see that the network finished training the first task, an optimal path was found and modules that construct this path were frozen. Then the network started the same process for the next task. Below is a screenshot of the video showing the path that was selected for the first task.

![Imgur](https://i.imgur.com/N1KIwg0.png)

### Result
|     | Control (c=0) | Experiment (c=0.152) |
|-----|---------------|----------------------|
| ACC | 0.315795      | 0.974935             |
| BWT | -0.74009      | 0                    |
| FWT | 0.889139      | 0.857017             |

Navigate to the PathNet notebook for the full result, we will take a look at the test result after training the first task and after the last task. 
```
Training Task 0 started...				Training Task 19 started...
Evaluating on task test sets				Evaluating on task test sets
Test Accuracy on Task Set 0: 0.977400004863739		Test Accuracy on Task Set 0: 0.977400004863739
Test Accuracy on Task Set 1: 0.9394999742507935		Test Accuracy on Task Set 1: 0.9718000292778015
Test Accuracy on Task Set 2: 0.9243999719619751		Test Accuracy on Task Set 2: 0.9775999784469604
Test Accuracy on Task Set 3: 0.6528000235557556		Test Accuracy on Task Set 3: 0.9761000275611877
Test Accuracy on Task Set 4: 0.5504000186920166		Test Accuracy on Task Set 4: 0.9609000086784363
Test Accuracy on Task Set 5: 0.39660000801086426	Test Accuracy on Task Set 5: 0.9782000184059143
Test Accuracy on Task Set 6: 0.30720001459121704	Test Accuracy on Task Set 6: 0.9785000085830688
Test Accuracy on Task Set 7: 0.23399999737739563	Test Accuracy on Task Set 7: 0.9761000275611877
Test Accuracy on Task Set 8: 0.18469999730587006	Test Accuracy on Task Set 8: 0.9753000140190125
Test Accuracy on Task Set 9: 0.14669999480247498	Test Accuracy on Task Set 9: 0.9754999876022339
Test Accuracy on Task Set 10: 0.14390000700950623	Test Accuracy on Task Set 10: 0.9800000190734863
Test Accuracy on Task Set 11: 0.13379999995231628	Test Accuracy on Task Set 11: 0.9761999845504761
Test Accuracy on Task Set 12: 0.13420000672340393	Test Accuracy on Task Set 12: 0.9703999757766724
Test Accuracy on Task Set 13: 0.1535000056028366	Test Accuracy on Task Set 13: 0.9767000079154968
Test Accuracy on Task Set 14: 0.19439999759197235	Test Accuracy on Task Set 14: 0.965499997138977
Test Accuracy on Task Set 15: 0.2085999995470047	Test Accuracy on Task Set 15: 0.9732000231742859
Test Accuracy on Task Set 16: 0.2728999853134155	Test Accuracy on Task Set 16: 0.9731000065803528
Test Accuracy on Task Set 17: 0.27410000562667847	Test Accuracy on Task Set 17: 0.9581000208854675
Test Accuracy on Task Set 18: 0.29980000853538513	Test Accuracy on Task Set 18: 0.974399983882904
Test Accuracy on Task Set 19: 0.2955999970436096	Test Accuracy on Task Set 19: 0.9801999926567078
Average Test Accuracy: 0.37122500091791155		Average Test Accuracy: 0.9737600058317184
Task 0 done.						Task 19 done.
```
From the test result provided above, we can see that the model does not “forget” what it “learned” before as test accuracy for task 0’s test data remains the same after training the first task and the last task. This behavior is expected as described in the above section, the same test data is being fed into the same path every time, since the modules that construct the path are frozen, their parameters do not change during later training, therefore, the network will produce the same test accuracy every time.

Additionally, it is important to point out the average test accuracy was improved significantly, from 37.12% after training only the first task to 97.38% after training all the task data. The figure below is a graph that presents the improvement of the average test accuracy. 

![Imgur](https://i.imgur.com/PuJ8e60.png)

Graph of accuracy on first task and average accuracy on tasks seen as a function of # of tasks trained for both control and experiment:

![Imgur](https://i.imgur.com/GPRg3hZ.png)

### Conculsion
PathNet performed really well in the scope of our project, it was able to prevent catastrophic forgetting and achieve high average test accuracy after training all tasks. However, there is a major disadvantage in the PathNet we constructed. As stated above, each test dataset is being fed into a specific path which is reserved for the task that the test data belongs to. Rather than determining the origin of the test data by its nature, our network is being told where the test data came from by an argument that was passed into the network with the test data when making predictions. Even though we had such information available to us in this project, this is not a usual case for most machine learning tasks out there. Therefore, we constructed another continuous learning model that does not need to know where the test data came from before testing, the model is called Synaptic Intelligence, detailed information is provided below.

## Synaptic Intelligence
### Principle Feature
Synaptic Intelligence is a regularization strategy for continuous learning. It defines a regularization term which is called the quadratic surrogate loss:

![Imgur](https://i.imgur.com/I4JS0ay.png)

This surrogate loss has an *Omega* term, which is the per parameter regularization strength and a *c* term which is the lambda for the regularization. *c* represents the trade off between new and old learnings. If the path integral (little omega defined below) were perfectly calculated, a c=1 would mean an equal weighting of new and old learnings. Omega is defined as:

![Imgur](https://i.imgur.com/yMr0q3p.png)

little omega is the parameter specific contribution to the change in total loss. It is the path integral of the gradient vector field along the parameter trajectory from the initial point in time to the final point in time. From the paper, little omega can be approximated as the the running sum of the product of the gradient with the with the parameter update.

### Result
|     | Control (c=0) | Experiment (c=0.152) |
|-----|---------------|----------------------|
| ACC | 0.43528       | 0.58579              |
| BWT | -0.59472      | -0.15066             |
| FWT | 0.851533      | 0.5347               |

Graph of accuracy on first task and average accuracy on tasks seen as a function of # of tasks trained for both control and experiment:

![Imgur](https://i.imgur.com/MuhRV4T.png)
### Conclusion
Synaptic Intelligence, like other regularization strategies, trades accuracy on new tasks for accuracy on old tasks. Comparing the control and experiment data, we can see that the control has higher forward transfer, but the experiment has higher backwards transfer. Average accuracy is still improved using Synaptic Intelligence (58.5% vs 43.5%).
## Reference
[1]. Github repo provided the data: https://github.com/facebookresearch/GradientEpisodicMemory <br>
[2]. Youtube video for PathNet visualization: https://youtu.be/7fHN5zA7R3o <br>
[3]. Fernando, Chrisantha, et al. “PathNet: Evolution Channels Gradient Descent in Super Neural Networks.” ArXiv.org, 30 Jan. 2017, arxiv.org/abs/1701.08734. <br>
[4]. Gitbub repo provided the code for PathNet: https://github.com/kimhc6028/pathnet-pytorch <br>
[5]. Gitbub repo provided the code for Synaptic Intelligence: https://github.com/GMvandeVen/continual-learning <br>
[6]. Lopez-Paz, David, and Marc'Aurelio Ranzato. “Gradient Episodic Memory for Continual Learning.” ArXiv.org, 4 Nov. 2017, arxiv.org/abs/1706.08840. <br>
