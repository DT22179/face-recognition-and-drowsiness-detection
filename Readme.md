The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms.
Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market
is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality
learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live
class scenario is yet an open-end challenge.In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention. Digital classrooms are conducted via video telephony software program (exZoom) where it’s not possible for medium scale class (25-50) to see all students and access the
mood. Because of this drawback, students are not focusing on content due to lack of surveillance. While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts
which can be analysed using deep learning algorithms. Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no
longer in the teacher’s brain rather translated in numbers that can be analysed and tracked.


## Problem Statement
### Face Recognition -This is a few shot learning live face attendance systems. The model should be able to real-time identify attending students in the live class based on few images of students.

### Drowsiness detector - This will detect facial landmarks and extract the eye regions. The algorithm should be able to identify students who are not attentive and drowsing in the class.


## Face - Recognition Library - 
1. Recognize and manipulate faces from Python or from the command line with the world's simplest face recognition library.

2. Built using dlib's state-of-the-art face recognition built with deep learning. The model has an accuracy of 99.38% on the Labeled Faces in the Wild benchmark.

3. This also provides a simple face_recognition command line tool that lets you do face recognition on a folder of images from the command line!


## Drowsiness Detection Using Transfer Learning - 
The reuse of a pre-trained model on a new problem is known as transfer learning in machine learning. A machine uses the knowledge learned from a prior assignment to increase prediction about a new task in transfer learning. You could, for example, use the information gained during training to distinguish beverages when training a classifier to predict whether an image contains cuisine.

The knowledge of an already trained machine learning model is transferred to a different but closely linked problem throughout transfer learning. For example, if you trained a simple classifier to predict whether an image contains a backpack, you could use the model’s training knowledge to identify other objects such as sunglasses.

![image](https://user-images.githubusercontent.com/44960814/166678520-8102f427-1d2b-46bb-bd9c-470653c91ab0.png)

Transfer learning offers a number of advantages, the most important of which are reduced training time, improved neural network performance (in most circumstances), and the absence of a large amount of data.

To train a neural model from scratch, a lot of data is typically needed, but access to that data isn’t always possible – this is when transfer learning comes in handy.

![image](https://user-images.githubusercontent.com/44960814/166678680-ca4bbedd-f768-4e88-9ed8-78b1e1982ddb.png)


Because the model has already been pre-trained, a good machine learning model can be generated with fairly little training data using transfer learning. This is especially useful in natural language processing, where huge labelled datasets require a lot of expert knowledge. Additionally, training time is decreased because building a deep neural network from the start of a complex task can take days or even weeks.

## Mobilenet Architecture

MobileNet is an efficient and portable CNN architecture that is used in real world applications. MobileNets primarily use depthwise seperable convolutions in place of the standard convolutions used in earlier architectures to build lighter models.MobileNets introduce two new global hyperparameters(width multiplier and resolution multiplier) that allow model developers to trade off latency or accuracy for speed and low size depending on their requirements.

MobileNets are built on depthwise seperable convolution layers.Each depthwise seperable convolution layer consists of a depthwise convolution and a pointwise convolution.Counting depthwise and pointwise convolutions as seperate layers, a MobileNet has 28 layers.A standard MobileNet has 4.2 million parameters which can be further reduced by tuning the width multiplier hyperparameter appropriately.
The size of the input image is 224 × 224 × 3.

![image](https://user-images.githubusercontent.com/44960814/166679154-3aa526ed-cf28-419e-b517-2faae76eea9f.png)


## Model Testing & Accuracy - 

![image](https://user-images.githubusercontent.com/44960814/166679368-715a4048-8f59-49d3-84bf-1aa9524cbf0f.png)

### Training Accuracy - 98.91%
### Validation Accuracy - 97.40%
