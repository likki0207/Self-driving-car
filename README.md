# Self Driving Car:
The Artificial Intelligence for this self driving car will be written in pytorch. You can just follow this instruction book to install it on WINDOWS as well to run this project.
pytorch is an optimized tensor library for deep learning using GPU and CPUs

Run the following commands on Anaconda prompt:
* conda install -c peterjc123 pytorch-cpu
* conda install -c conda-forge kivy

## Here we are going to implement deep Q-learning model:

![a](https://user-images.githubusercontent.com/68856803/89107341-05ea4c00-d44e-11ea-81c2-4775a37fab56.png)
![b](https://user-images.githubusercontent.com/68856803/89107397-7c874980-d44e-11ea-815b-8049d4793f49.png)

## Before training:
![before_training](https://user-images.githubusercontent.com/68856803/89097444-95670f00-d3fc-11ea-8e18-943b2c19a574.gif)

## After training:
![After_training](https://user-images.githubusercontent.com/68856803/89101105-bd656b00-d41a-11ea-9191-9cffc711cc1e.gif)





### In this project there are two python files and one kivy file(Kivy is a free and open source Python library for developing mobile apps and other multitouch application software with a natural user interface (NUI)):
(a) ai.py-> Here we will build the AI, this will be the deep Q-learning model.  This will contain the brain of the car

(b) map.py-> In this we will build the environmenet containing the map, the car and all the other features required

(c) car.kv-> This contains the objects like 
*size,shape,angle of the car
*sensors of the cars which are represented by ball1, ball2, ball3



## Execution:
First set the temperature parameter,T to be 0 and then run the map.py file. It means that we have not activated the brain of the car i.e.,AI. Hence the motion of the car would be random.
Now set the  temperature parameter,T to be 100 and then run the map.py file. It means that the the brain of the car has been activated and the car will start moving from the source point to the destination.(source is the top left corner and the destination is the bottom right corner)
T is basically a hyperparameter of neural networks used to control the randomness of predictions by scaling the logits before applying softmax. For example, in TensorFlow’s Magenta of LSTMs, temperature represents how much to divide the logits by before computing the softmax. Higher the T,higher is the probability for the event to take place.
