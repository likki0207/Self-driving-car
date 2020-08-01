# Self Driving Car
The Artificial Intelligence for this self driving car will be written in pytorch. You can just follow this instruction book to install it on WINDOWS as well to run this project.
pytorch is an optimized tensor library for deep learning using GPU and CPUs

Run the following commands on Anaconda prompt:
* conda install -c peterjc123 pytorch-cpu
* conda install -c conda-forge kivy

## In this project there are two python files and one kivy file
(a) ai.py-> Here we will build the AI, this will be the deep Q-learning model.  This will contain the brain of the car

(b) map.py-> In this we will build the environmenet containing the map, the car and all the other features required

(c) car.kv-> This contains the objects like 
*size,shape,angle of the car
*sensors of the cars which are represented by ball1, ball2, ball3

## Before training
![before_training](https://user-images.githubusercontent.com/68856803/89097444-95670f00-d3fc-11ea-8e18-943b2c19a574.gif)

## After training
