# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os # Useful when we want to load the model 
import torch
import torch.nn as nn   # for neural networks
import torch.nn.functional as F  # this will be having the loss function required for the softmax
import torch.optim as optim # contains the optimizer
import torch.autograd as autograd # To perform the stochastic gradient descent
from torch.autograd import Variable # We need this variable class to make some conversion from 
                                    # tensors which are like more advanced arrays 

# Creating the architecture of the Neural Network
#In this we will be making the neural network which will be the heart of the AI
# We'll be making a class which will be having 2 functions
class Network(nn.Module):
    
    def __init__(self,input_size,nb_action):
        # input_size is the total number of input neurons
        # nb_action is the total number of actions possible/output neurons(left,right,straight)
        super(Network,self).__init__()
        self.input_size=input_size
        self.nb_action=nb_action
        self.fc1=nn.Linear(input_size, 30) #full connection between the input and the first hidden layer
        self.fc2=nn.Linear(30, nb_action)  #full connection between the output and the first hidden layer
    # It will activate the neurons in the neural network;
    def forward(self, state):
        x=F.relu(self.fc1(state))
        q_values=self.fc2(x)
        return q_values

# Implementing Experience Replay : instead of only considering the current state that is only one
# state at time t, we wil consider more in the past i.e., we will put 100 less transitions into
# what we call the memory and that is why we are going to have a long term memory 

class ReplayMemory(object):
    # capacity will be 100
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        # memory will have the last 100 events and therefore it will be a simple list 
    
    def push(self, event): # This funtion performs 2 tasks.
    # (a) To append a new transition in the memory
    # (b) It will make sure that the memory will always have 100 transitions
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    # This will get the random samples from the memory and this function will return the random samples
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    # This will create a function which will select right action and each time. So basically
    # we will implement the part that will make the care to move in a right way
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100,it is the temperature parameter
        action = probs.multinomial(1)
        return action.data[0,0]
    
    # Training the deep neural network. We will perform forward propagation,back propagation
    # we will get our output, we are going to get the target. We will compare the output of the
    # target to compute the loss error and then we are going to back propagate loss error in 
    # neural network. Using stochastic gradient descent we will update the weights
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    # function to update everything as soon as AI reaches a new state
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    # compute the mean of all the rewards in the reward window
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.) # +1. because denominator should not be 0
    
    #function to save our model 
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")