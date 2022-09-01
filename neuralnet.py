# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques for the fall 2021 semester
# Modified by Kaiwen Hong for the Spring 2022 semester

"""
This is the main entry point for MP2. You should only modify code
within this file and neuralnet.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @param l(x,y) an () tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 2 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        # construct network architecture - use Linear and Sequential objects
        super(NeuralNet, self).__init__()
        
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.model = nn.Sequential( nn.Linear(in_size, 32), nn.ReLU(), nn.Linear(32, out_size) )
        self.optimizer = optim.SGD(self.model.parameters(), lrate, momentum=0.5) # additionally, you can inititalize an optimizer object to use in step() function

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # This can be done simply by calling your Sequential object definite in __init__()
        return self.model(x)
        # raise NotImplementedError("You need to write this part!")
        # return torch.ones(x.shape[0], 1)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """        
        yhat  = self.forward(x)
        loss_value = self.loss_fn(yhat, y)
        loss_value.backward()

        self.optimizer.step()
        self.optimizer.zero_grad() # call zero_grad() on your optimizer in order to clear the gradient buffer

        return loss_value.detach().cpu().numpy() # be sure to return loss_value.item() or loss_value.detach().cpu().numpy()

def fit(train_set, train_labels, dev_set, n_iter, batch_size=100):
    """ Fit a neural net. Use the full batch size.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of epoches of training
    @param batch_size: size of each batch to train on. (default 100)

    NOTE: This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # construct a NeuralNet object and iteratively call the neural net's step() to train the network
    # this should be done by feeding in batches of data determined by batch size

    N = train_set.shape[0]
    in_size_data = train_set.shape[1]
    #return net: a NeuralNet Object
    net = NeuralNet(lrate = .0001, loss_fn= nn.CrossEntropyLoss(), in_size = in_size_data, out_size=2)
    
    norm_train_set = (train_set - train_set.mean()) / train_set.std()
    losses = [] # array of total loss at the beginning and after each iterations (len(losses) == n_iter)
    
    for i in range(n_iter) : # each iteration/epoch
        # need to create the batch to feed to net
        batch_start = (i * batch_size) % N
        batch_end = ((i+1) * batch_size) % N
        x = norm_train_set[batch_start : batch_end]
        y = train_labels[batch_start: batch_end]
        losses.append(net.step(x,y))

    norm_dev_set = (dev_set - train_set.mean()) / train_set.std()
    net_return = net(norm_dev_set).detach().cpu().numpy()


    y_hats = [] # np array of binary labels for dev set
    for i in range(len(net_return)) :
        y_hats.append(np.argmax(net_return[i])) 
    y_hats = np.array(y_hats) # converting to np array from list

    return losses, y_hats, net
