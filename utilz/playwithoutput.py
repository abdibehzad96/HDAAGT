# This file is used to play with the output of the model
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
Predicteddata  = torch.load(os.path.join(os.getcwd(),'Pickled','Predicteddata.pt')).to('cpu')
Predicteddata = Predicteddata.numpy()
groundtruth = torch.load(os.path.join(os.getcwd(),'Pickled','groundtruth.pt')).to('cpu')
groundtruth = groundtruth.numpy()
loss_test = torch.load(os.path.join(os.getcwd(),'Pickled','loss_test.pt'))
epoch_losses = torch.load(os.path.join(os.getcwd(),'Pickled','epoch_losses.pt'))



# Plotting the loss
plt.plot(epoch_losses)
plt.title('Epoch Losses during batch Training')
plt.show()
# Plotting the predicted data
for i , data in enumerate(Predicteddata):
    if i < 100:
        plt.plot(data[:,0], data[:,1], 'r')
plt.title('Predicted Data')
plt.show()



# Plotting the ground truth
for i , data in enumerate(groundtruth):
    if i < 100:
        plt.plot(data[:, 0], data[:,1], 'b')
plt.title('Ground Truth')
plt.show()


# Plotting the predicted data and ground truth
for i , data in enumerate(Predicteddata):
    if i < 100:
        plt.plot(data[:,0], data[:,1], 'r')
        plt.plot(groundtruth[i][:,0], groundtruth[i][:,1], 'b')



