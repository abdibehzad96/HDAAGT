#This code will plot the data after loading it into train and test loaders

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import os
import math
import random
# from sklearn.metrics import mean_squared_error, r2_scor

# Define a custom dataset class
class Traj():
    def __init__(self, data, sw ,sl, shift):
        self.data = data
        self.ln = len(self.data[0])
        self.slidlen = self.nslides(sw ,sl, shift) if self.nslides(sw ,sl, shift) > 0 else 0
    
    def nslides(self, sw ,sl, shift):
        return (self.ln - sl - shift) // sw + 1

    def slid(self,i, sw ,sl, shift): # Sliding window = 2, sequence length = 10 and shift = 1
        inpt = [row[i*sw: i*sw + sl] for row in self.data]
        #trgt = [row[i*sw +sl: i*sw + 2*sl] for row in self.data[:2]]
        trgt = [row[i*sw + shift: i*sw + sl + shift] for row in self.data[:2]]
        #self.inpt.append(input)
        #self.targ.append(target)
        return inpt, trgt


class TrajDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
# Load the data
#data = torch.load(os.path.join(os.getcwd(),'Pickled','Predicteddata.pt'))
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
if __name__ == '__main__':
    # Load the data
    cwd = os.getcwd()
    csvpath = os.path.join(cwd,'Processed','Trj20231207T1831.csv')
    # # Hyperparameters
    
    input_size = 16
    hidden_size = 1024
    num_layers = 4
    output_size = 2
    epochs = 50
    learning_rate = 0.01
    batch_size = 1024  # Since each batch is a single set
    sequence_length = 100
    sw = 50
    sl = sequence_length
    shift = 10
    df = []
    q= 0

    prep_data = True # It's for preparing data
    add_class = True # It's for loading model


    with open(csvpath, 'r',newline='') as file:
        for line in file:
            row = line.strip().split(',')
            rowf = [float(element) for element in row]
            rowf = [0 if math.isnan(x) else x for x in rowf]
            df.append(rowf)

    #load brokentraj.csv file
    csvpath = os.path.join(cwd,'brokentraj.csv')
    with open(csvpath, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            blcklist = list(map(int,row))
    print('blacklist:', blcklist)

    # Create datasets
    if add_class:

        if torch.cuda.is_available():
            print("Using GPU")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputD = torch.empty(0, sl, input_size, device=device)
        targetD = torch.empty(0, sl, output_size, device=device)
        for i in range(len(df)//16): #len(df)//16
            if i not in blcklist:
                data = Traj(df[i*16:(i+1)*16], sw, sl, shift)
                for j in range(data.slidlen):
                    inpt, trgt = data.slid(j, sw, sl, shift)
                    inpt = torch.tensor(inpt, dtype=torch.float32).transpose(0,1).unsqueeze(0).to(device)
                    trgt = torch.tensor(trgt, dtype=torch.float32).transpose(0,1).unsqueeze(0).to(device)
                    inputD = torch.cat((inputD, inpt), dim=0)
                    targetD = torch.cat((targetD, trgt), dim=0)
        dataset = TrajDataset(inputD, targetD)
        #torch.save(dataset, os.path.join(os.getcwd(),'Pickled','datasetTorch.pt'))
    if False:
        dataset = torch.load(os.path.join(os.getcwd(),'Pickled','datasetTorch.pt'))
        print('dataset loaded')
        

    if prep_data:
        # Randomly split the data into train, test and validation sets
        train_size = int(0.95 * len(dataset))
        test_size = int(0.05 * len(dataset))
        val_size = len(dataset) - train_size - test_size

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        val_indices = indices[train_size + test_size:]

        # Define samplers for training and test sets
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        # # Create data loaders

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler)

n = 0

for x , y in train_loader:
    for i in range(len(x)):
        plt.plot(x[i,:,0].cpu().numpy(), x[i,:,1].cpu().numpy(), 'b')
        n += 1
print('n:', n)
plt.show()


n = 0
plt.figure()
for x , y in test_loader:
    for i in range(len(x)):
        plt.plot(x[i,:,0].cpu().numpy(), x[i,:,1].cpu().numpy(), 'b')
        n += 1
print('n:', n)
