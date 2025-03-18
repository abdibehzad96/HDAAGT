
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import os
import math

from utilz.utils import *
from models.encdec import *
#from socialconv.model import highwayNet
#from torchviz import make_dot

if __name__ == '__main__':

    # Load the data
    cwd = os.getcwd()
    csvpath = os.path.join(cwd,'Processed','Trj20240510T1529.csv')
    ct = datetime.datetime.now().strftime("%m-%d-%H-%M")


    # # Hyperparameters
    input_size, hidden_size, num_layers, output_size, epochs, patience_limit,  learning_rate = 10, 512, 1, 2, 300, 20, 0.01
    sl, sw , shift, batch_size = 20, 5, 20, 256
    future = 20
    # sw is sliding window that slides in the whole dataset by sw steps to create a new sequence
    LR = [learning_rate] #, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]
    HS = [hidden_size] #, 32, 64, 128, 256, 512, 1024]
    NL = [num_layers] #, 2, 3, 4]
    BS = [batch_size] #, 32, 64, 128, 256, 512, 1024]


    generate_data = True #  Add class by loading the CSV file
    load_data = not generate_data      # load the class from the saved file
    Train = True # It's for training the model with the prepared data
    test_in_epoch = True # It's for testing the model in each epoch
    model_from_scratch = True # It's for creating model from scratch
    load_the_model = not model_from_scratch # It's for loading model
    Seed = False # It's for setting the seed for the random number generator

    # check the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    savelog(f"Using {device} device", ct)


    # Create datasets
    if generate_data:
        df = loadcsv(csvpath)
        datasettr, datasettst, datasetval= def_class(df, sw, sl, shift, future, input_size, output_size, 
                                                     tr2img= False, ct= ct, imgsize=1024, device =device, normalize = False, seed= Seed)
        train_loader, test_loader, val_loader = prep_data(datasettr, datasettst, datasetval, batch_size)

    if load_data:
        train_loader = torch.load(os.path.join(os.getcwd(),'Pickled','train_loader.pt'))
        test_loader = torch.load(os.path.join(os.getcwd(),'Pickled','test_loader.pt'))
        val_loader = torch.load(os.path.join(os.getcwd(),'Pickled','val_loader.pt'))
        savelog('dataset loaded', ct)
    savelog(f"Train Dataset length : {len(datasettr)}, Test Dataset length : {len(datasettst)}, Val Dataset length : {len(datasetval)}", ct)

    if load_the_model:
        savelog("Loading model from the checkpoint", ct)
        mpath = os.path.join(os.getcwd(),'Processed','checkpoint.pth.tar')
        model, optimizer, criterion = load_model(mpath, input_size + output_size, hidden_size, output_size, future, num_layers, ct)
    else:
        savelog("Creating model from scratch",ct)



    #Training loop
    Hyperloss = []
    criterion = nn.MSELoss()
    if Train:
        savelog("Starting training phase", ct)
        for learning_rate in LR:
            for hidden_size in HS:
                for num_layers in NL:
                    for batch_size in BS:
                        model = ENCDEC(input_size + output_size, hidden_size, output_size, future, num_layers)
                        if torch.cuda.device_count() > 1:
                            savelog(f"Using {torch.cuda.device_count()} GPUs", ct)
                            model = nn.DataParallel(model)
                        model.to(device)
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate )
                        log = f"Learning rate: {learning_rate}, Hidden size: {hidden_size}, Number of layers: {num_layers}, Batch size: {batch_size}"
                        savelog(log, ct)
                        Best_model, predicteddata, test_loss, epoch_losses, target_gt, ego_gt, Max_Acc\
                            = train_model(model, optimizer, criterion, train_loader, test_loader, sl, future, output_size, \
                                          device, epochs,learning_rate, ct, patience_limit, batch_size, test_in_epoch = True)
                        Hyperloss += [epoch_losses]
                        code = f"LR{int(learning_rate*1000)}_HS{hidden_size}_NL{num_layers}_BS{batch_size} {ct}"
                        savelog(f"{code} with Max Acc of {Max_Acc}", f"summary {ct}")
                        if Max_Acc > 0.2:
                            savelog(f"Saving result of {code} with Max Acc of {Max_Acc}", ct)
                            #save output data as a pickle file
                            torch.save(Best_model, os.path.join(os.getcwd(),'Processed', code + 'Bestmodel.pth'))
                            torch.save(predicteddata, os.path.join(os.getcwd(),'Pickled', code + 'Predicteddata.pt'))
                            torch.save(test_loss, os.path.join(os.getcwd(),'Pickled', code + 'test_losses.pt'))
                            torch.save(epoch_losses, os.path.join(os.getcwd(),'Pickled', code + 'epoch_losses.pt'))
                            torch.save([sl, input_size, hidden_size, num_layers, output_size, epochs, batch_size,shift, sw, future],\
                                        os.path.join(os.getcwd(),'Pickled', code + 'model_parameters.pt'))
                            torch.save(target_gt, os.path.join(os.getcwd(),'Pickled', code + 'target_gt.pt'))
                            torch.save(ego_gt, os.path.join(os.getcwd(),'Pickled', code + 'ego_gt.pt'))
                            torch.save(Hyperloss, os.path.join(os.getcwd(),'Pickled', code + 'Hyperloss.pt'))
        # predicteddata, test_loss, epoch_losses, trainy, trainx, groundtruthx, groundtruthy = train_model(model, optimizer, criterion, train_loader, test_loader, sl, output_size, device, epochs, learning_rate, test_in_epoch = False)
        savelog(f"Training finished for {ct}!", ct)