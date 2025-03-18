
# export cuda_visible_devices=2,3

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import os
import math
from torch.nn.parallel import DataParallel
from utilz.utils import *
from models.train_test import *
from models.GGAT import *

class args:
    def __init__(self):
        self.arg = {}
    def add_argument(self, name, type, value, help):
        if name.startswith('--'):
            name = name[2:]
        self.arg[name] = {
            'value': value,
            'type': type,
            'help': help
        }
    def get(self, name):
        return self.arg[name]['value']
    def help(self, name):
        print(name, self.arg[name]['help'])
        return self.arg[name]['help']


if __name__ == '__main__':
    # Load the data
    cwd = os.getcwd()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    # # Hyperparameters
    hidden_size, num_layersGAT, num_layersGRU, expansion,  n_heads = 128, 1, 6, 2, 4

    learning_rate, schd_stepzise, gamma, epochs, batch_size, patience_limit, clip= 4e-4, 25, 0.15, 350, 8, 50, 0.25

    Headers = ['Frame', 'ID', 'BBx', 'BBy','W', 'L' , 'Cls','Tr1', 'Tr2', 'Tr3', 'Tr4', 'Zone', 'Xreal', 'Yreal']
    Columns_to_keep = [11,2,3,7,8,9,10] #['BBx', 'BBy','Tr1', 'Tr2', 'Tr3', 'Tr4']
    #  ['Vx', 'Vy', 'heading',xc,yc, Rc,Rc, SinX,CosX,SinY,CosY, Sin2X,Cos2X, Sin2Y, Cos2Y, Sin3X, Cos3X, Sin3Y, Cos3Y, Sin4X, Cos4X, Sin4Y, Cos4Y]
    Columns_to_Predict = [0,1] #['BBx', 'BBy','Xreal', 'Yreal','Vx', 'Vy']
    xyid = [1, 2] # the index of x and y  of the Columns_to_Keep in the columns for speed calculation
    TrfL_Columns = [7,8,9,10]
    NFeatures = len(Columns_to_keep)
    input_size = NFeatures +9 + 8
    output_size = len(Columns_to_Predict)
    Nusers, NZones = 32, 10
    Nnodes, NTrfL, sl, future, sw, sn  = Nusers, 4, 30, 30, 2, 10
    Centre = [512,512]
    dssc, dstar = 2,2
    sos = torch.cat((torch.tensor([10,1016,1016,7,7,7,7]), torch.zeros(17))).repeat(1,1,32, 1).to(device)
    eos = torch.cat((torch.tensor([11,1020,1020,8,8,8,8]), torch.zeros(17))).repeat(1,1,32, 1).to(device)
    # sos_target = torch.cat((torch.tensor([1,1]), torch.zeros(17))).repeat(1,1,32, 1).to(device)

    generate_data = False #  Add class by loading the CSV file
    # generate_data = True #  Add class by loading the CSV file
    loadData = not generate_data      # load the class from the saved file
    Train = True # It's for training the model with the prepared data
    test_in_epoch = True # It's for testing the model in each epoch
    model_from_scratch = True # It's for creating model from scratch
    load_the_model = not model_from_scratch # It's for loading model
    Seed = loadData # If true, it will use the predefined seed to load the indices
    only_test = False # It's for testing the model only
    concat = False
    
    csvpath ='/home/abdikhab/New_Idea_Traj_Pred/RawData/TrfZonXYCam.csv'
    Zoneconf_path ='/home/abdikhab/New_Idea_Traj_Pred/utilz/ZoneConf.yaml'
    dataset_path = os.path.join(cwd, 'Pickled')
    ct = datetime.datetime.now().strftime("%m-%d-%H-%M")
    print(f"Total  # of GPUs {torch.cuda.device_count()}")
    # spread the model to multiple GPUs
    print(f"Using {device} device")
    

    # Use argparse to get the parameters
    parser = argparse.ArgumentParser(description='Trajectory Prediction')
    # parser = args()
    parser.add_argument('--ct', type=str, default=ct, help='Current time')
    parser.add_argument('--Zonepath', type=str, default=Zoneconf_path, help='Zone Conf path')
    parser.add_argument('--Nfeatures', type=int, default=NFeatures, help='Number of features')
    parser.add_argument('--Nnodes', type=int, default=Nnodes, help='Number of nodes')
    parser.add_argument('--NZones', type=int, default=NZones, help='Number of zones')
    parser.add_argument('--NTrfL', type=int, default=NTrfL, help='Number of traffic lights')
    parser.add_argument('--sl', type=int, default=sl, help='Sequence length')
    parser.add_argument('--future', type=int, default=future, help='Future length')
    parser.add_argument('--sw', type=int, default=sw, help='Sliding window')
    parser.add_argument('--sn', type=int, default=sn, help='Sliding number')
    parser.add_argument('--Columns_to_keep', type=list, default=Columns_to_keep, help='Columns to keep')
    parser.add_argument('--Columns_to_Predict', type=list, default=Columns_to_Predict, help='Columns to predict')
    parser.add_argument('--TrfL_Columns', type=list, default=TrfL_Columns, help='Traffic light columns')
    parser.add_argument('--Nusers', type=int, default=Nusers, help='Number of maneuvers')
    parser.add_argument('--sos', type=int, default=sos, help='Start of sequence')
    parser.add_argument('--eos', type=int, default=eos, help='End of sequence')
    parser.add_argument('--xyidx', type=list, default=xyid, help='X and Y index')
    parser.add_argument('--Centre', type=list, default=Centre, help='Centre')

    parser.add_argument('--input_size', type=int, default=input_size, help='Input size')
    parser.add_argument('--hidden_size', type=int, default=hidden_size, help='Hidden size')
    parser.add_argument('--num_layersGAT', type=int, default=num_layersGAT, help='Number of layers')
    parser.add_argument('--num_layersGRU', type=int, default=num_layersGRU, help='Number of layers')
    parser.add_argument('--output_size', type=int, default=output_size, help='Output size')
    parser.add_argument('--n_heads', type=int, default=n_heads, help='Number of heads')
    parser.add_argument('--concat', type=bool, default=concat, help='Concat')
    parser.add_argument('--dropout', type=float, default=0.001, help='Dropout')
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2, help='Leaky relu slope')
    parser.add_argument('--expansion', type=int, default=expansion, help='Expantion')


    parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size')
    parser.add_argument('--patience_limit', type=int, default=patience_limit, help='Patience limit')
    parser.add_argument('--schd_stepzise', type=int, default=schd_stepzise, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=gamma, help='Scheduler Gamma')


    parser.add_argument('--only_test', type=bool, default=only_test, help='Only test')
    parser.add_argument('--generate_data', type=bool, default=generate_data, help='Generate data')
    parser.add_argument('--loadData', type=bool, default=loadData, help='Load data')
    parser.add_argument('--Train', type=bool, default=Train, help='Train')
    parser.add_argument('--test_in_epoch', type=bool, default=test_in_epoch, help='Test in epoch')
    parser.add_argument('--model_from_scratch', type=bool, default=model_from_scratch, help='Model from scratch')
    parser.add_argument('--load_the_model', type=bool, default=load_the_model, help='Load the model')
    parser.add_argument('--Seed', type=bool, default=Seed, help='Seed')
    parser.add_argument('--device', type=str, default=device, help='device')
    parser.add_argument('--dssc', type=int, default=dssc, help='downsample Scene')
    parser.add_argument('--dstar', type=int, default=dstar, help='downsample target')
    
    args = parser.parse_args()
    LR = [args.learning_rate] #, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]
    # LR = [1e-2]
    HS = [args.hidden_size] #, 32, 64, 128, 256, 512, 1024]
    NL = [num_layersGAT] #, 2, 3, 4]
    BS = [args.batch_size] #, 32, 64, 128, 256, 512, 1024]

    for arg in vars(args):
        savelog(f"{arg}, {getattr(args, arg)}", ct)
    # Create datasets
    Scenetr = Scenes(args, 0)
    Scenetst = Scenes(args, 1)
    Sceneval = Scenes(args, 2)
    savelog(f'Starting at:  {datetime.datetime.now().strftime("%H-%M-%S")}', ct)
    if loadData:
        savelog("Loading data ...", ct)
        Scenetr.load_class(dataset_path, cmnt = 'Train')
        Scenetst.load_class(dataset_path, cmnt = 'Test')
        Sceneval.load_class(dataset_path, cmnt = 'Validation')
        datamax, datamin = Scenetr.maxval, Scenetr.minval
        print("Data loaded with max and min of", datamax,"\n", datamin)
    else:
        savelog("Loading CSV file ...", ct)
        df = loadcsv(csvpath, Headers)
        Scenetr, Scenetst, Sceneval, datamax, datamin = def_class(Scenetr, Scenetst, Sceneval, df, args)
        # Scenetr.augmentation(4)
        Scenetr.addnoise(10, 4, 0.6)
        Scenetr.save_class(dataset_path, cmnt = 'Train')
        Scenetst.save_class(dataset_path, cmnt = 'Test')
        Sceneval.save_class(dataset_path, cmnt = 'Validation')
        print("Data saved with max and min of", datamax,"\n", datamin)
    savelog(f'Done at: {datetime.datetime.now().strftime("%H-%M-%S")}', ct)
    savelog(f"Train size: {len(Scenetr)}, Test size: {len(Scenetst)}, Validation size: {len(Sceneval)}", ct)
    print("Augmenting the data ...")
    
    train_loader, test_loader, val_loader = prep_data(Scenetr, Scenetst, Sceneval, args.batch_size)
    
    
    # model= nn.DataParallel(model).cuda()
    
    if load_the_model:
        savelog("Loading model from the checkpoint", ct)
        mpath = os.path.join(cwd,'Processed','checkpoint.pth.tar')
        model = model.load_state_dict(torch.load(mpath))
    else:
        savelog("Creating model from scratch",ct)

    
    # criterion = nn.CrossEntropyLoss()

    ZoneConf = Zoneconf()
    Hyperloss = []
    if Train:
        print("Starting training phase")
        for learning_rate in LR:
            for hidden_size in HS:
                for num_layers in NL:
                    for batch_size in BS:
                        model = GGAT(args)
                        # criterion = nn.MSELoss()
                        criterion = nn.CrossEntropyLoss()
                        savelog(f"The number of learnable parameter is {count_parameters(model=model)} !", ct)
                        # model = DataParallel(model, device_ids=[3,2,1])  # Wrap your model in DataParallel
                        model.to(device)
                        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                        print(f"Learning rate: {learning_rate}, Hidden size: {hidden_size}, Number of layers: {num_layers}, Batch size: {batch_size}")
                        Best_Model, predicteddata,Targetdata, test_loss, epoch_losses, Max_Acc =\
                            train_model(model, optimizer, criterion, train_loader, test_loader, clip, args, datamax, datamin, ZoneConf)
                        Hyperloss += [epoch_losses]
                        code = f"LR{int(learning_rate*100000)}_HS{hidden_size}_NL{num_layers}_BS{batch_size} {ct}"
                        savelog(f"{code} with Max Acc of {Max_Acc}", f"summary {ct}")
                        if Max_Acc > 0.001:
                            savelog(f"Saving result of {code} with Max Acc of {Max_Acc}", ct)
                            #save output data as a pickle file
                            torch.save(Best_Model, os.path.join(cwd,'Processed', code + 'Bestmodel.pth'))
                            torch.save(predicteddata, os.path.join(cwd,'Pickled', code + 'Predicteddata.pt'))
                            torch.save(test_loss, os.path.join(cwd,'Pickled', code + 'test_losses.pt'))
                            torch.save(epoch_losses, os.path.join(cwd,'Pickled', code + 'epoch_losses.pt'))
                            torch.save(Hyperloss, os.path.join(cwd,'Pickled', code + 'Hyperloss.pt'))
        # predicteddata, test_loss, epoch_losses, trainy, trainx, groundtruthx, groundtruthy = train_model(model, optimizer, criterion, train_loader, test_loader, sl, output_size, device, epochs, learning_rate, test_in_epoch = False)
        savelog(f"Training finished for {ct}!", ct)