
# export cuda_visible_devices=2,3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import yaml
import datetime
import argparse
# The rest is HDAAGT specific imports
from utilz.utils import *
from train_test import *
from models.HDAAGT import *

if __name__ == '__main__':
    # If the model is run through terminal
    parser = argparse.ArgumentParser(description="Run the main script with a config file.")
    parser.add_argument("-c", "--config", type=str, help="Path to the config.yaml file")
    args = parser.parse_args()
    # Otherwise we use the default directory to the config path
    if args.config is None:
        args.config = "configs/config.yaml"
        print(f"Using the '{args.config}' file")

    with  open(args.config ) as file:
        config = yaml.safe_load(file)
    cwd = os.getcwd()
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    ct = datetime.datetime.now().strftime("%m-%d-%H-%M") # Current time, used for saving the log
    print(f"Total  # of GPUs {torch.cuda.device_count()}")
    print(f"Using {device} device")

    
    ZoneConf = Zoneconf(config['Zoneconf_path']) # This is the setting for zone configuration, you can find the image file at 'data/Zone Configuration.jpg'
    # config non-indipendent parameters
    config['NFeatures']= len(config["Columns_to_keep"]) # This goes to the data preparation process
    config["input_size"] = config["NFeatures"] + 9 + 8 # This goes to the model, we add couple of more features to the input
    config["output_size"] = len(config['xy_indx']) # The number of columns to predict
    config['device'] = device
    config['ct'] = ct
    log_code = f"Learning rate: {config['learning_rate']}, Hidden size: {config['hidden_size']}, Batch size: {config['batch_size']}"
    if config['verbal']:
        for arg in config.items(): # Print the arguments to the log file
            savelog(f"{arg[0]} = {arg[1]}", ct)
    config['sos'] = torch.cat((torch.tensor(config['sos']) , torch.zeros(17)), dim=0).repeat(config['Nnodes'], 1).to(device)
    config['eos'] = torch.cat((torch.tensor(config['eos']), torch.zeros(17)), dim=0).repeat(config['Nnodes'],1).to(device)

    
    # Before Everything, make sure to assert all the parameters are correct, implement later
        # Later


    # Create datasets, we split the datasets to make sure there is no data leakage between training and test samples
    Scenetr = Scenes(config)
    Scenetst = Scenes(config)
    Sceneval = Scenes(config)
    savelog(f'Starting at:  {datetime.datetime.now().strftime("%H-%M-%S")}', ct)
    if not config['generate_data']:
        savelog("Loading data ...", ct)
        Scenetr.load_class('Pickled', cmnt = 'Train')
        Scenetst.load_class('Pickled', cmnt = 'Test')
        Sceneval.load_class('Pickled', cmnt = 'Validation')

    else:
        savelog("Loading CSV file ...", ct)
        Traffic_data = loadcsv(config['detection_path'], config['Headers'])
        Scenetr, Scenetst, Sceneval = Scene_Process(Scenetr, Scenetst, Sceneval, Traffic_data, config)
        Scenetr.addnoise(config['noise_multiply'], config['noise_amp'], config['noise_ratio'])
        Scenetr.save_class('Pickled', cmnt = 'Train')
        Scenetst.save_class('Pickled', cmnt = 'Test')
        Sceneval.save_class('Pickled', cmnt = 'Validation')
    savelog(f"Train size: {len(Scenetr)}, Test size: {len(Scenetst)}, Validation size: {len(Sceneval)}", ct)
    print("Augmenting the data ...")
    
    train_loader, test_loader, val_loader = DataLoader_Scene(Scenetr, Scenetst, Sceneval, config['batch_size'])
    

    model = HDAAGT(config).to(device) # Here we define our model
    savelog("Creating model from scratch",ct)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['schd_stepzise'], gamma=config['gamma'])
        
    if config['Load_Model']:
        savelog("Loading model from the checkpoint", ct)
        model.load_state_dict(torch.load(config['Load_Model_Path'], map_location=device))

    if config['Train']:
        savelog(f"The number of learnable parameter is {count_parameters(model=model)} !", ct)
        print(f"Learning rate: {config['learning_rate']}, Hidden size: {config['hidden_size']}, Batch size: {config['batch_size']}")
        Best_Model, train_loss, trainADE, trainFDE = train_model(model, optimizer, criterion, scheduler, train_loader,test_loader, config)
        savelog(f"{log_code} with Avg loss of {train_loss[-1]}, ADE of {trainADE[-1]}, FDE of {trainFDE[-1]}", f"summary {ct}")
        if train_loss[-1] < 1.5:
            savelog(f"Saving result of {log_code}", ct)
            torch.save(Best_Model.state_dict(), os.path.join(cwd,'Pickled', f'best_trained_model{ct}.pth'))
            torch.save(train_loss, os.path.join(cwd,'Pickled', f'epoch_losses{ct}.pt'))
        savelog(f"Training finished for {ct}!", ct)

    if config['Test']: # If not training, then test the model
        Topk_Selected_words, ADE, FDE = test_model(model, test_loader, config)
        savelog(f"Average Displacement Error: {ADE}, Final Displacement Error: {FDE}", ct)