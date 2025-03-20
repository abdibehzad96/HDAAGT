
# export cuda_visible_devices=2,3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import yaml
import datetime

# The rest is HDAAGT specific imports
from utilz.utils import *
from models.train_test import *
from models.HGAAGT import *

if __name__ == '__main__':
    # Load the data
    cwd = os.getcwd()
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    dataset_path = os.path.join(cwd, 'Pickled')
    ct = datetime.datetime.now().strftime("%m-%d-%H-%M") # Current time, used for saving the log
    print(f"Total  # of GPUs {torch.cuda.device_count()}")
    print(f"Using {device} device")
    with  open("configs/config.yaml") as file:
        config = yaml.safe_load(file)
    
    ZoneConf = Zoneconf(config['Zoneconf_path'])
    # config needs more information to be passed to the model
    config['NFeatures']= len(config["Columns_to_keep"]) # This goes to the data preparation process
    config["input_size"] = config["NFeatures"] + 9 + 8 # This goes to the model, we add couple of more features to the input
    config["output_size"] = len(config['Columns_to_Predict']) # The number of columns to predict
    config['device'] = device
    config['ct'] = ct
    log_code = f"Learning rate: {config['learning_rate']}, Hidden size: {config['hidden_size']}, Batch size: {config['batch_size']}"
    for arg in config.items(): # Print the arguments to the log file
        savelog(f"{arg[0]} = {arg[1]}", ct)
    config['sos'] = torch.cat((torch.tensor([10,1016,1016,7,7,7,7]) , torch.zeros(17)), dim=0).repeat(config['Nnodes'], 1).to(device)
    config['eos'] = torch.cat((torch.tensor([11,1020,1020,8,8,8,8]), torch.zeros(17)), dim=0).repeat(config['Nnodes'],1).to(device)

    
    # Before Everything, make sure to assert all the parameters are correct, implement later
    # Create datasets
    Scenetr = Scenes(config)
    Scenetst = Scenes(config)
    Sceneval = Scenes(config)
    savelog(f'Starting at:  {datetime.datetime.now().strftime("%H-%M-%S")}', ct)
    if not config['generate_data']:
        savelog("Loading data ...", ct)
        Scenetr.load_class(dataset_path, cmnt = 'Train')
        Scenetst.load_class(dataset_path, cmnt = 'Test')
        Sceneval.load_class(dataset_path, cmnt = 'Validation')

    else:
        savelog("Loading CSV file ...", ct)
        Traffic_data = loadcsv(config['detection_path'], config['Headers'])
        Scenetr, Scenetst, Sceneval = Scene_Process(Scenetr, Scenetst, Sceneval, Traffic_data, config)
        # Scenetr.augmentation(4)
        Scenetr.addnoise(config['noise_multiply'], config['noise_amp'], config['noise_ratio'])
        Scenetr.save_class(dataset_path, cmnt = 'Train')
        Scenetst.save_class(dataset_path, cmnt = 'Test')
        Sceneval.save_class(dataset_path, cmnt = 'Validation')
    savelog(f"Train size: {len(Scenetr)}, Test size: {len(Scenetst)}, Validation size: {len(Sceneval)}", ct)
    print("Augmenting the data ...")
    
    train_loader, test_loader, val_loader = DataLoader_Scene(Scenetr, Scenetst, Sceneval, config['batch_size'])
    

    model = HDAAGT(config)
    savelog("Creating model from scratch",ct)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['schd_stepzise'], gamma=config['gamma'])
        
    if config['Load_Model']:
        savelog("Loading model from the checkpoint", ct)
        model = model.load_state_dict(torch.load(config['Load_Model_Path'])).to(device)

    if config['Train']:
        savelog(f"The number of learnable parameter is {count_parameters(model=model)} !", ct)
        print(f"Learning rate: {config['learning_rate']}, Hidden size: {config['hidden_size']}, Batch size: {config['batch_size']}")
        train_loss, trainADE, trainFDE = train_model(model, optimizer, criterion, scheduler, train_loader, config)
        savelog(f"{log_code} with Avg loss of {train_loss[-1]}, ADE of {trainADE[-1]}, FDE of {trainFDE[-1]}", f"summary {ct}")
        if train_loss[-1] < 1.5:
            savelog(f"Saving result of {log_code}", ct)
            torch.save(model, os.path.join(cwd,'Processed', log_code + 'Bestmodel.pth'))
            torch.save(train_loss, os.path.join(cwd,'Pickled', log_code + 'epoch_losses.pt'))
        savelog(f"Training finished for {ct}!", ct)

    if config['Test']: # If not training, then test the model
        Topk_Selected_words, ADE, FDE = test_model(model, test_loader, config)
        savelog(f"Average Displacement Error: {ADE}, Final Displacement Error: {FDE}", ct)