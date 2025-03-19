
# export cuda_visible_devices=2,3
import torch
import torch.nn as nn
import torch.optim as optim
import os
import yaml
import datetime
# The rest is HDAAGT codes
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
    with  open("config.yaml") as file:
        config = yaml.safe_load(file)
    
    ZoneConf = Zoneconf(config['Zoneconf_path'])
    # config needs more information to be passed to the model
    config['NFeatures']= len(config["Columns_to_keep"]) # This goes to the data preparation process
    config["input_size"] = config["NFeatures"] + 9 + 8 # This goes to the model, we add couple of more features to the input
    config["output_size"] = len(config['Columns_to_Predict']) # The number of columns to predict
    config['device'] = device
    config['ct'] = ct


    for arg in config.items(): # Print the arguments to the log file
        savelog(f"{arg[0]} = {arg[1]}", ct)
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
        Traffic_data = loadcsv(config['csvpath'], config['Headers'])
        Scenetr, Scenetst, Sceneval = Scene_Process(Scenetr, Scenetst, Sceneval, Traffic_data, config)
        # Scenetr.augmentation(4)
        Scenetr.addnoise(config['noise_multiply'], config['noise_amp'], config['ratio'])
        Scenetr.save_class(dataset_path, cmnt = 'Train')
        Scenetst.save_class(dataset_path, cmnt = 'Test')
        Sceneval.save_class(dataset_path, cmnt = 'Validation')
    savelog(f"Train size: {len(Scenetr)}, Test size: {len(Scenetst)}, Validation size: {len(Sceneval)}", ct)
    print("Augmenting the data ...")
    
    train_loader, test_loader, val_loader = DataLoader_Scene(Scenetr, Scenetst, Sceneval, config['batch_size'])
    

    model = GGAT(config)
    if not config['model_from_scratch']:
        savelog("Loading model from the checkpoint", ct)
        mpath = os.path.join(cwd,'Processed','checkpoint.pth.tar')
        model = model.load_state_dict(torch.load(mpath))
    else:
        savelog("Creating model from scratch",ct)

    if config['Train']:
        print("Starting training phase")
        criterion = nn.CrossEntropyLoss()
        savelog(f"The number of learnable parameter is {count_parameters(model=model)} !", ct)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
        print(f"Learning rate: {config['learning_rate']}, Hidden size: {config['hidden_size']}, Batch size: {config['batch_size']}")
        Best_Model, predicteddata,Targetdata,test_loss, epoch_losses, Max_Acc =train_model(
            model, optimizer, criterion, train_loader, test_loader, config, ZoneConf)

        log_code = f"Learning rate: {config['learning_rate']}, Hidden size: {config['hidden_size']}, Batch size: {config['batch_size']}"
        savelog(f"{log_code} with Max Acc of {Max_Acc}", f"summary {ct}")
        if Max_Acc > 0.001:
            savelog(f"Saving result of {log_code} with Max Acc of {Max_Acc}", ct)
            torch.save(Best_Model, os.path.join(cwd,'Processed', log_code + 'Bestmodel.pth'))
            torch.save(predicteddata, os.path.join(cwd,'Pickled', log_code + 'Predicteddata.pt'))
            torch.save(test_loss, os.path.join(cwd,'Pickled', log_code + 'test_losses.pt'))
            torch.save(epoch_losses, os.path.join(cwd,'Pickled', log_code + 'epoch_losses.pt'))
        savelog(f"Training finished for {ct}!", ct)
