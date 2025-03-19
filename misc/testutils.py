import torch
from utilz.utils import *
import time
import argparse

csvpath ='/home/abdikhab/New_Idea_Traj_Pred/RawData/TrfZonXYCam.csv'
loadData = False
dataset_path = os.path.join(os.getcwd(), 'Pickled')
ct = datetime.datetime.now().strftime("%m-%d-%H-%M")
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

Headers = ['Frame', 'ID', 'BBx', 'BBy','W', 'L' , 'Cls','Tr1', 'Tr2', 'Tr3', 'Tr4', 'Zone', 'Xreal', 'Yreal']
Columns_to_keep = [4,5,6,12,13] #['W', 'L' , 'Cls', 'Xreal', 'Yreal']
TrfL_Columns = [7,8,9,10]
NFeatures = len(Columns_to_keep)
Nnodes, NZones, NTrfL, sl, future, sw, sn  = 32 + 9, 9, 4, 20, 30, 4, 4
generate_data = True #  Add class by loading the CSV file
load_data = not generate_data      # load the class from the saved file
Train = True # It's for training the model with the prepared data
test_in_epoch = True # It's for testing the model in each epoch
model_from_scratch = True # It's for creating model from scratch
load_the_model = not model_from_scratch # It's for loading model
Seed = False # It's for setting the seed for the random number generator
only_test = False # It's for testing the model only
input_size, hidden_size, num_layers, output_size, epochs, learning_rate, batch_size = 16, 16, 1, 2, 100, 1, 128
# Create datasets
parser = argparse.ArgumentParser(description='Trajectory Prediction')
parser.add_argument('--Nfeatures', type=int, default=NFeatures, help='Number of features')
parser.add_argument('--Nnodes', type=int, default=Nnodes, help='Number of nodes')
parser.add_argument('--NZones', type=int, default=NZones, help='Number of zones')
parser.add_argument('--NTrfL', type=int, default=NTrfL, help='Number of traffic lights')
parser.add_argument('--sl', type=int, default=sl, help='Sequence length')
parser.add_argument('--future', type=int, default=future, help='Future length')
parser.add_argument('--sw', type=int, default=sw, help='Sliding window')
parser.add_argument('--sn', type=int, default=sn, help='Sliding number')
parser.add_argument('--input_size', type=int, default=input_size, help='Input size')
parser.add_argument('--hidden_size', type=int, default=hidden_size, help='Hidden size')
parser.add_argument('--num_layers', type=int, default=num_layers, help='Number of layers')
parser.add_argument('--output_size', type=int, default=output_size, help='Output size')
parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size')
parser.add_argument('--generate_data', type=bool, default=generate_data, help='Generate data')
parser.add_argument('--load_data', type=bool, default=load_data, help='Load data')
parser.add_argument('--Train', type=bool, default=Train, help='Train')
parser.add_argument('--test_in_epoch', type=bool, default=test_in_epoch, help='Test in epoch')
parser.add_argument('--model_from_scratch', type=bool, default=model_from_scratch, help='Model from scratch')
parser.add_argument('--load_the_model', type=bool, default=load_the_model, help='Load the model')
parser.add_argument('--Seed', type=bool, default=Seed, help='Seed')
parser.add_argument('--only_test', type=bool, default=only_test, help='Only test')
parser.add_argument('--Columns_to_keep', type=list, default=Columns_to_keep, help='Columns to keep')
parser.add_argument('--TrfL_Columns', type=list, default=TrfL_Columns, help='Traffic light columns')
parser.add_argument('--device', type=str, default=device, help='device')
parser.add_argument('--ct', type=str, default=ct, help='Current time')
args = parser.parse_args()

Scenetr = Scenes(args)
Scenetst = Scenes(args)
Sceneval = Scenes(args)
if loadData:
    Scenetr.load_class(dataset_path)
    Scenetst.load_class(dataset_path)
    Sceneval.load_class(dataset_path)
else:
    df = loadcsv(csvpath, Headers)
    Scenetr, Scenetst, Sceneval = Scene_Process(Scenetr, Scenetst, Sceneval, df, args)
    Scenetr.save_class(dataset_path)
    Scenetst.save_class(dataset_path)
    Sceneval.save_class(dataset_path)
    
print(f"Train size: {len(Scenetr)}, Test size: {len(Scenetst)}, Validation size: {len(Sceneval)}")
train_loader, test_loader, val_loader = prep_data(Scenetr, Scenetst, Sceneval, batch_size)

