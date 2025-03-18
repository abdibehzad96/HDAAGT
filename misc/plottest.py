from utilz.utils import *
import argparse
cwd = os.getcwd()
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
# # Hyperparameters
hidden_size, num_layersGAT, num_layersGRU, expansion,  n_heads = 128, 1, 6, 2, 8

learning_rate, schd_stepzise, gamma, epochs, batch_size, patience_limit, clip= 5e-2, 15, 0.50, 200, 512, 10, 1

Headers = ['Frame', 'ID', 'BBx', 'BBy','W', 'L' , 'Cls','Tr1', 'Tr2', 'Tr3', 'Tr4', 'Zone', 'Xreal', 'Yreal']
Columns_to_keep = [2,3,7,8,9,10,11] #['BBx', 'BBy','W', 'L' , 'Cls','Tr1', 'Tr2', 'Tr3', 'Tr4']
#  ['Vx', 'Vy', 'heading',xc,yc, Rc,Rc, SinX,CosX,SinY,CosY, Sin2X,Cos2X, Sin2Y, Cos2Y, Sin3X, Cos3X, Sin3Y, Cos3Y, Sin4X, Cos4X, Sin4Y, Cos4Y]
Columns_to_Predict = [0,1] #['BBx', 'BBy','Xreal', 'Yreal','Vx', 'Vy']
xyid = [0, 1] # the index of x and y  of the Columns_to_Keep in the columns for speed calculation
TrfL_Columns = [7,8,9,10]
NFeatures = len(Columns_to_keep)
input_size = NFeatures
output_size = len(Columns_to_Predict)
Nusers, NZones = 32, 10
Nnodes, NTrfL, sl, future, sw, sn  = NZones + Nusers, 4, 20, 30, 2, 5
Centre = [512,512]

sos = torch.tensor([1022,1022], device=device)
eos = torch.tensor([1021,1021], device=device)

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
    
parser = argparse.ArgumentParser(description='Trajectory Prediction')

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

args = parser.parse_args()
Scenetr = Scenes(args, 0)
cwd = os.getcwd()
dataset_path = os.path.join(cwd, 'Pickled')
Scenetr.load_class(dataset_path, cmnt = 'Train')


import matplotlib.pyplot as plt


x,y = Scenetr.Scene[0,:,0,0], Scenetr.Scene[0,:,0,1]

plt.plot(x.cpu().numpy(), y.cpu().numpy(), 'ro')
plt.show()
plt.pause(5)
plt.close()