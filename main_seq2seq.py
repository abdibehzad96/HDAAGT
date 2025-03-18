from models.seq2seq import *
import torch.nn as nn
import torch
import torch.optim as optim
from utilz.utils import *


# Load the data
cwd = os.getcwd()
csvpath = os.path.join(cwd,'Processed','Trj20240115T2107.csv')
# # Hyperparameters
input_size, hidden_size, num_layers, output_size, epochs, learning_rate = 16, 16, 1, 2, 100, 1
sequence_length, sw , shift, batch_size = 20, 2, 5, 128

LR = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]
HS = [16, 32, 64, 128, 256, 512, 1024]
NL = [1, 2, 3, 4]
BS = [16, 32, 64, 128, 256, 512, 1024]

sl = sequence_length
add_class = True #  Add class by loading the CSV file
load_class = not add_class      # load the class from the saved file
Train = True # It's for training the model with the prepared data
test_in_epoch = True # It's for testing the model in each epoch
model_from_scratch = True # It's for creating model from scratch
load_the_model = not model_from_scratch # It's for loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create datasets

if add_class:
    df = loadcsv(csvpath)
    dataset, maxdata, mindata = def_class(df, sw, sl, shift, output_size, input_size, normalize = False)
    # inpt , target = parabola()
    # dataset = TrajDataset(inpt.to(device), target.to(device))
if load_class:
    dataset = torch.load(os.path.join(os.getcwd(),'Pickled','datasetTorch.pt'))
    normpar = torch.load(os.path.join(os.getcwd(),'Pickled','normpar.pt'))
    print('dataset loaded')
print("Dataset total length is", len(dataset))

train_loader, test_loader, val_loader = prep_data(dataset, batch_size)
# Define the model hyperparameters
INPUT_DIM = sequence_length
OUTPUT_DIM = sequence_length
HID_DIM = 512
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.01
DEC_DROPOUT = 0.01

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

# SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
# TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, 1024, 1024, device).to(device)
# Create the model instance
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)
LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = sequence_length+1)
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_loader, criterion)
    
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')
    
    # print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    model.load_state_dict(torch.load('tut6-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')