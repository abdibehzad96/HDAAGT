
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn.conv import GATConv, GATv2Conv
import yaml
from shapely.geometry import Point, Polygon
import re
import random
# Architecture Multi layer GATS concatenated at the end
from models.modules import *
from misc.not_used.Transformer import *

from utilz.utils import target_mask , create_src_mask
from misc.not_used.Transformer import *

# import torch.nn as nn
# import torch.nn.functional as F
# from pygcn.layers import GraphConvolution



def positional_encoding(x, d_model):

        # result.shape = (seq_len, d_model)
        result = torch.zeros(
            (x.size(1), d_model),
            dtype=torch.float,
            requires_grad=False
        )

        # pos.shape = (seq_len, 1)
        pos = torch.arange(0, x.size(1)).unsqueeze(1)

        # dim.shape = (d_model)
        dim = torch.arange(0, d_model, step=2)

        # Sine for even positions, cosine for odd dimensions
        result[:, 0::2] = torch.sin(pos / (10_000 ** (dim / d_model)))
        result[:, 1::2] = torch.cos(pos / (10_000 ** (dim / d_model)))
        return result.to(x.device)


def Spatial_encoding(x, d_model):

        result = torch.zeros_like(x, requires_grad=False)

        # pos.shape = (seq_len, 1)
        pos = torch.arange(0, x.size(2)).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(x.size(0),x.size(1), 1,1)

        # dim.shape = (d_model)
        dim = torch.arange(0, d_model, step=2)

        # Sine for even positions, cosine for odd dimensions
        result[:,:,:, 0::2] = torch.sin(pos / (10_000 ** (dim / d_model)))
        result[:,:,:, 1::2] = torch.cos(pos / (10_000 ** (dim / d_model)))
        return result.to(x.device)


class deepFF(nn.Module):
    def __init__(self, input_size, output_size, n_heads, n_nodes):
        super(deepFF, self).__init__()
        self.GAT1 = GraphAttentionV2Layer(in_features=input_size, out_features=output_size, n_heads=n_heads, n_nodes=n_nodes,
                    concat=False)
        self.hidden_size = output_size
        
        # self.dropout = nn.Dropout(0.1)
    def forward(self, h, adj):
        B, SL, Nnodes, _ = h.size()
        # x = h + Spatial_encoding(h, self.hidden_size)
        x = self.GAT1(h, adj)
        return x

class TrfEmb(nn.Module):
    def __init__(self, hidden_size, num_heads, n_nodes):
        super(TrfEmb, self).__init__()
        self.embdx = nn.Embedding(1024, hidden_size//2, padding_idx=0) #, padding_idx=0
        self.embdy= nn.Embedding(1024, hidden_size//2, padding_idx=0)
        self.embdTrL = nn.Embedding(9, hidden_size//8, padding_idx=0)
        self.embdZone = nn.Embedding(12,hidden_size//8, padding_idx=0)
        self.speed_HS = hidden_size - 5*(hidden_size//8)
        self.speedLN = nn.LayerNorm(17)

        # self.embdZone = nn.Embedding(12,hidden_size//8, padding_idx=0)

        self.scalespeed = nn.Parameter(torch.randn(17))
        self.speedLN1 = nn.LayerNorm(self.speed_HS)
        self.speedLN2 = nn.LayerNorm(self.speed_HS)
        self.speed = nn.Linear(17, self.speed_HS )
        self.speed2 = nn.Linear(self.speed_HS, self.speed_HS)
        self.speed3 = nn.Linear(self.speed_HS, self.speed_HS)

        # self.LN_Trf = nn.LayerNorm(hidden_size)
        self.Att = nn.MultiheadAttention(embed_dim= hidden_size, num_heads=num_heads, batch_first=True)
        self.AttTrf = nn.MultiheadAttention(embed_dim= hidden_size, num_heads=num_heads, batch_first=True)
        self.AttAll = nn.MultiheadAttention(embed_dim= hidden_size, num_heads=num_heads, batch_first=True)
        # self.AttAll2 = nn.MultiheadAttention(embed_dim= hidden_size, num_heads=num_heads, batch_first=True)

        self.FFTrfL = FeedForwardNetwork(d_model=hidden_size, out_dim=hidden_size)
        self.FFxy = FeedForwardNetwork(d_model=hidden_size, out_dim=hidden_size)
        self.FFAll = FeedForwardNetwork(d_model=hidden_size, out_dim=hidden_size)
        # self.FFAll2 = FeedForwardNetwork(d_model=hidden_size, out_dim=hidden_size)

        self.RezeroTrF = nn.Parameter(torch.rand(hidden_size))
        self.RezeroXY = nn.Parameter(torch.rand(hidden_size))
        self.RezeroALL = nn.Parameter(torch.rand(hidden_size))
        # self.RezeroALL2 = nn.Parameter(torch.zeros(hidden_size))
        self.RezeroALLFF = nn.Parameter(torch.rand(hidden_size))
        # self.RezeroALLFF2 = nn.Parameter(torch.zeros(hidden_size))

        self.RezeroSpeed = nn.Parameter(torch.rand(self.speed_HS))
        self.LNspeed = nn.LayerNorm(self.speed_HS)
        self.RezeroSpeed2 = nn.Parameter(torch.rand(self.speed_HS))
        self.LNspeed2 = nn.LayerNorm(self.speed_HS)
        self.dropout = nn.Dropout(p=0.1)
        self.dropout2= nn.Dropout(p=0.1)
        self.hidden_size = hidden_size
        self.d_model = hidden_size
        self.DeepFF = deepFF(hidden_size, hidden_size, n_heads=num_heads, n_nodes= n_nodes)
    def forward(self, x, src_mask, adj):
        B, Nnodes, SL, _ = x.size()
        embx= self.embdx(x[:,:,:,1].to(torch.long))
        emby= self.embdy(x[:,:,:,2].to(torch.long))
        embxy = torch.cat((embx, emby), dim=-1) #.reshape(-1, x.size(2), self.hidden_size)
        embTrf = self.embdTrL(x[:,:,:,3:7].to(torch.long)).flatten(-2)
        embZone = self.embdZone(x[:,:,:,0].to(torch.long))
        speed = self.speedLN(x[:,:,:,7:])
        speed1 = self.speed(speed)
        # speed2 = self.RezeroSpeed* self.speed2(F.leaky_relu(speed1)) + speed1
        speed2 = self.speedLN1(speed1 + self.speed2(F.leaky_relu(speed1)))
        speed = self.RezeroSpeed2* self.speed3(F.leaky_relu(speed2)) + speed2
        speed = self.speedLN2(speed2 + self.speed3(F.leaky_relu(speed2)))

        Traffic_embd = torch.cat((embTrf, embZone, speed), dim=-1).reshape(-1,x.size(2), self.hidden_size)
        # Traffic_embd = self.LN_Trf(Traffic_embd)
        
        Traffic_embd = Traffic_embd + positional_encoding(Traffic_embd,self.d_model)
        h = embxy.permute(0,2,1,3)
        # h= self.resnet(h)
        # h = h.permute(0,3,2,1)
        h = self.DeepFF(h, adj) #[B, SL, Nnodes, hidden_size]
        embxy = h.permute(0,2,1,3).reshape(B*Nnodes, SL, self.hidden_size)
        
        embxy = embxy + positional_encoding(embxy, self.d_model)
        
        TrfATT = self.AttTrf(Traffic_embd, Traffic_embd, Traffic_embd, key_padding_mask = src_mask)[0] #key_padding_mask = src_mask
        Traffic_embd = self.RezeroTrF*TrfATT + Traffic_embd
        Traffic_embd = self.FFTrfL(Traffic_embd)


        att_embxy = self.Att(embxy , embxy , embxy, need_weights=False, key_padding_mask = src_mask)[0] #key_padding_mask = src_mask
        att_embxy = self.dropout(att_embxy)
        embxy = self.RezeroXY*att_embxy + embxy
        embxy = self.FFxy(embxy)

        AttAll = self.AttAll(embxy, Traffic_embd, Traffic_embd,key_padding_mask = src_mask, need_weights=False)[0] #key_padding_mask = src_mask
        ALL_EMbedding = self.RezeroALL*AttAll + embxy

        ALL_EMbeddingFF = self.FFAll(ALL_EMbedding)
        ALL_EMbedding = self.dropout2(ALL_EMbedding + self.RezeroALLFF*ALL_EMbeddingFF)
        return ALL_EMbedding

class TargetEmb(nn.Module):
    def __init__(self, hidden_size):
        super(TargetEmb, self).__init__()
        self.embdx = nn.Embedding(1024, hidden_size//2, padding_idx=0) #, padding_idx=0
        self.embdy= nn.Embedding(1024, hidden_size//2, padding_idx=0)
        self.LN = nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size
    
    def forward(self, target):
        embx= self.embdx(target[:,:,:,0].to(torch.long))
        emby= self.embdy(target[:,:,:,1].to(torch.long))
        embxy = torch.cat((embx, emby), dim=-1).permute(0,2,1,3).reshape(-1, target.size(1), self.hidden_size)
        embxy = embxy + positional_encoding(embxy, self.hidden_size)
        # embxy = self.LN(embxy)
        return embxy
class Encoder(nn.Module):
    def __init__(self, hidden_size, num_heads, n_nodes):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.InitEmbed = TrfEmb(self.hidden_size, num_heads, n_nodes)
        # self.deepFF = deepFF(self.hidden_size, self.hidden_size)

    def forward(self, scene, adj,src_mask, B, SL, Nnodes):
        h = scene
        h = h.permute(0,2,1,3)
        h = self.InitEmbed(h, src_mask, adj)

        return h
    

class Projection(nn.Module):
    def __init__(self, hidden_size, output_size, Nnodes, future=1):
        super(Projection, self).__init__()
        self.Linear1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.Linear2 = nn.Linear(hidden_size, 1024*output_size)
        self.output_size = output_size
        self.Nnodes = Nnodes
        self.Rezero = nn.Parameter(torch.rand(hidden_size))
        # self.future = future
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, h):
        BN, SL, _ = h.size()
        h = self.Linear1(h)
        h = self.dropout(self.Rezero* h + h)
        return self.Linear2(h).reshape(BN//self.Nnodes, self.Nnodes, SL , self.output_size, 1024).permute(0,2,1,3,4)
    
class Decoderr(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Decoderr, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.conv1d = nn.Conv1d(1, 1, 8, 4,2, padding_mode='circular')
        # self.LN = nn.LayerNorm(hidden_size)
        # self.maxpool = nn.MaxPool1d(4,1,0)
        # self.LN = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8,1)
        

    def forward(self, x, hidden, cell):
        output, (hs, cs) = self.lstm(x, (hidden, cell))
        output = self.fc(hs.permute(1,2,0))
        output = self.fc2(F.tanh(output)).permute(0,2,1)
        return output, hs, cs

class GGAT(nn.Module):
    def __init__(self, args):
        super(GGAT, self).__init__()
        self.embed_size = args.input_size
        self.hidden_size = args.hidden_size
        self.d_model = args.hidden_size
        self.device = args.device
        self.num_layers = args.num_layersGRU
        self.output_size = args.output_size
        self.encoder = Encoder(self.hidden_size, num_heads=args.n_heads, n_nodes=args.Nnodes)
        self.EncoderLSTM = nn.LSTM(self.hidden_size,self.hidden_size, num_layers=32, batch_first=True)
        self.compressor = nn.Linear(self.hidden_size*32, self.hidden_size*2)
        self.targetembed = TargetEmb(self.hidden_size)
        self.Decoder = Decoderr(self.hidden_size, self.hidden_size, num_layers=32, output_size=self.hidden_size)
        self.proj = Projection(self.hidden_size, 2, 32, 1)
        self.fc = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8,1)
        
        #['BBx', 'BBy','W', 'L' , 'Cls','Tr1', 'Tr2', 'Tr3', 'Tr4', 'Zone']
    def forward(self, scene: torch.Tensor, src_mask, adj, target, future_len):
        B, SL, Nnodes, _ = scene.size()
        # TrgL = Target.size(1)
        enc_out = self.encoder(scene, adj, src_mask, B, SL, Nnodes)
        # enc_out = self.LN(enc_out)

        _, (hs,cs) = self.EncoderLSTM(enc_out)
        # outputs = self.FC(outputs)
        if target is not None:
            target = self.targetembed(target)
        decoder_input = target[:, :1]
        # decoder_input = self.fc(hs.permute(1,2,0))
        # decoder_input = self.fc2(F.tanh(decoder_input)).permute(0,2,1)
        # decoder_input = hs[0].unsqueeze(1)
        # decoder_input = enc_out[:,-1:]
        outputs = []
        # outputs.append(decoder_input)
        for t in range(future_len+1):
            next, hs, cs = self.Decoder(decoder_input, hs, cs)
            
            outputs.append(next)
            # Decide whether to use real or predicted values (Scheduled Sampling)
            if target.size(1) > 2 and random.random() < 0.25:
                decoder_input = target[:, t:t+1, :]  # Use actual data
            else:
                decoder_input = next # Use model's prediction

         
        # pred = torch.cat(outputs, dim=1).to(device=self.device)
        # pred = self.compressor(pred)
        out = self.proj(torch.cat(outputs, dim=1))
        return out


if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")