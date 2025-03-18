
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv, GATv2Conv
import yaml
from shapely.geometry import Point, Polygon
import re

# Architecture Multi layer GATS concatenated at the end
from models.GraphAttention import *
from models.Transformer import *
from models.minGRU import *


from models.Transformer import Decoder



class TransGAT(nn.Module):
    """
    Graph Attention Network (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.
    Consists of a 2-layer stack of Graph Attention Layers (GATs). The fist GAT Layer is followed by an ELU activation.
    And the second (final) layer is a GAT layer with a single attention head and softmax activation function. 
    """
    def __init__(self,
        args,
        ):
        """ Initializes the GAT model. 
        Args:
            in_features (int): number of input features per node.
            n_hidden (int): output size of the first Graph Attention Layer.
            n_heads (int): number of attention heads in the first Graph Attention Layer.
            num_classes (int): number of classes to predict for each node.
            concat (bool, optional): Wether to concatinate attention heads or take an average over them for the
                output of the first Graph Attention Layer. Defaults to False.
            dropout (float, optional): dropout rate. Defaults to 0.4.
            leaky_relu_slope (float, optional): alpha (slope) of the leaky relu activation. Defaults to 0.2.
        """

        super(TransGAT, self).__init__()
        # self.ZoneConf = Zoneconf(args.Zonepath)
        # self.expand = nn.ModuleList([FFGAT(args.input_size*(args.expansion**i), 
        #                                    args.Nnodes, args.n_heads, args.expansion,
        #                                     args.concat, args.dropout, args.leaky_relu_slope, args.device
        #                                     ) for i in range(args.num_layers)])

        # self.contract = nn.ModuleList([FFGAT(args.input_size*(args.expansion**(args.num_layers-i)),
        #                                     args.Nnodes, int(args.n_heads/args.expansion), 1/args.expansion,
        #                                     args.concat, args.dropout, args.leaky_relu_slope, args.device
        #                                     ) for i in range(args.num_layers)])
        self.src_pad_idx = 0 #src_pad_idx
        self.trg_pad_idx = 0 #trg_pad_idx
        self.trg_sos_idx = 0 #trg_sos_idx
        self.device = args.device
        self.sos = args.sos
        self.eos = args.eos
        self.constant = nn.ModuleList([FFGAT(args.input_size,
                                            args.Nnodes, args.n_heads, 1,
                                            args.concat, args.dropout, args.leaky_relu_slope, args.device
                                            ) for i in range(args.num_layers)])
        
        # self.embed_size = int(args.input_size*(args.expansion**(args.num_layers)-1)/(args.expansion-1))
        # self.embed_size = args.input_size*20*(args.num_layers+1)
        # self.embed_size = args.input_size*args.expansion**args.num_layers*20
        self.embed_size = args.input_size*(args.num_layers+1)
        self.dec_voc_size = 1024
        self.output_size = args.output_size
        self.num_layer = args.num_layers
        self.SL = args.sl-1
        self.NFs = args.input_size
        self.Nuser = args.Nuser
        self.future = args.future

        self.emb= nn.Embedding(self.dec_voc_size, self.embed_size//2)
        self.decoder = Decoder(d_model=self.embed_size,
                               n_head=args.n_heads,
                               max_len=self.future+2, # 2 for sos and eos
                               ffn_hidden=args.hidden_size,
                               dec_voc_size=self.dec_voc_size*2, # *2 for x and y
                               drop_prob=args.dropout,
                               n_layers= 4, #args.num_layers,
                               device=args.device)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.FC = nn.Linear(self.dec_voc_size, self.dec_voc_size)
        self.softmax = nn.Softmax(dim=-1)

        

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg): # [128,41,32,2]
        trg_pad_mask = (trg[:,:,0] != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) # was unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1] # trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask 
        return trg_mask
    
    def forward(self, scene: torch.Tensor, Adj_mat: torch.Tensor, target: torch.Tensor):
        """
        Performs a forward pass through the network.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        B, _, _, F = target.size()
        N= 32
        sos= self.sos.expand(B*N,1,F)
        eos= self.eos.expand(B*N,1,F)
        x = scene

        y = x
        for layer in self.constant:
            x = layer(x, Adj_mat)
            y = torch.cat((y, x), dim=-1)
        # target [Batch, FL, Users, xy]


        xuser = y[:,:,:32,:] #.permute(0,2,1,3)
        # xuser = xuser.reshape(B*N, self.SL, self.NFs*(self.num_layer+1))

        xtarget = target.permute(0,2,1,3).reshape(B*N, self.future, F)  #[Batch*Users, FL, xy] ---> [Batch, USers, FL, xy]
        xtarget = torch.cat((sos, xtarget, eos), dim=1)
        src_mask = self.make_src_mask(xuser[:,:,0,0])
        trg_mask = self.make_trg_mask(xtarget)
        xtarget = self.emb(xtarget.to(torch.long)).flatten(-2) # was [Batch*Users, FL,xy*embed_size]
        xtarget = xtarget.reshape(B,N, self.future + 2, self.embed_size)
        trg_mask = trg_mask.reshape(B,N, 1, self.future + 2, self.future + 2)
        
        out = self.decoder(xtarget, xuser, trg_mask, src_mask)
        
        out = torch.stack(out, dim=1)
        # out = self.decoder(xtarget, xuser, trg_mask, src_mask)
        # out = out.reshape(B*N, self.future + 2, 2, self.dec_voc_size)
        # out = self.FC(out)
        return out

class FFGAT(nn.Module):
    def __init__(self, input_size, Nnodes, n_heads, expantion, concat, dropout, leaky_relu_slope, device):
        super(FFGAT, self).__init__()
        self.norm1 = nn.LayerNorm(input_size)
        self.FC = nn.Linear(input_size, input_size)
        self.GAT1 = NewGATCell(in_features=input_size, out_features=input_size, n_heads=n_heads,
                    concat=concat, n_nodes=Nnodes, dropout=dropout, leaky_relu_slope=leaky_relu_slope)
        self.Relu = nn.ReLU()
        self.GAT2 = NewGATCell(in_features=input_size, out_features=int(input_size*expantion), n_heads=n_heads,
                    concat=concat, n_nodes=Nnodes, dropout=dropout, leaky_relu_slope=leaky_relu_slope)

        self.norm2 = nn.LayerNorm(int(input_size*expantion))
        self.norm3 = nn.LayerNorm(input_size)
        # self.GRU = stackedGRU(input_size, hidden_size=512, device=device)
        self.input_size = input_size


    def forward(self, x, adj):
        x = self.norm1(x)
        x = self.GAT1(x, adj)
        x = self.norm2(x)
        x = self.FC(x)
        x = self.Relu(x)
        x = self.norm3(x)
        x = self.GAT2(x, adj)
        # x = x.permute(0,2,1,3)[:,:32,:,:]
        # x = x.reshape(B*32, SL, F)
        # x = self.GRU(x, h0, None)
        # x = x.reshape(B, N, SL, F).permute(0,2,1,3)
        return x
    
    
    # def edge_index(self, adj):
    #     if self.edge_indx == []:
    #         for j in range(adj.size(0)):
    #             Iz , Jz = torch.where(adj[j]>0.1)
    #             self.edge_indx.append(torch.stack((Iz, Jz), dim=0))


class stackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(stackedGRU, self).__init__()
        self.LN = nn.LayerNorm(hidden_size, device=device)
        self.FC = nn.Linear(input_size, hidden_size)
        self.minLSTM = MinGRUcell(hidden_size, hidden_size)
        
        self.LN2 = nn.LayerNorm(hidden_size, device=device)
        self.FC2 = nn.Linear(hidden_size, hidden_size)
        self.minLSTM2 = MinGRUcell(hidden_size, hidden_size)
        self.activation = nn.ReLU()
    
    def forward(self, x, h0, seq):
        h = self.FC(x)
        h = self.activation(h)
        h = self.LN(h)
        h = self.minLSTM(h, h0, seq)
        h = self.FC2(h)
        h = self.activation(h)
        h = self.LN2(h)
        h = self.minLSTM2(h, h0, seq)
        return h
    
class FFlayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FFlayer, self).__init__()
        self.FC = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.FC2 = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.FC3 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.FC(x)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.FC2(x)
        x = self.relu(x)
        x = self.FC3(x)
        return x

class MinGRU(nn.Module):
    def __init__(self, args):
        super(MinGRU, self).__init__()
        self.embed_size = args.input_size *2**(args.num_layersGAT) #args.input_size*(args.num_layers+1)
        self.hidden_size = args.hidden_size
        self.device = args.device
        self.output_size = args.output_size
        self.num_layersGRU = args.num_layersGRU
        self.embd_list = [self.embed_size] + [args.hidden_size]*(args.num_layersGRU-1)



        self.constant = nn.ModuleList([FFGAT(args.input_size*2**(i),
                                        args.Nnodes, args.n_heads, 1,
                                        args.concat, args.dropout, args.leaky_relu_slope, args.device
                                        ) for i in range(args.num_layersGAT)])
        self.normFFGA = nn.LayerNorm(self.embed_size, device=self.device)
        
        self.FC0 = nn.Linear(self.embed_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


        # self.FC00 = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNormlast = nn.LayerNorm(self.hidden_size, device=self.device)
        # self.MinLSTM = nn.ModuleList([stackedGRU(self.hidden_size, self.hidden_size, self.device) for i in range(args.num_layersGRU)])
        self.MinLSTM = nn.GRU(self.hidden_size, self.hidden_size, num_layers=6, batch_first=True)
        self.normlstm = nn.LayerNorm(self.hidden_size, device=self.device)
        # self.LSTM = nn.GRU(self.hidden_size, self.hidden_size, num_layers=4, batch_first=True)
        self.FFlayer = FFlayer(self.hidden_size, self.hidden_size)
        self.FClst1 = nn.Linear(self.hidden_size//2, self.hidden_size//2)

        self.FClst2 = nn.Linear(self.hidden_size//2,self.hidden_size//2)

        # self.FClst3 = nn.Linear(self.hidden_size//2, self.hidden_size//2)
        self.FClst4 = nn.Linear(self.hidden_size//2, 1)

    def forward(self, scene: torch.Tensor, Adj_mat: torch.Tensor, target: torch.Tensor):
        h = scene
        B, SL, _, _ = scene.size()
        for layer in self.constant:
            x = layer(h, Adj_mat)
            h = torch.cat((h, x), dim=-1)

        h= h.permute(0,2,1,3).reshape(-1, SL, self.embed_size)
        # adj = Adj_mat.sum(dim=-1)[:,:,:32].permute(0,2,1).reshape(B*32, SL).unsqueeze(2)-1
        # h = h * adj
        
        h = self.FC0(h)

        h=self.MinLSTM(h)

        h = self.FFlayer(h[1])
        # h = self.relu(h)
        h = h.mean(dim=0)
        h = h.reshape(B,32, 2,self.hidden_size//2) # Here is the most important part, we have to separate the x and y and let them go through same layers

        out = self.FClst1(h)
        out= self.relu(out)
        out = self.FClst2(out)
        out= self.relu(out)
        # out = self.FClst3(out)
        out = self.FClst4(out)
        return out.squeeze(3)
    
if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")