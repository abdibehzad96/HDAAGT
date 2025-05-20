
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *


class Positional_Encoding_Layer(nn.Module):
    def __init__(self, hidden_size, num_att_heads, xy_indx, pos_embedding_dim, pos_embedding_dict_size, nnodes, sl, dropout = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.xy_indx = xy_indx

        # Embeddings of the positional features
        self.Postional_embeddings = nn.ModuleList()
        for i in range(len(xy_indx)):
            self.Postional_embeddings.append(nn.Embedding(pos_embedding_dict_size[i], pos_embedding_dim[i], padding_idx=0))

        # Temporal Convolution and Positional Attention layers
        self.TemporalConv = TemporalConv(2*hidden_size, sl)
        self.Position_Att = nn.MultiheadAttention(embed_dim= 3*hidden_size, num_heads=num_att_heads, batch_first=True)
        self.Pos_FF = FeedForwardNetwork(d_model= 3*hidden_size, out_dim=3*hidden_size)
        self.Position_Rezero = nn.Parameter(torch.zeros(3*hidden_size))


        # The recent results showed that placing DAAG layer here is more efficient than putting it after the Encoder
        self.DAAG = DAAG_Layer(in_features=2*hidden_size, out_features=2*hidden_size, n_heads= num_att_heads, n_nodes = nnodes, concat=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Scene, Scene_mask, adj):
        B, SL, Nnodes, _ = Scene.size()

        # Embedding the positional features
        positional_embedding = []
        for n, indx in enumerate(self.xy_indx):
            positional_embedding.append(self.Postional_embeddings[n](Scene[:,:,:,indx].long()))
        Pos_embd = torch.cat(positional_embedding, dim=-1)

        # DAAG layer
        Pos_embd = self.DAAG(Pos_embd, adj) 
        Pos_embd = Pos_embd.permute(0,2,1,3).reshape(B*Nnodes, SL, 2*self.hidden_size)

        # Leaky_Residual 
        Leaky_residual = self.TemporalConv(Pos_embd)
        Pos_embd= torch.cat((Pos_embd,Leaky_residual), dim=-1)

        # Positional Encoding + Attention
        Pos_embd = Pos_embd + positional_encoding(Pos_embd, 3*self.hidden_size)
        Pos_embd_att = self.Position_Att(Pos_embd , Pos_embd , Pos_embd, need_weights=False, key_padding_mask = Scene_mask)[0] #key_padding_mask = src_mask
        Pos_embd = self.Position_Rezero*Pos_embd_att + Pos_embd
        Pos_embd = self.dropout(Pos_embd)
        return Pos_embd, Leaky_residual


class Traffic_Encoding_Layer(nn.Module):
    def __init__(self, hidden_size, num_att_heads, Traffic_indx, trf_embedding_dim, trf_embedding_dict_size, Num_linear_inputs, Linear_indx, dropout = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.Traffic_indx = Traffic_indx
        self.Linear_indx = Linear_indx
        self.Linear_Embedding_dim = 2*hidden_size - sum(trf_embedding_dim)

        # Embeddings of the Traffic features
        self.Traffic_embeddings = nn.ModuleList()
        for i in range(len(Traffic_indx)):
            self.Traffic_embeddings.append(nn.Embedding(trf_embedding_dict_size[i], trf_embedding_dim[i], padding_idx=0))
        
        # For the Linear features, we consider the followinng fully connected layers
        self.LE_LN1 = nn.LayerNorm(Num_linear_inputs)
        self.Linear_Embedding1 = nn.Linear(Num_linear_inputs, self.Linear_Embedding_dim)
        self.LE_LN2 = nn.LayerNorm(self.Linear_Embedding_dim)
        self.Linear_Embedding2 = nn.Linear(self.Linear_Embedding_dim, self.Linear_Embedding_dim)
        self.LE_LN3 = nn.LayerNorm(self.Linear_Embedding_dim)
        self.Linear_Embedding3 = nn.Linear(self.Linear_Embedding_dim, self.Linear_Embedding_dim)

        # Traffic Attention
        self.Traffic_Att = nn.MultiheadAttention(embed_dim= 3*hidden_size, num_heads=num_att_heads, batch_first=True)
        self.Traffic_FF = FeedForwardNetwork(d_model=3* hidden_size, out_dim=3* hidden_size)
        self.LE_Rezero2 = nn.Parameter(torch.zeros(self.Linear_Embedding_dim))
        self.LE_Rezero3 = nn.Parameter(torch.zeros(self.Linear_Embedding_dim))
        self.Traffic_Rezero = nn.Parameter(torch.zeros(3*hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, Scene, Scene_mask, Leaky_residual):
        B, SL, Nnodes, _ = Scene.size()

        # Embedding the Traffic features
        Traffic_embedding = []
        for n, indx in enumerate(self.Traffic_indx):
            Traffic_embedding.append(self.Traffic_embeddings[n](Scene[:,:,:,indx].long()))
        Trf_embd = torch.cat(Traffic_embedding, dim=-1)
        # Linear Embedding
        Lin_embd = self.LE_LN1(Scene[:,:,:,self.Linear_indx])
        Lin_embd1 = self.Linear_Embedding1(Lin_embd)
        Lin_embd2 = self.LE_Rezero2* self.Linear_Embedding2(F.leaky_relu(Lin_embd1)) + Lin_embd1
        Lin_embd = self.LE_Rezero3* self.Linear_Embedding3(F.leaky_relu(Lin_embd2)) + Lin_embd2
        Trf_embd = torch.cat((Trf_embd, Lin_embd), dim=-1).reshape(B*Nnodes, SL, 2*self.hidden_size)
        Trf_embd = torch.cat((Trf_embd, Leaky_residual), dim =-1)
        
        
        # Traffic Attention
        Trf_embd = Trf_embd + positional_encoding(Trf_embd, 3*self.hidden_size)
        Trf_embd_att = self.Traffic_Att(Trf_embd, Trf_embd, Trf_embd, key_padding_mask = Scene_mask)[0] #key_padding_mask = src_mask
        Trf_embd = self.dropout(self.Traffic_Rezero*Trf_embd_att + Trf_embd)
        Trf_embd = self.Traffic_FF(Trf_embd)
        Trf_embd = self.dropout(Trf_embd)
        return Trf_embd

class Mixed_Attention_Layer(nn.Module):
    def __init__(self, hidden_size, num_att_heads):
        super().__init__()
        self.Mixed_Att = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_att_heads, batch_first=True)
        self.Mixed_FF = FeedForwardNetwork(d_model=hidden_size, out_dim=hidden_size)
        self.Mixed_Rezero = nn.Parameter(torch.zeros(hidden_size))
        self.Mixed_Rezero2 = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, Pos_embd, Traffic_embd, Scene_mask):
        # Cross Attention
        Att_mixed = self.Mixed_Att(Pos_embd, Traffic_embd, Traffic_embd, key_padding_mask = Scene_mask, need_weights=False)[0] #key_padding_mask = src_mask
        Mixed = self.Mixed_Rezero*Att_mixed + Pos_embd
        Mixed_FF = self.Mixed_FF(Mixed)
        Mixed = Mixed + self.Mixed_Rezero2*Mixed_FF
        Mixed = self.dropout(Mixed)
        return Mixed

class Encoder_DAAG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        num_heads = config['num_heads']
        self.xy_indx = config['xy_indx']
        self.Traffic_indx = config['Traffic_indx']
        # self.Linear_indx = [x for x in list(range(config['input_size'])) if x not in self.xy_indx + self.Traffic_indx]
        self.Linear_indx = config['Linear_indx']
        self.Positional_Encoding_Layer = Positional_Encoding_Layer(hidden_size = self.hidden_size, num_att_heads = num_heads, 
                                                                   xy_indx = self.xy_indx, pos_embedding_dim = config['pos_embedding_dim'], 
                                                                   pos_embedding_dict_size = config['pos_embedding_dict_size'], nnodes = config['Nnodes'], sl = config['sl']//config['dwn_smple'])
        self.Traffic_Encoding_Layer = Traffic_Encoding_Layer(hidden_size = self.hidden_size,
                                                             num_att_heads = num_heads, Traffic_indx = self.Traffic_indx, trf_embedding_dim = config['trf_embedding_dim'],
                                                             trf_embedding_dict_size = config['trf_embedding_dict_size'], Num_linear_inputs = len(self.Linear_indx),
                                                             Linear_indx = self.Linear_indx)
        self.Mixed_Att_Layer = Mixed_Attention_Layer(hidden_size = 3*self.hidden_size, num_att_heads = num_heads)


    def forward(self, Scene, Scene_mask, Adj_mat):
        Pos_embd, Leaky_Res = self.Positional_Encoding_Layer(Scene, Scene_mask, Adj_mat)
        Traffic_embd = self.Traffic_Encoding_Layer(Scene, Scene_mask, Leaky_Res)
        Mixed = self.Mixed_Att_Layer(Pos_embd, Traffic_embd, Scene_mask)
        return Mixed


class Projection(nn.Module):
    def __init__(self, hidden_size, output_size, output_dict_size, Nnodes):
        super(Projection, self).__init__()
        self.output_dict_size = output_dict_size
        self.Nnodes = Nnodes
        self.LN = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size*output_dict_size)
        self.output_size = output_size
        

    def forward(self, x):
        B, SL, _ = x.size()
        x = self.LN(x)
        x = self.linear2(x).reshape(B//self.Nnodes, self.Nnodes,  SL, self.output_size, self.output_dict_size).permute(0,2,1,3,4) # [B, SL, Nnodes, output_size, output_dict_size]
        return x


class HDAAGT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        output_size = len(config['xy_indx'])
        Nnodes = config['Nnodes']
        self.encoder = Encoder_DAAG(config)
        self.proj = Projection(3*self.hidden_size, output_size, output_dict_size= config['output_dict_size'], Nnodes= Nnodes)
    def forward(self, scene: torch.Tensor, src_mask, adj_mat: torch.Tensor):
        enc_out = self.encoder(scene, src_mask, adj_mat)
        proj = self.proj(enc_out)
        return proj

if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")