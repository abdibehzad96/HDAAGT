
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *

class TrafficEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        num_heads = config['num_heads']
        self.xy_indx = config['xy_indx']
        self.Traffic_indx = config['Traffic_indx']
        self.Linear_indx = torch.arange(len(config['NFeatures']), config['input_size'])
        self.Postional_embeddings = nn.ModuleList()
        self.Traffic_embeddings = nn.ModuleList()
        for i in config['Traffic_indx']:
            self.Traffic_embeddings.append(nn.Embedding(config['Embedding_dict_size'][i], config['Embedding_model_dim'][i]), padding_idx=0)
        for i in config['xy_indx']:
            self.Postional_embeddings.append(nn.Embedding(config['Embedding_dict_size'][i], config['Embedding_model_dim'][i]), padding_idx=0)


        # For the Linear embeddings, we consider the followinng fully connected layers
        self.LE_LN1 = nn.LayerNorm(config['Num_linear_inputs'])
        self.Linear_Embedding1 = nn.Linear(config['input_size'], self.hidden_size)
        self.LE_LN2 = nn.LayerNorm( self.hidden_size)
        self.Linear_Embedding2 = nn.Linear( self.hidden_size, self.hidden_size)
        self.LE_LN3 = nn.LayerNorm( self.hidden_size)
        self.Linear_Embedding3 = nn.Linear( self.hidden_size, self.hidden_size)

        # 3 Multihead attention layers for the position, traffic and mixed embeddings
        self.Position_Att = nn.MultiheadAttention(embed_dim=3* self.hidden_size, num_heads=num_heads, batch_first=True)
        self.Traffic_Att = nn.MultiheadAttention(embed_dim= 3* self.hidden_size, num_heads=num_heads, batch_first=True)
        self.Mixed_Att = nn.MultiheadAttention(embed_dim= 3* self.hidden_size, num_heads=num_heads, batch_first=True)

        self.Traffic_FF = FeedForwardNetwork(d_model=3* self.hidden_size, out_dim=3* self.hidden_size)
        self.Pos_FF = FeedForwardNetwork(d_model=3* self.hidden_size, out_dim=3* self.hidden_size)
        self.Mixed_FF = FeedForwardNetwork(d_model=3* self.hidden_size, out_dim=3* self.hidden_size)


        self.LE_Rezero2 = nn.Parameter(torch.zeros(self.hidden_size))
        self.LE_Rezero3 = nn.Parameter(torch.zeros(self.hidden_size))
        self.Traffic_Rezero = nn.Parameter(torch.zeros(3* self.hidden_size))
        self.Position_Rezero = nn.Parameter(torch.zeros(3* self.hidden_size))
        self.Mixed_Rezero = nn.Parameter(torch.zeros(3* self.hidden_size))
        self.Mixed_Rezero2 = nn.Parameter(torch.zeros(3* self.hidden_size))
        
        self.DAAG = DAAG_Layer(in_features=2*self.hidden_size, out_features=2*self.hidden_size, n_heads= num_heads, n_nodes = config['Nnodes'], concat=True)

        self.TemporalConv = Classifier(2* self.hidden_size)
        self.Temporal = nn.LSTM(2* self.hidden_size,  self.hidden_size, num_heads, batch_first=True)

        self.LN = nn.LayerNorm( self.hidden_size*3)
        self.LN1 = nn.LayerNorm( self.hidden_size)
        self.LNembx = nn.LayerNorm(3* self.hidden_size)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, src_mask, adj):
        x = x.permute(0,2,1,3)
        B, Nnodes, SL, _ = x.size()


        # Embeddings for the position and traffic
        positional_embedding = []
        Traffic_embedding = []
        for i in self.xy_indx:
            positional_embedding.append(self.Postional_embeddings[i](x[:,:,:,i].long()))
        for i in self.Traffic_indx:
            Traffic_embedding.append(self.Traffic_embeddings[i](x[:,:,:,i].long()))
        Pos_embd = torch.cat(positional_embedding, dim=-1)
        Trf_embd = torch.cat(Traffic_embedding, dim=-1)

        Lin_embd = self.LE_LN1(x[:,:,:,self.Linear_indx])
        Lin_embd1 = self.Linear_Embedding1(Lin_embd)
        Lin_embd2 = self.LE_Rezero2* self.Linear_Embedding2(F.leaky_relu(Lin_embd1)) + Lin_embd1
        Lin_embd = self.LE_Rezero3* self.Linear_Embedding3(F.leaky_relu(Lin_embd2)) + Lin_embd2

        Pos_embd = Pos_embd.permute(0,2,1,3)  
        Pos_embd = self.DAAG(Pos_embd, adj)
        Pos_embd = Pos_embd.permute(0,2,1,3).reshape(B*Nnodes, SL, 2*self.hidden_size)

        stlstm , _ = self.Temporal(Pos_embd)
        convtemp = self.TemporalConv(Pos_embd)

        Traffic_embd = torch.cat((Trf_embd, Lin_embd), dim=-1).reshape(-1, SL, 2*self.hidden_size)
        Traffic_embd = torch.cat((Traffic_embd, convtemp), dim =-1)
        Traffic_embd = self.dropout(Traffic_embd)
        
        Traffic_embd = Traffic_embd + positional_encoding(Traffic_embd, 3*self.hidden_size)
        Pos_embd= torch.cat((Pos_embd,stlstm), dim=-1)
        Pos_embd = Pos_embd + positional_encoding(Pos_embd, 3*self.hidden_size)
        TrfATT = self.Traffic_Att(Traffic_embd, Traffic_embd, Traffic_embd, key_padding_mask = src_mask)[0] #key_padding_mask = src_mask
        Traffic_embd = self.dropout(self.Traffic_Rezero*TrfATT + Traffic_embd)
        Traffic_embd = self.Traffic_FF(Traffic_embd)
        att_embxy = self.Position_Att(Pos_embd , Pos_embd , Pos_embd, need_weights=False, key_padding_mask = src_mask)[0] #key_padding_mask = src_mask
        Pos_embd = self.Position_Rezero*att_embxy + Pos_embd
        Pos_embd = self.dropout(Pos_embd)
        AttAll = self.Mixed_Att(Pos_embd, Traffic_embd, Traffic_embd, key_padding_mask = src_mask, need_weights=False)[0] #key_padding_mask = src_mask
        ALL_EMbedding = self.Mixed_Rezero*AttAll + Pos_embd
        ALL_EMbeddingFF = self.Mixed_FF(ALL_EMbedding)
        ALL_EMbedding = ALL_EMbedding + self.Mixed_Rezero2*ALL_EMbeddingFF
        return ALL_EMbedding


class Projection(nn.Module):
    def __init__(self, hidden_size, output_size, embedsize):
        super(Projection, self).__init__()
        self.LN = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size*embedsize)
        self.output_size = output_size
        self.embedsize = embedsize

    def forward(self, x):
        B, SL, _ = x.size()
        x = self.LN(x)
        x = self.linear2(x).reshape(B//32, 32,  SL, self.output_size, self.embedsize).permute(0,2,1,3,4)
        return x


class HDAAGT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.encoder = TrafficEmbedding(config)
        self.proj = Projection(self.hidden_size*3, 2, 1024)
    def forward(self, scene: torch.Tensor, src_mask, adj_mat: torch.Tensor):
        enc_out = self.encoder(scene, src_mask, adj_mat)
        DAAG_out = self.DAAG(enc_out, adj_mat)
        proj = self.proj(DAAG_out)
        return proj



def greedy(model, scene, adj, Target, Nseq, device):
        B, SL, Nnodes, _ = scene.size()
        scene_mask = create_src_mask(scene)
        scene_mask = scene_mask.permute(0,2,1)
        src_mask = scene_mask.reshape(-1, SL)
        trgt_mask = target_mask(Target,16)
        # trgt_mask = trgt_mask.permute(0,4,1,2,3)
        # trgt_mask = trgt_mask.reshape(-1, Target.size(1), Target.size(1))

        enc_out = model.encoder(scene, adj, src_mask, B, scene.size(1), Nnodes)

        
        # output= F.one_hot(pred_trgt.to(torch.int64), num_classes=1024).long().to(self.device)
        output =torch.empty(B, 0, Nnodes, 2).to(device)
        for i in range(Nseq):
            trgt_mask = target_mask(Target,16)
            # trgt_mask = trgt_mask.permute(0,4,1,2,3)
            # trgt_mask = trgt_mask.reshape(-1, i+1, i+1)
            Nxt_trgt = model.decoder(Target, trgt_mask, enc_out, src_mask, B, Nnodes, i+1)
            Nxt_trgt = model.proj(Nxt_trgt[:,-1].unsqueeze(1))
            next_word = torch.softmax(Nxt_trgt, dim=-1)
            topk_val, topk_indices = torch.topk(next_word, 5, dim=-1)
            avg_next = (topk_val*topk_indices).sum(-1)/topk_val.sum(-1)
            # next_word = torch.argmax(next_word, dim=-1)
            # next_word = topk_indices[:,:,:,:,0]
            # Nxt_trgt = F.one_hot(next_word.to(torch.int64), num_classes=1024).float().to(device)
            Target = torch.cat((Target, avg_next), dim=1)
            output = torch.cat((output,  topk_indices[:,:,:,:,0]), dim=1)
            # output = torch.cat((output, Nxt_trgt), dim=1)
        return Target[:,1:], output


if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")