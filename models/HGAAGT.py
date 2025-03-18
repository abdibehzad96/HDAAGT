
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *
# from utilz.utils import target_mask , create_src_mask

class FFGAT(nn.Module):
    def __init__(self, args):
        s = args.hidden_size
    def edge_index(self, adj):
        edge_indx = torch.zeros_like(adj)
        indices = torch.nonzero(adj> 0.1, as_tuple=True)
        edge_indx[indices] = 1
        return edge_indx
    
    def convert_to_edge_list(self, adj):
        # Initialize an empty list to store the edge list
        edge_list = []
        # Iterate through the batch of the adjacency matrix
        for b in range(adj.shape[0]):
            # Iterate through the sequence of the adjacency matrix
            # for s in range(adj.shape[1]):
                Ixy = torch.nonzero(adj[b,0]>0.1, as_tuple=False).t()
                edge_list.append(Ixy)
        # Return the edge list
        return edge_list

class convdeep(nn.Module):
    def __init__(self, hidden_size):
        super(convdeep, self).__init__()
        self.conv1 = nn.Conv1d(hidden_size,hidden_size,3, 1, 1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size,5,1,2)
        self.conv3 = nn.Conv1d(hidden_size,hidden_size,7,1,3)
        self.conv4 = nn.Conv2d(3,1,)

    def forward(self, h):
        # x shape is [B, SL, F]
        x = h.permute(0,2,1)
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x))
        x3 = F.leaky_relu(self.conv3(x))
        x  = x1 + x2 + x3
        return x.permute(0,2,1)

class Classifier(nn.Module):
    def __init__(self, hidden_size, no = 1):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv1d(17,  17, 7, 2, 3)
        self.conv2 = nn.Conv1d(8,  17, 7, 2, 3)
        self.conv3 = nn.Conv1d(6,  17, 7, 2, 3)
        self.out = nn.Linear(hidden_size//2, hidden_size//2)
        self.LN = nn.LayerNorm(hidden_size)
        self.Rezero = nn.Parameter(torch.zeros(hidden_size//2))
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, h):
        # h = h.unsqueeze(1)
        x0 = F.relu(self.conv1(h))
        x1 = F.relu(self.conv2(h[:,1::2]))
        x2 = F.relu(self.conv3(h[:,1::3]))
        # x3 = F.relu(self.conv4(h[:,1::7]))
        # x3 = F.leaky_relu(self.conv4(h[:,::6]))
        x = self.dropout(x0 + x1 + x2)
        # x = self.LN(x)
        x = self.out(x)*self.Rezero + x
        return x

class deepFF(nn.Module):
    def __init__(self, input_size, output_size, n_heads):
        super(deepFF, self).__init__()
        self.GAT1 = DAAG_Layer(in_features=input_size, out_features=output_size, n_heads=n_heads, n_nodes=32,
                    concat=True)
        self.hidden_size = output_size
        
        # self.dropout = nn.Dropout(0.1)
    def forward(self, h, adj):
        B, SL, Nnodes, _ = h.size()
        # x = h + Spatial_encoding(h, self.hidden_size)
        x = self.GAT1(h, adj)
        return x

class TrfEmb(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(TrfEmb, self).__init__()
        self.embdx = nn.Embedding(1024, hidden_size, padding_idx=0)
        # self.emby = nn.Embedding(1024, hidden_size//2, padding_idx=0)
        self.embdTrL = nn.Embedding(9, hidden_size//8, padding_idx=0)
        self.embdZone = nn.Embedding(12, hidden_size//2, padding_idx=0)
        self.speed_HS = hidden_size #- 5*(hidden_size//8)
        self.speedLN = nn.LayerNorm(17)
        self.speedLN1 = nn.LayerNorm(self.speed_HS)
        self.speedLN2 = nn.LayerNorm(self.speed_HS)
        self.speed = nn.Linear(17, self.speed_HS )
        self.speed2 = nn.Linear(self.speed_HS, self.speed_HS)
        self.speed3 = nn.Linear(self.speed_HS, self.speed_HS)

        self.Att = nn.MultiheadAttention(embed_dim=3* hidden_size, num_heads=num_heads, batch_first=True)
        self.AttTrf = nn.MultiheadAttention(embed_dim= 3*hidden_size, num_heads=num_heads, batch_first=True)
        self.AttAll = nn.MultiheadAttention(embed_dim= 3*hidden_size, num_heads=num_heads, batch_first=True)
        
        # self.TemporalConv = convdeep(hidden_size)
        
        # self.TemporalFC = nn.Linear(3*hidden_size, hidden_size)
        self.FFTrfL = FeedForwardNetwork(d_model=3*hidden_size, out_dim=3*hidden_size)
        self.FFxy = FeedForwardNetwork(d_model=3*hidden_size, out_dim=3*hidden_size)
        self.FFAll = FeedForwardNetwork(d_model=3*hidden_size, out_dim=3*hidden_size)
        self.RezeroTrF = nn.Parameter(torch.zeros(3*hidden_size))
        self.RezeroXY = nn.Parameter(torch.zeros(3*hidden_size))
        self.RezeroALL = nn.Parameter(torch.zeros(3*hidden_size))
        self.RezeroALLFF = nn.Parameter(torch.zeros(3*hidden_size))
        self.RezeroSpeed = nn.Parameter(torch.zeros(self.speed_HS))
        self.RezeroSpeed2 = nn.Parameter(torch.zeros(self.speed_HS))
        self.hidden_size = hidden_size
        self.d_model = hidden_size
        self.DeepFF = deepFF(2*hidden_size, 2*hidden_size, n_heads=2*num_heads)
        self.TemporalConv = Classifier(2*hidden_size)
        self.Temporal = nn.LSTM(2*hidden_size, hidden_size, num_heads, batch_first=True)

        self.LN = nn.LayerNorm(hidden_size*3)
        self.LN1 = nn.LayerNorm(hidden_size)
        self.LNembx = nn.LayerNorm(3*hidden_size)
        

        self.dropout1 = nn.Dropout(0.25) 
        self.dropout2 = nn.Dropout(0.25)
        self.dropoutTrf = nn.Dropout(0.25)
    def forward(self, x, src_mask, adj, trgt_mask):
        B, Nnodes, SL, _ = x.size()
        embxy= self.embdx(x[:,:,:,1:3].long()).flatten(-2)

        # embxy = torch.cat((embx, emby), dim = -1)# .reshape(-1,x.size(2), self.hidden_size) # [B* Nnodes, SL, hidden_size]
        embTrf = self.embdTrL(x[:,:,:,3:7].long()).flatten(-2)
        embZone = self.embdZone(x[:,:,:,0].long())
        speed = x[:,:,:,7:]
        speed = self.speedLN(speed)
        speed1 = self.speed(speed)
        speed2 = self.RezeroSpeed* self.speed2(F.leaky_relu(speed1)) + speed1
        # speed2 = self.speedLN1(self.speed2(F.leaky_relu(speed1)) + speed1) 
        speed = self.RezeroSpeed2* self.speed3(F.leaky_relu(speed2)) + speed2
        # speed = self.speedLN2(self.speed3(F.leaky_relu(speed2)) + speed2)
        
        # h = embxy.reshape(B, Nnodes, SL, self.hidden_size)
        h = embxy.permute(0,2,1,3)  
        h = self.DeepFF(h, adj)
        embxy = h.permute(0,2,1,3).reshape(B*Nnodes, SL, 2*self.hidden_size)

        stlstm , _ = self.Temporal(embxy)
        convtemp = self.TemporalConv(embxy)

        Traffic_embd = torch.cat((embTrf,embZone, speed), dim=-1).reshape(-1, SL, 2*self.hidden_size)
        Traffic_embd = torch.cat((Traffic_embd, convtemp), dim =-1)
        Traffic_embd = self.dropoutTrf(Traffic_embd)
        

        Traffic_embd = Traffic_embd + positional_encoding(Traffic_embd,3*self.d_model)

        embxy= torch.cat((embxy,stlstm), dim=-1)
        # embxy = self.LN(embxy)
        # embxy = self.TemporalFC(embxy)

        
        embxy = embxy + positional_encoding(embxy, 3*self.d_model)



        TrfATT = self.AttTrf(Traffic_embd, Traffic_embd, Traffic_embd, key_padding_mask = src_mask)[0] #key_padding_mask = src_mask
        Traffic_embd = self.dropout2(self.RezeroTrF*TrfATT + Traffic_embd)
        Traffic_embd = self.FFTrfL(Traffic_embd)


        att_embxy = self.Att(embxy , embxy , embxy, need_weights=False, key_padding_mask = src_mask)[0] #key_padding_mask = src_mask

        embxy = self.RezeroXY*att_embxy + embxy

        # embxy = self.FFxy(embxy)
        embxy = self.dropout1(embxy)



        AttAll = self.AttAll(embxy, Traffic_embd, Traffic_embd, key_padding_mask = src_mask, need_weights=False)[0] #key_padding_mask = src_mask
        ALL_EMbedding = self.RezeroALL*AttAll + embxy



        ALL_EMbeddingFF = self.FFAll(ALL_EMbedding)
        # ALL_EMbedding = self.LNLast(ALL_EMbedding + ALL_EMbeddingFF)
        ALL_EMbedding = ALL_EMbedding + self.RezeroALLFF*ALL_EMbeddingFF
        # att = att.reshape(x.size(0), 32,20, self.hidden_size)
        # ALL_EMbedding = self.LN(ALL_EMbedding)
        # _ ,ALL_EMbedding = self.Temporal(ALL_EMbedding)

        return ALL_EMbedding


class TargetEmb(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(TargetEmb, self).__init__()
        self.embdx= nn.Embedding(1024, hidden_size//2, padding_idx=0)
        # self.embdy = nn.Embedding(1024, hidden_size//2, padding_idx=0)
        self.Att = nn.MultiheadAttention(embed_dim= hidden_size, num_heads=num_heads, batch_first=True)
        self.FF = FeedForwardNetwork(d_model=hidden_size, out_dim=hidden_size)
        self.ReZero = nn.Parameter(torch.zeros(hidden_size))
        self.hidden_size = hidden_size
        self.d_model = hidden_size
        self.n_head = num_heads

    def forward(self, trgt, trgt_mask, src_mask):
        trgtxy= self.embdx(trgt.long()).flatten(-2).reshape(-1,trgt.size(2), self.hidden_size)
        # trgty= self.embdy(trgt[:,:,:,1])
        # trgtxy = torch.cat((trgtx, trgty), dim=-1).reshape(-1,trgt.size(2), self.hidden_size)
        trgtxy = trgtxy + positional_encoding(trgtxy, self.d_model)
        att_trgt = self.Att(trgtxy, trgtxy, trgtxy,attn_mask=trgt_mask, need_weights=False)[0] #, attn_mask=trgt_mask, attn_mask=trgt_mask.repeat_interleave(self.n_head, dim=0)
        trgtxy = self.ReZero*att_trgt + trgtxy
        trgtxy = self.FF(trgtxy)
        return trgtxy
    
class Encoder(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.InitEmbed = TrfEmb(self.hidden_size, num_heads)

    def forward(self, scene, adj,src_mask, B, SL, Nnodes, trgt_mask):
        h = scene
        h = h.permute(0,2,1,3)
        adj_zero = torch.ones_like(adj[:,0]).unsqueeze(1)
        adj_mat = torch.cat((adj_zero, adj, adj_zero), dim=1)
        h = self.InitEmbed(h, src_mask, adj_mat, trgt_mask)
        return h
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.TargetEmb = TargetEmb(self.hidden_size, num_heads)
        self.AttAll = nn.MultiheadAttention(embed_dim= hidden_size, num_heads=num_heads, batch_first=True)
        self.ReZeroTrgt = nn.Parameter(torch.zeros(self.hidden_size))
        self.FF = FeedForwardNetwork(d_model=hidden_size, out_dim=hidden_size)
        self.ReZeroFF = nn.Parameter(torch.zeros(self.hidden_size))
        

    def forward(self, Target, trgt_mask, enc_out,src_mask, B, Nnodes, TrgLen):
        trgt = Target.permute(0,2,1,3) # [B, SL, Nnodes, 2] ---> [B, Nnodes, SL, 2]
        trgt = self.TargetEmb(trgt, trgt_mask, src_mask)
        Att = self.AttAll(trgt, enc_out, enc_out, need_weights=False)[0] #key_padding_mask = src_mask
        trgt = self.ReZeroTrgt*Att + trgt
        trgtFF = self.FF(trgt)
        trgt = trgt + self.ReZeroFF*trgtFF
        trgt = trgt.reshape(B, Nnodes, TrgLen, self.hidden_size).permute(0,2,1,3)
        return trgt

class Projection(nn.Module):
    def __init__(self, hidden_size, output_size, embedsize):
        super(Projection, self).__init__()
        self.LN = nn.LayerNorm(hidden_size)
        # self.linear1 = nn.Linear(hidden_size, hidden_size*2)
        self.linear2 = nn.Linear(hidden_size, output_size*embedsize)
        self.output_size = output_size
        self.embedsize = embedsize

    
    def forward(self, x):
        B, SL, Nnodes = x.size()
        # x = self.linear1(x)
        # x = F.gelu(x)
        x = self.LN(x)
        x = self.linear2(x).reshape(B//32, 32,  SL, self.output_size, self.embedsize).permute(0,2,1,3,4)
        return x


class GGAT(nn.Module):
    def __init__(self, args):
        super(GGAT, self).__init__()
        self.embed_size = args.input_size
        self.hidden_size = args.hidden_size
        self.d_model = args.hidden_size
        self.device = args.device
        self.num_layers = args.num_layersGRU
        self.output_size = args.output_size
        self.encoder = Encoder(self.hidden_size, num_heads=args.n_heads)
        # self.decoder = Decoder(self.hidden_size, num_heads=args.n_heads)
        self.proj = Projection(self.hidden_size*3, 2, 1024)
        
        #['BBx', 'BBy','W', 'L' , 'Cls','Tr1', 'Tr2', 'Tr3', 'Tr4', 'Zone']
    def forward(self, scene: torch.Tensor, src_mask, adj, Target, trgt_mask):
        B, SL, Nnodes, _ = scene.size()
        TrgL = Target.size(1)
        enc_out = self.encoder(scene, adj, src_mask, B, SL, Nnodes, trgt_mask)
        # h = self.decoder(Target,trgt_mask, enc_out,src_mask, B, Nnodes, TrgL)
        h = self.proj(enc_out)
        return h



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