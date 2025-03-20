import torch
from torch import nn
import torch.nn.functional as F


################################
###         FFN LAYER       ###
################################

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, out_dim):
            super().__init__()

            self.linear1 = nn.Linear(d_model, out_dim)
            self.relu = nn.LeakyReLU(negative_slope=0.01)
            self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))


################################
###         DAAG LAYER       ###
################################
class DAAG_Layer(nn.Module):
    r"""
    ## Decision-Aware Attention Graph Transformer (DAAGT) Layer
    This is a single graph attention v2 layer.
    author: @Behzad Abdi
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int, n_nodes: int,
                concat: bool = True,
                dropout: float = 0.01,
                leaky_relu_negative_slope: float = 0.1,
                share_weights: bool = False):
        r"""
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        * `share_weights` if set to `True`, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()

        self.is_concat = concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        self.n_nodes = n_nodes

        # Calculate the number of dimensions per head
        if concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads,  bias=False)
        
        self.linear_v = nn.Linear(in_features, self.n_hidden * n_heads,  bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.att = nn.Linear(n_nodes + self.n_hidden, self.n_hidden)
        self.att_mh_1 = nn.Parameter(torch.Tensor(1, n_heads, n_nodes))
        self.scoreatt = nn.Linear(1,n_nodes)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.FF = FeedForwardNetwork(self.n_hidden, self.n_hidden)
        self.Rezero = nn.Parameter(torch.zeros(self.n_hidden))
        self.LN = nn.LayerNorm(n_nodes)
    def forward(self, h: torch.Tensor, Adj_mat: torch.Tensor):
        r"""
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        B, SL, n_nodes, _ = h.shape
        Adj_mat = Adj_mat - torch.eye(n_nodes).to(Adj_mat.device).repeat(B, SL, 1, 1)
        Adj_mat = Adj_mat.unsqueeze(-2) < 0.1
        q = self.linear_l(h).view(B, SL, n_nodes, self.n_heads, self.n_hidden)
        k = self.linear_r(-h).view(B, SL, n_nodes, self.n_heads, self.n_hidden)
        v = self.linear_v(h).view(B, SL, n_nodes, self.n_heads, self.n_hidden)
        score = torch.einsum("bsihf, bsjhf->bsihj", [q, k])/ ((self.n_hidden*self.n_heads*n_nodes)**0.5)
        score = score.masked_fill(Adj_mat, -1e6)
        score = torch.exp(score).sum(-1).unsqueeze(-1)
        score = self.scoreatt(score)
        score = self.LN(score)
        score = F.softmax(score, dim=-1)
        score = torch.cat((score, v), dim=-1)
        score = self.att(score)
        scoreFF = self.FF(score)
        score = score + self.Rezero * scoreFF
        
        if self.is_concat:
            return score.flatten(-2)
        else:
            return score.mean(-2)
        

class TemporalConv(nn.Module):
    def __init__(self, hidden_size, no = 1):
        super(TemporalConv, self).__init__()
        self.conv1 = nn.Conv1d(17,  17, 7, 2, 3)
        self.conv2 = nn.Conv1d(8,  17, 7, 2, 3)
        self.conv3 = nn.Conv1d(6,  17, 7, 2, 3)
        self.out = nn.Linear(hidden_size//2, hidden_size//2)
        self.LN = nn.LayerNorm(hidden_size)
        self.Rezero = nn.Parameter(torch.zeros(hidden_size//2))
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, h):
        x0 = F.relu(self.conv1(h))
        x1 = F.relu(self.conv2(h[:,1::2]))
        x2 = F.relu(self.conv3(h[:,1::3]))
        x = self.dropout(x0 + x1 + x2)
        x = self.out(x)*self.Rezero + x
        return x
    
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

def target_mask(trgt, num_head = 0, device="cuda:3"):
    B, SL, Nnodes, _= trgt.size()
    mask = 1- torch.triu(torch.ones(SL, SL, device=trgt.device), diagonal=1)# Upper triangular matrix
    return mask == 0

def target_mask0(trgt, num_head, device="cuda:3"):
    B, SL, Nnodes, _= trgt.size()
    mask = 1- torch.triu(torch.ones(SL, SL, device=trgt.device), diagonal=1).unsqueeze(0).unsqueeze(0)  # Upper triangular matrix
    mask = mask.unsqueeze(4) * (trgt[:,:,:,1] != 0).unsqueeze(1).unsqueeze(3)
    if num_head > 1:
        mask = mask.repeat_interleave(num_head,dim=0)
    return mask == 0

def create_src_mask(src): # src => [B, SL, Nnodes, Features]
    mask = src[:,:,:, 1] == 0
    return mask.permute(0,2,1).reshape(-1, src.size(1)) # out => [B* Nnodes, SL]

def attach_sos_eos(src, sos, eos): # src => [B, SL0, Nnodes, Features]
    return torch.cat((sos.repeat(src.size(0),1,1,1), src, eos.repeat(src.size(0),1,1,1)), dim=1) # out => [B, SL0+2, Nnodes, Features]

