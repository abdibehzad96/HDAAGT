import torch
from torch import nn
import torch.nn.functional as F


################################
###         GAT LAYER        ###
################################

class BatchedGraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.

        This operation can be mathematically described as:

            e_ij = a(W h_i, W h_j)
            α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k(exp(e_ik))     
            h_i' = σ(Σ_j(α_ij W h_j))
            
            where h_i and h_j are the feature vectors of nodes i and j respectively, W is a learnable weight matrix,
            a is an attention mechanism that computes the attention coefficients e_ij, and σ is an activation function.

    """
    def __init__(self, in_features: int, out_features: int, n_heads: int, n_nodes: int, concat: bool = False, dropout: float = 0.4, leaky_relu_slope: float = 0.2):
        super(BatchedGraphAttentionLayer, self).__init__()

        self.n_heads = n_heads # Number of attention heads
        self.concat = concat # wether to concatenate the final attention heads
        self.dropout = dropout # Dropout rate
        self.n_nodes = n_nodes # Number of nodes in the graph
        self.training = False # Training mode
        if concat: # concatenating the attention heads
            self.out_features = out_features # Number of output features per node
            assert out_features % n_heads == 0 # Ensure that out_features is a multiple of n_heads
            self.n_hidden = out_features // n_heads
        else: # averaging output over the attention heads (Used in the main paper)
            self.n_hidden = out_features

        #  A shared linear transformation, parametrized by a weight matrix W is applied to every node
        #  Initialize the weight matrix W 
        # self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))
        self.W = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)



        # Initialize the attention weights a
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))

        self.leakyrelu = nn.LeakyReLU(negative_slope=leaky_relu_slope) # LeakyReLU activation function
        self.softmax = nn.Softmax(dim=3) # softmax activation function to the attention coefficients
        self.sigmoid = nn.Sigmoid() # sigmoid activation function for the adjacency matrix
        self.socialpool = nn.Conv2d(20, 20, kernel_size=n_nodes, stride=1, padding=0, bias=False)
        self.reset_parameters() # Reset the parameters


    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_normal_(self.a)
        # nn.init.xavier_normal_(self.learn_adj)
    

    def _get_attention_scores(self, h_transformed: torch.Tensor): # this attention is along the features of a single node
        """calculates the attention scores e_ij for all pairs of nodes (i, j) in the graph
        in vectorized parallel form. for each pair of source and target nodes (i, j),
        the attention score e_ij is computed as follows:

            e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) 

            where || denotes the concatenation operation, and a and W are the learnable parameters.

        Args:
            h_transformed (torch.Tensor): Transformed feature matrix with shape (n_nodes, n_heads, n_hidden),
                where n_nodes is the number of nodes and out_features is the number of output features per node.

        Returns:
            torch.Tensor: Attention score matrix with shape (n_heads, n_nodes, n_nodes), where n_nodes is the number of nodes.
        """
        
        source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])

        target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])


        # broadcast add 
        # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
        e = source_scores + target_scores.mT
        return self.sigmoid(e)

    def forward(self,  h: torch.Tensor, adj_mat: torch.Tensor):
        """
        Performs a graph attention layer operation.

        Args:
            h (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """
        n_nodes = h.shape[2] # was h.size(0)
        batch_size = h.shape[0]
        sl = h.shape[1]
        
        # Apply linear transformation to node feature -> W h
        # output shape (n_nodes, n_hidden * n_heads)
        # h_transformed = torch.matmul(h, self.W)
        h_transformed = self.W(h)
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)

        # splitting the heads by reshaping the tensor and putting heads dim first
        # output shape (batch, sl, n_heads, n_nodes, n_hidden)
        h_transformed = h_transformed.view(batch_size, sl, n_nodes, self.n_heads, self.n_hidden).permute(0, 1, 3, 2, 4) # was .view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)
        
        # getting the attention scores
        # output shape (batch, sl, n_heads, n_nodes, n_nodes)
        e = self._get_attention_scores(h_transformed)
        # th = self.threshold * torch.mean(e)
        # Set the attention score for non-existent edges to -9e15 (MASKING NON-EXISTENT EDGES)
        # rep_adj_mat = adj_mat.unsqueeze(2).repeat(1,1,self.n_heads,1,1)
        # connectivity_mask = -1e15 * torch.ones_like(e)
        # threshold = self.socialpool(adj_mat)
        threshold = adj_mat < 0.05
        e = e.masked_fill(threshold.unsqueeze(2), -9e15)
        # e = torch.where(rep_adj_mat > self.threshold, e, connectivity_mask) # masked attention scores
        # attention coefficients are computed as a softmax over the rows
        # for each column j in the attention score matrix e

        attention = F.softmax(e, dim=-1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        
        # final node embeddings are computed as a weighted average of the features of its neighbors
        # output shape (batch, sl, n_heads, n_nodes, n_hidden)
        h_prime = torch.matmul(attention, h_transformed)

        # concatenating/averaging the attention heads
        # output shape (batch, sl, n_nodes, out_features)
        if self.concat:
            h_prime = h_prime.contiguous().view(batch_size, sl, n_nodes, self.out_features)
        else:
            h_prime = h_prime.mean(dim=2)

        return h_prime
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, out_dim):
            super().__init__()

            self.linear1 = nn.Linear(d_model, out_dim)
            self.relu = nn.LeakyReLU(negative_slope=0.01)
            self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))


class NewGATCell(nn.Module):
    """
    Graph Attention Layer (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.

        This operation can be mathematically described as:

            e_ij = a(W h_i, W h_j)
            α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k(exp(e_ik))     
            h_i' = σ(Σ_j(α_ij W h_j))
            
            where h_i and h_j are the feature vectors of nodes i and j respectively, W is a learnable weight matrix,
            a is an attention mechanism that computes the attention coefficients e_ij, and σ is an activation function.

    """
    def __init__(self, in_features: int, out_features: int, n_heads: int, n_nodes: int, concat: bool = False, dropout: float = 0.14, leaky_relu_slope: float = 0.2):
        super(NewGATCell, self).__init__()

        self.n_heads = n_heads # Number of attention heads
        self.concat = concat # wether to concatenate the final attention heads
        self.dropout = dropout # Dropout rate
        self.n_nodes = n_nodes # Number of nodes in the graph
        self.training = False # Training mode
        if concat: # concatenating the attention heads
            self.out_features = out_features # Number of output features per node
            assert out_features % n_heads == 0 # Ensure that out_features is a multiple of n_heads
            self.n_hidden = out_features // n_heads
        else: # averaging output over the attention heads (Used in the main paper)
            self.n_hidden = out_features

        #  A shared linear transformation, parametrized by a weight matrix W is applied to every node
        #  Initialize the weight matrix W 
        self.W = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        

        # Initialize the attention weights a
        self.a = nn.Linear(2*self.n_hidden, 1, bias=False)

        self.socialpool = nn.Conv1d(n_nodes,n_nodes, kernel_size=n_nodes, stride=1, padding=0)
        # self.socialpool = nn.Linear(n_nodes, n_nodes, bias=False)
        self.tanh = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1*leaky_relu_slope) # LeakyReLU activation function
        self.softmax = nn.Softmax(dim=3) # softmax activation function to the attention coefficients
        self.sigmoid = nn.Sigmoid() # sigmoid activation function for the adjacency matrix

    def forward(self,  h: torch.Tensor, adj_mat: torch.Tensor):
        """
        Performs a graph attention layer operation.

        Args:
            h (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """

        n_nodes = h.shape[2] # was h.size(0)
        batch_size = h.shape[0]
        sl = h.shape[1]
        
        # Apply linear transformation to node feature -> W h
        # output shape (n_nodes, n_hidden * n_heads)
        # h_transformed = torch.matmul(h, self.W)
        h_transformed = self.W(h)
        # h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)

        # splitting the heads by reshaping the tensor and putting heads dim first
        # output shape (batch, sl, n_heads, n_nodes, n_hidden)
        h_transformed = h_transformed.view(batch_size, sl, n_nodes, self.n_heads, self.n_hidden)
        h_tr_repeat = h_transformed.repeat(1,1,n_nodes,1,1)
        h_rep_interleave = h_transformed.repeat_interleave(n_nodes, dim=2)

        h_concat = torch.cat((h_tr_repeat, h_rep_interleave), dim=-1)
        h_concat = h_concat.view(batch_size, sl, n_nodes, n_nodes, self.n_heads, 2*self.n_hidden)

        e = self.leakyrelu(self.a(h_concat)).squeeze(-1)


        # rep_adj_mat = adj_mat.unsqueeze(4)
        threshold = self.socialpool(adj_mat.mean(dim = 1))
        # threshold = adj_mat < 0.05
        threshold = adj_mat < threshold.unsqueeze(1)
        e = e.masked_fill(threshold.unsqueeze(4), -9e15)
        # e = e*(adj_mat).unsqueeze(4)

        a = self.softmax(e)
        # a = F.dropout(a, self.dropout, training=self.training)

        att_res = torch.einsum('bsijh,bsjhf->bsihf', a, h_transformed)


        if self.concat:
            att_res = att_res.reshape(batch_size, sl, n_nodes, self.out_features)
        else:
            att_res = att_res.mean(dim=3)

        return att_res
    

class GraphAttentionV2Layer(nn.Module):
    r"""
    ## Graph attention v2 layer
    This is a single graph attention v2 layer.
    A GATv2 is made up of multiple such layers.
    It takes
    $$\mathbf{h} = \{ \overrightarrow{h_1}, \overrightarrow{h_2}, \dots, \overrightarrow{h_N} \}$$,
    where $\overrightarrow{h_i} \in \mathbb{R}^F$ as input
    and outputs
    $$\mathbf{h'} = \{ \overrightarrow{h'_1}, \overrightarrow{h'_2}, \dots, \overrightarrow{h'_N} \}$$,
    where $\overrightarrow{h'_i} \in \mathbb{R}^{F'}$.
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

        # self.socialpool = nn.Conv2d(32,32, kernel_size=(1,32), stride=1, padding=0)    # This worked perfectly
        # self.adjAtt = nn.MultiheadAttention(embed_dim= n_nodes, num_heads=n_heads, batch_first=True) # we want to train the attention weights
    def forward(self, h0: torch.Tensor, adj_mat: torch.Tensor):
        r"""
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        B, SL, n_nodes, _ = h0.shape
        adj_mat = adj_mat - torch.eye(n_nodes).to(adj_mat.device).repeat(B, SL, 1, 1)
        adj_mat = adj_mat.unsqueeze(-2) < 0.1
        # h = h.reshape(-1, 32, 128)
        # The initial transformations,
        # $$\overrightarrow{{g_l}^k_i} = \mathbf{W_l}^k \overrightarrow{h_i}$$
        # $$\overrightarrow{{g_r}^k_i} = \mathbf{W_r}^k \overrightarrow{h_i}$$
        # for each head.
        # We do two linear transformations and then split it up for each head.
        # h = self.LN(h0)
        h = h0
        q = self.linear_l(h).view(B, SL, n_nodes, self.n_heads, self.n_hidden)
        k = self.linear_r(-h).view(B, SL, n_nodes, self.n_heads, self.n_hidden)
        v = self.linear_v(h).view(B, SL, n_nodes, self.n_heads, self.n_hidden)
        score = torch.einsum("bsihf, bsjhf->bsihj", [q, k])/ ((self.n_hidden*self.n_heads*n_nodes)**0.5)
        score = score.masked_fill(adj_mat, -1e6)
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