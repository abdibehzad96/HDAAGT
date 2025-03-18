import torch
import torch.nn as nn
import torch.nn.functional as F




def parallel_scan_log(log_coeffs, log_values):
    # log_coeffs: (batch_size, seq_len, input_size)
    # log_values: (batch_size, seq_len + 1, input_size)
    a_star = F.pad(torch.cumsum(log_coeffs, dim=-1), (0,0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star 
    return torch.exp(log_h[:,1:])

def parallel_scan(coeffs, values):
    # coeffs: (batch_size, seq_len, input_size)
    # values: (batch_size, seq_len + 1, input_size)
    a_star = F.pad(torch.cumsum(coeffs, dim=-1), (0,0, 1, 0))
    h0_plus_b_star = torch.cumsum(values - a_star, dim=1)
    h = a_star + h0_plus_b_star
    return h[:,1:]

def log_g(x):
    return torch.where(x >= 0, (F.relu(x)+0.5).log(), -F.softplus(-x, threshold=1))

def g(x):
    return torch.where(x >= 0, x+0.5, torch.sigmoid(x))


class MinGRUcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinGRUcell, self).__init__()
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)


    def forward(self, x, h0, Nseq = False):
        # x: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)
        if Nseq is False:
            z = torch.sigmoid(self.linear_z(x))
            tilde_h = self.linear_h(x)
            h =  parallel_scan((1 - z), torch.cat([h0, z * tilde_h], dim=1))
            return h, h[:,-1,:].unsqueeze(1)
        else:
            z = torch.sigmoid(self.linear_z(x))
            tilde_h = self.linear_h(x)
            h = z * tilde_h + (1 - z) *h0 
        return h, h
    


class MinGRUcell_log(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinGRUcell_log, self).__init__()
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, h0, Nseq = False):
        # x: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)
        if Nseq is False:
            k = self.linear_z(x)
            log_z = -F.softplus(-k)
            log_coeffs = -F.softplus(k)
            log_h_0 = log_g(h0)
            log_tilde_h = log_g(self.linear_h(x))
            h = parallel_scan_log(log_coeffs,torch.cat([log_h_0, log_z + log_tilde_h], dim=1))
            return h, h[:,-1,:].unsqueeze(1)
        else:
            z = torch.sigmoid(self.linear_z(x))
            tilde_h = g(self.linear_h(x))
            h = z * tilde_h + (1 - z) *h0 
        return h, h
    



class minLSTMPar(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(minLSTMPar, self).__init__()
        self.linear_f = nn.Linear(input_size, hidden_size)
        self.linear_i = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def forward(self, x, h_0, seq = None):
        # x: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)
        if seq is None:
            diff = F.softplus(-self.linear_f(x)) - F.softplus(-self.linear_i(x))
            log_f = -F.softplus(diff)
            log_i = -F.softplus(-diff)
            log_h_0 = log_g(h_0)
            log_tilde_h = log_g(self.linear_h(x))
            h = parallel_scan_log(log_f, torch.cat([log_h_0, log_i + log_tilde_h], dim=1))
        else:
            f_t = torch.sigmoid(self.linear_f(x))
            i_t = torch.sigmoid(self.linear_i(x))
            tilde_h_t = g(self.linear_h(x))
            f_prime_t = f_t / (f_t + i_t)
            i_prime_t = i_t / (f_t + i_t)
            h = f_prime_t * h_0 + i_prime_t * tilde_h_t
        return h
    
#### Big Attention:: The next thing to do is to make the sequence prediction to xt-1 --(predict)--> xt not this style ###### Done



    # def forward(self, x, h_0):
    #     # x: (batch_size, seq_len, input_size)
    #     # h_0: (batch_size, 1, hidden_size)
    #     i = torch.sigmoid(self.linear_i(x))
    #     tilde_h = self.linear_h(x)
    #     f_prime = f / (f + i)
    #     i_prime = i / (f + i)
    #     h = parallel_scan(f_prime,
    #     torch.cat([h_0, i_prime * tilde_h], dim=1))
    #     return h
class MinLSTMCell(nn.Module):
    def __init__(self, units, input_shape):
        super(MinLSTMCell, self).__init__()
        self.units = units
        self.input_shape = input_shape
        
        # Initialize the linear layers for the forget gate, input gate, and hidden state transformation
        self.linear_f = nn.Linear(self.input_shape, self.units)
        self.linear_i = nn.Linear(self.input_shape, self.units)
        self.linear_h = nn.Linear(self.input_shape, self.units)

    def forward(self, pre_h, x_t):
        """
        pre_h: (batch_size, units) - previous hidden state (h_prev)
        x_t: (batch_size, input_size) - input at time step t
        """

        # Forget gate: f_t = sigmoid(W_f * x_t)
        f_t = torch.sigmoid(self.linear_f(x_t))  # (batch_size, units)

        # Input gate: i_t = sigmoid(W_i * x_t)
        i_t = torch.sigmoid(self.linear_i(x_t))  # (batch_size, units)

        # Hidden state: tilde_h_t = W_h * x_t
        tilde_h_t = self.linear_h(x_t)  # (batch_size, units)

        # Normalize the gates
        sum_f_i = f_t + i_t
        f_prime_t = f_t / sum_f_i  # (batch_size, units)
        i_prime_t = i_t / sum_f_i  # (batch_size, units)

        # New hidden state: h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t
        h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t  # (batch_size, units)

        return h_t  # (batch_size, units)
    
