# import
import torch
import torch.nn as nn
import math

# RNN
class Elman_RNNCell(nn.Module):
    """
    Elman_RNNCell.

    h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)

    @ : matrix multiplication, * : element-wise product
    """
    def __init__(self, input_size, hidden_size):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = nn.Parameter(torch.empty(input_size, hidden_size)) 
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size)) 
        self.b_h  = nn.Parameter(torch.empty(hidden_size))

        self.init_weights()
        
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x_t, h_prev):
        h_next = torch.tanh(
            torch.matmul(x_t, self.W_xh) +
            torch.matmul(h_prev, self.W_hh) +
            self.b_h
        )
        return h_next

class Elman_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = Elman_RNNCell(input_size, hidden_size)

        self.fc  = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, X):
        # (Batch, Sequence_Length, Feature)
        batch_size, seq_len, _ = X.shape 

        # init hidden tensor of t=0
        h_t = torch.zeros(batch_size, self.hidden_size).to(X.device)

        # Forward pass
        for t in range(seq_len):
            x_t = X[:, t, :] # Batch, Feature
            h_t = self.rnn_cell(x_t, h_t)
        
        out = self.fc(h_t)
        return out

# LSTM
class LSTM_Cell(nn.Module):
    """
    LSTM_Cell.

    f_t = sigmoid(x_t @ W_xf + h_{t-1} @ W_hf + b_f)
    i_t = sigmoid(x_t @ W_xi + h_{t-1} @ W_hi + b_i)
    o_t = sigmoid(x_t @ W_xo + h_{t-1} @ W_ho + b_o)
    c_candidates = tanh(x_t @ W_xc + h_{t-1} @ W_hc + b_c)
    c_t = f_t * c_{t-1} + i_t * c_candidates
    h_t = o_t * tanh(c_t)

    @ : matrix multiplication, * : element-wise product
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget Gate
        self.W_hf = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xf = nn.Parameter(torch.empty(input_size, hidden_size))
        self.b_f = nn.Parameter(torch.empty(hidden_size))
        # Input Gate
        self.W_hi = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xi = nn.Parameter(torch.empty(input_size, hidden_size))
        self.b_i = nn.Parameter(torch.empty(hidden_size))
        # Output Gate
        self.W_ho = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xo = nn.Parameter(torch.empty(input_size, hidden_size))
        self.b_o = nn.Parameter(torch.empty(hidden_size))
        # Cell Candidates
        self.W_hc = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xc = nn.Parameter(torch.empty(input_size, hidden_size))
        self.b_c = nn.Parameter(torch.empty(hidden_size))
        
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x_t, h_prev, c_prev):

        # Forget Gate
        f_t = torch.matmul(h_prev, self.W_hf) + torch.matmul(x_t, self.W_xf) + self.b_f
        f_t = torch.sigmoid(f_t)
        # Input Gate
        i_t = torch.matmul(h_prev, self.W_hi) + torch.matmul(x_t, self.W_xi) + self.b_i
        i_t = torch.sigmoid(i_t)
        # Output Gate
        o_t = torch.matmul(h_prev, self.W_ho) + torch.matmul(x_t, self.W_xo) + self.b_o
        o_t = torch.sigmoid(o_t)
        # Cell Candidates
        c_candidates = torch.matmul(h_prev, self.W_hc) + torch.matmul(x_t, self.W_xc) + self.b_c
        c_candidates = torch.tanh(c_candidates)

        # C Next
        c_t = f_t * c_prev + i_t * c_candidates
        # H Next
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTM_Cell(input_size, hidden_size)

        # Fully Connected Layer
        self.fc  = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
    
    def forward(self, X):
        # (Batch, Sequence_Length, Feature)
        batch_size, seq_len, _ = X.shape

        # init hidden_state, cell_state of t=0
        h_t = torch.zeros(batch_size, self.hidden_size).to(X.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(X.device)

        for t in range(seq_len):
            x_t = X[:, t, :] # Batch, Feature
            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)

        out = self.fc(h_t)
        return out

# GRU
class GRU_Cell(nn.Module):
    """
    GRU_Cell.

    z_t = sigmoid(x_t @ W_xz + h_{t-1} @ W_hz + b_z)
    r_t = sigmoid(x_t @ W_xr + h_{t-1} @ W_hr + b_r)
    h_tilde = tanh(x_t @ W_xh + (r_t * h_{t-1}) @ W_hh + b_h)
    h_t = z_t * h_tilde + (1 - z_t) * h_{t-1}

    @ : matrix multiplication, * : element-wise product
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update Gate
        self.W_xz = nn.Parameter(torch.empty(input_size, hidden_size))
        self.W_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_z = nn.Parameter(torch.empty(hidden_size))

        # Reset Gate
        self.W_xr = nn.Parameter(torch.empty(input_size, hidden_size))
        self.W_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.empty(hidden_size))

        # Candidate Hidden State
        self.W_xh = nn.Parameter(torch.empty(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.empty(hidden_size))

        self.init_weights()
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x_t, h_prev):
        # Update Gate
        z_t = torch.matmul(h_prev, self.W_hz) + torch.matmul(x_t, self.W_xz) + self.b_z
        z_t = torch.sigmoid(z_t)
        # Reset Gate
        r_t = torch.matmul(h_prev, self.W_hr) + torch.matmul(x_t, self.W_xr) + self.b_r
        r_t = torch.sigmoid(r_t)
        # Candidate Hidden State
        h_t = torch.matmul((h_prev * r_t), self.W_hh) + torch.matmul(x_t, self.W_xh) + self.b_h
        h_t = torch.tanh(h_t)
        # H Next
        h_t = z_t * h_t + (1-z_t) * h_prev
        return h_t

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_cell = GRU_Cell(input_size, hidden_size)

        # Fully Connected Layer
        self.fc  = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
    
    def forward(self, X):
        # (Batch, Sequence_Length, Feature)
        batch_size, seq_len, _ = X.shape

        # init hidden tensor of t=0
        h_t = torch.zeros(batch_size, self.hidden_size).to(X.device)

        for t in range(seq_len):
            x_t = X[:, t, :] # Batch, Feature
            h_t = self.gru_cell(x_t, h_t)

        out = self.fc(h_t)
        return out


# Build LSTM
def build_lstm(**param):
    return LSTM(**param)

# Build RNN
def build_rnn(**param):
    return Elman_RNN(**param)

# Build GRU
def build_gru(**param):
    return GRU(**param)