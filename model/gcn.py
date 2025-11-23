import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU


'''
Basic GCN architecture

hyperparameters:
- architecture
    - layers
    - layer sizes
    - activation function

- learning rate
'''

class GCN(torch.nn.Module):
    def __init__(self, input_dim = 32, hidden_layers = [128,128], out_dim = 128, targets=1):

        '''
        hidden_layers;
        - list containing the dimension of the hidden feature vectors for each layer
        - each element in the list represents one layer, thus,
          len(hidden_layers) indicates the number of hidden layers
        '''

        super().__init__()
        torch.manual_seed(42)
        
        self.input_dim = input_dim
        self.hidden_layers = torch.nn.ModuleList()
        self.out_dim = out_dim
        self.targets = targets

        self.initial_layer = GCNConv(self.input_dim, hidden_layers[0])
        
        for i, n in enumerate(hidden_layers):
            n_in = n
            
            if i == len(hidden_layers) - 1:
                n_out = out_dim
            else:
                n_out = hidden_layers[i+1]
            
            layer = GCNConv(n_in, n_out)

            self.hidden_layers.append(layer)

        self.activation = torch.nn.LeakyReLU()

        self.out = Linear(2*out_dim, targets)

    def forward(self, x, edge_index, batch_index):
        #input layer
        h = self.initial_layer(x, edge_index)
        h = self.activation(h)
        
        for layer in self.hidden_layers:
            h = layer(h, edge_index)
            h = self.activation(h)
        
        #global mean (average) pooling, and global max pooling concatenated
        h = torch.cat([gap(h, batch_index), gmp(h, batch_index)], dim=1)

        out = self.out(h)

        return out, h