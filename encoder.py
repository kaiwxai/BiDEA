import torch
import torch.nn as nn
import torch.nn.functional as F

class Temporal_Convolution_Block(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(Temporal_Convolution_Block, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)    
        z = z.tanh()
        return z


class ST_Adaptive_Fusion(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(ST_Adaptive_Fusion, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        # self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        #                                    for _ in range(num_hidden_layers - 1))

        #FIXME:
        # self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.node_embeddings2 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, 4, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))
            # self.adj_matrix = nn.Parameter(adj_matrix, requires_grad=True)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z, rw_adj_in, rw_adj_out):
        z = self.linear_in(z)
        z = z.relu()
       
        if self.g_type == 'agc':
            z = self.agc(z, rw_adj_in, rw_adj_out)
        else:
            raise ValueError('Check g_type argument')
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 64, 1])

    def agc(self, z, rw_adj_in, rw_adj_out):
        node_num = self.node_embeddings1.shape[0]
        supports1 = F.softmax(F.relu(torch.mm(self.node_embeddings1, self.node_embeddings1.transpose(0, 1))), dim=1)
        supports2 = F.softmax(F.relu(torch.mm(self.node_embeddings2, self.node_embeddings2.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports1.device), supports1, F.softmax(rw_adj_in.mean(dim=0)/z.shape[0]), F.softmax(rw_adj_out.mean(dim=0)/z.shape[0])]
        supports = torch.stack(support_set, dim=0)
        
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings1, self.weights_pool) 
        bias = torch.matmul(self.node_embeddings1, self.bias_pool)                     
        x_g = torch.einsum("knm,bmc->bknc", supports, z)     
        x_g = x_g.permute(0, 2, 1, 3)  
        z = torch.einsum('bnki,nkio->bno', x_g, weights)+bias     
        return z

