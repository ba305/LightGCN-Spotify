import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GNN(torch.nn.Module):
    def __init__(self, embedding_dim, num_nodes, num_layers):
        super(GNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        self.embeddings = torch.nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.embedding_dim)
        torch.nn.init.normal_(self.embeddings.weight, std=0.1)

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(LightGCN())

        self.sigmoid = torch.sigmoid

    def forward(self):
        pass

    def gnn_propagation(self, edge_index_mp):
        x = self.embeddings.weight
        # assert x.shape[0] == 801 # temporary
        # assert x.shape[0] == data.num_nodes # make sure data and nn.Embedding have same nodes. 
                                            # Otherwise may be issue with train/val/test split (each split should contain all nodes)

        # orig = x.clone()
        x_at_each_layer = [x] # start with layer-0 embeddings
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index_mp)
            x_at_each_layer.append(x)
            # import ipdb; ipdb.set_trace()
        final_embs = torch.stack(x_at_each_layer, dim=0).mean(dim=0) # multi-scale diffusion
        return final_embs

    def predict_scores(self, edge_index, embs):
        scores = embs[edge_index[0,:], :] * embs[edge_index[1,:], :]
        scores = scores.sum(dim=1)
        scores = self.sigmoid(scores)
        return scores

    def calc_loss(self, data_mp, data_pos, data_neg, epoch):
        # Perform GNN propagation to get final embeddings
        # import ipdb; ipdb.set_trace()
        final_embs = self.gnn_propagation(data_mp.edge_index) # only run propagation on the message-passing edges

        # Get edge prediction scores for all positive and negative evaluation edges
        pos_scores = self.predict_scores(data_pos.edge_index, final_embs)
        neg_scores = self.predict_scores(data_neg.edge_index, final_embs)

        # Calculate loss
        all_scores = torch.cat([pos_scores, neg_scores], dim=0)
        all_labels = torch.cat([torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])], dim=0)
        loss_fn = torch.nn.BCELoss() # TODO: CHANGE TO BPR LOSS
        loss = loss_fn(all_scores, all_labels)
        
        # if epoch == 1000:
        #     import ipdb; ipdb.set_trace()
        return loss



class LightGCN(MessagePassing):
    def __init__(self):
        super(LightGCN, self).__init__(aggr='add') # aggregation is defined here

    def forward(self, x, edge_index):
        # x has shape [N, input_dim]
        # edge_index has shape [2, E]

        # Computing degrees for normalization term
        row, col = edge_index
        deg = degree(col)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Begin propagation
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_dim]
        # import ipdb; ipdb.set_trace()
        return norm.view(-1, 1) * x_j
