import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from utils import recall_at_k


class GNN(torch.nn.Module):
    """
    Overall graph neural network. Consists of learnable user/item (i.e., playlist/song) embeddings
    and LightGCN layers.
    """
    def __init__(self, embedding_dim, num_nodes, num_playlists, num_layers):
        super(GNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes # total number of nodes (songs + playlists) in dataset
        self.num_playlists = num_playlists # total number of playlists in dataset
        self.num_layers = num_layers

        self.embeddings = torch.nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.embedding_dim)
        torch.nn.init.normal_(self.embeddings.weight, std=0.1)

        self.layers = torch.nn.ModuleList() # LightGCN layers
        for _ in range(self.num_layers):
            self.layers.append(LightGCN())

        self.sigmoid = torch.sigmoid

    def forward(self):
        raise NotImplementedError("forward() has not been implemented for the GNN class. Do not use")

    def gnn_propagation(self, edge_index_mp):
        """
        Performs the linear embedding propagation (carried out by the LightGCN layers) and calculates final (multi-scale) embeddings
        for each user/item, which are calculated as a weighted sum of that user/item's embeddings at each layer (from
        0 to self.num_layers). Technically, the weighted sum here is just the average, which is what the LightGCN authors use.

        args:
          edge_index_mp: a tensor of all (undirected) edges in the graph, which is used for message passing/propagation and
              calculating the multi-scale embeddings. (In contrast to the evaluation/supervision edges, which are distinct
              from the message passing edges and will be used for calculating loss/performance metrics).
        returns:
          final multi-scale embeddings for all users/items
        """
        x = self.embeddings.weight
        # assert x.shape[0] == 801 # temporary
        # assert x.shape[0] == data.num_nodes # make sure data and nn.Embedding have same nodes. 
                                            # Otherwise may be issue with train/val/test split (each split should contain all nodes)

        # orig = x.clone()
        x_at_each_layer = [x] # stores embeddings from each layer. Start with layer-0 embeddings
        for i in range(self.num_layers): # now performing the GNN propagation
            x = self.layers[i](x, edge_index_mp)
            x_at_each_layer.append(x)
        final_embs = torch.stack(x_at_each_layer, dim=0).mean(dim=0) # multi-scale diffusion
        return final_embs

    def predict_scores(self, edge_index, embs):
        """
        Calculates predicted scores for each playlist/song pair in the list of edges. Uses dot product of their embeddings.

        args:
          edge_index: tensor of edges (between playlists and songs) whose scores we will calculate.
          embs: node embeddings used to calculate predicted scores (should typically be the multi-scale embeddings calculated
              by self.gnn_propagation())
        returns:
          predicted scores for each playlist/song pair in edge_index
        """
        scores = embs[edge_index[0,:], :] * embs[edge_index[1,:], :] # taking dot product for each playlist/song pair
        scores = scores.sum(dim=1)
        scores = self.sigmoid(scores)
        return scores

    def calc_loss(self, data_mp, data_pos, data_neg):
        """
        The main training step. Performs GNN propagation on message passing edges, to get multi-scale embeddings.
        Then predicts scores for each training example, and calculates Bayesian Personalized Ranking (BPR) loss.

        args:
          data_mp: tensor of edges used for message passing / calculating multi-scale embeddings
          data_pos: set of positive edges that will be used during loss calculation
          data_neg: set of negative edges that will be used during loss calculation
        returns:
          loss calculated on the positive/negative training edges
        """
        # Perform GNN propagation on message passing edges to get final embeddings
        final_embs = self.gnn_propagation(data_mp.edge_index) # only run propagation on the message-passing edges

        # Get edge prediction scores for all positive and negative evaluation edges
        pos_scores = self.predict_scores(data_pos.edge_index, final_embs)
        neg_scores = self.predict_scores(data_neg.edge_index, final_embs)

        # # Calculate loss (binary cross-entropy). Commenting out, but can use instead of BPR if desired.
        # all_scores = torch.cat([pos_scores, neg_scores], dim=0)
        # all_labels = torch.cat([torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])], dim=0)
        # loss_fn = torch.nn.BCELoss()
        # loss = loss_fn(all_scores, all_labels)

        # Calculate loss (more efficient approximation to BPR, similar to the one used in official LightGCN implementation at 
        # https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/model.py#L202)
        loss = -torch.log(self.sigmoid(pos_scores - neg_scores)).mean() # mean vs. sum
        return loss

    def evaluation(self, data_mp, data_pos, k):
        """
        Performs evaluation on validation or test set. Calculates recall@k.

        args:
          data_mp: message passing edges to use for propagation/calculating multi-scale embeddings
          data_pos: positive edges to use for scoring metrics. There should be no overlap between these edges and the 
            ones in data_mp
          k: value of k to use for recall@k
        returns:
          recall@k
        """
        # Run propagation on the message-passing edges to get multi-scale embeddings
        final_embs = self.gnn_propagation(data_mp.edge_index)

        # Get embeddings of all unique playlists in the batch of evaluation edges
        assert (torch.unique(data_pos.edge_index[0,:]) == torch.unique_consecutive(data_pos.edge_index[0,:]).sort()[0]).all()
        unique_playlists = torch.unique_consecutive(data_pos.edge_index[0,:])
        playlist_emb = final_embs[unique_playlists, :] # has shape [number of playlists in batch, 64]
        
        # Get embeddings of ALL songs in dataset
        song_emb = final_embs[self.num_playlists:, :] # has shape [total number of songs in dataset, 64]

        # All ratings for each playlist in batch to each song in entire dataset (using dot product as the scoring function)
        ratings = self.sigmoid(torch.matmul(playlist_emb, song_emb.t())) # has shape [number of playlists in batch, total number of songs in dataset]
                                                                         # where entry i,j is the predicted rating of song j for playlist i
        # Calculate recall@k. This will be a dictionary of playlist idx -> recall
        result = recall_at_k(ratings.cpu(), k, self.num_playlists, data_pos.edge_index.cpu(), 
                             unique_playlists.cpu(), data_mp.edge_index.cpu())
        return result


class LightGCN(MessagePassing):
    """
    A single LightGCN layer. Extends the MessagePassing class from PyTorch Geometric
    """
    def __init__(self):
        super(LightGCN, self).__init__(aggr='add') # aggregation function  is defined here. can also write your own
                                                   # by overriding self.aggregation()

    def forward(self, x, edge_index):
        """
        Performs the LightGCN message passing/aggregation/update to get updated node embeddings

        args:
          x: current embeddings (shape: [N, emb_dim])
          edge_index: message passing edges (shape: [2, E])
        returns:
          updated embeddings
        """
        # Computing node degrees for normalization term in LightGCN (see LightGCN paper for details on this normalization term)
        row, col = edge_index
        deg = degree(col)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Begin propagation. Will perform message passing and aggregation (which is specified in the aggr parameter in __init__)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        """
        Specifies how to perform message passing during GNN propagation. For LightGCN, we simply pass along each
        source node's embedding to the target node, normalized by the normalization term for that node.
        args:
          x_j: node embeddings of the source nodes, which will be passed to the target node (shape: [E, emb_dim])
          norm: the normalization terms we calculated in forward() and passed into propagate()
        returns:
          
        """
        # Here we are just multiplying the x_j's by the normalization terms (using some broadcasting)
        return norm.view(-1, 1) * x_j
