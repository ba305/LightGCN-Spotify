"""
This file contains the main training/evaluation loop that trains LightGCN on the Spotify dataset.
This is the main file that you should be running.
"""

import torch
import numpy as np
import os
import json
from lightgcn import GNN
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data, Dataset
from utils import sample_negative_edges


def train(model, data_mp, loader, opt, num_playlists, num_nodes, device):
    """
    Main training loop

    args:
       model: the GNN model
       data_mp: message passing edges to use for performing propagation/calculating multi-scale embeddings
       loader: DataLoader that loads in batches of supervision/evaluation edges
       opt: the optimizer
       num_playlists: the number of playlists in the entire dataset
       num_nodes: the number of nodes (playlists + songs) in the entire dataset
       device: whether to run on CPU or GPU
    returns:
       the training loss for this epoch
    """
    total_loss = 0
    total_examples = 0
    model.train()
    for batch in loader:
        del batch.batch; del batch.ptr # delete unwanted attributes
        
        opt.zero_grad()
        negs = sample_negative_edges(batch, num_playlists, num_nodes)  # sample negative edges
        data_mp, batch, negs = data_mp.to(device), batch.to(device), negs.to(device)
        loss = model.calc_loss(data_mp, batch, negs)
        loss.backward()
        opt.step()

        num_examples = batch.edge_index.shape[1]
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    avg_loss = total_loss / total_examples
    return avg_loss

def test(model, data_mp, loader, k, device, save_dir, epoch):
    """
    Evaluation loop for validation/testing.

    args:
       model: the GNN model
       data_mp: message passing edges to use for propagation/calculating multi-scale embeddings
       loader: DataLoader that loads in batches of evaluation (i.e., validation or test) edges
       k: value of k to use for recall@k
       device: whether to use CPU or GPU
       save_dir: directory to save multi-scale embeddings for later analysis. If None, doesn't save any embeddings.
       epoch: the number of the current epoch
    returns:
       recall@k for this epoch
    """
    model.eval()
    all_recalls = {}
    with torch.no_grad():
        # Save multi-scale embeddings if save_dir is not None
        data_mp = data_mp.to(device)
        if save_dir is not None:
            embs_to_save = gnn.gnn_propagation(data_mp.edge_index)
            torch.save(embs_to_save, os.path.join(save_dir, f"embeddings_epoch_{epoch}.pt"))

        # Run evaluation
        for batch in loader:
            del batch.batch; del batch.ptr # delete unwanted attributes

            batch = batch.to(device)
            recalls = model.evaluation(data_mp, batch, k)
            for playlist_idx in recalls:
                assert playlist_idx not in all_recalls
            all_recalls.update(recalls)
    recall_at_k = np.mean(list(all_recalls.values()))
    return recall_at_k


class PlainData(Data):
    """
    Custom Data class for use in PyG. Basically the same as the original Data class from PyG, but
    overrides the __inc__ method because otherwise the DataLoader was incrementing indices unnecessarily.
    Now it functions more like the original DataLoader from PyTorch itself.
    See here for more information: https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    """
    def __inc__(self, key, value, *args, **kwargs):
        return 0

class SpotifyDataset(Dataset):
    """
    Dataset object containing the Spotify supervision/evaluation edges. This will be used by the DataLoader to load
    batches of edges to calculate loss or evaluation metrics on. Here, get(idx) will return ALL edges of the graph
    corresponding to playlist "idx." This is because when calculating metrics such as recall@k, we need all of the
    playlist's positive edges in the same batch.
    """
    def __init__(self, root, edge_index, transform=None, pre_transform=None):
        self.edge_index = edge_index
        self.unique_idxs = torch.unique(edge_index[0,:]).tolist() # playlists will all be in row 0, b/c sorted by RandLinkSplit
        self.num_nodes = len(self.unique_idxs)
        super().__init__(root, transform, pre_transform)

    def len(self):
        return self.num_nodes

    def get(self, idx):
        edge_index = self.edge_index[:, self.edge_index[0,:] == idx]
        return PlainData(edge_index=edge_index)



if __name__ == "__main__":
    # Random seed
    seed_everything(5)

    # Load data
    base_dir = "../data/dataset_large"
    data = torch.load(os.path.join(base_dir, "data_object.pt"))
    with open(os.path.join(base_dir, "dataset_stats.json"), 'r') as f:
        stats = json.load(f)
    num_playlists, num_nodes = stats["num_playlists"], stats["num_nodes"]

    # Train/val/test split (70-15-15). Need to specify is_undirected=True so that it knows to avoid data leakage from 
    # reverse edges (e.g., [4,5] and [5,4] should stay in the same split since they are basically the same edge).
    # Also set add_negative_train_samples=False and neg_sampling_ratio=0 since we have our own negative sampling implementation.
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=False, neg_sampling_ratio=0,
                                num_val=0.15, num_test=0.15)
    train_split, val_split, test_split = transform(data)
    # Confirm that every node appears in every set above
    assert train_split.num_nodes == val_split.num_nodes and train_split.num_nodes == test_split.num_nodes

    # For each split, we have a set of message passing edges (for GNN propagation/getting final multi-scale node embeddings),
    # and also a set of evaluation edges (used to calculate loss/performance metrics). For message passing edges, store in
    # Data object. For eval edges, put them in a SpotifyDataset object so we can load them in in batches with a DataLoader.
    train_ev = SpotifyDataset('temp', edge_index=train_split.edge_label_index)
    train_mp = Data(edge_index=train_split.edge_index)

    val_ev = SpotifyDataset('temp', edge_index=val_split.edge_label_index)
    val_mp = Data(edge_index=val_split.edge_index)

    test_ev = SpotifyDataset('temp', edge_index=test_split.edge_label_index)    
    test_mp = Data(edge_index=test_split.edge_index)

    # Training hyperparameters
    epochs = 500         # number of training epochs
    k = 250              # value of k for recall@k. It is important to set this to a reasonable value!
    num_layers = 3       # number of LightGCN layers (i.e., number of hops to consider during propagation)
    batch_size = 2048    # batch size. refers to the # of playlists in the batch (each will come with all of its edges)
    embedding_dim = 64   # dimension to use for the playlist/song embeddings
    save_emb_dir = None  # path to save multi-scale embeddings during test(). If None, will not save any embeddings

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create DataLoaders for the supervision/evaluation edges (one each for train/val/test sets)
    train_loader = DataLoader(train_ev, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ev, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ev, batch_size=batch_size, shuffle=False)

    # Initialize GNN model
    gnn = GNN(embedding_dim=embedding_dim, num_nodes=data.num_nodes, num_playlists=num_playlists, num_layers=num_layers).to(device)

    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3) # using Adam optimizer

    all_train_losses = [] # list of (epoch, training loss)
    all_val_recalls = []  # list of (epoch, validation recall@k)

    # Main training loop
    for epoch in range(epochs):
        train_loss = train(gnn, train_mp, train_loader, opt, num_playlists, num_nodes, device)
        all_train_losses.append((epoch, train_loss))
        
        if epoch % 5 == 0:
            val_recall = test(gnn, val_mp, val_loader, k, device, save_emb_dir, epoch)
            all_val_recalls.append((epoch, val_recall))
            print(f"Epoch {epoch}: train loss={train_loss}, val_recall={val_recall}")
        else:
            print(f"Epoch {epoch}: train loss={train_loss}")


    print()

    # Print best validation recall@k value
    best_val_recall = max(all_val_recalls, key = lambda x: x[1])
    print(f"Best validation recall@k: {best_val_recall[1]} at epoch {best_val_recall[0]}")

    # Print final recall@k on test set
    test_recall = test(gnn, test_mp, test_loader, k, device, None, None)
    print(f"Test set recall@k: {test_recall}")