import torch
import numpy as np
import os
import json
from lightgcn import GNN
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data, Dataset
# from torch.utils.data import Dataset, DataLoader


seed_everything(5)


def sample_negative_edges(batch, num_playlists, num_nodes):
    # Randomly samples songs for each playlist. Doesn't currently check if they are true negatives
    negs = []
    for i in batch.edge_index[0,:]: # will all be playlists
        assert i < num_playlists
        rand = torch.randint(num_playlists, num_nodes, (1,))
        negs.append(rand.item())
    edge_index_negs = torch.row_stack([batch.edge_index[0,:], torch.LongTensor(negs)])
    return Data(edge_index=edge_index_negs)


def train(model, data_mp, loader, opt, num_playlists, num_nodes, epoch, device):
    total_loss = 0
    total_examples=0
    model.train()
    for batch in loader:
        del batch.batch; del batch.ptr # delete unwanted attributes
        
        opt.zero_grad()
        negs = sample_negative_edges(batch, num_playlists, num_nodes)
        data_mp, batch, negs = data_mp.to(device), batch.to(device), negs.to(device)
        loss = model.calc_loss(data_mp, batch, negs, epoch)
        loss.backward()
        opt.step()

        num_examples = batch.edge_index.shape[1] # maybe it should be 2x this because of negatives
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        # import ipdb; ipdb.set_trace()
    avg_loss = total_loss / total_examples
    return avg_loss

def test(model, data_mp, loader, k, device, save_dir, epoch):
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
    # Overriding the __inc__ method so that the DataLoader stops increasing indices unnecessarily.
    # See here for more information: https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    def __inc__(self, key, value, *args, **kwargs):
        return 0

class SpotifyDataset(Dataset):
    def __init__(self, root, edge_index, transform=None, pre_transform=None):
        self.edge_index = edge_index
        self.unique_idxs = torch.unique(edge_index[0,:]).tolist() # playlists will all be in 0 row, songs will be in 1, b/c sorted by RandLinkSplit
        self.num_nodes = len(self.unique_idxs)
        ### LATER: CAN possible abandon the usage of edge_index. Don't necessarily need that for these evaluation edges, can just use tuples or something
        super().__init__(root, transform, pre_transform)

    def len(self):
        return self.num_nodes

    def get(self, idx):
        node_at_idx = self.unique_idxs[idx] # actually should usually just be the same as idx so maybe not necessary
        edge_index = self.edge_index[:, self.edge_index[0,:] == node_at_idx]
        # could also do negative sampling here?
        return PlainData(edge_index=edge_index) ##, num_nodes=len(torch.unique(edge_index)))



if __name__ == "__main__":
    # Load data
    base_dir = "../data/dataset_small"
    data = torch.load(os.path.join(base_dir, "data_object.pt"))
    with open(os.path.join(base_dir, "dataset_stats.json"), 'r') as f:
        stats = json.load(f)
    num_playlists, num_nodes = stats["num_playlists"], stats["num_nodes"]

    # Train/val/test split
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=False, neg_sampling_ratio=0)
    train_split, val_split, test_split = transform(data)
    # Need to make sure all nodes appear in all sets above
    assert train_split.num_nodes == val_split.num_nodes and train_split.num_nodes == test_split.num_nodes

    # For each of train/val/test splits, we will have a set of message passing edges (used for GNN propagation and to get
    # the final multi-scale node embeddings), and also a set of evaluation edges (used to calculate loss/performance metrics)
    train_ev = SpotifyDataset('temp', edge_index=train_split.edge_label_index)
    train_mp = Data(edge_index=train_split.edge_index)

    val_ev = SpotifyDataset('temp', edge_index=val_split.edge_label_index)
    val_mp = Data(edge_index=val_split.edge_index)

    test_ev = SpotifyDataset('temp', edge_index=test_split.edge_label_index)    
    test_mp = Data(edge_index=test_split.edge_index)

    epochs = 500
    k = 150 # for recall@k
    num_layers = 3
    batch_size = 2048

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_ev, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ev, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ev, batch_size=batch_size, shuffle=False)

    gnn = GNN(embedding_dim=64, num_nodes=data.num_nodes, num_playlists=num_playlists, num_layers=num_layers).to(device)

    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)

    save_embeddings_dir = None

    all_train_losses = [] # list of (epoch, training loss)
    all_val_recalls = []  # list of (epoch, validation recall@k)
    for epoch in range(epochs):
        train_loss = train(gnn, train_mp, train_loader, opt, num_playlists, num_nodes, epoch, device)
        all_train_losses.append((epoch, train_loss))
        if epoch % 5 == 0:
            val_recall = test(gnn, val_mp, val_loader, k, device, save_embeddings_dir, epoch)
            all_val_recalls.append((epoch, val_recall))
            print(f"Epoch {epoch}: train loss={train_loss}, val_recall={val_recall}")
        else:
            print(f"Epoch {epoch}: train loss={train_loss}")


print()

best_val_recall = max(all_val_recalls, key = lambda x: x[1])
print(f"Best validation recall@k: {best_val_recall[1]} at epoch {best_val_recall[0]}")

test_recall = test(gnn, test_mp, test_loader, k, device, None, None)
print(f"Test set recall@k: {test_recall}")