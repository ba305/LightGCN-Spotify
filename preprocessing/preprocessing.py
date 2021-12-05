import json
import random
import numpy as np
import os
import snap
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected, degree
from torch_geometric.loader import DataLoader

random.seed(5)
np.random.seed(5)


##### Read in data and create SNAP graph
data_dir = '/Users/benalexander/Downloads/Song datasets/spotify_million_playlist_dataset/data'
data_files = os.listdir(data_dir)
data_files = sorted(data_files, key=lambda x: int(x.split(".")[2].split("-")[0]))

data_files = data_files[:1] # TODO: Decide how many files to use

G = snap.TUNGraph().New()

# First add all playlist IDs as nodes
for data_file in data_files:
    with open(os.path.join(data_dir, data_file), 'r') as f:
        d = json.load(f)['playlists']
        for playlist in d:
            G.AddNode(playlist['pid'])

maxPlaylistPid = max([x.GetId() for x in G.Nodes()]) # will start song IDs after this ID
assert maxPlaylistPid == len([x for x in G.Nodes()]) - 1 # checks that the sorting above was correct


# Now add song IDs as nodes, and also add the edges
currSongIdx = maxPlaylistPid + 1
songToId = {}
# Note: some playlists have same song multiple times. I will ignore those and just add 1 edge
for data_file in data_files:
    with open(os.path.join(data_dir, data_file), 'r') as f:
        d = json.load(f)['playlists']
        for playlist in d:
            for song in playlist['tracks']:
                track_uri = song['track_uri']
                if track_uri not in songToId:
                    songToId[track_uri] = currSongIdx
                    assert not G.IsNode(currSongIdx)
                    G.AddNode(currSongIdx)
                    currSongIdx += 1
                G.AddEdge(playlist['pid'], songToId[track_uri])


##### Get K-Core subgraph
num_pl_orig = len([x for x in G.Nodes() if x.GetId() <= maxPlaylistPid])
num_song_orig = len([x for x in G.Nodes() if x.GetId() > maxPlaylistPid])
print("Original graph:")
print(f"Num nodes: {len([x for x in G.Nodes()])} ({num_pl_orig} playlists, {num_song_orig} unique songs)")
print(f"Num edges: {len([x for x in G.Edges()])}")
print()

K = 10
kcore = G.GetKCore(K)

num_pl_kcore = len([x for x in kcore.Nodes() if x.GetId() <= maxPlaylistPid])
num_song_kcore = len([x for x in kcore.Nodes() if x.GetId() > maxPlaylistPid])
print(f"K-core graph with K={K}:")
print(f"Num nodes: {len([x for x in kcore.Nodes()])} ({num_pl_kcore} playlists, {num_song_kcore} unique songs)")
print(f"Num edges: {len([x for x in kcore.Edges()])}")


###### Need to re-index new graph to have nodes in continuous sequence
cnt = 0
oldToNewId_playlist = {}
oldToNewId_song = {}
for NI in kcore.Nodes(): # will be in sorted order already
    old_id = NI.GetId()
    assert old_id not in oldToNewId_song and old_id not in oldToNewId_playlist # each should only appear once
    new_id = cnt
    if old_id <= num_pl_orig - 1:
        oldToNewId_playlist[old_id] = new_id
    else:
        oldToNewId_song[old_id] = new_id
    cnt += 1

# Nodes in the for loop above should be in sorted order, so all playlists will still be before all songs. Here are a few simple checks
assert max(oldToNewId_playlist.values()) == num_pl_kcore-1
assert len(oldToNewId_playlist.values()) == num_pl_kcore
assert max(oldToNewId_song.values()) == len([x for x in kcore.Nodes()]) - 1
assert len(oldToNewId_song.values()) == num_song_kcore

songToId = {k: oldToNewId_song[v] for k,v in songToId.items() if v in oldToNewId_song} # will contain new IDs, and only for songs that are still left in the graph


# import ipdb; ipdb.set_trace()

#### Edge counts printed above should maybe be multiplied by 2 because only includes each edge once

all_edges = []
for EI in tqdm(kcore.Edges()):
    # When creating graph, made edges from playlist -> song, so should all be in that order. Double-checking this with assert statements below:
    assert (EI.GetSrcNId() in oldToNewId_playlist) and (EI.GetSrcNId() not in oldToNewId_song)
    assert (EI.GetDstNId() in oldToNewId_song) and (EI.GetDstNId() not in oldToNewId_playlist)
    edge_info = [oldToNewId_playlist[EI.GetSrcNId()], oldToNewId_song[EI.GetDstNId()]] # using new node IDs instead of old ones
    all_edges.append(edge_info)
    all_edges.append(edge_info[::-1]) # also add the edge in the opposite direction b/c undirected
edge_idx = torch.Tensor(all_edges)

data = Data(edge_index = edge_idx.t().contiguous(), num_nodes=kcore.GetNodes())

torch.save(data, 'data_object')



# # No longer using this
# class SpotifyDataset(Dataset):
#     def __init__(self, root, data_object, num_playlists, transform=None, pre_transform=None):
#         self.data = data_object
#         self.num_playlists = num_playlists

#         # Degrees of all nodes
#         # TODO: edge_index should be a LongTensor I think
#         self.degrees = degree(self.data.edge_index[0, :].to(torch.int64)).to(torch.int32).tolist()

#         super().__init__(root, transform, pre_transform)

#     def len(self):
#         return self.data.num_nodes

#     def get(self, idx):
#         # Positive edges
#         edge_index_pos = self.data.edge_index[:, torch.where(self.data.edge_index[1,:] == idx)[0]]
#         y_pos = torch.ones(edge_index_pos.shape[1])
        
#         # Sample negative edges. If idx is a playlist, will randomly sample from all songs. If idx is a song, will randomly sample from all playlists
#         # Note that all playlist indices come before all song indices, which is why the code below works
#         if idx <= self.num_playlists - 1: # then idx is a playlist
#             rand_songs = torch.randint(self.num_playlists, self.data.num_nodes+1, (len(y_pos), ))
#         else: # then idx is a song
#             rand_songs = torch.randint(0, self.num_playlists, (len(y_pos), ))

#         edge_index_neg = torch.row_stack([rand_songs, edge_index_pos[1,:]])
#         y_neg = torch.zeros(edge_index_neg.shape[1])

#         edge_index = torch.hstack([edge_index_pos, edge_index_neg])
#         y = torch.hstack([y_pos, y_neg])

#         # Also store degrees of all source and target nodes
#         deg_i = torch.Tensor([self.degrees[x] for x in edge_index[0, :].to(torch.int32).tolist()])
#         deg_j = torch.Tensor([self.degrees[x] for x in edge_index[1, :].to(torch.int32).tolist()])

#         return Data(edge_index=edge_index, y=y, num_nodes=len(torch.unique(edge_index)), deg_i=deg_i, deg_j=deg_j)

# dataset = SpotifyDataset('test/spotify', data, num_pl_kcore)
# dataset[0]
# loader = DataLoader(dataset, batch_size=4, shuffle=True)


# import ipdb; ipdb.set_trace()