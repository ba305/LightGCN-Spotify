"""
This script performs preprocessing on the full Spotify Million Playlist Dataset, which can be downloaded
from https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
(although you will need to make an account).

Specifically, it first uses the SNAP library to calculate the K-core subgraph. It then performs some
re-indexing so that the remaining nodes have sequential indexing. Then it converts the graph to the Data
format from PyTorch Geometric and saves to a file. Along the way, it also stores some information about the
songs/playlists in the resulting dataset, and saves these to a file. That info will be used later if you want
to perform some analysis on the resulting embeddings, etc.

Outputs (all will be saved in save_dir):
  data_object.pt: resulting graph in PyTorch Geometric's Data format. Playlists will have indices from 0...num_playlists-1,
     and songs will have indices from num_playlists...num_nodes-1
  playlist_info.json: JSON file mapping playlist ID to some information about that playlist. Can be used later for analysis
  song_info.json: JSON file mapping song ID to some information about that song. Can be used later for analysis
"""

import json
import random
import numpy as np
import os
import snap
from tqdm import tqdm
import torch
from torch_geometric.data import Data

random.seed(5)
np.random.seed(5)


##### SET THESE VALUES BEFORE RUNNING
data_dir = '/Users/benalexander/Downloads/Song datasets/spotify_million_playlist_dataset/data' # path to Spotify dataset files
NUM_FILES_TO_USE = 30 # will create dataset based on the first NUM_FILES_TO_USE files from the full dataset
save_dir = '.'        # directory to save the new dataset files after preprocessing
K = 35                # value of K for the K-core graph


# Read in data files from Spotify dataset
data_files = os.listdir(data_dir)
data_files = sorted(data_files, key=lambda x: int(x.split(".")[2].split("-")[0]))
data_files = data_files[:NUM_FILES_TO_USE]

# Create undirected SNAP graph
G = snap.TUNGraph().New()

# First add all playlist IDs as nodes in G
for data_file in data_files:
    with open(os.path.join(data_dir, data_file), 'r') as f:
        d = json.load(f)['playlists']
        for playlist in d:
            G.AddNode(playlist['pid'])

# Playlists so far will have indices from 0...num_playlists-1. We will start indexing the songs from num_playlists
# onwards. Thus, later on, if we know num_playlists, we can easily identify if a node is a playlist or song based on
# whether its index is < or >= num_playlists
maxPlaylistPid = max([x.GetId() for x in G.Nodes()])     # will start song IDs after this ID
assert maxPlaylistPid == len([x for x in G.Nodes()]) - 1 # checks that the sorting of files name above was correct.
                                                         # otherwise this will fail because we read in the pid's in wrong order


# Now read through the data again. Add new nodes for songs that don't already have nodes. Also add edges between playlists
# and songs as we come to them.
currSongIdx = maxPlaylistPid + 1 # start songs idxs here. (see above for explanation)
playlistInfo = {} # maps the playlist ID (pid) to information about that playlist
songToId = {} # maps the song URI to its new index (which we are generating, unlike the pid above) and other info about the song
# Note: some playlists have same song multiple times. I will ignore those and just add 1 edge
for data_file in data_files:
    with open(os.path.join(data_dir, data_file), 'r') as f:
        d = json.load(f)['playlists']
        for playlist in d:
            # plalistInfo will just store some info about the playlist so we can perform analysis later (after training)
            playlistInfo[playlist['pid']] = {'name': playlist['name']}

            # Loop through songs. Add new node for a song if it doesn't already have one in G. Add playlist-song edges as well.
            for song in playlist['tracks']:
                track_uri, track_name = song['track_uri'], song['track_name']
                artist_name, artist_uri = song['artist_name'], song['artist_uri']

                # First time seeing this song. Add a new node to the graph
                if track_uri not in songToId:
                    songToId[track_uri] = {'songid': currSongIdx, 'track_name': track_name, 'artist_name': artist_name,
                                           'artist_uri': artist_uri}
                    assert not G.IsNode(currSongIdx)
                    G.AddNode(currSongIdx)
                    currSongIdx += 1
                # Add edge between the current playlist and song
                G.AddEdge(playlist['pid'], songToId[track_uri]['songid'])


# Print some stats about graph G before calculating K-core subgraph
num_pl_orig = len([x for x in G.Nodes() if x.GetId() <= maxPlaylistPid])
num_song_orig = len([x for x in G.Nodes() if x.GetId() > maxPlaylistPid])
print("Original graph:")
print(f"Num nodes: {len([x for x in G.Nodes()])} ({num_pl_orig} playlists, {num_song_orig} unique songs)")
print(f"Num edges: {len([x for x in G.Edges()])} (undirected)")
print()

# Get K-Core subgraph
kcore = G.GetKCore(K)
if kcore.Empty():
    raise Exception(f"No Core exists for K={K}")

# Print the same stats as above, but after calculating K-core subgraph
num_pl_kcore = len([x for x in kcore.Nodes() if x.GetId() <= maxPlaylistPid])
num_song_kcore = len([x for x in kcore.Nodes() if x.GetId() > maxPlaylistPid])
print(f"K-core graph with K={K}:")
print(f"Num nodes: {len([x for x in kcore.Nodes()])} ({num_pl_kcore} playlists, {num_song_kcore} unique songs)")
kcore_num_edges = len([x for x in kcore.Edges()])
print(f"Num edges: {kcore_num_edges} (undirected)")


# Need to re-index new graph to have nodes in continuous sequence. After finding the K-core, we will have lost a lot of
# nodes, so indices will no longer be from 0...num_nodes. That will cause some issues later in PyG if we don't fix it here.
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

# A few error checks. Nodes in the for loop above should be in sorted order, so all playlists will still be before all songs.
assert max(oldToNewId_playlist.values()) == num_pl_kcore-1
assert len(oldToNewId_playlist.values()) == num_pl_kcore
assert max(oldToNewId_song.values()) == len([x for x in kcore.Nodes()]) - 1
assert len(oldToNewId_song.values()) == num_song_kcore


# Just rearranging the info saved above to get useful information about songs and playlists in our K-Core graph.
# These will only be used for analyzing our results after training the model
songInfo = {} # will map new song index/ID -> a dictionary containing some information about that song
for track_uri, info in songToId.items():
    if info['songid'] in oldToNewId_song: # only keeping songs that ended up in the K-Core graph
        new_id = oldToNewId_song[info['songid']]
        songInfo[new_id] = {'track_uri': track_uri, 'track_name': info['track_name'], 'artist_uri': info['artist_uri'],
                            'artist_name': info['artist_name']}
playlistInfo = {oldToNewId_playlist[k]: v for k,v in playlistInfo.items() if k in oldToNewId_playlist}


# Convert snap graph to a format that can be used in PyG. Basically converting to edge_index and storing in a PyG Data object
all_edges = []
for EI in tqdm(kcore.Edges()):
    # When creating graph, made edges from playlist -> song, so should all be in that order. Double-checking this with assert statements below:
    assert (EI.GetSrcNId() in oldToNewId_playlist) and (EI.GetSrcNId() not in oldToNewId_song)
    assert (EI.GetDstNId() in oldToNewId_song) and (EI.GetDstNId() not in oldToNewId_playlist)
    edge_info = [oldToNewId_playlist[EI.GetSrcNId()], oldToNewId_song[EI.GetDstNId()]] # using new node IDs instead of old ones
    all_edges.append(edge_info)
    all_edges.append(edge_info[::-1]) # also add the edge in the opposite direction b/c undirected
edge_idx = torch.LongTensor(all_edges)

data = Data(edge_index = edge_idx.t().contiguous(), num_nodes=kcore.GetNodes())

# Save Data object (for training model), some dataset stats/metadata, and song/playlist info (used for post-training analysis)
torch.save(data, os.path.join(save_dir, 'data_object.pt'))
stats = {'num_playlists': num_pl_kcore, 'num_nodes': num_pl_kcore + num_song_kcore, 'kcore_value_k': K,
         'num_spotify_files_used': NUM_FILES_TO_USE, 'num_edges_directed': 2*kcore_num_edges, 'num_edges_undirected': kcore_num_edges}
with open(os.path.join(save_dir, 'dataset_stats.json'), 'w') as f:
    json.dump(stats, f)
with open(os.path.join(save_dir, 'playlist_info.json'), 'w') as f:
    json.dump(playlistInfo, f)
with open(os.path.join(save_dir, 'song_info.json'), 'w') as f:
    json.dump(songInfo, f)
