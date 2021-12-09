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


##### Read in data files from full Spotify Million Playlist Dataset
data_dir = '/Users/benalexander/Downloads/Song datasets/spotify_million_playlist_dataset/data'
data_files = os.listdir(data_dir)
data_files = sorted(data_files, key=lambda x: int(x.split(".")[2].split("-")[0]))


NUM_FILES_TO_USE = 10 # will create dataset based on the first NUM_FILES_TO_USE files in the full dataset
data_files = data_files[:NUM_FILES_TO_USE]

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
playlistInfo = {} # maps the playlist ID (pid) to information about that playlist
songToId = {} # maps the song URI to its new index (which we are generating, unlike the pid above) and other info about the song
# Note: some playlists have same song multiple times. I will ignore those and just add 1 edge
for data_file in data_files:
    with open(os.path.join(data_dir, data_file), 'r') as f:
        d = json.load(f)['playlists']
        for playlist in d:
            # plalistInfo will just store some info about the playlist so we can perform analysis later (after training)
            playlistInfo[playlist['pid']] = {'name': playlist['name'], 'songs': []}
            for song in playlist['tracks']:
                track_uri, track_name = song['track_uri'], song['track_name']
                artist_name, artist_uri = song['artist_name'], song['artist_uri']

                playlistInfo[playlist['pid']]['songs'].append((track_uri, track_name, artist_uri, artist_name))

                # First time seeing this song. Add a new node to the graph
                if track_uri not in songToId:
                    songToId[track_uri] = {'songid': currSongIdx, 'track_name': track_name, 'artist_name': artist_name,
                                           'artist_uri': artist_uri}
                    assert not G.IsNode(currSongIdx)
                    G.AddNode(currSongIdx)
                    currSongIdx += 1
                # Add edge between the current playlist and song
                G.AddEdge(playlist['pid'], songToId[track_uri]['songid'])


##### Get K-Core subgraph
num_pl_orig = len([x for x in G.Nodes() if x.GetId() <= maxPlaylistPid])
num_song_orig = len([x for x in G.Nodes() if x.GetId() > maxPlaylistPid])
print("Original graph:")
print(f"Num nodes: {len([x for x in G.Nodes()])} ({num_pl_orig} playlists, {num_song_orig} unique songs)")
print(f"Num edges: {len([x for x in G.Edges()])} (undirected)")
print()

K = 20
kcore = G.GetKCore(K)

num_pl_kcore = len([x for x in kcore.Nodes() if x.GetId() <= maxPlaylistPid])
num_song_kcore = len([x for x in kcore.Nodes() if x.GetId() > maxPlaylistPid])
print(f"K-core graph with K={K}:")
print(f"Num nodes: {len([x for x in kcore.Nodes()])} ({num_pl_kcore} playlists, {num_song_kcore} unique songs)")
print(f"Num edges: {len([x for x in kcore.Edges()])} (undirected)")


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


# Just rearranging the info saved above to get useful information about songs and playlists in our K-Core graph.
# These will only be used for analyzing our results after training the model
songInfo = {} # will map new song index/ID -> a dictionary containing some information about that song
for track_uri, info in songToId.items():
    if info['songid'] in oldToNewId_song: # only keeping songs that ended up in the K-Core graph
        new_id = oldToNewId_song[info['songid']]
        songInfo[new_id] = {'track_uri': track_uri, 'track_name': info['track_name'], 'artist_uri': info['artist_uri'],
                            'artist_name': info['artist_name']}
playlistInfo = {oldToNewId_playlist[k]: v for k,v in playlistInfo.items() if k in oldToNewId_playlist}


# Convert snap graph to a format that can be used in PyG. Basically just converting to edge_index
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

# Save data object, plus song/playlist info
save_dir = '.'
torch.save(data, os.path.join(save_dir, 'data_object'))
with open(os.path.join(save_dir, 'playlist_info.json'), 'w') as f:
    json.dump(playlistInfo, f)
with open(os.path.join(save_dir, 'song_info.json'), 'w') as f:
    json.dump(songInfo, f)
