import torch
import numpy as np
from torch_geometric.data import Data


def sample_negative_edges(batch, num_playlists, num_nodes):
    # Randomly samples songs for each playlist. Doesn't currently check if they are true negatives, since that is
    # computationally expensive. This is fine in our case, because we will never be sampling more than ~100
    # songs for a playlist (out of thousands of songs), so although we will accidentally sample some positive songs,
    # it will be an acceptably small number. However, if that is not the case for your dataset, please consider
    # sampling true negatives only. Here we sample 1 negative edge for each positive edge in the graph, so we will
    # end up having a balanced 1:1 ratio of positive to negative edges.
    negs = []
    for i in batch.edge_index[0,:]:  # looping over playlists
        assert i < num_playlists     # just ensuring that i is a playlist
        rand = torch.randint(num_playlists, num_nodes, (1,))  # randomly sample a song
        negs.append(rand.item())
    edge_index_negs = torch.row_stack([batch.edge_index[0,:], torch.LongTensor(negs)])
    return Data(edge_index=edge_index_negs)


def recall_at_k(all_ratings, k, num_playlists, ground_truth, unique_playlists, data_mp):
   """
   Calculates recall@k during validation/testing for a single batch.

   args:
     all_ratings: array of shape [number of playlists in batch, number of songs in whole dataset]
     k: the value of k to use for recall@k
     num_playlists: the number of playlists in the dataset
     ground_truth: array of shape [2, X] where each column is a pair of (playlist_idx, positive song idx). This is the
        batch that we are calculating metrics on.
     unique_playlists: 1D vector of length [number of playlists in batch], which specifies which playlist corresponds
        to each row of all_ratings
     data_mp: an array of shape [2, Y]. This is all of the known message-passing edges. We will use this to make sure we
        don't recommend songs that are already known to be in the playlist.
   returns:
     Dictionary of playlist ID -> recall@k on that playlist
   """
   # We don't want to recommend songs that are already known to be in the playlist. 
   # Set those to a low rating so they won't be recommended
   known_edges = data_mp[:, data_mp[0,:] < num_playlists] # removing duplicate edges (since data_mp is undirected). also makes it so
                                                          # that for each column, playlist idx is in row 0 and song idx is in row 1
   playlist_to_idx_in_batch = {playlist: i for i, playlist in enumerate(unique_playlists.tolist())}
   exclude_playlists, exclude_songs = [], [] # already-known playlist/song links. Don't want to recommend these again
   for i in range(known_edges.shape[1]): # looping over all known edges
      pl, song = known_edges[:,i].tolist()
      if pl in playlist_to_idx_in_batch: # don't need the edges in data_mp that are from playlists that are not in this batch
         exclude_playlists.append(playlist_to_idx_in_batch[pl])
         exclude_songs.append(song - num_playlists) # subtract num_playlists to get indexing into all_ratings correct
   all_ratings[exclude_playlists, exclude_songs] = -10000 # setting to a very low score so they won't be recommended

   # Get top k recommendations for each playlist
   _, top_k = torch.topk(all_ratings, k=k, dim=1)
   top_k += num_playlists # topk returned indices of songs in ratings, which doesn't include playlists.
                          # Need to shift up by num_playlists to get the actual song indices
    
   # Calculate recall@k
   ret = {}
   for i, playlist in enumerate(unique_playlists):
      pos_songs = ground_truth[1, ground_truth[0, :] == playlist]

      k_recs = top_k[i, :] # top k recommendations for playlist
      recall = len(np.intersect1d(pos_songs, k_recs)) / len(pos_songs)
      ret[playlist] = recall
   return ret
