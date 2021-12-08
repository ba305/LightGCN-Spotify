import torch
import numpy as np


def recall_at_k(all_ratings, k, num_playlists, ground_truth, unique_playlists, data_mp):
   """
   args:
     all_ratings: array of shape [number of playlists in batch, number of songs in whole dataset]
     k: the value of k for recall@k
     num_playlists: the number of playlists in the dataset
     ground_truth: array of shape [2, X] where each column is a pair of (playlist_idx, positive song). Used to
        calculate metrics
     unique_playlists: 1D vector of length [number of playlists in batch], which specifies which playlist corresponds
        to each row of all_ratings
     data_mp: an array of shape [2, Y]. This is all of the known message-passing edges. We will use this to make sure we
        don't recommend songs that are already known to be in the playlist.
   returns:
     recall@k
   """
   # We don't want to recommend songs that are already known to be in the playlist. 
   # Set those to a low rating so they won't be recommended
   known_edges = data_mp[:, data_mp[0,:] < num_playlists] # removing duplicate edges (since data_mp is undirected) 
                                                           # and also gets all playlists to be in row 0 and all songs in row 1
   playlist_to_idx_in_batch = {playlist: i for i, playlist in enumerate(unique_playlists.tolist())}
   exclude_playlists = []
   exclude_songs = []
   for i in range(known_edges.shape[1]):
      pl, song = known_edges[:,i].tolist()
      if pl in playlist_to_idx_in_batch: # don't need the edges in data_mp that are from playlists that are not in this batch
         exclude_playlists.append(playlist_to_idx_in_batch[pl])
         exclude_songs.append(song - num_playlists) # subtract num_playlists to get indexing into all_ratings correct
   all_ratings[exclude_playlists, exclude_songs] = -10000 # setting to a very low score

   # Get top k recommendations for each song
   _, top_k = torch.topk(all_ratings, k=k, dim=1)
   top_k += num_playlists # topk returned indices of songs in ratings, which doesn't include playlists. Need to shift up by num_playlists
    
   # Calculate recall@k
   ret = {}
   for i, playlist in enumerate(unique_playlists):
      pos_songs = ground_truth[1, ground_truth[0, :] == playlist]
      assert len(pos_songs) > 0

      k_recs = top_k[i, :]
      recall = len(np.intersect1d(pos_songs, k_recs)) / len(pos_songs)
      ret[playlist] = recall
   return ret
