import math
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
import itertools

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


import pandas as pd
#Centroids of clusters/labels
def getCentroidDists(embed,clusType):
	""" Compute inter-distances for a set of label centroids (all pairwise distances)

	Parameters
	----------
	embed : Numpy array for latent space (n_obs x n_features or n_latent)
	clusType : Numpy array of list of labels for a class (n_obs, )

	Returns
	-------
	dists: List of pairwise distances between centroids of labels in clusType
	"""
	clusters = np.unique(clusType)

	centroids = np.zeros((len(clusters),embed.shape[1]))

	for i in range(len(clusters)):

		sub_data = embed[clusType == clusters[i],:]
		centroid = sub_data.mean(axis=0)

		centroids[i,:] = list(centroid)

	dists = pairwise_distances(centroids,centroids,metric='l1')

	return dists.flatten().tolist()

#Get distances to centroids of clusters/labels
def getCentroidDists_oneVsAll(embed,clusType,clus):
	""" Compute inter-distances for one label versus the remaining (subset of all pairwise distances)

	Parameters
	----------
	embed : Numpy array for latent space (n_obs x n_features or n_latent)
	clusType : Numpy array of list of labels for a class (n_obs, )
	clus : Specific label (single string input) from which to calculate inter-distances

	Returns
	-------
	dists : List of distances between centroid of clus label to all other lable centroids in clusType
	"""
	clusters = np.unique(clusType)

	centroids = np.zeros((len(clusters),embed.shape[1]))

	comp = embed[clusType == clus,:]
	comp_centroid = comp.mean(axis=0)

	for i in range(len(clusters)):

		sub_data = embed[clusType == clusters[i],:]
		centroid = sub_data.mean(axis=0)

		centroids[i,:] = list(centroid)

	dists = pairwise_distances(comp_centroid.reshape(1, -1),centroids,metric='l1')

	return dists.flatten().tolist()


def getIntraVar(embed, outLab, inLab):
	""" Compute intra-distances for an inner set of labels (averaged distances)

	Parameters
	----------
	embed : Numpy array for latent space (n_obs x n_features or n_latent)
	outLab : 1D array for outer label (e.g. cell type) (n_obs, )
	inLab : 1D array for inner label (e.g. sex) (n_obs, )

	Returns
	-------
	avg_dists : List of average pairwise distances within labels in inLab
	"""
	outs = np.unique(outLab)
	avg_dists = []

	for i in outs:

		sub = embed[outLab == i,:]

		sub_ins = inLab[outLab == i]
		ins = np.unique(sub_ins)

		for j in ins:

			sub_i = sub[sub_ins == j,:]
			if sub_i.shape[0] > 1:
				
				d = pairwise_distances(sub_i,sub_i,metric='l1')
				np.fill_diagonal(d, np.nan)
				d = d[~np.isnan(d)].reshape(d.shape[0], d.shape[1] - 1)

				f_d = d.flatten().tolist()

				avg_dists += [np.mean(f_d)] 

	return avg_dists

def getInterVar(embed, outLab, inLab):
	""" Compute inter-distances for an inner set of labels (averaged distances)

	Parameters
	----------
	embed : Numpy array for latent space (n_obs x n_features or n_latent)
	outLab : 1D array for outer label (e.g. cell type) (n_obs, )
	inLab : 1D array for inner label (e.g. sex) (n_obs, )

	Returns
	-------
	avg_dists: List of average pairwise distances between labels in inLab
	"""


	outs = np.unique(outLab)
	avg_dists = []

	for i in outs:
		sub = embed[outLab == i,:]

		sub_ins = inLab[outLab == i]
		ins = np.unique(sub_ins)

		pairs = list(itertools.combinations(ins, 2))
		for p in pairs:

			sub_1 = sub[sub_ins == p[0],:]
			sub_2 = sub[sub_ins == p[1],:]
			avg_dists += [np.mean(pairwise_distances(sub_1,sub_2,metric='l1').flatten().tolist())]



	return avg_dists



def getNeighbors(embed, n_neigh = 15, p=1):
	"""Get indices of nearest neighbors for all points in embedding 

	Parameters
	----------
	embed : Numpy array for latent space (n_obs x n_features or n_latent)
	n_neigh : No. of neighbors for each cell. Default is 15
	p : Distance metric (1= Manhattan, 2= Euclidean) (see options in sklearn.neighbors.DistanceMetric). Default is 1

	Returns
	-------
	indices : Matrix of n_obs x n_neigh with indices of nearest neighbors for each obs
	"""
	nbrs = NearestNeighbors(n_neighbors=n_neigh, p=p).fit(embed)
	distances, indices = nbrs.kneighbors(embed)

	return indices

def getIntersect(orig, new):
	"""Get fraction of neighbors intersecting with ambient space neighbors

	Parameters
	----------
	orig : Original/ambient space nearest neighbor indices, from getNeighbors() (n_obs x n_neigh)
	new : Latent/Comparison space nearest neighbor indices, from getNeighbors() (n_obs x n_neigh)

	Returns
	-------
	frac : List of fraction of neighbors shared in new space, for each obs"""
	frac = [0]*new.shape[0]
	for i in range(new.shape[0]):
		inter = set(orig[i,:]).intersection(new[i,:])
		frac[i] = len(inter)/orig.shape[1]

	return frac


def getJaccard(orig, new):
	"""Get jaccard distance of neighbors intersecting with those in ambient space

	Parameters
	----------
	orig : Original/ambient space nearest neighbor indices, from getNeighbors() (n_obs x n_neigh)
	new : Latent/Comparison space nearest neighbor indices, from getNeighbors() (n_obs x n_neigh)

	Returns
	-------
	frac : List of Jaccard distances for each obs
	"""
	frac = [0]*new.shape[0]
	for i in range(new.shape[0]):
		inter = set(orig[i,:]).intersection(new[i,:])
		frac[i] = 1 - len(inter)/len(set(orig[i,:]).union(new[i,:]))

	return frac


def frac_unique_neighbors(latent, cluster_label, metric = 1,neighbors = 30):
	""" Calculates the fraction of nearest neighbors for each point (cell) with same label as point itself

	Parameters
	----------
	latent : Numpy array of latent space (n_obs x n_latent)
	cluster_label : Numpy array of list of labels for all n_obs (n_obs, )
	metrics : Distance metric, 1 = manhattan (see options in sklearn.neighbors.DistanceMetric)
	neighbors : No. of nearest neighbors to consider

	Returns
	-------
	frac_neighbors : Dictionary with each key a unique label in the class cluster_label, and
	each value a list of the fraction of neighbors in the same label (for each cell in that label)
	unique_clusters : Dictionary with each key a unique label in the category cluster_label and
	each value a list of the unique labels of each cell's neighbors (for each cell in that label)
	"""
	cats = pd.Categorical(cluster_label)
	# Get nearest neighbors in embedding
	n = neighbors
	neigh = NearestNeighbors(n_neighbors=n, p=metric)
	
	clusters = np.unique(cluster_label)
	unique_clusters = {}
	frac_neighbors = {}
	X_full  = latent
	neigh.fit(X_full)

	for c in clusters:
		X  = latent[cats == c, :]
		# Find n nearest neighbor cells (L1 distance)
		kNeigh = neigh.kneighbors(X)
		matNeigh = kNeigh[1]
		frac = np.zeros(matNeigh.shape[0])
		#How many of top n neighbors come from same cluster in the labeled data (out of n neighbors)
		unique_clusters[c] = np.unique([cats[matNeigh[i]] for i in range(0, len(frac))])
		frac_neighbors[c] = [cats[matNeigh[i]].value_counts()[c]/n for i in range(0,len(frac))]

	return frac_neighbors, unique_clusters

