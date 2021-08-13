#pip3 install torch
#pip3 install anndata
#pip3 install matplotlib
#pip3 install scikit-learn
#pip3 install torchsummary
#pip3 install --quiet umap-learn

import torch
from torch import nn, optim
import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt
import random
from torchsummary import summary
from collections import Counter
import itertools

class autoencoder(nn.Module):
	"""
	Create autoencoder architecture
	Returns: autoencoder object
    """
	def __init__(self,n_input: int, n_hidden: int, n_output: int, dropout_rate = 0.1):
		super(autoencoder,self).__init__()

		#Encoder
		self.encoder = nn.Sequential(nn.Linear(n_input, n_hidden),
			#Parameter value from scVI original tensorflow implementation
			nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001),
			nn.ReLU(True),
			nn.Dropout(p=dropout_rate),
			nn.Linear(n_hidden, n_output))

		#Linear decoder
		self.decoder = nn.Linear(n_output, n_input, bias=False)

	def forward(self, x):
		z = self.encoder(x)
		return self.decoder(z), z


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MCML():
	"""
	Create object for fitting NCA model
	Returns: NCA model object
    """

	def __init__(self, n_latent = 10, n_hidden = 128, epochs = 100,batch_size = 128, lr = 1e-3, weight_decay=1e-5):
		#super(NN_NCA, self).__init__()

		#torch.manual_seed(0)

		self.n_latent = n_latent
		self.epochs = epochs
		self.n_hidden = n_hidden
		self.model = None
		self.batch_size = batch_size
		self.lr = lr
		self.weight_decay = weight_decay

		self.set_weights = False
		self.weights = None
		self.Losses = None
		self.test_losses = None




	def normLabels(self, Y): #allLabs
		"""
		Parameters:
	    Y : 2d np array (from lists of lists), columns represent each class of labels

	    Returns :
	    Normalized booleam mask for pairwise comparisons between points for each label class
	    """
		#Weight mask for each label type by block area
		masks = []

		#Number of label types (Classes)
		num_labs = Y.shape[0] 

		#Number of cells/observations
		cells = Y.shape[1]

		m0 = np.zeros((cells,cells))
		maxVal = 0

		for i in range(num_labs):
			allLabs = Y[i,:]
			counts = Counter(allLabs)

			#Counts for each unique label
			area_counts = {k:  v for k, v in list(counts.items())} #**2 
			
			#Find all cell pairs with the same label
			m = allLabs[:, np.newaxis] == allLabs[np.newaxis, :]

			#Count number of labels (for each unique label)
			for k, v in list(area_counts.items()):
				#if '*nan*' not in k:
				if 'nan' != str(k): 
					#masks += [m*(allLabs == k)*(1/v)] #*(1/maxVal) *maxVal
					
					maxVal += 1
					m0 += m*(allLabs == k)*(1/v)

		if maxVal != 0:
			m0 = m0*(1/maxVal)

		return torch.from_numpy(m0).float().to(device) #np.array(masks)





	def multiLabelMask(self,Y, Y_b_cont, dim_cont, cont): 
		"""
		Parameters:
	    Y : 2d np array (from lists of lists), columns represent each class of labels
	    Y_b_cont : Additional numpy array of lists for continuous labels
		dim_cont : List of dimension for each continuous label (one value per multi-dim label)
		cont : Boolean, if continuous labels are present

	    Returns :
	    Masks for pairwise comparisons between points for each label class
	    """

	    #Loop through all continuous classes of labels
		if cont: #Continuous classes
			Y_b_cont = torch.from_numpy(Y_b_cont).float().to(device)
			Y_b_cont = torch.transpose(Y_b_cont,0, 1)

			n_obs = Y_b_cont.size()[0]
			if dim_cont is None:
				dim_cont = [1]*Y_b_cont.size()[1]

			weights = torch.empty(n_obs, n_obs*len(dim_cont),dtype=torch.float,device=device)
			i = 0
			dim_i = 0
			while dim_i < len(dim_cont):
				end = i + dim_cont[dim_i] 
				cont_lab =  Y_b_cont[:,i:end]

				#Calculate distances
				cont_dists = self.pairwise_dists(cont_lab,cont_lab) #n_obs x n_obs

				#Set diag to inf
				cont_dists.diagonal().copy_(np.inf*torch.ones(len(cont_dists)))

				cont_dists = torch.nan_to_num(cont_dists, nan=np.inf) #Use nans for data missing labels

				#Softmax on negative distances
				cont_dists = self.softmax(-cont_dists)



				#Add to tensor matrix object
				s = n_obs*dim_i
				e = n_obs*(dim_i+1)
				weights[:,s:e] = cont_dists

				
				i += dim_cont[dim_i]
				dim_i += 1
		else:
			weights = None




		masks = self.normLabels(Y) #np.array(masks) Y[0]



	
		return masks, weights

	def pairwise_dists(self,z1,z2,p=2.0):
		"""
		Parameters:
		z1 : Input matrix 1
		z2 : Input matrix 2
		p : Distance metric (1=manhattan, 2=euclidean)
		Returns :
		Pairwise distance matrix between z1 and z2
		"""
		d1 = z1.clone()
		d2 = z2.clone()
		dist = torch.cdist(d1, d2, p=p)
		#dist = torch.clamp(dist, min=0)
		return dist.clone()


	def softmax(self, p):
		"""
		Parameters:
		p : n_obs x n_obs probability matrix
		Returns :
		Softmax of matrix p
		"""
		#Based on sklearn NCA implementation

		#Subtract max prob from each row for numerical stability
		p = p.clone()
		max_prob, max_indexes = torch.max(p,dim=1,keepdim=True)
		p = p - max_prob.expand_as(p)
		p = torch.exp(p)
		sum_p = torch.sum(p,dim=1,keepdim=True)
		p = p / sum_p.expand_as(p)
		return p

	def lossFunc(self, recon_batch, X_b, z, masks, weights, cont, lab_weights, fracNCA):
		"""
		Parameters:
		recon_batch : Reconstruction from decoder for mini-batch
		X_b : Mini-batch of X
		z : Latent space
		masks : Array of pairwise masks
		weights : Pairwise weights from continuous label distances
		cont : Boolean, if continuous labels are present
		lab_weights : Weights for each label's masks in loss calculation
		fracNCA : Fraction of NCA cost in loss calculation
		Returns :
		Loss value with NCA cost and Reconstruction loss
		"""
		losses = []

		#Reconstruction loss
		recon_loss_b = torch.norm(recon_batch-X_b) #L2 norm


		losses += [recon_loss_b]

		#Calculate distances
		p_ij = self.pairwise_dists(z,z)

		#Set diag to inf
		p_ij.diagonal().copy_(np.inf*torch.ones(len(p_ij)))

		#Softmax on negative distances
		p_ij = self.softmax(-p_ij)

		#Calculate masked p_ij (over multiple discrete labels)
		masked_pij = p_ij * masks

		losses += [torch.sum(masked_pij)]
	



		if cont:# Continuous labels
			n_obs = int(weights.size()[0])
			num_weights = int(weights.size()[1])/n_obs
			num_weights = int(num_weights)

			for n in range(0,num_weights):
				s = n_obs*n
				e = n_obs*(n+1)

				weight_calc = p_ij * weights[:,s:e]

				m, m_indexes = torch.max(weights[:,s:e],dim=1)

				masked_pij = weight_calc / torch.sum(m)#/ weight_calc.size()[0] #masked_pij + 
				cont_max, inds = torch.max(masked_pij,dim=1)
				
				losses += [torch.sum(masked_pij)]
				#losses += [torch.sum(cont_max)]


		

		#Loss with NCA cost and Euclidean distance (reconstruction loss)

		lossVals = torch.stack(losses,dim=0)
		

		scaled_losses = lossVals

		p_sum = torch.sum(scaled_losses[1]) #Don't really need sum here

		#print(p_sum)

		if cont:
			p_sum_cont = torch.sum(scaled_losses[2:2+num_weights])
		else:
			p_sum_cont = torch.tensor(0, device=device)



		loss = -10*fracNCA*(p_sum + p_sum_cont) + (1-fracNCA)*(scaled_losses[0])



		
		return p_sum, p_sum_cont, scaled_losses[0], loss 


	def getLoadings(self):
		"""
		Returns :
		Weights from the decoder layer, matrix of n_features x n_hidden
		"""
		if self.model != None:
			return self.model.decoder.weight.detach().cpu().numpy()
		else:
			return None

	def plotLosses(self, figsize=(15,4),fname=None,axisFontSize=11,tickFontSize=10):
		"""
		Parameters:
		figsize : Tuple for figure size
		fname : Name for file to save figure to, if None plot is displayed
		axisFontSize : Font size for axis labels
		tickFontSize : Font size for tick labels
		Returns :
		Plot of each loss term over epochs
		"""
		fig, axs = plt.subplots(1, self.Losses.shape[1],figsize=figsize)
		titles = ['Discrete NCA','Continuous NCA','Reconstruction','Total Loss']
		if(isinstance(self.test_losses, np.ndarray)):

			for i in range(self.Losses.shape[1]):
				axs[i].plot(self.Losses[:,i],label='Train Loss')
				axs[i].plot(self.test_losses[:,i],label='Test Loss')
				axs[i].set_title(titles[i],fontsize=axisFontSize)
				plt.setp(axs[i].get_xticklabels(), fontsize=tickFontSize)
				plt.setp(axs[i].get_yticklabels(), fontsize=tickFontSize)

				axs[i].grid(False)
			plt.legend(prop={'size': axisFontSize})
			plt.xlabel('Epoch',fontsize=axisFontSize)
			plt.ylabel('Loss',fontsize=axisFontSize)

		else:
			for i in range(self.Losses.shape[1]):
				axs[i].plot(self.Losses[:,i])
				axs[i].set_title(titles[i],fontsize=axisFontSize)
				#axs[i].tick_params(axis="x", fontsize=tickFontSize) 
				plt.setp(axs[i].get_xticklabels(), fontsize=tickFontSize)
				plt.setp(axs[i].get_yticklabels(), fontsize=tickFontSize)

				axs[i].grid(False)

			plt.xlabel('Epoch',fontsize=axisFontSize)
			plt.ylabel('Loss',fontsize=axisFontSize)


		fig.tight_layout()
		if(fname != None):
			plt.savefig(fname)
		else:
			plt.show()


	def fit(self,X,Y,Y_cont = None, dim_cont = None, lab_weights = None, fracNCA = 0.8, silent = False, ret_loss = False):
		"""
		Parameters:
		X : Input data as numpy array (n_obs x n_features)
		Y : Label matrix, numpy array of lists. Col is label, Row is each label class. (n_classes x n_obs)
		Y_cont : Additional numpy array of lists for continuous labels (optional)
		dim_cont : List of dimension for each continuous label (one value per multi-dim label) (optional)
		lab_weights : Weights for each label's masks in loss calculation (optional) (currently not used)
		fracNCA : Fraction of NCA cost in loss calculation (default is 0.8)
		silent : Print average loss per epoch (default is False)
		ret_loss : Boolean to return matrix of loss values over epochs

		Returns :
		Latent space representation of X
		"""

		cont = isinstance(Y_cont, np.ndarray)


		iters_per_epoch = int(np.ceil(X.shape[0] / self.batch_size))

		model = autoencoder(X.shape[1], self.n_hidden, self.n_latent).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

		#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose = not silent)

		#Print model summary
		
		# print("Num Trainable Parameters: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
		# summary(model, (self.batch_size,X.shape[1]), self.batch_size)

		X = torch.from_numpy(X).float().to(device)

		#Get Cartesian product of labels across classes
		#Y = self.mergeLabels(Y)
		
	
		loss_values = []
		for e in range(self.epochs):

			#Shuffle data
			permutation = torch.randperm(X.size()[0])

			model.train()
			allLosses = torch.tensor(0,device=device)

			with torch.autograd.set_detect_anomaly(True):
				for b in range(iters_per_epoch):

					indices = permutation[b*self.batch_size:(b+1)*self.batch_size]
					X_b, Y_b = X[indices], Y[:,indices]

					if cont:
						Y_b_cont = Y_cont[:,indices]
					else:
						Y_b_cont = None

		


					#Make y_mask
					masks, weights = self.multiLabelMask(Y_b, Y_b_cont, dim_cont, cont)

					#Set grad to zero, compute loss, take gradient step
					optimizer.zero_grad()
					recon_batch, z = model(X_b)
					losses = self.lossFunc(recon_batch, X_b, z, masks,weights, cont, lab_weights, fracNCA) #*****
					
					
					losses[-1].backward()

					allLosses = allLosses + torch.stack(losses,dim=0)
	

					optimizer.step()


			#scheduler.step(allLosses[-1].item())

			if silent != True:
				print('====> Epoch: {} Average loss: {:.4f}'.format(e, allLosses[-1].item() / len(X)))

			loss_values.append([allLosses[i].item() / len(X) for i in range(len(allLosses))])



		model.eval()
		recon_batch, z = model(X)
		self.model = model
		self.Losses = np.array(loss_values)
		if ret_loss:
			return np.array(loss_values), z.detach().cpu().numpy()
		else:
			return z.detach().cpu().numpy()


	def trainTest(self,X,Y,Y_cont = None, dim_cont = None, lab_weights = None, trainFrac = 0.8, fracNCA = 0.8, silent = False):
		"""
		Parameters:
		X : Input data as numpy array (n_obs x n_features)
		Y : Label matrix, numpy array of lists. Each col is label, rows are value of obs in that label class. ()
		Y_cont : Additional numpy array of lists for continuous labels (optional)
		dim_cont : List of dimension for each continuous label (one value per multi-dim label) (optional)
		lab_weights : Weights for each label's masks in loss calculation (optional) (currently not used)
		trainFrac : Fraction of X to use for training
		fracNCA : Fraction of NCA cost in loss calculation (default is 0.8)
		silent : Print average loss per epoch (default is False)

		Returns :
		Loss values from training and validation batches of X
		"""

		# Plot training and testing loss
		cont = isinstance(Y_cont, np.ndarray)
		

		#Y = self.mergeLabels(Y)

		trainSize = int(np.floor(trainFrac*X.shape[0]))
		trainInd = random.sample(range(0,X.shape[0]), trainSize) 
		testInd = [i not in trainInd for i in range(0,X.shape[0])]

		X_train = X[trainInd,:]
		Y_train = Y[:,trainInd]

		X_test = X[testInd,:]
		Y_test = Y[:,testInd]

		if cont:
			Y_c_train = Y_cont[:,trainInd]
			Y_c_test = Y_cont[:,testInd]
		else:
			Y_c_train = None
			Y_c_test = None

	


		#print(X.shape)
		iters_per_epoch = int(np.ceil(X_train.shape[0] / self.batch_size))

		model = autoencoder(X_train.shape[1], self.n_hidden, self.n_latent).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

		#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose = not silent)


		X_train = torch.from_numpy(X_train).float().to(device)
		X_test = torch.from_numpy(X_test).float().to(device)
		#print(X.size())
		loss_values = []
		test_loss_values = []
		for e in range(self.epochs):

			#Shuffle data
			permutation = torch.randperm(X_train.size()[0])

			model.train()
			allLosses = torch.tensor(0,device=device) 

			with torch.autograd.set_detect_anomaly(True):
				for b in range(iters_per_epoch):

					#Choose batch

					indices = permutation[b*self.batch_size:(b+1)*self.batch_size]
					X_b, Y_b = X_train[indices], Y_train[:,indices]
					if cont:
						Y_b_cont = Y_c_train[:,indices]
					else:
						Y_b_cont = None

			

					#Make y_mask
					masks, weights = self.multiLabelMask(Y_b, Y_b_cont, dim_cont, cont)

					#Set grad to zero, compute loss, take gradient step
					optimizer.zero_grad()
					recon_batch, z = model(X_b)
					losses = self.lossFunc(recon_batch, X_b, z, masks, weights, cont, lab_weights, fracNCA)

		

					losses[-1].backward()
					
					allLosses = allLosses + torch.stack(losses,dim=0)
					optimizer.step()



			test_losses = self.test(model, X_test, Y_test, Y_c_test, dim_cont, lab_weights = lab_weights, fracNCA = fracNCA, silent = silent)

			#scheduler.step(test_losses[-1])

			if silent != True:
				print('====> Epoch: {} Average loss: {:.4f}'.format(e, allLosses[-1] / len(X_train)))

			loss_values.append([allLosses[i].item() / len(X_train) for i in range(len(allLosses))])
			test_loss_values.append(test_losses)

		self.Losses = np.array(loss_values)
		self.test_losses = np.array(test_loss_values)

		return np.array(loss_values), np.array(test_loss_values)


	def test(self, model, X, Y, Y_cont = None, dim_cont = None, lab_weights = None, fracNCA = 0.8, silent = False):
			
		cont = isinstance(Y_cont, np.ndarray)
		#Shuffle data
		permutation = torch.randperm(X.size()[0])
		iters_per_epoch = int(np.ceil(X.size()[0] / self.batch_size))

		model.eval()
		allLosses = torch.tensor(0,device=device) 

		with torch.no_grad():

			for b in range(iters_per_epoch):

				#Choose batch
				indices = permutation[b*self.batch_size:(b+1)*self.batch_size]
				X_b, Y_b = X[indices], Y[:,indices]

				if cont:
					Y_b_cont = Y_cont[:,indices]
				else:
					Y_b_cont = None

				

				#Make y_mask
				masks, weights = self.multiLabelMask(Y_b, Y_b_cont, dim_cont, cont)

				#Set grad to zero, compute loss, take gradient step
				recon_batch, z = model(X_b)
				losses = self.lossFunc(recon_batch, X_b, z, masks, weights, cont, lab_weights, fracNCA)

				
				allLosses = allLosses + torch.stack(losses,dim=0)


		test_loss = allLosses[-1]/len(X)

		if silent != True:
			print('====> Test set loss: {:.4f}'.format(test_loss))


		return [allLosses[i].item() / len(X) for i in range(len(allLosses))]




class bMCML(MCML):
	"""
	Create object for fitting biased reconstruction model
	Returns: Biased Recon model object
    """

	def __init__(self, n_latent = 10, n_hidden = 128, epochs = 100,batch_size = 400, lr = 1e-3, weight_decay=1e-5):
		super().__init__(n_latent, n_hidden, epochs,batch_size, lr, weight_decay)

	def getDist(self, embed, outLab, inLab):
		"""
		Get average distances/variances between cells within labels.
		Currently only for internal (intra-) variances rather than inter-label distances.
		
		outLab: 1D numpy array, outer label (e.g. cell type)
		inLab: 1D numpy array, inner label (e.g. sex)
		Returns: 
		Average L1 pairwise distances between cells within inLab labels
	    """
		outs = np.unique(outLab)
		avg_dists = []

		for i in outs:
			pos = outLab == i
			pInd = np.where(pos)[0]

			subInd = torch.tensor(pInd,dtype=torch.int64,device=device)

			sub = torch.index_select(embed, 0, subInd)#embed[outLab == i,:]

			sub_ins = inLab[outLab == i]
			ins = np.unique(sub_ins)

			for j in ins:

				subPos = sub_ins == j
				newInd = np.where(subPos)[0]
				newTInd = torch.tensor(newInd,dtype=torch.int64,device=device)


				sub_i = torch.index_select(sub, 0, newTInd)#sub[sub_ins == j,:]
				if sub_i.size()[0] > 1:
					#lens += [len([i for i in pairwise_distances(sub_i,sub_i,metric='l1').flatten().tolist() if i !=0])]
					d = self.pairwise_dists(sub_i,sub_i,p=1.0)
					
					#np.fill_diagonal(d, np.nan)
					#d = d[~np.isnan(d)].reshape(d.shape[0], d.shape[1] - 1)

					#f_d = d.flatten().tolist()

					#Ignore diagonal (zero) elements
					d = d.masked_select(~torch.eye(d.size()[0], dtype=bool,device=device)).view(d.size()[0], d.size()[0] - 1)
					avg_dists += [torch.mean(d)] #[np.mean([i for i in pairwise_distances(sub_i,sub_i,metric='l1').flatten().tolist() if i !=0])]

		
		avgVals = torch.stack(avg_dists,dim=0)

		return avgVals




	def lossFunc(self, recon_batch, X_b, Yout_b, Yin_b):
		"""
		Parameters:
		recon_batch : Reconstruction from decoder for mini-batch
		X_b : Mini-batch of X
		Yout_b : Batch selection of outer-labels
		Yin_b : Batch selection of inner-labels
		
		Returns :
		Loss value with Biased Reconstruction loss
		"""
		losses = []

		

		# ----- Try maximizing label correlation (Pearson) between recon and X input ----- (minimize negative)

		#Calculate variances for recon_batch and X_b 
		rDists = self.getDist(recon_batch, Yout_b, Yin_b)
		xDists = self.getDist(X_b, Yout_b, Yin_b)

		vx = rDists - torch.mean(rDists)
		vy = xDists - torch.mean(xDists)

		recon_loss_b = -1*(torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))

		
		#recon_loss_b = 
		losses += [recon_loss_b]

		
		lossVals = torch.stack(losses,dim=0)
		
		scaled_losses = lossVals 


		loss = scaled_losses[0]

		
		return scaled_losses[0], loss

	def plotLosses(self, figsize=(15,4),fname=None,axisFontSize=11,tickFontSize=10):
		"""
		Parameters:
		figsize : Tuple for figure size
		fname : Name for file to save figure to, if None plot is displayed
		axisFontSize : Font size for axis labels
		tickFontSize : Font size for tick labels
		Returns :
		Plot of each loss term over epochs
		"""
		fig, axs = plt.subplots(1, self.Losses.shape[1],figsize=figsize)
		titles = ['Biased Reconstruction','Total Loss']
		if(isinstance(self.test_losses, np.ndarray)):

			for i in range(self.Losses.shape[1]):
				axs[i].plot(self.Losses[:,i],label='Train Loss')
				axs[i].plot(self.test_losses[:,i],label='Test Loss')
				axs[i].set_title(titles[i],fontsize=axisFontSize)
				plt.setp(axs[i].get_xticklabels(), fontsize=tickFontSize)
				plt.setp(axs[i].get_yticklabels(), fontsize=tickFontSize)

				axs[i].grid(False)
			plt.legend(prop={'size': axisFontSize})
			plt.xlabel('Epoch',fontsize=axisFontSize)
			plt.ylabel('Loss',fontsize=axisFontSize)

		else:
			for i in range(self.Losses.shape[1]):
				axs[i].plot(self.Losses[:,i])
				axs[i].set_title(titles[i],fontsize=axisFontSize)
				plt.setp(axs[i].get_xticklabels(), fontsize=tickFontSize)
				plt.setp(axs[i].get_yticklabels(), fontsize=tickFontSize)

				axs[i].grid(False)

			plt.xlabel('Epoch',fontsize=axisFontSize)
			plt.ylabel('Loss',fontsize=axisFontSize)


		fig.tight_layout()
		if(fname != None):
			plt.savefig(fname)
		else:
			plt.show()


	def fit(self, X, Yout, Yin, silent = False, ret_loss = False):
		"""
		Parameters:
		X : Input data as numpy array (obs x features)
		Y_out : Outer label matrix, numpy array of list. Outer label within which to do correlation calculation.
		Y_in : Inner label matrix, numpy array of list. Label to do correlation calculation on, within Outer label (e.g. sexes within cell types)
		
		silent : Print average loss per epoch (default is False)
		ret_loss : Boolean to return loss values over epochs

		Returns :
		Latent space representation of X
		"""


		iters_per_epoch = int(np.ceil(X.shape[0] / self.batch_size))

		model = autoencoder(X.shape[1], self.n_hidden, self.n_latent).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


		X = torch.from_numpy(X).float().to(device)
		
	
		loss_values = []
		for e in range(self.epochs):

			#Shuffle data
			permutation = torch.randperm(X.size()[0])

			model.train()
			allLosses = torch.tensor(0,device=device)

			with torch.autograd.set_detect_anomaly(True):
				for b in range(iters_per_epoch):

					indices = permutation[b*self.batch_size:(b+1)*self.batch_size]
					#X_b, Yout_b, Yin_b = X[indices], Yout[:,indices], Yin[:,indices]
					X_b, Yout_b, Yin_b = X[indices], Yout[indices], Yin[indices]


					
					#Set grad to zero, compute loss, take gradient step
					optimizer.zero_grad()
					recon_batch, z = model(X_b)
					losses = self.lossFunc(recon_batch, X_b, Yout_b, Yin_b) #*****

					
				
					losses[-1].backward()

					allLosses = allLosses + torch.stack(losses,dim=0)
	

					optimizer.step()


			#scheduler.step(allLosses[-1].item())

			if silent != True:
				print('====> Epoch: {} Average loss: {:.4f}'.format(e, allLosses[-1].item() / len(X)))

			loss_values.append([allLosses[i].item() / len(X) for i in range(len(allLosses))])



		model.eval()
		recon_batch, z = model(X)
		self.model = model
		self.Losses = np.array(loss_values)
		if ret_loss:
			return np.array(loss_values), z.detach().cpu().numpy()
		else:
			return z.detach().cpu().numpy()


	def trainTest(self,X,Yout, Yin, trainFrac = 0.8, silent = False):
		"""
		Parameters:
		X : Input data as numpy array (obs x features)
		Y : Label matrix, numpy array of lists. Each col is label, rows represent value of obs in that label class.
		Y_out : Outer label matrix, numpy array of list. Outer label within which to do correlation calculation.
		Y_in : Inner label matrix, numpy array of list. Label to do correlation calculation on, within Outer label (e.g. sexes within cell types)
		
		trainFrac : fraction of X used for training
		silent : Print average loss per epoch (default is False)

		Returns :
		Loss values from training and validation batches of X
		"""



		trainSize = int(np.floor(trainFrac*X.shape[0]))
		trainInd = random.sample(range(0,X.shape[0]), trainSize) 
		testInd = [i not in trainInd for i in range(0,X.shape[0])]

		X_train = X[trainInd,:]
		Y_train_out = Yout[:,trainInd]
		Y_train_in = Yin[:,trainInd]


		X_test = X[testInd,:]
		Y_test_out = Yout[:,testInd]
		Y_test_in = Yin[:,testInd]


		#print(X.shape)
		iters_per_epoch = int(np.ceil(X_train.shape[0] / self.batch_size))

		model = autoencoder(X_train.shape[1], self.n_hidden, self.n_latent).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


		X_train = torch.from_numpy(X_train).float().to(device)
		X_test = torch.from_numpy(X_test).float().to(device)
		#print(X.size())
		loss_values = []
		test_loss_values = []
		for e in range(self.epochs):

			#Shuffle data
			permutation = torch.randperm(X_train.size()[0])

			model.train()
			allLosses = torch.tensor(0,device=device) 

			with torch.autograd.set_detect_anomaly(True):
				for b in range(iters_per_epoch):

					#Choose batch

					indices = permutation[b*self.batch_size:(b+1)*self.batch_size]
					X_b, Yout_b, Yin_b = X_train[indices], Y_train_out[:,indices], Y_train_in[:,indices]
					

					#Set grad to zero, compute loss, take gradient step
					optimizer.zero_grad()
					recon_batch, z = model(X_b)
					losses  = self.lossFunc(recon_batch, X_b, Yout_b, Yin_b) #*****


					losses[-1].backward()

					allLosses = allLosses + torch.stack(losses,dim=0)
					optimizer.step()



			test_losses = self.test(model, X_test, Y_test_out, Y_test_in, silent = silent)
			
			if silent != True:
				print('====> Epoch: {} Average loss: {:.4f}'.format(e, allLosses[-1] / len(X_train)))

			loss_values.append([allLosses[i].item() / len(X_train) for i in range(len(allLosses))])
			test_loss_values.append(test_losses)

		self.Losses = np.array(loss_values)
		self.test_losses = np.array(test_loss_values)

		return np.array(loss_values), np.array(test_loss_values)


	def test(self, model, X, Yout, Yin, silent = False):
			

		#Shuffle data
		permutation = torch.randperm(X.size()[0])
		iters_per_epoch = int(np.ceil(X.size()[0] / self.batch_size))

		model.eval()
		allLosses = torch.tensor(0,device=device) 

		with torch.no_grad():

			for b in range(iters_per_epoch):

				#Choose batch
				indices = permutation[b*self.batch_size:(b+1)*self.batch_size]
				X_b, Yout_b, Yin_b = X[indices], Yout[:,indices], Yin[:,indices]

			
				#Set grad to zero, compute loss, take gradient step
				recon_batch, z = model(X_b)
				losses = self.lossFunc(recon_batch, X_b, Yout_b, Yin_b)


				
				allLosses = allLosses + torch.stack(losses,dim=0)


		test_loss = allLosses[-1]/len(X)

		if silent != True:
			print('====> Test set loss: {:.4f}'.format(test_loss))


		return [allLosses[i].item() / len(X) for i in range(len(allLosses))]



