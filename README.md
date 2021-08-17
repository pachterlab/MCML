# MCML

MCML is a toolkit for semi-supervised dimensionality reduction and quantitative analysis of Multi-Class, Multi-Label data. We demonstrate its use for single-cell datasets though the method can use any matrix as input.

MCML __modules__ include the __MCML__ and __bMCML__ algorithms for dimensionality reduction, and MCML __tools__ include functions for quantitative analysis of inter- and intra- distances between labeled groups and nearest neighbor metrics in the latent or ambient space. The __modules__ are autoencoder-based neural networks with label-aware cost functions for weight optimization.

Briefly, __MCML__ adapts the [Neighborhood Component Analysis algorithm](https://www.cs.toronto.edu/~hinton/absps/nca.pdf) to utilize mutliple classes of labels for each observation (cell) to embed observations of the same labels close to each other. This essentially optimizes the latent space for k-Nearest Neighbors (KNN) classification.

__bMCML__ demonstrates targeted reconstruction error, which optimizes for recapitulation of intra-label distances (the pairwise distances between cells within the same label). 

__tools__ include functions for inter- and intra-label distance calculations as well as metrics on the labels of n the k nearest neighbors of each observation. These can be performed on any latent or ambient space (matrix) input. 

Requirements
------------

You need Python 3.6 or later to run MCML.  You can have multiple Python
versions (2.x and 3.x) installed on the same system without problems.

In Ubuntu, Mint and Debian you can install Python 3 like this:

    $ sudo apt-get install python3 python3-pip

For other Linux distributions, macOS and Windows, packages are available at

  https://www.python.org/getit/


Quick start
-----------

MCML can be installed using pip:

    $ python3 -m pip install -U MCML

If you want to run the latest version of the code, you can install from git:

    $ python3 -m pip install -U git+git://github.com/pachterlab/MCML.git


Examples
-----------

Example data download:

    $ wget --quiet https://caltech.box.com/shared/static/i66kelel9ouep3yw8bn2duudkqey190j
    $ mv i66kelel9ouep3yw8bn2duudkqey190j mat.mtx
    $ wget --quiet https://caltech.box.com/shared/static/dcmr36vmsxgcwneh0attqt0z6qm6vpg6
    $ mv dcmr36vmsxgcwneh0attqt0z6qm6vpg6 metadata.csv
    
Extract matrix (obs x features) and labels for each obs:
```python
>>> import pandas as pd
>>> import scipy.io as sio
>>> import numpy as np

>>> mat = sio.mmread('mat.mtx') #Is a centered and scaled matrix (scaling input is optional)
>>> mat.shape
(3850, 1999)

>>> meta = pd.read_csv('metadata.csv')
>>> meta.head()
 Unnamed: 0          sample_name  smartseq_cluster_id  smartseq_cluster  ... n_genes percent_mito pass_count_filter  pass_mito_filter
0  SM-GE4R2_S062_E1-50  SM-GE4R2_S062_E1-50                   46   Nr5a1_9|11 Rorb  ...    9772          0.0              True              True
1  SM-GE4SI_S356_E1-50  SM-GE4SI_S356_E1-50                   46   Nr5a1_9|11 Rorb  ...    8253          0.0              True              True
2  SM-GE4SI_S172_E1-50  SM-GE4SI_S172_E1-50                   46   Nr5a1_9|11 Rorb  ...    9394          0.0              True              True
3   LS-15034_S07_E1-50   LS-15034_S07_E1-50                   42  Nr5a1_4|7 Glipr1  ...   10643          0.0              True              True
4   LS-15034_S28_E1-50   LS-15034_S28_E1-50                   42  Nr5a1_4|7 Glipr1  ...   10550          0.0              True              True

>>> cellTypes = list(meta.smartseq_cluster)
>>> sexLabels = list(meta.sex_label)
>>> len(sexLabels)
3850
```

<br/><br/>

To run the __MCML__ algorithm for dimensionality reduction (Python 3):

```python
>>> from MCML.modules import MCML, bMCML

>>> mcml = MCML(n_latent = 50) #Initialize MCML class

>>> latentMCML = mcml.fit(mat, np.array([cellTypes,sexLabels]) , fracNCA = 0.8 , silent = True) #Run MCML
>>> latentMCML.shape
(3850, 50)

>>> mcml.plotLosses(figsize=(10,3),axisFontSize=10,tickFontSize=8) #Plot loss over epochs

```
This incorporates both the cell type and sex labels into the latent space construction.


<br/><br/>

To run the __bMCML__ algorithm for dimensionality reduction (Python 3):

```python
>>> bmcml = bMCML(n_latent = 50) #Initialize bMCML class


>>> latentbMCML = bmcml.fit(mat, np.array(cellTypes), np.array(sexLabels), silent=True) #Run bMCML
>>> latentbMCML.shape
(3850, 50)

>>> bmcml.plotLosses(figsize=(10,3),axisFontSize=10,tickFontSize=8) #Plot loss over epochs

```
__bMCML__ is optimizing for the intra-distances of the sex labels i.e. the pairwise distances of cells in each sex for each cell type.


<br/><br/>

To use the metrics available in __tools__:

```python
>>> from MCML import tools as tl

#Pairwise distances between centroids of cells in each label
>>> cDists = tl.getCentroidDists(mat, np.array(cellTypes)) 
>>> len(cDists)
784

#Avg pairwise distances between cells of both sexes, for each cell type
>>> interDists = tl.getInterVar(mat, np.array(cellTypes), np.array(sexLabels))  
>>> len(interDists)
27

#Avg pairwise distances between cells of the same sex, for each cell type
>>> intraDists = tl.getIntraVar(mat, np.array(cellTypes), np.array(sexLabels)) 
>>> len(intraDists)
53

```

<br/><br/>

To see further details of all inputs and outputs for all functions use: 

```python
>>> help(MCML)
>>> help(bMCML)
>>> help(tl)
```


License
-------

MCML is licensed under the terms of the BSD License (see the file
LICENSE).
