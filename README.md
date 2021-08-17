# MCML

MCML is a toolkit for semi-supervised dimensionality reduction and quantitative analysis of Multi-Class, Multi-Label data. We describe its uses for singl-cell datasets here...
MCML _modules_ include the _MCML_ and _bMCML_ algorithms described in ... for dimensionality reduction, and MCML _tools_ include functions for quantitative analysis of inter- and intra- distances between labeled groups and nearest neighbor metrics in the latent or ambient space. 

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

    $ python3 -m pip install -U 

If you want to run the latest version of the code, you can install from git:

    $ python3 -m pip install -U git+git://github.com/pachterlab/MCML.git


Examples
-----------

Example data download:

    $ wget --quiet  (smartseq.mtx)
    $ wget --quiet  (celltypelabels)
    $ wget --quiet  (sexlabels)
    
And (optional) pre-processing:
```python
>>> read()
>>> show size of matrix

>>> read labels
>>> show labels

>>> read labels
>>> show labels

>>> save labels
>>> save labels

```

To run the _MCML_ or _bMCML_ algorithms for dimensionality reduction (Python 3):

```python
>>> from MCML.modules import MCML, bMCML

>>> run
>>>
>>> plot_losses

```


To use the metrics available in _tools_:

```python
>>> from MCML import tools as tl

>>> 
>>>
>>>
>>>
```

To see the descriptions all inputs and outputs for all functions use: 

```python
>>> help(MCML)
>>> help(bMCML)
>>> help(tl)
```


License
-------

MCML is licensed under the terms of the BSD 2 clause License (see the file
LICENSE).
