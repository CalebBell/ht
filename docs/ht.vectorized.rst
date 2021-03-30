Support for numpy arrays (ht.vectorized)
========================================


Basic module which wraps all ht functions with numpy's vectorize.
All other object - dicts, classes, etc - are not wrapped. Supports star 
imports; so the same objects exported when importing from the main library
will be imported from here. 

>>> from ht.vectorized import *

Inputs do not need to be numpy arrays; they can be any iterable:

>>> import ht.vectorized
>>> ht.vectorized.LMTD([100, 101], 60., 30., 40.2)
array([43.20040929, 43.60182765])

