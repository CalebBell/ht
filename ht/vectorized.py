'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from fluids.numerics import FakePackage
from fluids.numerics import numpy as np

import ht

"""Basic module which wraps all ht functions with numpy's vectorize.
All other object - dicts, classes, etc - are not wrapped. Supports star
imports; so the same objects exported when importing from the main library
will be imported from here.

>>> from ht.vectorized import *

Inputs do not need to be numpy arrays; they can be any iterable:

>>> import ht.vectorized
>>> ht.vectorized.LMTD([100, 101], 60., 30., 40.2)
array([ 43.20040929,  43.60182765])

Note that because this needs to import ht itself, ht.vectorized
needs to be imported separately; the following will cause an error:

>>> import ht
>>> ht.vectorized # Won't work, has not been imported yet

The correct syntax is as follows:

>>> import ht.vectorized # Necessary
>>> from ht.vectorized import * # May be used without first importing ht
"""

__all__ = []


__funcs = {}

if isinstance(np, FakePackage):
    pass
else:
    import types
    for name in dir(ht):
        obj = getattr(ht, name)
        if isinstance(obj, types.FunctionType):
            obj = np.vectorize(obj)
        elif isinstance(obj, str):
            continue
        __all__.append(name)
        __funcs.update({name: obj})
globals().update(__funcs)




