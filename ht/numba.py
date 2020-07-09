# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
SOFTWARE.'''

from __future__ import division
import sys
import importlib.util
import types
import numpy as np
import inspect
import numba
import ht
import fluids
import fluids.numba
normal_fluids = fluids
normal = ht

orig_file = __file__
caching = False
'''


'''
__all__ = []
__funcs = {}

numerics = fluids.numba.numerics
replaced = fluids.numba.numerics_dict.copy()
cache_blacklist = []

def transform_complete_ht(replaced, __funcs, __all__, normal, vec=False):
    __funcs.update(normal_fluids.numba.__dict__.copy())
    new_mods = normal_fluids.numba.transform_module(normal, __funcs, replaced, vec=vec, cache_blacklist=cache_blacklist)
    if vec:
        conv_fun = numba.vectorize
    else:
        conv_fun = numba.jit
    
    to_change_full_output = []
    
    to_change = {}
    to_change.update({k: 'full_output' for k in to_change_full_output})
#    to_change['hx.Ntubes_Phadkeb'] = 'square_C1s is None'
    to_change['boiling_nucleic.Gorenflo'] = 'h0 is None: # NUMBA: DELETE'

    for s, bad_branch in to_change.items():
        mod, func = s.split('.')
        source = inspect.getsource(getattr(getattr(normal, mod), func))
        fake_mod = __funcs[mod]
        source = normal_fluids.numba.remove_branch(source, bad_branch)
        normal_fluids.numba.numba_exec_cacheable(source, fake_mod.__dict__, fake_mod.__dict__)
        new_func = fake_mod.__dict__[func]
        obj = conv_fun(cache=caching)(new_func)
        __funcs[func] = obj
        globals()[func] = obj
        obj.__doc__ = ''
    __funcs['hx']._load_coeffs_Phadkeb()
    to_change = ['air_cooler.Ft_aircooler', 'hx.Ntubes_Phadkeb', 'boiling_nucleic.h_nucleic_methods']
    normal_fluids.numba.transform_lists_to_arrays(normal, to_change, __funcs)

        
    __funcs['hx']._load_coeffs_Phadkeb()

    for mod in new_mods:
        mod.__dict__.update(__funcs)

transform_complete_ht(replaced, __funcs, __all__, normal, vec=False)



globals().update(__funcs)
globals().update(replaced)

__name__ = 'ht.numba'
__file__ = orig_file
