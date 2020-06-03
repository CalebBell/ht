# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from ht import *
import ht.vectorized
from math import *
from fluids.constants import *
from fluids.numerics import assert_close, assert_close1d
import pytest
try:
    import numba
    import ht.numba
    import ht.numba_vectorized
except:
    numba = None
import numpy as np

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_core_misc():
    assert_close(ht.numba.LMTD(100., 60., 20., 60, counterflow=False),
                 ht.LMTD(100., 60., 20., 60, counterflow=False))
    
    assert_close(ht.numba.fin_efficiency_Kern_Kraus(0.0254, 0.05715, 3.8E-4, 200, 58),
                 ht.fin_efficiency_Kern_Kraus(0.0254, 0.05715, 3.8E-4, 200, 58),)
    
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_conv_tank():
    assert_close(ht.Stein_Schmidt(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, rhow=971.8),
                 ht.numba.Stein_Schmidt(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, rhow=971.8))



@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_Ntubes_Phadkeb():
    # Extremely impressive performance
    Bundles = np.linspace(1, 2, 5)
    Dos = np.linspace(.028, .029, 5)
    pitches = np.linspace(.036, .037, 5)
    Ntps = np.linspace(2, 2, 5, dtype=np.int64)
    angles = np.linspace(45, 45, 5, dtype=np.int64)
    
    assert_close(ht.numba_vectorized.Ntubes_Phadkeb(Bundles, Dos, pitches, Ntps, angles), 
                 [ 558,  862, 1252, 1700, 2196])