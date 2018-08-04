# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import numpy as np
from numpy.testing import assert_allclose


def test_conv_tube_bank():
    f = Kern_f_Re(np.linspace(10, 1E6, 10))
    f_values = [6.0155491322862771, 0.19881943524161752, 0.1765198121811164, 0.16032260681398205, 0.14912064432650635, 0.14180674990498099, 0.13727374873569789, 0.13441446600494875, 0.13212172689902535, 0.12928835660421958]
    assert_allclose(f, f_values)

    dP = dP_Kern(11., 995., 0.000803, 0.584, 0.1524, 0.0254, .019, 22, 0.000657)
    assert_allclose(dP, 18980.58768759033)

    dP = dP_Kern(m=11., rho=995., mu=0.000803, DShell=0.584, LSpacing=0.1524, pitch=0.0254, Do=.019, NBaffles=22)
    assert_allclose(dP, 19521.38738647667)

    # TODO Splines

    dP1 = dP_Zukauskas(Re=13943., n=7, ST=0.0313, SL=0.0343, D=0.0164, rho=1.217, Vmax=12.6)
    dP2 = dP_Zukauskas(Re=13943., n=7, ST=0.0313, SL=0.0313, D=0.0164, rho=1.217, Vmax=12.6)
    assert_allclose([dP1, dP2], [235.22916169118335, 217.0750033117563])


def test_baffle_correction_Bell():
    Jc = baffle_correction_Bell(0.82)
    assert_allclose(Jc, 1.1258554691854046, 5e-4)
    
    # Check the match is reasonably good
    from ht.conv_tube_bank import Bell_baffle_configuration_Fcs, Bell_baffle_configuration_Jcs
    errs = np.array([(baffle_correction_Bell(Fc)-Jc)/Jc for Fc, Jc in zip(Bell_baffle_configuration_Fcs, Bell_baffle_configuration_Jcs)])
    assert np.abs(errs).sum()/len(errs) < 1e-3
    
    Jc = baffle_correction_Bell(0.1, 'chebyshev')   
    assert_allclose(Jc, 0.61868011359447)
     
    Jc = baffle_correction_Bell(0.82, 'HEDH')
    assert_allclose(Jc, 1.1404)
    
    
def test_baffle_leakage_Bell():
    Jl = baffle_leakage_Bell(1, 1, 4)
    assert_allclose(Jl, 0.5159239501898142, rtol=1e-3)
    
    Jl = baffle_leakage_Bell(1, 1, 8)
    assert_allclose(Jl, 0.6820523047494141, rtol=1e-3)

    Jl = baffle_leakage_Bell(1, 3, 8)
    assert_allclose(Jl, 0.5906621282470395, rtol=1e-3)
    
    # Silent clipping
    Jl = baffle_leakage_Bell(1, .0001, .00001)
    assert_allclose(Jl,  0.16072739052053492)
    
    Jl = baffle_leakage_Bell(1, 3, 8, method='HEDH')
    assert_allclose(Jl, 0.5530236260777133)


def test_bundle_bypassing_Bell():
    Jb = bundle_bypassing_Bell(0.5, 5, 25)
    assert_allclose(Jb, 0.8469611760884599, rtol=1e-3)
    Jb = bundle_bypassing_Bell(0.5, 5, 25, laminar=True)
    assert_allclose(Jb, 0.8327442867825271, rtol=1e-3)
    
    Jb = bundle_bypassing_Bell(0.99, 5, 25, laminar=True)
    assert_allclose(Jb, 0.7786963825447165, rtol=1e-3)
    
    Jb = bundle_bypassing_Bell(0.5, 5, 25, method='HEDH')
    assert_allclose(Jb, 0.8483210970579099)
    
    Jb = bundle_bypassing_Bell(0.5, 5, 25, method='HEDH', laminar=True)
    assert_allclose(0.8372305924553625, Jb)