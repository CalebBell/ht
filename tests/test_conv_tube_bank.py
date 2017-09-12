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
