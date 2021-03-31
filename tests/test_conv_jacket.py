# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids.numerics import assert_close, assert_close1d, assert_close2d
import pytest


def test_conv_jacket():
    # actual example
    h = Lehrer(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, dT=20.)
    assert_close(h, 2922.128124761829)
    # no wall correction
    h = Lehrer(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, dT=20.)
    assert_close(h, 2608.8602693706853)

    # with isobaric expansion, all cases
    h = Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', isobaric_expansion=0.000303)
    assert_close(h, 3269.4389632666557)

    h = Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', inletlocation='top', isobaric_expansion=0.000303)
    assert_close(h, 2566.1198726589996)

    h = Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=-20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', isobaric_expansion=0.000303)
    assert_close(h, 3269.4389632666557)

    h = Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=-20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', inletlocation='bottom', isobaric_expansion=0.000303)
    assert_close(h, 2566.1198726589996)


    ### Stein Schmidt

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, 971.8)
    assert_close(h, 5695.204169808863)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, 971.8, inlettype='radial')
    assert_close(h, 1217.1449686341773)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, 971.8, inletlocation='top')
    assert_close(h, 5675.841635061595)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, 971.8, inletlocation='bottom')
    assert_close(h, 5695.2041698088633)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 971.8, 4178.1, 0.615, 798E-6, 355E-6, 995.7, inletlocation='bottom')
    assert_close(h, 5694.9722658952096)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 971.8, 4178.1, 0.615, 798E-6, 355E-6, 995.7, inletlocation='top')
    assert_close(h, 5676.0744960391157)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 971.8, 4178.1, 0.615, 798E-6, 355E-6)
    assert_close(h, 5685.532991556428)

    h = Stein_Schmidt(.1, 0.6, 0.65, 0.6, 0.025, 971.8, 4178.1, 0.615, 798E-6)
    assert_close(h, 151.78819106776797)