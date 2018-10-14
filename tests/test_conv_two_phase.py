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

from numpy.testing import assert_allclose
import pytest


def test_Davis_David():
    h = Davis_David(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mul=1E-3)
    assert_allclose(h, 1437.3282869955121)


def test_Elamvaluthi_Srinivas():
    h = Elamvaluthi_Srinivas(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mug=1E-5, mu_b=1E-3, mu_w=1.2E-3)
    assert_allclose(h, 3901.2134471578584)


def test_Groothuis_Hendal():
    h = Groothuis_Hendal(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mug=1E-5, mu_b=1E-3, mu_w=1.2E-3)
    assert_allclose(h, 1192.9543445455754)
    
    h = Groothuis_Hendal(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mug=1E-5, mu_b=1E-3, mu_w=1.2E-3, water=True)
    assert_allclose(h, 6362.8989677634545)


def test_Hughmark():
    h = Hughmark(m=1, x=.9, D=.3, L=.5, alpha=.9, Cpl=2300, kl=0.6, mu_b=1E-3, mu_w=1.2E-3)
    assert_allclose(h, 212.7411636127175)


def test_Knott():
    h = Knott(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mu_b=1E-3, mu_w=1.2E-3, L=4)
    assert_allclose(h, 4225.536758045839)


def test_Kudirka_Grosh_McFadden():
    h = Kudirka_Grosh_McFadden(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mug=1E-5, mu_b=1E-3, mu_w=1.2E-3)
    assert_allclose(h, 303.9941255903587)


def test_Martin_Sims():
    h = Martin_Sims(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, hl=141.2)
    assert_allclose(h, 5563.280000000001)
    
    h = Martin_Sims(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mu_b=1E-3, mu_w=1.2E-3, L=24)
    assert_allclose(h, 5977.505465781747)


def test_Ravipudi_Godbold():
    h = Ravipudi_Godbold(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mug=1E-5, mu_b=1E-3, mu_w=1.2E-3)
    assert_allclose(h, 299.3796286459285)


def test_Aggour():
    h = Aggour(m=1, x=.9, D=.3, alpha=.9, rhol=1000, Cpl=2300, kl=.6, mu_b=1E-3)
    assert_allclose(h, 420.9347146885667)
    
    h = Aggour(m=.1, x=.9, D=.3, alpha=.9, rhol=1000, Cpl=2300, kl=.6, mu_b=1E-3, mu_w=1.2E-3, L=4)
    assert_allclose(h, 33.64542760558181)
    

def test_h_two_phase():
    h = h_two_phase(m=1, x=.9, D=.3, alpha=.9, rhol=1000, Cpl=2300, kl=.6, mu_b=1E-3, mu_w=1.2E-3, L=5, method='Aggour')
    assert_allclose(h, 420.9347146885667)
    
    