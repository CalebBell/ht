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
from fluids import *
from ht import *
from ht.boiling_nucleic import _angles_Stephan_Abdelsalam
import numpy as np

from numpy.testing import assert_allclose
import pytest



### Condensation

def test_h_Nusselt_laminar():
    h = Nusselt_laminar(370., 350., 7.0, 585., 0.091, 158.9E-6, 776900., 0.1)
    assert_allclose(h, 1482.206403453679)
    h_angle = [Nusselt_laminar(Tsat=370., Tw=350., rhog=7.0, rhol=585., kl=0.091, mul=158.9E-6, Hvap=776900., L=0.1, angle=float(i)) for i in np.linspace(0, 90, 8)]
    h_angle_values = np.array([0.0, 1018.0084987903685, 1202.9623322809389, 1317.0917328126477, 1393.7567182107628, 1444.0629692910647, 1472.8272516024929, 1482.206403453679])
    assert_allclose(h_angle, h_angle_values)


def test_h_Boyko_Kruzhilin():
    h_xs = [Boyko_Kruzhilin(m=0.35, rhog=6.36, rhol=582.9, kl=0.098,
	    mul=159E-6, Cpl=2520., D=0.03, x=i) for i in np.linspace(0,1,11)]
    h_xs_values = [1190.3309510899785, 3776.3883678904836, 5206.2779830848758, 6320.5657791981021, 7265.9323628276288, 8101.7278671405438, 8859.0188940546595, 9556.4866502932564, 10206.402815353165, 10817.34162173243, 11395.573750069829]
    assert_allclose(h_xs, h_xs_values)


def test_Akers_Deans_Crosser():
    h = Akers_Deans_Crosser(m=0.35, rhog=6.36, rhol=582.9, kl=0.098,  mul=159E-6, Cpl=2520., D=0.03, x=0.85)
    assert_allclose(h, 7117.24177265201)
    h = Akers_Deans_Crosser(m=0.01, rhog=6.36, rhol=582.9, kl=0.098,  mul=159E-6, Cpl=2520., D=0.03, x=0.85)
    assert_allclose(h, 737.5654803081094)

def test_h_kinetic():
    h = h_kinetic(300, 1E5, 18.02, 2441674)
    assert_allclose(h, 30788845.562480535, rtol=1e-5)


def test_Cavallini_Smith_Zecchin():
    assert_allclose(Cavallini_Smith_Zecchin(m=1, x=0.4, D=.3, rhol=800, rhog=2.5, mul=1E-5, mug=1E-3, kl=0.6, Cpl=2300), 5578.218369177804)


def test_Shah():
    assert_allclose(Shah(m=1, x=0.4, D=.3, rhol=800, mul=1E-5, kl=0.6, Cpl=2300, P=1E6, Pc=2E7), 2561.2593415479214)
    # In Shaw's second paper, they used the following definition. However, it
    # is just rearanged differently. It was coded to verify this, and is left
    # in case further sources list it in different forms.
    def Shah3(m, x, D, rhol, mul, kl, Cpl, P, Pc):
        Pr = P/Pc
        G = m/(pi/4*D**2)
        Prl = Prandtl(Cp=Cpl, k=kl, mu=mul)
        Rel = G*D/mul
        hL = kl/D*(0.023*Rel**0.8*Prl**0.4)
        hl = hL*(1-x)**0.8
        Z = (1/x -1)**0.8*Pr**0.4
        h_TP = hl*(1 + 3.8/Z**0.95)
        return h_TP
    assert_allclose(Shah(m=1, x=0.4, D=.3, rhol=800, mul=1E-5, kl=0.6, Cpl=2300, P=1E6, Pc=2E7), 2561.2593415479214)
    # The following is in the review of Balcular (2011), and incorrect.
    #    Pr = P/Pc
    #    G = m/(pi/4*D**2)
    #    Prl = Prandtl(Cp=Cpl, k=kl, mu=mul)
    #    Rel = G*D*(1-x)/mul
    #    hl = kl/D*(0.023*(Rel/(1-x))**0.8*Prl**0.4)
    #    hsf = hl*(1-x)**0.8
    #    Co = (1/x-1)**0.8*(rhog/rhol)**0.5
    #    return hsf*1.8/Co**0.8
