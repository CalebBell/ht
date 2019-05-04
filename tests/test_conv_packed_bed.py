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
from numpy.testing import assert_allclose


def test_Nu_packed_bed_Gnielinski():
    Nu = Nu_packed_bed_Gnielinski(8E-4, 0.4, 1, 1E3, 1E-3, 0.7)
    assert_allclose(Nu, 61.37823202546954)

    # fa=2 test
    Nu = Nu_packed_bed_Gnielinski(8E-4, 0.4, 1, 1E3, 1E-3, 0.7, 2)
    assert_allclose(Nu, 64.60866528996795)


def test_Nu_Wakao_Kagei():
    Nu = Nu_Wakao_Kagei(2000, 0.7)
    assert_allclose(Nu, 95.40641328041248)

def test_Nu_Achenbach():
    Nu = Nu_Achenbach(2000, 0.7, 0.4)
    assert_allclose(Nu, 117.70343608599121)

def test_Nu_KTA():
    Nu = Nu_KTA(2000, 0.7, 0.4)
    assert_allclose(Nu, 102.08516480718129)