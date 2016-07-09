# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.'''

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