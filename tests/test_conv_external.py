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
import numpy as np
from numpy.testing import assert_allclose
import pytest

### Conv external

def test_Nu_cylinder_Zukauskas():
    Nu = Nu_cylinder_Zukauskas(7992, 0.707, 0.69)
    assert_allclose(Nu, 50.523612661934386)

    Nus_allRe = [Nu_cylinder_Zukauskas(Re, 0.707, 0.69) for Re in np.logspace(0, 6, 8)]
    Nus_allRe_values = [0.66372630070423799, 1.4616593536687801, 3.2481853039940831, 8.7138930573143227, 26.244842388228189, 85.768869004450067, 280.29503021904566, 1065.9610995854582]
    assert_allclose(Nus_allRe, Nus_allRe_values)

    Nu_highPr = Nu_cylinder_Zukauskas(7992, 42.)
    assert_allclose(Nu_highPr, 219.24837219760443)



def test_Nu_cylinder_Churchill_Bernstein():
    Nu = Nu_cylinder_Churchill_Bernstein(6071, 0.7)
    assert_allclose(Nu, 40.63708594124974)


def test_Nu_cylinder_Sanitjai_Goldstein():
    Nu = Nu_cylinder_Sanitjai_Goldstein(6071, 0.7)
    assert_allclose(Nu, 40.38327083519522)


def test_Nu_cylinder_Fand():
    Nu = Nu_cylinder_Fand(6071, 0.7)
    assert_allclose(Nu, 45.19984325481126)


def test_Nu_cylinder_McAdams():
    Nu = Nu_cylinder_McAdams(6071, 0.7)
    assert_allclose(Nu, 46.98179235867934)


def test_Nu_cylinder_Whitaker():
    Nu = Nu_cylinder_Whitaker(6071, 0.7)
    assert_allclose(Nu, 45.94527461589126)
    Nu = Nu_cylinder_Whitaker(6071, 0.7, 1E-3, 1.2E-3)
    assert_allclose(Nu, 43.89808146760356)


def test_Nu_cylinder_Perkins_Leppert_1962():
    Nu = Nu_cylinder_Perkins_Leppert_1962(6071, 0.7)
    assert_allclose(Nu, 49.97164291175499)
    Nu = Nu_cylinder_Perkins_Leppert_1962(6071, 0.7, 1E-3, 1.2E-3)
    assert_allclose(Nu, 47.74504603464674)


def test_Nu_cylinder_Perkins_Leppert_1964():
    Nu = Nu_cylinder_Perkins_Leppert_1964(6071, 0.7)
    assert_allclose(Nu, 53.61767038619986)
    Nu = Nu_cylinder_Perkins_Leppert_1964(6071, 0.7, 1E-3, 1.2E-3)
    assert_allclose(Nu, 51.22861670528418)