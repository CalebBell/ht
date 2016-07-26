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


def test_Ravipudi_Godbold():
    h = Ravipudi_Godbold(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mug=1E-5, mu_b=1E-3, mu_w=1.2E-3)
    assert_allclose(h, 299.3796286459285)


def test_Aggour():
    h = Aggour(m=1, x=.9, D=.3, alpha=.9, rhol=1000, Cpl=2300, kl=.6, mu_b=1E-3)
    assert_allclose(h, 420.9347146885667)
    
    h = Aggour(m=.1, x=.9, D=.3, alpha=.9, rhol=1000, Cpl=2300, kl=.6, mu_b=1E-3, mu_w=1.2E-3, L=4)
    assert_allclose(h, 33.64542760558181)