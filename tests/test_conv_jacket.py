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


def test_conv_jacket():
    # actual example
    h = Lehrer(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, dT=20.)
    assert_allclose(h, 2922.128124761829)
    # no wall correction
    h = Lehrer(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, dT=20.)
    assert_allclose(h, 2608.8602693706853)

    # with isobaric expansion, all cases
    h = Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', isobaric_expansion=0.000303)
    assert_allclose(h, 3269.4389632666557)

    h = Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', inletlocation='top', isobaric_expansion=0.000303)
    assert_allclose(h, 2566.1198726589996)

    h = Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=-20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', isobaric_expansion=0.000303)
    assert_allclose(h, 3269.4389632666557)

    h = Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=-20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', inletlocation='bottom', isobaric_expansion=0.000303)
    assert_allclose(h, 2566.1198726589996)


    ### Stein Schmidt

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, 971.8)
    assert_allclose(h, 5695.1871940874225)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, 971.8, inlettype='radial')
    assert_allclose(h, 1217.1449686341773)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, 971.8, inletlocation='top')
    assert_allclose(h, 5675.824588428565)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, 971.8, inletlocation='bottom')
    assert_allclose(h, 5695.1871940874225)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 971.8, 4178.1, 0.615, 798E-6, 355E-6, 995.7, inletlocation='bottom')
    assert_allclose(h, 5694.955289327642)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 971.8, 4178.1, 0.615, 798E-6, 355E-6, 995.7, inletlocation='top')
    assert_allclose(h, 5676.0574502620975)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 971.8, 4178.1, 0.615, 798E-6, 355E-6)
    assert_allclose(h, 5685.515980483362)

    h = Stein_Schmidt(.1, 0.6, 0.65, 0.6, 0.025, 971.8, 4178.1, 0.615, 798E-6)
    assert_allclose(h, 146.80846173206865)