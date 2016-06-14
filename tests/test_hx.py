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


### TODO hx requires testing, but perhaps first improvement
def test_hx():
    Nt_perry = [[Ntubes_Perrys(DBundle=1.184, Ntp=i, do=.028, angle=j) for i in [1,2,4,6]] for j in [30, 45, 60, 90]]
    Nt_values = [[1001, 973, 914, 886], [819, 803, 784, 769], [1001, 973, 914, 886], [819, 803, 784, 769]]
    assert_allclose(Nt_perry, Nt_values)
#    angle = 30 or 60 and ntubes = 1.5 raise exception

    with pytest.raises(Exception):
        Ntubes_Perrys(DBundle=1.184, Ntp=5, do=.028, angle=30)
    with pytest.raises(Exception):
        Ntubes_Perrys(DBundle=1.184, Ntp=5, do=.028, angle=45)


    VDI_t = [[Ntubes_VDI(DBundle=1.184, Ntp=i, do=.028, pitch=.036, angle=j) for i in [1,2,4,6,8]] for j in [30, 45, 60, 90]]
    VDI_values = [[983, 966, 929, 914, 903], [832, 818, 790, 778, 769], [983, 966, 929, 914, 903], [832, 818, 790, 778, 769]]
    assert_allclose(VDI_t, VDI_values)
    with pytest.raises(Exception):
        Ntubes_VDI(DBundle=1.184, Ntp=5, do=.028, pitch=.036, angle=30)
    with pytest.raises(Exception):
        Ntubes_VDI(DBundle=1.184, Ntp=2, do=.028, pitch=.036, angle=40)

    # TODO: Phadke

    Ntubes_HEDH_c = [Ntubes_HEDH(DBundle=1.200-.008*2, do=.028, pitch=.036, angle=i) for i in [30, 45, 60, 90]]
    assert_allclose(Ntubes_HEDH_c, [928, 804, 928, 804])
    with pytest.raises(Exception):
        Ntubes_HEDH(DBundle=1.200-.008*2, do=.028, pitch=.036, angle=20)


    methods = Ntubes(DBundle=1.2, do=0.025, AvailableMethods=True)
    Ntubes_calc = [Ntubes(DBundle=1.2, do=0.025, Method=i) for i in methods]
    assert Ntubes_calc == [1285, 1272, 1340, 1297, None]

    assert_allclose(Ntubes(DBundle=1.2, do=0.025), 1285)

    with pytest.raises(Exception):
        Ntubes(DBundle=1.2, do=0.025, Method='failure')


    D_VDI =  [[D_for_Ntubes_VDI(Nt=970, Ntp=i, do=0.00735, pitch=0.015, angle=j) for i in [1, 2, 4, 6, 8]] for j in [30, 60, 45, 90]]
    D_VDI_values = [[0.489981989464919, 0.5003600119829544, 0.522287673753684, 0.5311570964003711, 0.5377131635291736], [0.489981989464919, 0.5003600119829544, 0.522287673753684, 0.5311570964003711, 0.5377131635291736], [0.5326653264480428, 0.5422270203444146, 0.5625250342473964, 0.5707695340997739, 0.5768755899087357], [0.5326653264480428, 0.5422270203444146, 0.5625250342473964, 0.5707695340997739, 0.5768755899087357]]
    assert_allclose(D_VDI, D_VDI_values)

    with pytest.raises(Exception):
        D_for_Ntubes_VDI(Nt=970, Ntp=5., do=0.00735, pitch=0.015, angle=30.)
    with pytest.raises(Exception):
        D_for_Ntubes_VDI(Nt=970, Ntp=2., do=0.00735, pitch=0.015, angle=40.)
