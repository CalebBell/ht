# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from numpy.testing import assert_allclose
import ht
import ht.vectorized
import numpy as np


def test_LMTD_vect():
    dTlms = [ht.LMTD(T, 60., 30., 40.2) for T in [100, 101]]
    dTlms_vect = ht.vectorized.LMTD([100, 101], 60., 30., 40.2)
    assert_allclose(dTlms, dTlms_vect)
