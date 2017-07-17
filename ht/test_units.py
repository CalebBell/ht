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
import types
import numpy as np
from numpy.testing import assert_allclose
import pytest
import fluids
import ht
from ht.units import *



def assert_pint_allclose(value, magnitude, units):
    assert_allclose(value.to_base_units().magnitude, magnitude)
    assert dict(value.dimensionality) == units

def test_sample_cases():
    pass

def test_custom_wraps():
    pass


def test_check_signatures():
    from fluids.units import check_args_order
    for name in dir(ht):
        obj = getattr(ht, name)
        if isinstance(obj, types.FunctionType) and obj not in [ht.qmax_boiling, ht.h_nucleic, ht.get_tube_TEMA, ht.Ntubes, ht.Nu_conv_internal, ht.R_to_k, ht.check_tubing_TEMA]:
            check_args_order(obj)
