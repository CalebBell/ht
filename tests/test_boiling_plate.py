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
from ht import *
import numpy as np
from numpy.testing import assert_allclose
import pytest

def test_h_boiling_Amalfi():
    h = h_boiling_Amalfi(m=3E-5, x=.4, Dh=0.00172, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, mug=7.11E-6, sigma=0.02, Hvap=9E5, q=1E5, A_channel_flow=0.0003)
    assert_allclose(h, 776.0781179096225)
    
    h = h_boiling_Amalfi(m=3E-5, x=.4, Dh=0.0172, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, mug=7.11E-6, sigma=0.02, Hvap=9E5, q=1E5, A_channel_flow=0.0003)
    assert_allclose(h, 527.4075513650002)