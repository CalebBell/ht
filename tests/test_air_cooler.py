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
from ht.boiling_nucleic import _angles_Stephan_Abdelsalam
from numpy.testing import assert_allclose
import pytest

### Air Cooler

def test_air_cooler_Ft():
    Ft_1 = Ft_aircooler(Thi=93, Tho=52, Tci=35, Tco=54.59, Ntp=2, rows=4)
    assert_allclose(Ft_1, 0.9570456123827129)
    Ft_2 = Ft_aircooler(Thi=125., Tho=45., Tci=25., Tco=95., Ntp=1, rows=4)
    assert_allclose(Ft_2, 0.5505093604092708)
    Ft_many = [[Ft_aircooler(Thi=125., Tho=80., Tci=25., Tco=95., Ntp=i, rows=j) for i in range(1,6)] for j in range(1, 6)]
    Ft_values = [[0.6349871996666123, 0.9392743008890244, 0.9392743008890244, 0.9392743008890244, 0.9392743008890244], [0.7993839562360742, 0.9184594715750571, 0.9392743008890244, 0.9392743008890244, 0.9392743008890244], [0.8201055328279105, 0.9392743008890244, 0.9784008071402877, 0.9392743008890244, 0.9392743008890244], [0.8276966706732202, 0.9392743008890244, 0.9392743008890244, 0.9828365967034366, 0.9392743008890244], [0.8276966706732202, 0.9392743008890244, 0.9392743008890244, 0.9392743008890244, 0.9828365967034366]]
    assert_allclose(Ft_many, Ft_values)