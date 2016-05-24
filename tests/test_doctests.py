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

if __name__ == '__main__':
    import doctest
    doctest.testmod(core)
    doctest.testmod(hx)
    doctest.testmod(conv_internal)
    doctest.testmod(conv_jacket)
    doctest.testmod(boiling_nucleic)
    doctest.testmod(condensation)
    doctest.testmod(air_cooler)
    doctest.testmod(radiation)
    doctest.testmod(insulation)
    doctest.testmod(conduction)
    doctest.testmod(conv_free_immersed)
    doctest.testmod(conv_tube_bank)
    doctest.testmod(conv_packed_bed)
    doctest.testmod(conv_external)


