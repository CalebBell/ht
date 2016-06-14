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

### Nucleic boiling


def test_boiling_nucleic_Rohsenow():
    h_calc = [Rohsenow(Te=i, Cpl=4180, kl=0.688, mul=2.75E-4, sigma=0.0588, Hvap=2.25E6, rhol=958, rhog=0.597, Csf=0.013, n=1) for i in [4.3, 9.1, 13]]
    h_values = [2860.6242230238613, 12811.697777642301, 26146.321995188344]
    assert_allclose(h_calc, h_values)
    q_test = Rohsenow(Te=4.9, Cpl=4217., kl=0.680, mul=2.79E-4, sigma=0.0589, Hvap=2.257E6, rhol=957.854, rhog=0.595593, Csf=0.011, n=1.26)*4.9
    assert_allclose(18245.91080863059, q_test)
    h_with_defaults = Rohsenow(5, 4180, 0.688, 2.75E-4, 0.0588, 2.25E6, 958, 0.597)
    assert_allclose(h_with_defaults, 1316.2269561541964)


def test_boiling_nucleic_McNelly():
    # Water matches expectations, ammonia is somewhat distant. Likely just
    # error in the text's calculation.
    h_McNelly1 = McNelly(4.3, 101325, 4180., 0.688, 0.0588, 2.25E6, 958., 0.597)
    h_McNelly2 = McNelly(9.1, 101325., 4472., 0.502, 0.0325, 1.37E6, 689., 0.843)
    assert_allclose([h_McNelly1, h_McNelly2], [533.8056972951352, 6387.3951029225855])


def test_boiling_nucleic_Forster_Zuber():
    # All examples are for water from [1]_ and match.
    # 4th example is from [3]_ and matches completely.
    FZ1 = Forster_Zuber(Te=4.3, dPSat=3906*4.3, Cpl=4180., kl=0.688, mul=0.275E-3, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)
    FZ2 = Forster_Zuber(Te=9.1, dPSat=3906*9.1, Cpl=4180., kl=0.688, mul=0.275E-3, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)
    FZ3 = Forster_Zuber(Te=13, dPSat=3906*13, Cpl=4180., kl=0.688, mul=0.275E-3, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)
    FZ4 = Forster_Zuber(16.2, 106300., 2730., 0.086, 156E-6, .0082, 272E3, 567., 18.09)
    FZ_values = [3519.9239897462644, 7393.507072909551, 10524.54751261952, 5512.279068294656]
    assert_allclose([FZ1, FZ2, FZ3, FZ4], FZ_values)


def test_boiling_nucleic_Montinsky():
    # Fourth example is from [4]_ and matches to within the error of the algebraic
    # manipulation rounding.
    # First three examples are for water, ammonia, and benzene, from [1]_, and
    # match to within 20%.
    W_Te = [Montinsky(Te=i, P=101325., Pc=22048321.0) for i in [4.3, 9.1, 13]]
    W_Te_values = [1185.0509770292663, 6814.079848742471, 15661.924462897328]
    assert_allclose(W_Te, W_Te_values)
    A_Te = [Montinsky(Te=i, P=101325., Pc=112E5) for i in [4.3, 9.1, 13]]
    A_Te_values = [377.04493949460635, 2168.0200886557072, 4983.118427770712]
    assert_allclose(A_Te, A_Te_values)
    B_Te = [Montinsky(i, 101325., 48.9E5) for i in [4.3, 9.1, 13]]
    B_Te_values = [96.75040954887533, 556.3178536987874, 1278.6771501657056]
    assert_allclose(B_Te, B_Te_values)
    assert_allclose(Montinsky(16.2, 310.3E3, 2550E3), 2423.2656339862583)


def test_boiling_nucleic_Stephan_Abdelsalam():
    # Stephan Abdelsalam function
    with pytest.raises(Exception):
        Stephan_Abdelsalam(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6,  sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, angle=35, correlation='fail')

    h_SA = [Stephan_Abdelsalam(16.2, 437.5, 2730., 0.086, mul=156E-6, sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, correlation=i) for i in _angles_Stephan_Abdelsalam.keys()]
    h_values = [30571.788078886435, 84657.98595551957, 3548.8050360907037, 21009.03422203015, 26722.441071108373]
    h_SA.sort()
    h_values.sort()
    assert_allclose(h_SA, h_values)



def test_boiling_nucleic_HEDH_Taborek():
    h = HEDH_Taborek(16.2, 310.3E3, 2550E3)
    assert_allclose(h, 1397.272486525486)


def test_boiling_nucleic_Bier():
    h_W = [Bier(i, 101325., 22048321.0) for i in [4.3, 9.1, 13]]
    h_W_values = [1290.5349471503353, 7420.6159464293305, 17056.026492351128]
    assert_allclose(h_W, h_W_values)
    h_B = [Bier(i, 101325., 48.9E5) for i in [4.3, 9.1, 13]]
    h_B_values = [77.81190344679615, 447.42085661013226, 1028.3812069865799]
    assert_allclose(h_B, h_B_values)


def test_boiling_nucleic_Cooper():
    h_W = [Cooper(i, 101325., 22048321.0, 18.02) for i in [4.3, 9.1, 13]]
    h_W_values = [1558.1435442153575, 7138.700876530947, 14727.09551225091]
    assert_allclose(h_W, h_W_values)
    h_B = [Cooper(i,101325., 48.9E5, 78.11184) for i in [4.3, 9.1, 13]]
    h_B_values = [504.57942247904055, 2311.7520711767947, 4769.130145905329]
    assert_allclose(h_B, h_B_values)


def test_h_nucleic():
  # TODO
    pass


def test_qmax_Zuber():
    q_calc_ex = Zuber(sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09, K=0.149)
    assert_allclose(q_calc_ex, 444307.22304342285)
    q_max = Zuber(8.2E-3, 272E3, 567, 18.09, 0.18)
    assert_allclose(q_max, 536746.9808578263)


def test_qmax_Serth_HEDH():
    qmax = Serth_HEDH(D=0.0127, sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09)
    assert_allclose(qmax, 351867.46522901946)
    # Test K calculated as a function of R
    qmax = Serth_HEDH(0.00127, 8.2E-3, 272E3, 567, 18.09)
    assert_allclose(qmax, 440111.4740326096)


def test_HEDH_Montinsky():
    assert_allclose(HEDH_Montinsky(310.3E3, 2550E3), 398405.66545181436)


def test_qmax_nucleic():
  # TODO
    pass
