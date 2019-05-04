# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

from __future__ import division
from ht import *
from numpy.testing import assert_allclose
import pytest

### Conduction
# Nothing is necessary, it's all in doctests

def test_conduction():
    assert_allclose(R_to_k(R=0.05, t=0.025), 0.5)
    assert_allclose(k_to_R(k=0.5, t=0.025), 0.05)
    assert_allclose(k_to_thermal_resistivity(0.25), 4.0)
    assert_allclose(thermal_resistivity_to_k(4), 0.25)

    Rs = [R_value_to_k(0.12), R_value_to_k(0.71, SI=False)]
    assert_allclose(Rs, [0.2116666666666667, 0.20313787163983468])
    assert_allclose(R_value_to_k(1., SI=False)/R_value_to_k(1.), 5.678263341113488)


    values =  k_to_R_value(R_value_to_k(0.12)), k_to_R_value(R_value_to_k(0.71, SI=False), SI=False)
    assert_allclose(values, [0.11999999999999998, 0.7099999999999999])

    assert_allclose(R_cylinder(0.9, 1., 20, 10), 8.38432343682705e-05)

    ### Shape Factors

    assert_allclose(S_isothermal_sphere_to_plane(1, 100), 6.298932638776527)
    assert_allclose(S_isothermal_pipe_to_plane(1, 100, 3), 3.146071454894645)
    assert_allclose(S_isothermal_pipe_normal_to_plane(1, 100), 104.86893910124888)
    assert_allclose(S_isothermal_pipe_to_isothermal_pipe(.1, .2, 1, 1), 1.188711034982268)
    assert_allclose(S_isothermal_pipe_to_two_planes(.1, 5, 1), 1.2963749299921428)
    assert_allclose(S_isothermal_pipe_eccentric_to_isothermal_pipe(.1, .4, .05, 10), 47.709841915608976)


def test_cylindrical_heat_transfer():
    data = cylindrical_heat_transfer(Ti=453.15, To=301.15, hi=1e12, ho=22.697193, Di=0.0779272, ts=[0.0054864, .05], ks=[56.045, 0.0598535265])
    expect = {'Q': 73.12000884069367,
     'Rs': [0.00022201030738405449, 1.189361782070256],
     'Ts': [453.15, 453.1226455779877, 306.578530147744],
     'UA': 0.48105268974140575,
     'U_inner': 1.9649599487726137,
     'U_outer': 0.8106078714663484,
     'q': 123.21239646288495}
    
    for k, v in expect.items():
        assert_allclose(v, data[k])


def test_insulation():
    rho_tot = sum([i[0] for i in building_materials.values()])
    k_tot = sum([i[1] for i in building_materials.values()])
    Cp_tot = sum([i[2] for i in building_materials.values()])
    ans = [213240.48, 1132.7733999999994, 164486]
    assert_allclose([rho_tot, k_tot, Cp_tot], ans)

    assert_allclose(0.036, ASHRAE_k(ID='Mineral fiber'))

    k_VDIs = [refractory_VDI_k('Fused silica', i) for i in [None, 200, 1000, 1500]]
    assert_allclose(k_VDIs, [1.44, 1.44, 1.58074, 1.73])

    Cp_VDIs = [refractory_VDI_Cp('Fused silica', i) for i in [None, 200, 1000, 1500]]
    assert_allclose(Cp_VDIs, [917.0, 917.0, 956.78225, 982.0])


    k = k_material('Mineral fiber')
    assert_allclose(k, 0.036)
    k = k_material('stainless steel')
    assert_allclose(k, 17.0)

    rho = rho_material('Mineral fiber')
    assert_allclose(rho, 30.0)

    rho = rho_material('stainless steel')
    assert_allclose(rho, 7900.0)

    rho = rho_material('Board, Asbestos/cement')
    assert_allclose(rho, 1900.0)


    Cp = Cp_material('Mineral fiber')
    assert_allclose(Cp, 840.0)

    Cp = Cp_material('stainless steel')
    assert_allclose(Cp, 460.0)

    with pytest.raises(Exception):
        rho_material('Clay tile, hollow, 1 cell deep')
    with pytest.raises(Exception):
        Cp_material('Siding, Aluminum, steel, or vinyl, over sheathing foil-backed')


@pytest.mark.slow
def test_insulation_fuzz():
    assert_allclose(sum([ASHRAE_k(ID) for ID in ASHRAE]), 102.33813464784427)
    
    k_tot = sum([k_material(ID) for ID in materials_dict])
    assert_allclose(k_tot, 1436.251534647845)

    rho = sum([rho_material(mat) for mat in materials_dict if (materials_dict[mat] == 1 or materials_dict[mat]==3 or ASHRAE[mat][0])])
    assert_allclose(rho, 473135.98)

    Cp = sum([Cp_material(mat) for mat in materials_dict if ( materials_dict[mat] == 1 or materials_dict[mat]==3 or ASHRAE[mat][1])])
    assert_allclose(Cp, 353115.0)

    # fuzzy matching is slow
    assert nearest_material('stainless steel') == 'Metals, stainless steel'
    assert nearest_material('stainless wood') == 'Metals, stainless steel'
    assert nearest_material('asdfasdfasdfasdfasdfasdfads ') == 'Expanded polystyrene, molded beads'

    assert nearest_material('stainless steel', complete=True) == 'Metals, stainless steel'

