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
import numpy as np

from numpy.testing import assert_allclose
import pytest




### Free convection immersed

def test_Nu_vertical_plate_Churchill():
    Nu = Nu_vertical_plate_Churchill(0.69, 2.63E9)
    assert_allclose(Nu, 147.16185223770603)


def test_Nu_sphere_Churchill():
    Nu_Res = [Nu_sphere_Churchill(.7, i) for i in np.logspace(0, 10, 11)]
    Nu_Res_values = [2.415066377224484, 2.7381040025746382, 3.3125553308635283, 4.3340933312726548, 6.1507272232235417, 9.3821675084055443, 15.145453144794978, 25.670869440317578, 47.271761310748289, 96.479305628419823, 204.74310854292045]
    assert_allclose(Nu_Res, Nu_Res_values)


def test_Nu_vertical_cylinder_Griffiths_Davis_Morgan():
    Nu_all = [[Nu_vertical_cylinder_Griffiths_Davis_Morgan(i, 1E9, j) for i in (0.999999, 1.000001)] for j in (True, False, None)]
    Nu_all_values = [[127.7046167347578, 127.7047079158867], [119.14469068641654, 119.14475025877677], [119.14469068641654, 127.7047079158867]]
    assert_allclose(Nu_all, Nu_all_values)


def test_Nu_vertical_cylinder_Jakob_Linke_Morgan():
    Nu_all = [[Nu_vertical_cylinder_Jakob_Linke_Morgan(i, 1E8, j) for i in (0.999999, 1.000001)] for j in (True, False, None)]
    Nu_all_values = [[59.87647599476619, 59.87651591243016], [55.499986124994805, 55.5000138749948], [55.499986124994805, 59.87651591243016]]
    assert_allclose(Nu_all, Nu_all_values)


def test_Nu_vertical_cylinder_Carne_Morgan():
    Nu_all = [[Nu_vertical_cylinder_Carne_Morgan(i, 2E8, j) for i in (0.999999, 1.000001)] for j in (True, False, None)]
    Nu_all_values = [[216.88764905616722, 216.88781389084312], [225.77302655456344, 225.77315298749372], [225.77302655456344, 216.88781389084312]]
    assert_allclose(Nu_all, Nu_all_values)


def test_Nu_vertical_cylinder_Eigenson_Morgan():
    Grs = [1.42E9, 1.43E9, 2.4E10, 2.5E10]
    Nus_expect = [85.22908647061865, 85.47896057139417, 252.35445465640387, 256.64456353698154]
    Nus = [Nu_vertical_cylinder_Eigenson_Morgan(0.7, Gr) for Gr in Grs]
    assert_allclose(Nus, Nus_expect)


def test_Nu_vertical_cylinder_Touloukian_Morgan():
    Nus = [Nu_vertical_cylinder_Touloukian_Morgan(.7, i) for i in (5.7E10, 5.8E10)]
    assert_allclose(Nus, [324.47395664562873, 223.80067132541936])


def test_Nu_vertical_cylinder_McAdams_Weiss_Saunders():
    Nus = [Nu_vertical_cylinder_McAdams_Weiss_Saunders(.7, i) for i in [1.42E9,  1.43E9]]
    assert_allclose(Nus, [104.76075212013542, 130.04331889690818])    
    
    
def test_Nu_vertical_cylinder_Kreith_Eckert():
    Nus = [Nu_vertical_cylinder_Kreith_Eckert(.7, i) for i in [1.42E9, 1.43E9]]
    assert_allclose(Nus, [98.54613123165282, 83.63593679160734])
    
    
def test_Nu_vertical_cylinder_Hanesian_Kalish_Morgan():
    Nu = Nu_vertical_cylinder_Hanesian_Kalish_Morgan(.7, 1E7)
    assert_allclose(Nu, 18.014150492696604)


def test_Nu_vertical_cylinder_Al_Arabi_Khamis():
    Nus = [Nu_vertical_cylinder_Al_Arabi_Khamis(.71, i, 10, 1) for i in [3.6E9, 3.7E9]]
    assert_allclose(Nus, [185.32314790756703, 183.89407579255627])


def test_Nu_vertical_cylinder_Popiel_Churchill():
    Nu = Nu_vertical_cylinder_Popiel_Churchill(0.7, 1E10, 2.5, 1)
    assert_allclose(Nu, 228.8979005514989)


def test_Nu_horizontal_cylinder_Churchill_Chu():
    Nu = Nu_horizontal_cylinder_Churchill_Chu(0.69, 2.63E9)
    assert_allclose(Nu, 139.13493970073597)


def test_Nu_horizontal_cylinder_Kuehn_Goldstein():
    Nu = Nu_horizontal_cylinder_Kuehn_Goldstein(0.69, 2.63E9)
    assert_allclose(Nu, 122.99323525628186)
    
    
def test_Nu_horizontal_cylinder_Morgan():
    Nus = [Nu_horizontal_cylinder_Morgan(.9, i) for i in (1E-2, 1E2, 1E4, 1E7, 1E10)]
    Nus_expect = [0.5136293570857408, 1.9853087795801612, 4.707783879945983, 26.290682760247975, 258.0315247153301]
    assert_allclose(Nus, Nus_expect)
    
    
def test_Nu_horizontal_cylinder():
    Nu = Nu_horizontal_cylinder(0.72, 1E7)
    assert_allclose(Nu, 24.864192615468973)
    
    
def test_Nu_vertical_cylinder():
    Nu = Nu_vertical_cylinder(0.72, 1E7)
    assert_allclose(Nu, 30.562236756513943)
    
