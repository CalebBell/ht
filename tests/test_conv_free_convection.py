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
from ht.boiling_nucleic import _angles_Stephan_Abdelsalam
import numpy as np

from numpy.testing import assert_allclose
import pytest




### Free convection immersed

def test_Nu_horizontal_plate_McAdams():
    Nu = Nu_horizontal_plate_McAdams(5.54, 3.21e8, buoyancy=True)
    assert_allclose(Nu, 181.73121274384457)
    
    Nu = Nu_horizontal_plate_McAdams(5.54, 3.21e8, buoyancy=False)
    assert_allclose(Nu, 55.44564799362829)
    
    Nu = Nu_horizontal_plate_McAdams(.01, 3.21e8, buoyancy=True)
    assert_allclose(Nu, 22.857041558492334)
    Nu = Nu_horizontal_plate_McAdams(.01, 3.21e8, buoyancy=False)
    assert_allclose(Nu, 11.428520779246167)

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
    Nu = Nu_horizontal_cylinder(Pr=0.72, Gr=1E7)
    assert_allclose(Nu, 24.864192615468973)
    
    l = Nu_horizontal_cylinder(0.72, 1E7, AvailableMethods=True)
    assert len(l) == 3
    
    with pytest.raises(Exception):
        Nu_horizontal_cylinder(Pr=0.72, Gr=1E7, Method='BADMETHOD')

    
def test_Nu_vertical_cylinder():
    Nu = Nu_vertical_cylinder(0.72, 1E7)
    assert_allclose(Nu, 30.562236756513943)
    
    Nu = Nu_vertical_cylinder(0.72, 1E7, L=1, D=.1)
    assert_allclose(Nu, 36.82833881084525)
    
    with pytest.raises(Exception):
        Nu_vertical_cylinder(0.72, 1E7, Method='BADMETHOD')
    
    l = Nu_vertical_cylinder(0.72, 1E7, L=1, D=.1, AvailableMethods=True)
    assert len(l) == 11
    

def test_Nu_coil_Xin_Ebadian():
    Nu = Nu_coil_Xin_Ebadian(0.7, 2E4, horizontal=False)
    assert_allclose(Nu, 4.755689726250451)
    Nu = Nu_coil_Xin_Ebadian(0.7, 2E4, horizontal=True)
    assert_allclose(Nu, 5.2148597687849785)


def test_Nu_vertical_helical_coil_Prabhanjan_Rennie_Raghavan():
    Nu = Nu_vertical_helical_coil_Prabhanjan_Rennie_Raghavan(4.4, 1E11)
    assert_allclose(Nu, 720.6211067718227)