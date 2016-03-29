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

### Air Cooler

def test_air_cooler_Ft():
    Ft_1 = Ft_aircooler(Thi=93, Tho=52, Tci=35, Tco=54.59, Ntp=2, rows=4)
    assert_allclose(Ft_1, 0.9570456123827129)
    Ft_2 = Ft_aircooler(Thi=125., Tho=45., Tci=25., Tco=95., Ntp=1, rows=4)
    assert_allclose(Ft_2, 0.5505093604092708)
    Ft_many = [[Ft_aircooler(Thi=125., Tho=80., Tci=25., Tco=95., Ntp=i, rows=j) for i in range(1,6)] for j in range(1, 6)]
    Ft_values = [[0.6349871996666123, 0.9392743008890244, 0.9392743008890244, 0.9392743008890244, 0.9392743008890244], [0.7993839562360742, 0.9184594715750571, 0.9392743008890244, 0.9392743008890244, 0.9392743008890244], [0.8201055328279105, 0.9392743008890244, 0.9784008071402877, 0.9392743008890244, 0.9392743008890244], [0.8276966706732202, 0.9392743008890244, 0.9392743008890244, 0.9828365967034366, 0.9392743008890244], [0.8276966706732202, 0.9392743008890244, 0.9392743008890244, 0.9392743008890244, 0.9828365967034366]]
    assert_allclose(Ft_many, Ft_values)


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


### Condensation

def test_h_Nusselt_laminar():
    h = Nusselt_laminar(370, 350, 7.0, 585., 0.091, 158.9E-6, 776900, 0.1)
    assert_allclose(h, 1482.5066124858113)
    h_angle = [Nusselt_laminar(Tsat=370, Tw=350, rhog=7.0, rhol=585., kl=0.091, mul=158.9E-6, Hvap=776900, L=0.1, angle=i) for i in np.linspace(0, 90, 8)]
    h_angle_values = [0.0, 1018.2146882558925, 1203.2059826636548, 1317.3584991910789, 1394.0390124677751, 1444.3554526761984, 1473.12556096256, 1482.5066124858113]
    assert_allclose(h_angle, h_angle_values)


def test_h_Boyko_Kruzhilin():
    h_xs = [Boyko_Kruzhilin(m=0.35, rhog=6.36, rhol=582.9, kl=0.098,
	    mul=159E-6, Cpl=2520., D=0.03, x=i) for i in np.linspace(0,1,11)]
    h_xs_values = [1190.3309510899785, 3776.3883678904836, 5206.2779830848758, 6320.5657791981021, 7265.9323628276288, 8101.7278671405438, 8859.0188940546595, 9556.4866502932564, 10206.402815353165, 10817.34162173243, 11395.573750069829]
    assert_allclose(h_xs, h_xs_values)


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



### Conv external

def test_Nu_cylinder_Zukauskas():
    Nu = Nu_cylinder_Zukauskas(7992, 0.707, 0.69)
    assert_allclose(Nu, 50.523612661934386)

    Nus_allRe = [Nu_cylinder_Zukauskas(Re, 0.707, 0.69) for Re in np.logspace(0, 6, 8)]
    Nus_allRe_values = [0.66372630070423799, 1.4616593536687801, 3.2481853039940831, 8.7138930573143227, 26.244842388228189, 85.768869004450067, 280.29503021904566, 1065.9610995854582]
    assert_allclose(Nus_allRe, Nus_allRe_values)

    Nu_highPr = Nu_cylinder_Zukauskas(7992, 42.)
    assert_allclose(Nu_highPr, 219.24837219760443)

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


def test_Nu_cylinder_Churchill_Bernstein():
    Nu = Nu_cylinder_Churchill_Bernstein(6071, 0.7)
    assert_allclose(Nu, 40.63708594124974)


def test_Nu_cylinder_Sanitjai_Goldstein():
    Nu = Nu_cylinder_Sanitjai_Goldstein(6071, 0.7)
    assert_allclose(Nu, 40.38327083519522)


def test_Nu_cylinder_Fand():
    Nu = Nu_cylinder_Fand(6071, 0.7)
    assert_allclose(Nu, 45.19984325481126)


def test_Nu_cylinder_McAdams():
    Nu = Nu_cylinder_McAdams(6071, 0.7)
    assert_allclose(Nu, 46.98179235867934)


def test_Nu_cylinder_Whitaker():
    Nu = Nu_cylinder_Whitaker(6071, 0.7)
    assert_allclose(Nu, 45.94527461589126)
    Nu = Nu_cylinder_Whitaker(6071, 0.7, 1E-3, 1.2E-3)
    assert_allclose(Nu, 43.89808146760356)


def test_Nu_cylinder_Perkins_Leppert_1962():
    Nu = Nu_cylinder_Perkins_Leppert_1962(6071, 0.7)
    assert_allclose(Nu, 49.97164291175499)
    Nu = Nu_cylinder_Perkins_Leppert_1962(6071, 0.7, 1E-3, 1.2E-3)
    assert_allclose(Nu, 47.74504603464674)


def test_Nu_cylinder_Perkins_Leppert_1964():
    Nu = Nu_cylinder_Perkins_Leppert_1964(6071, 0.7)
    assert_allclose(Nu, 53.61767038619986)
    Nu = Nu_cylinder_Perkins_Leppert_1964(6071, 0.7, 1E-3, 1.2E-3)
    assert_allclose(Nu, 51.22861670528418)

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

### Giving up ono conv_free immersed for now, TODO

### conv_internal
def test_Nu_const():
    assert_allclose(laminar_T_const(), 3.66)
    assert_allclose(laminar_Q_const(), 48/11.)


def test_laminar_entry_region():
    Nu = laminar_entry_thermal_Hausen(100000, 1.1, 5, .5)
    assert_allclose(Nu, 39.01352358988535)

    Nu = laminar_entry_Seider_Tate(Re=100000, Pr=1.1, L=5, Di=.5)
    assert_allclose(Nu, 41.366029684589265)
    Nu_wall = laminar_entry_Seider_Tate(100000, 1.1, 5, .5, 1E-3, 1.2E-3)
    assert_allclose(Nu_wall, 40.32352264095969)

    Nu = laminar_entry_Baehr_Stephan(100000, 1.1, 5, .5)
    assert_allclose(Nu, 72.65402046550976)

def test_turbulent_complicated():
    Nu1 = turbulent_Dittus_Boelter(1E5, 1.2, True, False)
    Nu2 = turbulent_Dittus_Boelter(Re=1E5, Pr=1.2, heating=False, revised=False)
    Nu3 = turbulent_Dittus_Boelter(Re=1E5, Pr=1.2, heating=False)
    Nu4 = turbulent_Dittus_Boelter(Re=1E5, Pr=1.2)
    Nu_values = [261.3838629346147, 279.89829163640354, 242.9305927410295, 247.40036409449127]
    assert_allclose([Nu1, Nu2, Nu3, Nu4], Nu_values)

    Nu1 = turbulent_Sieder_Tate(Re=1E5, Pr=1.2)
    Nu2 = turbulent_Sieder_Tate(1E5, 1.2, 0.01, 0.067)
    assert_allclose([Nu1, Nu2], [286.9178136793052, 219.84016455766044])

    Nus = [turbulent_entry_Hausen(1E5, 1.2, 0.154, i) for i in np.linspace(0,1,11)]
    Nus_values = [np.inf, 507.39810608575436, 400.1002551153033, 356.83464396632377, 332.50684459222612, 316.60088883614151, 305.25121748064328, 296.67481510644825, 289.92566421612082, 284.45128111774227, 279.90553997822707]
    assert_allclose(Nus, Nus_values)

def test_turbulent_simple():
    Nu = turbulent_Colburn(1E5, 1.2)
    assert_allclose(Nu, 244.41147091200068)

    Nu = turbulent_Drexel_McAdams(1E5, 0.6)
    assert_allclose(Nu, 171.19055301724387)

    Nu = turbulent_von_Karman(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 255.7243541243272)

    Nu = turbulent_Prandtl(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 256.073339689557)

    Nu = turbulent_Friend_Metzner(1E5, 100., 0.0185)
    assert_allclose(Nu, 1738.3356262055322)

    Nu = turbulent_Petukhov_Kirillov_Popov(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 250.11935088905105)

    Nu = turbulent_Webb(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 239.10130376815872)

    Nu = turbulent_Sandall(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 229.0514352970239)

    Nu = turbulent_Gnielinski(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 254.62682749359632)

    Nu = turbulent_Gnielinski_smooth_1(1E5, 1.2)
    assert_allclose(Nu, 227.88800494373442)

    Nu = turbulent_Gnielinski_smooth_2(1E5, 7.)
    assert_allclose(Nu, 577.7692524513449)

    Nu = turbulent_Churchill_Zajic(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 260.5564907817961)

    Nu = turbulent_ESDU(1E5, 1.2)
    assert_allclose(Nu, 232.3017143430645)

def test_turbulent_rough():
    Nu = turbulent_Martinelli(1E5, 100., 0.0185)
    assert_allclose(Nu, 887.1710686396347)

    Nu = turbulent_Nunner(1E5, 0.7, 0.0185, 0.005)
    assert_allclose(Nu, 101.15841010919947)

    Nu = turbulent_Dipprey_Sabersky(1E5, 1.2, 0.0185, 1E-3)
    assert_allclose(Nu, 288.33365198566656)

    Nu = turbulent_Gowen_Smith(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 131.72530453824106)

    Nu = turbulent_Kawase_Ulbrecht(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 389.6262247333975)

    Nu = turbulent_Kawase_De(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 296.5019733271324)

    Nu = turbulent_Bhatti_Shah(1E5, 1.2, 0.0185, 1E-3)
    assert_allclose(Nu, 302.7037617414273)

# TODO meta function  Nu internal

def test_conv_jacket():
    # actual example
    h = Lehrer(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, dT=20.)
    assert_allclose(h, 2922.128124761829)
    # no wall correction
    h = Lehrer(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, dT=20.)
    assert_allclose(h, 2608.8602693706853)

    # with isobaric expansion, all cases
    h = Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', isobaric_expansion=0.000303)
    assert_allclose(h, 3269.4389632666557)

    h = Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', inletlocation='top', isobaric_expansion=0.000303)
    assert_allclose(h, 2566.1198726589996)

    h = Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=-20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', isobaric_expansion=0.000303)
    assert_allclose(h, 3269.4389632666557)

    h = Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=-20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', inletlocation='bottom', isobaric_expansion=0.000303)
    assert_allclose(h, 2566.1198726589996)


    ### Stein Schmidt

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, 971.8)
    assert_allclose(h, 5695.1871940874225)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, 971.8, inlettype='radial')
    assert_allclose(h, 1217.1449686341773)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, 971.8, inletlocation='top')
    assert_allclose(h, 5675.824588428565)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 995.7, 4178.1, 0.615, 798E-6, 355E-6, 971.8, inletlocation='bottom')
    assert_allclose(h, 5695.1871940874225)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 971.8, 4178.1, 0.615, 798E-6, 355E-6, 995.7, inletlocation='bottom')
    assert_allclose(h, 5694.955289327642)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 971.8, 4178.1, 0.615, 798E-6, 355E-6, 995.7, inletlocation='top')
    assert_allclose(h, 5676.0574502620975)

    h = Stein_Schmidt(2.5, 0.6, 0.65, 0.6, 0.025, 971.8, 4178.1, 0.615, 798E-6, 355E-6)
    assert_allclose(h, 5685.515980483362)

    h = Stein_Schmidt(.1, 0.6, 0.65, 0.6, 0.025, 971.8, 4178.1, 0.615, 798E-6)
    assert_allclose(h, 146.80846173206865)


def test_Nu_packed_bed():
    Nu = Nu_packed_bed_Gnielinski(8E-4, 0.4, 1, 1E3, 1E-3, 0.7)
    assert_allclose(Nu, 61.37823202546954)

    # fa=2 test
    Nu = Nu_packed_bed_Gnielinski(8E-4, 0.4, 1, 1E3, 1E-3, 0.7, 2)
    assert_allclose(Nu, 64.60866528996795)

def test_conv_tube_bank():
    f = Kern_f_Re(np.linspace(10, 1E6, 10))
    f_values = [6.0155491322862771, 0.19881943524161752, 0.1765198121811164, 0.16032260681398205, 0.14912064432650635, 0.14180674990498099, 0.13727374873569789, 0.13441446600494875, 0.13212172689902535, 0.12928835660421958]
    assert_allclose(f, f_values)

    dP = dP_Kern(11., 995., 0.000803, 0.584, 0.1524, 0.0254, .019, 22, 0.000657)
    assert_allclose(dP, 18980.58768759033)

    dP = dP_Kern(m=11., rho=995., mu=0.000803, DShell=0.584, LSpacing=0.1524, pitch=0.0254, Do=.019, NBaffles=22)
    assert_allclose(dP, 19521.38738647667)

    # TODO Splines

    dP1 = dP_Zukauskas(Re=13943., n=7, ST=0.0313, SL=0.0343, D=0.0164, rho=1.217, Vmax=12.6)
    dP2 = dP_Zukauskas(Re=13943., n=7, ST=0.0313, SL=0.0313, D=0.0164, rho=1.217, Vmax=12.6)
    assert_allclose([dP1, dP2], [235.22916169118335, 217.0750033117563])

def test_core():
    dT = LMTD(100., 60., 30., 40.2)
    assert_allclose(dT, 43.200409294131525)
    dT = LMTD(100., 60., 30., 40.2, counterflow=False)
    assert_allclose(dT, 39.75251118049003)


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

def test_insulation():
    rho_tot = sum([i[0] for i in building_materials.values()])
    k_tot = sum([i[1] for i in building_materials.values()])
    Cp_tot = sum([i[2] for i in building_materials.values()])
    ans = [213240.48, 1201.7044, 164486]
    assert_allclose([rho_tot, k_tot, Cp_tot], ans)

    assert_allclose(0.036, ASHRAE_k(ID='Mineral fiber'))
    assert_allclose(sum([ASHRAE_k(ID) for ID in ASHRAE]), 102.33813464784427)

    k_VDIs = [refractory_VDI_k('Fused silica', i) for i in [None, 200, 1000, 1500]]
    assert_allclose(k_VDIs, [1.44, 1.44, 1.58074, 1.73])

    Cp_VDIs = [refractory_VDI_Cp('Fused silica', i) for i in [None, 200, 1000, 1500]]
    assert_allclose(Cp_VDIs, [917.0, 917.0, 956.78225, 982.0])

    assert nearest_material('stainless steel') == 'Metals, stainless steel'
    assert nearest_material('stainless wood') == 'Metals, stainless steel'
    assert nearest_material('asdfasdfasdfasdfasdfasdfads ') == 'Expanded polystyrene, molded beads'

    assert nearest_material('stainless steel', complete=True) == 'Metals, stainless steel'


    k = k_material('Mineral fiber')
    assert_allclose(k, 0.036)
    k = k_material('stainless steel')
    assert_allclose(k, 17.0)

    k_tot = sum([k_material(ID) for ID in materials_dict])
    assert_allclose(k_tot, 1505.18253465)

    rho = rho_material('Mineral fiber')
    assert_allclose(rho, 30.0)

    rho = rho_material('stainless steel')
    assert_allclose(rho, 7900.0)

    rho = rho_material('Board, Asbestos/cement')
    assert_allclose(rho, 1900.0)

    rho = sum([rho_material(mat) for mat in materials_dict if (materials_dict[mat] == 1 or materials_dict[mat]==3 or ASHRAE[mat][0])])
    assert_allclose(rho, 473135.98)

    Cp = Cp_material('Mineral fiber')
    assert_allclose(Cp, 840.0)

    Cp = Cp_material('stainless steel')
    assert_allclose(Cp, 460.0)

    Cp = sum([Cp_material(mat) for mat in materials_dict if ( materials_dict[mat] == 1 or materials_dict[mat]==3 or ASHRAE[mat][1])])
    assert_allclose(Cp, 353115.0)

    with pytest.raises(Exception):
        rho_material('Clay tile, hollow, 1 cell deep')
    with pytest.raises(Exception):
        Cp_material('Siding, Aluminum, steel, or vinyl, over sheathing foil-backed')


def test_radiation():
    assert_allclose(q_rad(1., 400), 1451.613952, rtol=1e-05)
    assert_allclose(q_rad(.85, 400, 305.), 816.7821722650002, rtol=1e-05)

    assert_allclose(blackbody_spectral_radiance(800., 4E-6), 1311692056.2430143, rtol=1e-05)