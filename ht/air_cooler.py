'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
SOFTWARE.
'''

from math import atan, log10, sin

from fluids.constants import hp, minute
from fluids.core import Prandtl, Reynolds

from ht.conv_tube_bank import ESDU_tube_row_correction
from ht.core import LMTD, WALL_FACTOR_PRANDTL, fin_efficiency_Kern_Kraus, wall_factor

__all__ = ['Ft_aircooler', 'air_cooler_noise_GPSA',
           'air_cooler_noise_Mukherjee', 'h_Briggs_Young',
           'h_ESDU_high_fin', 'h_ESDU_low_fin', 'h_Ganguli_VDI', 'dP_ESDU_high_fin',
           'dP_ESDU_low_fin']

fin_densities_inch = [7, 8, 9, 10, 11] # fins/inch
fin_densities = [275.6, 315.0, 354.3, 393.7, 433.1] # [round(i/0.0254, 1) for i in fin_densities_inch]
ODs = [1, 1.25, 1.5, 2] # Actually though, just use TEMA. API 661 says 1 inch min.
fin_heights = [0.010, 0.012, 0.016] # m


tube_orientations = ['vertical (inlet at bottom)', 'vertical (inlet at top)', 'horizontal', 'inclined']

_fan_diameters = [0.71, 0.8, 0.9, 1.0, 1.2, 1.24, 1.385, 1.585, 1.78, 1.98, 2.22, 2.475, 2.775, 3.12, 3.515, 4.455, 4.95, 5.545, 6.24, 7.03, 7.92, 8.91, 9.9, 10.4, 11.1, 12.4, 13.85, 15.85]

fan_ring_types = ['straight', 'flanged',  'bell', '15 degree cone', '30 degree cone']

fin_constructions = ['extruded', 'embedded', 'L-footed', 'overlapped L-footed', 'externally bonded', 'knurled footed']

headers = ['plug', 'removable cover', 'removable bonnet', 'welded bonnet']
configurations = ['forced draft', 'natural draft', 'induced-draft (top drive)', 'induced-draft (bottom drive)']




# Coefs are from: Roetzel and Nicole - 1975 - Mean Temperature Difference for Heat Exchanger Design A General Approximate Explicit Equation
# Checked twice.

_crossflow_1_row_1_pass = [[-4.62E-1, -3.13E-2, -1.74E-1, -4.2E-2],
                           [5.08E0, 5.29E-1, 1.32E0, 3.47E-1],
                           [-1.57E1, -2.37E0, -2.93E0, -8.53E-1],
                           [1.72E1, 3.18E0, 1.99E0, 6.49E-1]]

_crossflow_2_rows_1_pass = [[-3.34E-1, -1.54E-1, -8.65E-2, 5.53E-2],
                            [3.3E0, 1.28E0, 5.46E-1, -4.05E-1],
                            [-8.7E0, -3.35E0, -9.29E-1, 9.53E-1],
                            [8.7E0, 2.83E0, 4.71E-1, -7.17E-1]]

_crossflow_3_rows_1_pass = [[-8.74E-2, -3.18E-2, -1.83E-2, 7.1E-3],
                            [1.05E0, 2.74E-1, 1.23E-1, -4.99E-2],
                            [-2.45E0, -7.46E-1, -1.56E-1, 1.09E-1],
                            [3.21E0, 6.68E-1, 6.17E-2, -7.46E-2]]

_crossflow_4_rows_1_pass = [[-4.14E-2, -1.39E-2, -7.23E-3, 6.1E-3],
                            [6.15E-1, 1.23E-1, 5.66E-2, -4.68E-2],
                            [-1.2E0, -3.45E-1, -4.37E-2, 1.07E-1],
                            [2.06E0, 3.18E-1, 1.11E-2, -7.57E-2]]

_crossflow_2_rows_2_pass = [[-2.35E-1, -7.73E-2, -5.98E-2, 5.25E-3],
                            [2.28E0, 6.32E-1, 3.64E-1, -1.27E-2],
                            [-6.44E0, -1.63E0, -6.13E-1, -1.14E-2],
                            [6.24E0, 1.35E0, 2.76E-1, 2.72E-2]]

_crossflow_3_rows_3_pass = [[-8.43E-1, 3.02E-2, 4.8E-1, 8.12E-2],
                            [5.85E0, -9.64E-3, -3.28E0, -8.34E-1],
                            [-1.28E1, -2.28E-1, 7.11E0, 2.19E0],
                            [9.24E0, 2.66E-1, -4.9E0, -1.69E0]]

_crossflow_4_rows_4_pass = [[-3.39E-1, 2.77E-2, 1.79E-1, -1.99E-2],
                            [2.38E0, -9.99E-2, -1.21E0, 4E-2],
                            [-5.26E0, 9.04E-2, 2.62E0, 4.94E-2],
                            [3.9E0, -8.45E-4, -1.81E0, -9.81E-2]]

_crossflow_4_rows_2_pass = [[-6.05E-1, 2.31E-2, 2.94E-1, 1.98E-2],
                            [4.34E0, 5.9E-3, -1.99E0, -3.05E-1],
                            [-9.72E0, -2.48E-1, 4.32, 8.97E-1],
                            [7.54E0, 2.87E-1, -3E0, -7.31E-1]]




def Ft_aircooler(Thi, Tho, Tci, Tco, Ntp=1, rows=1):
    r'''Calculates log-mean temperature difference correction factor for
    a crossflow heat exchanger, as in an Air Cooler. Method presented in [1]_,
    fit to other's nonexplicit work. Error is < 0.1%. Requires number of rows
    and tube passes as well as stream temperatures.

    .. math::
        F_T = 1 - \sum_{i=1}^m \sum_{k=1}^n a_{i,k}(1-r_{1,m})^k\sin(2i\arctan R)

    .. math::
        R = \frac{T_{hi} - T_{ho}}{T_{co}-T_{ci}}

    .. math::
        r_{1,m} = \frac{\Delta T_{lm}}{T_{hi} - T_{ci}}

    Parameters
    ----------
    Thi : float
        Temperature of hot fluid in [K]
    Tho : float
        Temperature of hot fluid out [K]
    Tci : float
        Temperature of cold fluid in [K]
    Tco : float
        Temperature of cold fluid out [K]
    Ntp : int
        Number of passes the tubeside fluid will flow through [-]
    rows : int
        Number of rows of tubes [-]

    Returns
    -------
    Ft : float
        Log-mean temperature difference correction factor [-]

    Notes
    -----
    This equation assumes that the hot fluid is tubeside, as in the case of air
    coolers. The model is not symmetric, so ensure to switch around the inputs
    if using this function for other purposes.

    This equation appears in [1]_. It has been verified.
    For some cases, approximations are made to match coefficients with the
    number of tube passes and rows provided.
    16 coefficients are used for each case; 8 cases are considered:

    * 1 row 1 pass
    * 2 rows 1 pass
    * 2 rows 2 passes
    * 3 rows 1 pass
    * 3 rows 3 passes
    * 4 rows 1 pass
    * 4 rows 2 passes
    * 4 rows 4 passes

    Examples
    --------
    >>> Ft_aircooler(Thi=125., Tho=45., Tci=25., Tco=95., Ntp=1, rows=4)
    0.550509360409

    References
    ----------
    .. [1] Roetzel, W., and F. J. L. Nicole. "Mean Temperature Difference for
       Heat Exchanger Design-A General Approximate Explicit Equation." Journal
       of Heat Transfer 97, no. 1 (February 1, 1975): 5-8.
       doi:10.1115/1.3450288
    '''
    dTlm = LMTD(Thi=Thi, Tho=Tho, Tci=Tci, Tco=Tco)
    rlm = dTlm/(Thi-Tci)
    R = (Thi-Tho)/(Tco-Tci)
#    P = (Tco-Tci)/(Thi-Tci)

    if Ntp == 1 and rows == 1:
        coefs = _crossflow_1_row_1_pass
    elif Ntp == 1 and rows == 2:
        coefs = _crossflow_2_rows_1_pass
    elif Ntp == 1 and rows == 3:
        coefs = _crossflow_3_rows_1_pass
    elif Ntp == 1 and rows == 4:
        coefs = _crossflow_4_rows_1_pass
    elif Ntp == 1 and rows > 4:
        # A reasonable assumption
        coefs = _crossflow_4_rows_1_pass
    elif Ntp == 2 and rows == 2:
        coefs = _crossflow_2_rows_2_pass
    elif Ntp == 3 and rows == 3:
        coefs = _crossflow_3_rows_3_pass
    elif Ntp == 4 and rows == 4:
        coefs = _crossflow_4_rows_4_pass
    elif Ntp > 4 and rows > 4 and Ntp == rows:
        # A reasonable assumption
        coefs = _crossflow_4_rows_4_pass
    elif Ntp  == 2 and rows == 4:
        coefs = _crossflow_4_rows_2_pass
    else:
        # A bad assumption, but hey, gotta pick something.
        coefs = _crossflow_4_rows_2_pass
    tot = 0.0
    atanR2 = 2.0*atan(R)
    N = len(coefs)
    sine_terms = [0.0]*N
    for i in range(N):
        sine_terms[i] = sin((i + 1.)*atanR2)
    x0 = one_m_rlm_orig = 1.0 - rlm
    for k in range(N):
        coeffs_k = coefs[k]
        tot_i = 0.0
        for i in range(N):
            tot_i += coeffs_k[i]*sine_terms[i]
        tot += tot_i*x0
        x0 *= one_m_rlm_orig
    return 1. - tot


def air_cooler_noise_GPSA(tip_speed, power):
    r'''Calculates the noise generated by an air cooler bay with one fan
    according to the GPSA handbook [1]_.

    .. math::
        \text{PWL[dB(A)]} = 56 + 30\log_{10}\left( \frac{\text{tip speed}
        [m/min]}{304.8 [m/min]}\right) + 10\log_{10}( \text{power}[hp])

    Parameters
    ----------
    tip_speed : float
        Tip speed of the air cooler fan blades, [m/s]
    power : float
        Shaft power of single fan motor, [W]

    Returns
    -------
    noise : float
        Sound pressure level at 1 m from source, [dB(A)]

    Notes
    -----
    Internal units are in m/minute, and hp.

    Examples
    --------
    Example problem from GPSA [1]_.

    >>> air_cooler_noise_GPSA(tip_speed=3177/minute, power=25.1*hp)
    100.5368047795

    References
    ----------
    .. [1] GPSA. "Engineering Data Book, SI." 13th edition. Gas Processors
       Suppliers Association (2012).
    '''
    tip_speed = tip_speed*minute # convert tip speed to m/minute
    power = power/hp # convert power from W to hp
    return 56.0 + 30.0*log10(tip_speed/304.8) + 10.0*log10(power)


def air_cooler_noise_Mukherjee(tip_speed, power, fan_diameter, induced=False):
    r'''Calculates the noise generated by an air cooler bay with one fan
    according to [1]_.

    .. math::
        \text{SPL[dB(A)]} = 46 + 30\log_{10}\text{(tip speed)}[m/s]
        + 10\log_{10}( \text{power}[hp]) - 20 \log_{10}(D_{fan})

    Parameters
    ----------
    tip_speed : float
        Tip speed of the air cooler fan blades, [m/s]
    power : float
        Shaft power of single fan motor, [W]
    fan_diameter : float
        Diameter of air cooler fan, [m]
    induced : bool
        Whether the air cooler is forced air (False) or induced air (True), [-]

    Returns
    -------
    noise : float
        Sound pressure level at 1 m from source (p0=2E-5 Pa), [dB(A)]

    Notes
    -----
    Internal units are in m/minute, hp, and m.

    If the air cooler is induced, the sound pressure level is reduced by 3 dB.

    Examples
    --------
    >>> air_cooler_noise_Mukherjee(tip_speed=3177/minute, power=25.1*hp, fan_diameter=4.267)
    99.1102632909

    References
    ----------
    .. [1] Mukherjee, R., and Geoffrey Hewitt. Practical Thermal Design of
       Air-Cooled Heat Exchangers. New York: Begell House Publishers Inc.,U.S.,
       2007.
    '''
    noise = 46.0 + 30.0*log10(tip_speed) + 10.0*log10(power/hp) - 20.0*log10(fan_diameter)
    if induced:
        noise -= 3.0
    return noise


def h_Briggs_Young(m, A, A_min, A_increase, A_fin, A_tube_showing,
                   tube_diameter, fin_diameter, fin_thickness, bare_length,
                   rho, Cp, mu, k, k_fin):
    r'''Calculates the air side heat transfer coefficient for an air cooler
    or other finned tube bundle with the formulas of Briggs and Young [1]_,
    [2]_ [3]_.

    .. math::
        Nu = 0.134Re^{0.681} Pr^{0.33}\left(\frac{S}{h}\right)^{0.2}
        \left(\frac{S}{b}\right)^{0.1134}

    Parameters
    ----------
    m : float
        Mass flow rate of air across the tube bank, [kg/s]
    A : float
        Surface area of combined finned and non-finned area exposed for heat
        transfer, [m^2]
    A_min : float
        Minimum air flow area, [m^2]
    A_increase : float
        Ratio of actual surface area to bare tube surface area
        :math:`A_{increase} = \frac{A_{tube}}{A_{bare, total/tube}}`, [-]
    A_fin : float
        Surface area of all fins in the bundle, [m^2]
    A_tube_showing : float
        Area of the bare tube which is exposed in the bundle, [m^2]
    tube_diameter : float
        Diameter of the bare tube, [m]
    fin_diameter : float
        Outer diameter of each tube after including the fin on both sides,
        [m]
    fin_thickness : float
        Thickness of the fins, [m]
    bare_length : float
        Length of bare tube between two fins
        :math:`\text{bare length} = \text{fin interval} - t_{fin}`, [m]
    rho : float
        Average (bulk) density of air across the tube bank, [kg/m^3]
    Cp : float
        Average (bulk) heat capacity of air across the tube bank, [J/kg/K]
    mu : float
        Average (bulk) viscosity of air across the tube bank, [Pa*s]
    k : float
        Average (bulk) thermal conductivity of air across the tube bank,
        [W/m/K]
    k_fin : float
        Thermal conductivity of the fin, [W/m/K]

    Returns
    -------
    h_bare_tube_basis : float
        Air side heat transfer coefficient on a bare-tube surface area as if
        there were no fins present basis, [W/K/m^2]

    Notes
    -----
    The limits on this equation are 1000 < Re < 8000 ,
    11.13 mm < D_o < 40.89 mm, 1.42 mm < fin height < 16.57 mm,
    0.33 mm < fin thickness < 2.02 mm, 1.30 mm < fin pitch < 4.06 mm, and
    24.49 mm < normal pitch < 111 mm.

    Examples
    --------
    >>> from fluids.geometry import AirCooledExchanger
    >>> from scipy.constants import inch
    >>> AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=20, tube_length=3,
    ... tube_diameter=1*inch, fin_thickness=0.000406, fin_density=1/0.002309,
    ... pitch_normal=.06033, pitch_parallel=.05207,
    ... fin_height=0.0159, tube_thickness=(.0254-.0186)/2,
    ... bundles_per_bay=1, parallel_bays=1, corbels=True)

    >>> h_Briggs_Young(m=21.56, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin,
    ... A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter,
    ... fin_diameter=AC.fin_diameter, bare_length=AC.bare_length,
    ... fin_thickness=AC.fin_thickness,
    ... rho=1.161, Cp=1007., mu=1.85E-5, k=0.0263, k_fin=205)
    1422.872240323

    References
    ----------
    .. [1] Briggs, D.E., and Young, E.H., 1963, "Convection Heat Transfer and
       Pressure Drop of Air Flowing across Triangular Banks of Finned Tubes",
       Chemical Engineering Progress Symp., Series 41, No. 59. Chem. Eng. Prog.
       Symp. Series No. 41, "Heat Transfer - Houston".
    .. [2] Mukherjee, R., and Geoffrey Hewitt. Practical Thermal Design of
       Air-Cooled Heat Exchangers. New York: Begell House Publishers Inc.,U.S.,
       2007.
    .. [3] Kroger, Detlev. Air-Cooled Heat Exchangers and Cooling Towers:
       Thermal-Flow Performance Evaluation and Design, Vol. 1. Tulsa, Okl:
       PennWell Corp., 2004.
    '''
    fin_height = 0.5*(fin_diameter - tube_diameter)

    V_max = m/(A_min*rho)

    Re = Reynolds(V=V_max, D=tube_diameter, rho=rho, mu=mu)
    Pr = Prandtl(Cp=Cp, mu=mu, k=k)

    Nu = 0.134*Re**0.681*Pr**(1/3.)*(bare_length/fin_height)**0.2*(bare_length/fin_thickness)**0.1134

    h = k/tube_diameter*Nu
    efficiency = fin_efficiency_Kern_Kraus(Do=tube_diameter, D_fin=fin_diameter,
                                           t_fin=fin_thickness, k_fin=k_fin, h=h)
    h_total_area_basis = (efficiency*A_fin + A_tube_showing)/A*h
    h_bare_tube_basis = h_total_area_basis*A_increase

    return h_bare_tube_basis


def h_ESDU_high_fin(m, A, A_min, A_increase, A_fin, A_tube_showing,
                    tube_diameter, fin_diameter, fin_thickness, bare_length,
                    pitch_parallel, pitch_normal, tube_rows,
                    rho, Cp, mu, k, k_fin, Pr_wall=None):
    r'''Calculates the air side heat transfer coefficient for an air cooler
    or other finned tube bundle with the formulas of [2]_ as presented in [1]_.

    .. math::
        Nu = 0.242 Re^{0.658} \left(\frac{\text{bare length}}
        {\text{fin height}}\right)^{0.297}
        \left(\frac{P_1}{P_2}\right)^{-0.091} P_r^{1/3}\cdot F_1\cdot F_2

    .. math::
        h_{A,total} = \frac{\eta A_{fin} + A_{bare, showing}}{A_{total}} h

    .. math::
        h_{bare,total} = A_{increase} h_{A,total}

    Parameters
    ----------
    m : float
        Mass flow rate of air across the tube bank, [kg/s]
    A : float
        Surface area of combined finned and non-finned area exposed for heat
        transfer, [m^2]
    A_min : float
        Minimum air flow area, [m^2]
    A_increase : float
        Ratio of actual surface area to bare tube surface area
        :math:`A_{increase} = \frac{A_{tube}}{A_{bare, total/tube}}`, [-]
    A_fin : float
        Surface area of all fins in the bundle, [m^2]
    A_tube_showing : float
        Area of the bare tube which is exposed in the bundle, [m^2]
    tube_diameter : float
        Diameter of the bare tube, [m]
    fin_diameter : float
        Outer diameter of each tube after including the fin on both sides,
        [m]
    fin_thickness : float
        Thickness of the fins, [m]
    bare_length : float
        Length of bare tube between two fins
        :math:`\text{bare length} = \text{fin interval} - t_{fin}`, [m]
    pitch_parallel : float
        Distance between tube center along a line parallel to the flow;
        has been called `longitudinal` pitch, `pp`, `s2`, `SL`, and `p2`, [m]
    pitch_normal : float
        Distance between tube centers in a line 90° to the line of flow;
        has been called the `transverse` pitch, `pn`, `s1`, `ST`, and `p1`, [m]
    tube_rows : int
        Number of tube rows per bundle, [-]
    rho : float
        Average (bulk) density of air across the tube bank, [kg/m^3]
    Cp : float
        Average (bulk) heat capacity of air across the tube bank, [J/kg/K]
    mu : float
        Average (bulk) viscosity of air across the tube bank, [Pa*s]
    k : float
        Average (bulk) thermal conductivity of air across the tube bank,
        [W/m/K]
    k_fin : float
        Thermal conductivity of the fin, [W/m/K]
    Pr_wall : float, optional
        Prandtl number at the wall temperature; provide if a correction with
        the defaults parameters is desired; otherwise apply the correction
        elsewhere, [-]

    Returns
    -------
    h_bare_tube_basis : float
        Air side heat transfer coefficient on a bare-tube surface area as if
        there were no fins present basis, [W/K/m^2]

    Notes
    -----
    The tube-row count correction factor is 1 for four or more rows, 0.92 for
    three rows, 0.84 for two rows, and 0.76 for one row according to [1]_.

    The property correction factor can be disabled by not specifying
    `Pr_wall`. A Prandtl number exponent of 0.26 is recommended in [1]_ for
    heating and cooling for both liquids and gases.

    Examples
    --------
    >>> from fluids.geometry import AirCooledExchanger
    >>> from scipy.constants import inch
    >>> AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=20, tube_length=3,
    ... tube_diameter=1*inch, fin_thickness=0.000406, fin_density=1/0.002309,
    ... pitch_normal=.06033, pitch_parallel=.05207,
    ... fin_height=0.0159, tube_thickness=(.0254-.0186)/2,
    ... bundles_per_bay=1, parallel_bays=1, corbels=True)

    >>> h_ESDU_high_fin(m=21.56, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin,
    ... A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter,
    ... fin_diameter=AC.fin_diameter, bare_length=AC.bare_length,
    ... fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows,
    ... pitch_normal=AC.pitch_normal, pitch_parallel=AC.pitch_parallel,
    ... rho=1.161, Cp=1007., mu=1.85E-5, k=0.0263, k_fin=205)
    1390.88891804

    References
    ----------
    .. [1] Hewitt, G. L. Shires, T. Reg Bott G. F., George L. Shires, and T.
       R. Bott. Process Heat Transfer. 1st edition. Boca Raton: CRC Press,
       1994.
    .. [2] "High-Fin Staggered Tube Banks: Heat Transfer and Pressure Drop for
       Turbulent Single Phase Gas Flow." ESDU 86022 (October 1, 1986).
    .. [3] Rabas, T. J., and J. Taborek. "Survey of Turbulent Forced-Convection
       Heat Transfer and Pressure Drop Characteristics of Low-Finned Tube Banks
       in Cross Flow."  Heat Transfer Engineering 8, no. 2 (January 1987):
       49-62.
    '''
    fin_height = 0.5*(fin_diameter - tube_diameter)

    V_max = m/(A_min*rho)
    Re = Reynolds(V=V_max, D=tube_diameter, rho=rho, mu=mu)
    Pr = Prandtl(Cp=Cp, mu=mu, k=k)
    Nu = 0.242*Re**0.658*(bare_length/fin_height)**0.297*(pitch_normal/pitch_parallel)**-0.091*Pr**(1/3.)

    if tube_rows < 2:
        F2 = 0.76
    elif tube_rows < 3:
        F2 = 0.84
    elif tube_rows < 4:
        F2 = 0.92
    else:
        F2 = 1.0

    Nu *= F2
    if Pr_wall is not None:
        F1 = wall_factor(Pr=Pr, Pr_wall=Pr_wall, Pr_heating_coeff=0.26,
                         Pr_cooling_coeff=0.26,
                         property_option=WALL_FACTOR_PRANDTL)
        Nu *= F1

    h = k/tube_diameter*Nu
    efficiency = fin_efficiency_Kern_Kraus(Do=tube_diameter, D_fin=fin_diameter,
                                           t_fin=fin_thickness, k_fin=k_fin, h=h)

    h_total_area_basis = (efficiency*A_fin + A_tube_showing)/A*h
    h_bare_tube_basis =  h_total_area_basis*A_increase
    return h_bare_tube_basis


def h_ESDU_low_fin(m, A, A_min, A_increase, A_fin,
                   A_tube_showing, tube_diameter,
                   fin_diameter, fin_thickness, bare_length,
                   pitch_parallel, pitch_normal, tube_rows,
                   rho, Cp, mu, k, k_fin, Pr_wall=None):
    r'''Calculates the air side heat transfer coefficient for an air cooler
    or other finned tube bundle with low fins using the formulas of [1]_ as
    presented in [2]_ (and also [3]_).

    .. math::
        Nu = 0.183Re^{0.7} \left(\frac{\text{bare length}}{\text{fin height}}
        \right)^{0.36}\left(\frac{p_1}{D_{o}}\right)^{0.06}
        \left(\frac{\text{fin height}}{D_o}\right)^{0.11}
        Pr^{0.36} \cdot F_1\cdot F_2

    .. math::
        h_{A,total} = \frac{\eta A_{fin} + A_{bare, showing}}{A_{total}} h

    .. math::
        h_{bare,total} = A_{increase} h_{A,total}

    Parameters
    ----------
    m : float
        Mass flow rate of air across the tube bank, [kg/s]
    A : float
        Surface area of combined finned and non-finned area exposed for heat
        transfer, [m^2]
    A_min : float
        Minimum air flow area, [m^2]
    A_increase : float
        Ratio of actual surface area to bare tube surface area
        :math:`A_{increase} = \frac{A_{tube}}{A_{bare, total/tube}}`, [-]
    A_fin : float
        Surface area of all fins in the bundle, [m^2]
    A_tube_showing : float
        Area of the bare tube which is exposed in the bundle, [m^2]
    tube_diameter : float
        Diameter of the bare tube, [m]
    fin_diameter : float
        Outer diameter of each tube after including the fin on both sides,
        [m]
    fin_thickness : float
        Thickness of the fins, [m]
    bare_length : float
        Length of bare tube between two fins
        :math:`\text{bare length} = \text{fin interval} - t_{fin}`, [m]
    pitch_parallel : float
        Distance between tube center along a line parallel to the flow;
        has been called `longitudinal` pitch, `pp`, `s2`, `SL`, and `p2`, [m]
    pitch_normal : float
        Distance between tube centers in a line 90° to the line of flow;
        has been called the `transverse` pitch, `pn`, `s1`, `ST`, and `p1`, [m]
    tube_rows : int
        Number of tube rows per bundle, [-]
    rho : float
        Average (bulk) density of air across the tube bank, [kg/m^3]
    Cp : float
        Average (bulk) heat capacity of air across the tube bank, [J/kg/K]
    mu : float
        Average (bulk) viscosity of air across the tube bank, [Pa*s]
    k : float
        Average (bulk) thermal conductivity of air across the tube bank,
        [W/m/K]
    k_fin : float
        Thermal conductivity of the fin, [W/m/K]
    Pr_wall : float, optional
        Prandtl number at the wall temperature; provide if a correction with
        the defaults parameters is desired; otherwise apply the correction
        elsewhere, [-]

    Returns
    -------
    h_bare_tube_basis : float
        Air side heat transfer coefficient on a bare-tube surface area as if
        there were no fins present basis, [W/K/m^2]

    Notes
    -----
    The tube-row count correction factor `F2` can be disabled by setting `tube_rows`
    to 10. The property correction factor `F1` can be disabled by not specifying
    `Pr_wall`. A Prandtl number exponent of 0.26 is recommended in [1]_ for
    heating and cooling for both liquids and gases.

    There is a third correction factor in [1]_ for tube angles not 30, 45, or
    60 degrees, but it is not fully explained and it is not shown in [2]_.
    Another correction factor is in [2]_ for flow at an angle; however it would
    not make sense to apply it to finned tube banks due to the blockage by the
    fins.

    Examples
    --------
    >>> from fluids.geometry import AirCooledExchanger
    >>> AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=8, tube_length=0.5,
    ... tube_diameter=0.0164, fin_thickness=0.001, fin_density=1/0.003,
    ... pitch_normal=0.0313, pitch_parallel=0.0271, fin_height=0.0041, corbels=True)

    >>> h_ESDU_low_fin(m=0.914, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin,
    ... A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter,
    ... fin_diameter=AC.fin_diameter, bare_length=AC.bare_length,
    ... fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows,
    ... pitch_normal=AC.pitch_normal, pitch_parallel=AC.pitch_parallel,
    ... rho=1.217, Cp=1007., mu=1.8E-5, k=0.0253, k_fin=15)
    553.85383647

    References
    ----------
    .. [1] Hewitt, G. L. Shires, T. Reg Bott G. F., George L. Shires, and T.
       R. Bott. Process Heat Transfer. 1st edition. Boca Raton: CRC Press,
       1994.
    .. [2] "High-Fin Staggered Tube Banks: Heat Transfer and Pressure Drop for
       Turbulent Single Phase Gas Flow." ESDU 86022 (October 1, 1986).
    .. [3] Rabas, T. J., and J. Taborek. "Survey of Turbulent Forced-Convection
       Heat Transfer and Pressure Drop Characteristics of Low-Finned Tube Banks
       in Cross Flow."  Heat Transfer Engineering 8, no. 2 (January 1987):
       49-62.
    '''
    fin_height = 0.5*(fin_diameter - tube_diameter)

    V_max = m/(A_min*rho)
    Re = Reynolds(V=V_max, D=tube_diameter, rho=rho, mu=mu)
    Pr = Prandtl(Cp=Cp, mu=mu, k=k)
    Nu = (0.183*Re**0.7*(bare_length/fin_height)**0.36
          *(pitch_normal/fin_diameter)**0.06
          *(fin_height/fin_diameter)**0.11*Pr**0.36)

    staggered = abs(1 - pitch_normal/pitch_parallel) > 0.05
    F2 = ESDU_tube_row_correction(tube_rows=tube_rows, staggered=staggered)
    Nu *= F2
    if Pr_wall is not None:
        F1 = wall_factor(Pr=Pr, Pr_wall=Pr_wall, Pr_heating_coeff=0.26,
                         Pr_cooling_coeff=0.26,
                         property_option=WALL_FACTOR_PRANDTL)
        Nu *= F1

    h = k/tube_diameter*Nu
    efficiency = fin_efficiency_Kern_Kraus(Do=tube_diameter,
                                           D_fin=fin_diameter,
                                           t_fin=fin_thickness,
                                           k_fin=k_fin, h=h)
    h_total_area_basis = (efficiency*A_fin + A_tube_showing)/A*h
    h_bare_tube_basis = h_total_area_basis*A_increase
    return h_bare_tube_basis


def h_Ganguli_VDI(m, A, A_min, A_increase, A_fin,
                  A_tube_showing, tube_diameter,
                  fin_diameter, fin_thickness, bare_length,
                  pitch_parallel, pitch_normal, tube_rows,
                  rho, Cp, mu, k, k_fin):
    r'''Calculates the air side heat transfer coefficient for an air cooler
    or other finned tube bundle with the formulas of [1]_ as modified in [2]_.

    Inline:

    .. math::
        Nu_d = 0.22Re_d^{0.6}\left(\frac{A}{A_{tube,only}}\right)^{-0.15}Pr^{1/3}

    Staggered:

    .. math::
        Nu_d = 0.38 Re_d^{0.6}\left(\frac{A}{A_{tube,only}}\right)^{-0.15}Pr^{1/3}

    Parameters
    ----------
    m : float
        Mass flow rate of air across the tube bank, [kg/s]
    A : float
        Surface area of combined finned and non-finned area exposed for heat
        transfer, [m^2]
    A_min : float
        Minimum air flow area, [m^2]
    A_increase : float
        Ratio of actual surface area to bare tube surface area
        :math:`A_{increase} = \frac{A_{tube}}{A_{bare, total/tube}}`, [-]
    A_fin : float
        Surface area of all fins in the bundle, [m^2]
    A_tube_showing : float
        Area of the bare tube which is exposed in the bundle, [m^2]
    tube_diameter : float
        Diameter of the bare tube, [m]
    fin_diameter : float
        Outer diameter of each tube after including the fin on both sides,
        [m]
    fin_thickness : float
        Thickness of the fins, [m]
    bare_length : float
        Length of bare tube between two fins
        :math:`\text{bare length} = \text{fin interval} - t_{fin}`, [m]
    pitch_parallel : float
        Distance between tube center along a line parallel to the flow;
        has been called `longitudinal` pitch, `pp`, `s2`, `SL`, and `p2`, [m]
    pitch_normal : float
        Distance between tube centers in a line 90° to the line of flow;
        has been called the `transverse` pitch, `pn`, `s1`, `ST`, and `p1`, [m]
    tube_rows : int
        Number of tube rows per bundle, [-]
    rho : float
        Average (bulk) density of air across the tube bank, [kg/m^3]
    Cp : float
        Average (bulk) heat capacity of air across the tube bank, [J/kg/K]
    mu : float
        Average (bulk) viscosity of air across the tube bank, [Pa*s]
    k : float
        Average (bulk) thermal conductivity of air across the tube bank,
        [W/m/K]
    k_fin : float
        Thermal conductivity of the fin, [W/m/K]

    Returns
    -------
    h_bare_tube_basis : float
        Air side heat transfer coefficient on a bare-tube surface area as if
        there were no fins present basis, [W/K/m^2]

    Notes
    -----
    The VDI modifications were developed in comparison with HTFS and HTRI data
    according to [2]_.

    For cases where the tube row count is less than four, the coefficients are
    modified in [2]_. For the inline case, 0.2 replaces 0.22. For the stagered
    cases, the coefficient is 0.2, 0.33, 0.36 for 1, 2, or 3 tube rows
    respectively.

    The model is also showin in [4]_.

    Examples
    --------
    Example 12.1 in [3]_:

    >>> from fluids.geometry import AirCooledExchanger
    >>> from scipy.constants import foot, inch
    >>> AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=56, tube_length=36*foot,
    ... tube_diameter=1*inch, fin_thickness=0.013*inch, fin_density=10/inch,
    ... angle=30, pitch_normal=2.5*inch, fin_height=0.625*inch, corbels=True)

    >>> h_Ganguli_VDI(m=130.70315, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin,
    ... A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter,
    ... fin_diameter=AC.fin_diameter, bare_length=AC.bare_length,
    ... fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows,
    ... pitch_parallel=AC.pitch_parallel, pitch_normal=AC.pitch_normal,
    ... rho=1.2013848, Cp=1009.0188, mu=1.9304793e-05, k=0.027864828, k_fin=238)
    969.285081857

    References
    ----------
    .. [1] Ganguli, A., S. S. Tung, and J. Taborek. "Parametric Study of
       Air-Cooled Heat Exchanger Finned Tube Geometry." In AIChE Symposium
       Series, 81:122-28, 1985.
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    .. [3] Serth, Robert W., and Thomas Lestina. Process Heat Transfer:
       Principles, Applications and Rules of Thumb. Academic Press, 2014.
    .. [4] Kroger, Detlev. Air-Cooled Heat Exchangers and Cooling Towers:
       Thermal-Flow Performance Evaluation and Design, Vol. 1. Tulsa, Okl:
       PennWell Corp., 2004.
    '''
    V_max = m/(A_min*rho)

    Re = Reynolds(V=V_max, D=tube_diameter, rho=rho, mu=mu)
    Pr = Prandtl(Cp=Cp, mu=mu, k=k)

    if abs(1 - pitch_normal/pitch_parallel) < 0.05: # in-line, with a tolerance of 0.05 proximity
        if tube_rows < 4:
            coeff = 0.2
        else:
            coeff = 0.22
    else: # staggered
        if tube_rows == 1:
            coeff = 0.2
        elif tube_rows == 2:
            coeff = 0.33
        elif tube_rows == 3:
            coeff = 0.36
        else:
            coeff = 0.38

    # VDI example shows the ratio is of the total area, to the original bare tube area
    # Serth example would match Nu = 47.22 except for lazy rounding
    Nu = coeff*Re**0.6*Pr**(1/3.)*(A_increase)**-0.15
    h = k/tube_diameter*Nu
    efficiency = fin_efficiency_Kern_Kraus(Do=tube_diameter, D_fin=fin_diameter,
                                           t_fin=fin_thickness, k_fin=k_fin, h=h)
    h_total_area_basis = (efficiency*A_fin + A_tube_showing)/A*h
    h_bare_tube_basis = h_total_area_basis*A_increase
    return h_bare_tube_basis


def dP_ESDU_high_fin(m, A_min, A_increase, flow_area_contraction_ratio,
                     tube_diameter, pitch_parallel, pitch_normal, tube_rows,
                     rho, mu):
    r'''Calculates the air-side pressure drop for a high-finned tube bank
    according to the ESDU [1]_ method, as described in [2]_. This includes the
    effects of friction of the fin, and acceleration.

    .. math::
        \Delta P = (K_{acc} + n_{rows} K_{f}) \frac{1}{2}\rho v_{max}^2

    .. math::
        K_{f} = 4.567 Re_D^{-0.242} \left(\frac{A}{A_{tube,only}}
        \right)^{0.504} \left(\frac{p_1}{D_o}\right)^{-0.376}
        \left(\frac{p_2}{D_{o}}\right)^{-0.546}

    .. math::
        K_{acc} = 1 + \text{(flow area contraction ratio)}^2

    Parameters
    ----------
    m : float
        Mass flow rate of air across the tube bank, [kg/s]
    A_min : float
        Minimum air flow area, [m^2]
    A_increase : float
        Ratio of actual surface area to bare tube surface area
        :math:`A_{increase} = \frac{A_{tube}}{A_{bare, total/tube}}`, [-]
    flow_area_contraction_ratio : float
        Ratio of `A_min` to `A_face`, [-]
    tube_diameter : float
        Diameter of the bare tube, [m]
    pitch_parallel : float
        Distance between tube center along a line parallel to the flow;
        has been called `longitudinal` pitch, `pp`, `s2`, `SL`, and `p2`, [m]
    pitch_normal : float
        Distance between tube centers in a line 90° to the line of flow;
        has been called the `transverse` pitch, `pn`, `s1`, `ST`, and `p1`, [m]
    tube_rows : int
        Number of tube rows per bundle, [-]
    rho : float
        Average (bulk) density of air across the tube bank, [kg/m^3]
    mu : float
        Average (bulk) viscosity of air across the tube bank, [Pa*s]

    Returns
    -------
    dP : float
        Overall pressure drop across the finned tube bank, [Pa]

    Notes
    -----
    The data used by the ESDU covered:
        * fin density 4 to 11/inch
        * tube outer diameters 3/8 to 2 inches
        * fin heights 1/3 to 5/8 inches
        * fin tip to fin root diameters 1.2 to 2.4
        * Reynolds numbers 5000 to 50000

    [1]_ claims 72% of experimental points were within 10% of the results of
    the correlation.

    The Reynolds number used in this equation is that based on `V_max`,
    calculated using the minimum flow area.

    Examples
    --------
    >>> from fluids.geometry import AirCooledExchanger
    >>> AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=8, tube_length=0.5,
    ... tube_diameter=0.0164, fin_thickness=0.001, fin_density=1/0.003,
    ... pitch_normal=0.0313, pitch_parallel=0.0271, fin_height=0.0041, corbels=True)

    >>> dP_ESDU_high_fin(m=0.914, A_min=AC.A_min, A_increase=AC.A_increase, flow_area_contraction_ratio=AC.flow_area_contraction_ratio, tube_diameter=AC.tube_diameter, pitch_parallel=AC.pitch_parallel, pitch_normal=AC.pitch_normal, tube_rows=AC.tube_rows, rho=1.217,  mu=0.000018)
    485.630768779

    References
    ----------
    .. [1] "High-Fin Staggered Tube Banks: Heat Transfer and Pressure Drop for
       Turbulent Single Phase Gas Flow." ESDU (October 1, 1986).
    .. [2] Hewitt, G. L. Shires T. Reg Bott G. F., George L. Shires, and
       T. R. Bott. Process Heat Transfer. 1E. Boca Raton: CRC Press, 1994.
    '''
    Vmax = m/(A_min*rho)
    Re = Vmax*tube_diameter*rho/mu
    Kf = (4.567*Re**-0.242*(A_increase)**0.504
          *(pitch_normal/tube_diameter)**-0.376
          *(pitch_parallel/tube_diameter)**-0.546)
    Ka = 1.0 + flow_area_contraction_ratio*flow_area_contraction_ratio
    dP = (Ka + tube_rows*Kf)*0.5*rho*Vmax*Vmax
    return dP


def dP_ESDU_low_fin(m, A_min, A_increase, flow_area_contraction_ratio,
                    tube_diameter, fin_height, bare_length, pitch_parallel,
                    pitch_normal, tube_rows, rho, mu):
    r'''Calculates the air-side pressure drop for a low-finned tube bank
    according to the ESDU [1]_ method, as described in [2]_. This includes the
    effects of friction of the fin, and acceleration.

    .. math::
        \Delta P = (K_{acc} + n_{rows} K_{f}) \frac{1}{2}\rho v_{max}^2

    .. math::
        K_{f} = 4.71 Re_D^{-0.286} \left(\frac{\text{fin height}}
        {\text{bare length}}\right)^{0.51}
        \left(\frac{p_1 - D_o}{p_2 - D_o}\right)^{0.536}
        \left(\frac{D_o}{p_1 - D_o}\right)^{0.36}

    .. math::
        K_{acc} = 1 + \text{(flow area contraction ratio)}^2

    Parameters
    ----------
    m : float
        Mass flow rate of air across the tube bank, [kg/s]
    A_min : float
        Minimum air flow area, [m^2]
    A_increase : float
        Ratio of actual surface area to bare tube surface area
        :math:`A_{increase} = \frac{A_{tube}}{A_{bare, total/tube}}`, [-]
    flow_area_contraction_ratio : float
        Ratio of `A_min` to `A_face`, [-]
    tube_diameter : float
        Diameter of the bare tube, [m]
    fin_height : float
        Height above bare tube of the tube fins, [m]
    bare_length : float
        Length of bare tube between two fins
        :math:`\text{bare length} = \text{fin interval} - t_{fin}`, [m]
    pitch_parallel : float
        Distance between tube center along a line parallel to the flow;
        has been called `longitudinal` pitch, `pp`, `s2`, `SL`, and `p2`, [m]
    pitch_normal : float
        Distance between tube centers in a line 90° to the line of flow;
        has been called the `transverse` pitch, `pn`, `s1`, `ST`, and `p1`, [m]
    tube_rows : int
        Number of tube rows per bundle, [-]
    rho : float
        Average (bulk) density of air across the tube bank, [kg/m^3]
    mu : float
        Average (bulk) viscosity of air across the tube bank, [Pa*s]

    Returns
    -------
    dP : float
        Overall pressure drop across the finned tube bank, [Pa]

    Notes
    -----
    Low fins are fins which were formed on the tube outside wall, normally
    by the cold rolling process. The data used by the ESDU covered:

    * fin density 11 to 32/inch
    * tube outer diameters 0.5 to 1.25 inches
    * fin heights 0.03 to 0.1 inches
    * Reynolds numbers 1000 to 80000

    [1]_ compared this correlation with 81 results and obtained a standard
    deviation of 7.7%.

    The Reynolds number used in this equation is that based on `V_max`,
    calculated using the minimum flow area.

    Examples
    --------
    >>> from fluids.geometry import AirCooledExchanger
    >>> AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=8, tube_length=0.5,
    ... tube_diameter=0.0164, fin_thickness=0.001, fin_density=1/0.003,
    ... pitch_normal=0.0313, pitch_parallel=0.0271, fin_height=0.0041, corbels=True)

    >>> dP_ESDU_low_fin(m=0.914, A_min=AC.A_min, A_increase=AC.A_increase,
    ... flow_area_contraction_ratio=AC.flow_area_contraction_ratio,
    ... tube_diameter=AC.tube_diameter, fin_height=AC.fin_height,
    ... bare_length=AC.bare_length, pitch_parallel=AC.pitch_parallel,
    ... pitch_normal=AC.pitch_normal, tube_rows=AC.tube_rows, rho=1.217,
    ... mu=0.000018)
    464.5433141865

    References
    ----------
    .. [1] "High-Fin Staggered Tube Banks: Heat Transfer and Pressure Drop for
       Turbulent Single Phase Gas Flow." ESDU (October 1, 1986).
    .. [2] Hewitt, G. L. Shires T. Reg Bott G. F., George L. Shires, and
       T. R. Bott. Process Heat Transfer. 1E. Boca Raton: CRC Press, 1994.
    '''
    Vmax = m/(A_min*rho)
    Re = Vmax*tube_diameter*rho/mu
    Kf = (4.72*Re**-0.286*(fin_height/bare_length)**0.51
          *((pitch_normal-tube_diameter)/(pitch_parallel-tube_diameter))**0.536
          *(tube_diameter/(pitch_normal-tube_diameter))**0.36)
    Ka = 1.0 + flow_area_contraction_ratio*flow_area_contraction_ratio
    dP = (Ka + tube_rows*Kf)*0.5*rho*Vmax*Vmax
    return dP


"""Three more correlations -

Heat Transfer and Pressure Drop Characteristics of Dry Tower Extended Surfaces: Data Analysis and Correlation. Pacific Northwest Laboratory, 1976.
* said to be in common use in http://www.thermopedia.com/content/551/

Cao, Eduardo. Heat Transfer in Process Engineering. McGraw Hill Professional, 2009.

Kroger - Mirković

"""
