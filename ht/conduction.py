# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from math import log, pi, acosh, cosh
from fluids.constants import inch, foot, hour, Btu, degree_Fahrenheit
try:
    from pprint import pprint
except:
    pass

__all__ = ['R_to_k', 'k_to_R', 'k_to_thermal_resistivity',
'thermal_resistivity_to_k', 'R_value_to_k', 'k_to_R_value', 'R_cylinder',
'S_isothermal_sphere_to_plane', 'S_isothermal_pipe_to_plane',
'S_isothermal_pipe_normal_to_plane',
'S_isothermal_pipe_to_isothermal_pipe', 'S_isothermal_pipe_to_two_planes',
'S_isothermal_pipe_eccentric_to_isothermal_pipe',
'cylindrical_heat_transfer']


def R_to_k(R, t, A=1.):
    r'''Returns the thermal conductivity of a substance given its thickness
    and thermal resistance.

    .. math::
        k = \frac{t}{RA}

    Parameters
    ----------
    R : float
        Thermal resistance of a substance, (K/W) if A is 1 m^2, otherwise must
        be [m^2*K/W]
    t : float
        Thickness of the substance used in the measurement of R, [m]
    A : float, optional
        Area; normally 1, [m^2]

    Returns
    -------
    k : float
        Thermal conductivity of a substance [W/m/K]

    Examples
    --------
    >>> R_to_k(R=0.05, t=0.025)
    0.5

    Notes
    -----
    When solving problems of changing areas, this value may be calculated with
    an area other than 1 m^2. Values in tables reported as properties of
    materials are often divided by area already; the conversion holds if A is 1.

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    return t/(A*R)


def k_to_R(k, t, A=1.):
    r'''Returns the thermal resistance of a substance given its thickness
    and thermal conductivity.

    .. math::
        R = \frac{t}{kA}

    Parameters
    ----------
    k : float
        Thermal conductivity of a substance [W/m/K]
    t : float
        Thickness of the substance for a given value of R, [m]
    A : float, optional
        Area; normally 1, [m^2]

    Returns
    -------
    R : float
        Thermal resistance of a substance [K/W]

    Examples
    --------
    >>> k_to_R(k=0.5, t=0.025)
    0.05

    Notes
    -----
    When solving problems of changing areas, this value may be calculated with
    an area other than 1 m^2. Values in tables reported as properties of
    materials are often divided by area already; the conversion holds if A is 1.

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    return t/(k*A)


def k_to_thermal_resistivity(k):
    r'''Returns the thermal resistivity of a substance given its thermal
    conductivity.

    .. math::
        r = \frac{1}{k}

    Parameters
    ----------
    k : float
        Thermal conductivity of a substance [W/m/K]

    Returns
    -------
    r : float
        Thermal resistivity of a substance [m*K/W]

    Examples
    --------
    >>> k_to_thermal_resistivity(0.25)
    4.0

    Notes
    -----
    Do not confuse this with thermal resistance! Often not introduced in heat
    transfer textbooks to avoid further confusion. Used almost exclusively
    as a description of solids. Thermal resistivity has different units than
    R-value, but is of the same dimensionality.

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    return 1./k


def thermal_resistivity_to_k(r):
    r'''Returns the thermal resistivity of a substance given its thermal
    conductivity.

    .. math::
        k = \frac{1}{r}

    Parameters
    ----------
    r : float
        Thermal resistivity of a substance [m*K/W]

    Returns
    -------
    k : float
        Thermal conductivity of a substance [W/m/K]

    Examples
    --------
    >>> thermal_resistivity_to_k(4)
    0.25

    Notes
    -----
    Do not confuse this with thermal resistance! Often not introduced in heat
    as a description of solids. Thermal resistivity has different units than
    R-value, but is of the same dimensionality.

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    return 1./r


def R_value_to_k(R_value, SI=True):
    r'''Returns the thermal conductivity of a substance given its R-value,
    which can be in either SI units of m^2 K/(W*inch) or the Imperial units
    of ft^2 deg F*h/(BTU*inch).

    Parameters
    ----------
    R_value : float
        R-value of a substance [m^2 K/(W*inch) or ft^2 deg F*h/(BTU*inch)]
    SI : bool, optional
        Whether to use the SI conversion or not

    Returns
    -------
    k : float
        Thermal conductivity of a substance [W/m/K]

    Notes
    -----
    If given input is SI, it is divided by 0.0254 (multiplied by 39.37) and
    then inversed. Otherwise, it is multiplied by 6.93347 and then inversed.

    Examples
    --------
    >>> R_value_to_k(0.12), R_value_to_k(0.71, SI=False)
    (0.2116666666666667, 0.20313787163983468)

    >>> R_value_to_k(1., SI=False)/R_value_to_k(1.)
    5.678263341113488

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    if SI:
        r = R_value/inch
    else:
        r = R_value*foot**2*degree_Fahrenheit*hour/Btu/inch
    return thermal_resistivity_to_k(r)


def k_to_R_value(k, SI=True):
    r'''Returns the R-value of a substance given its thermal conductivity,
    Will return R-value in SI units unless SI is false. SI units are
    m^2 K/(W*inch); Imperial units of R-value are ft^2 deg F*h/(BTU*inch).

    Parameters
    ----------
    k : float
        Thermal conductivity of a substance [W/m/K]
    SI : bool, optional
        Whether to use the SI conversion or not

    Returns
    -------
    R_value : float
        R-value of a substance [m^2 K/(W*inch) or ft^2 deg F*h/(BTU*inch)]

    Notes
    -----
    Provides the reverse conversion of R_value_to_k.

    Examples
    --------
    >>> k_to_R_value(R_value_to_k(0.12)), k_to_R_value(R_value_to_k(0.71, SI=False), SI=False)
    (0.11999999999999998, 0.7099999999999999)

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    r = k_to_thermal_resistivity(k)
    if SI:
        return r*inch
    else:
        return r/(foot**2*degree_Fahrenheit*hour/Btu/inch)


def R_cylinder(Di, Do, k, L):
    r'''Returns the thermal resistance `R` of a cylinder of constant thermal
    conductivity `k`, of inner and outer diameter `Di` and `Do`, and with a
    length `L`.

    .. math::
        (hA)_{\text{cylinder}}=\frac{k}{\ln(D_o/D_i)} \cdot 2\pi L\\
        R_{\text{cylinder}}=\frac{1}{(hA)_{\text{cylinder}}}=
        \frac{\ln(D_o/D_i)}{2\pi Lk}

    Parameters
    ----------
    Di : float
        Inner diameter of the cylinder, [m]
    Do : float
        Outer diameter of the cylinder, [m]
    k : float
        Thermal conductivity of the cylinder, [W/m/K]
    L : float
        Length of the cylinder, [m]

    Returns
    -------
    R : float
        Thermal resistance [K/W]

    Examples
    --------
    >>> R_cylinder(0.9, 1., 20, 10)
    8.38432343682705e-05

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    hA = k*2*pi*L/log(Do/Di)
    return 1./hA

### Shape Factors

def S_isothermal_sphere_to_plane(D, Z):
    r'''Returns the Shape factor `S` of a sphere of constant temperature
    and of outer diameter `D` which is `Z` distance from an infinite plane.

    .. math::
        S = \frac{2\pi D}{1 - \frac{D}{4Z}}

    Parameters
    ----------
    D : float
        Diameter of the sphere, [m]
    Z : float
        Distance from the middle of the sphere to the infinite plane, [m]

    Returns
    -------
    S : float
        Shape factor [m]

    Examples
    --------
    >>> S_isothermal_sphere_to_plane(1, 100)
    6.298932638776527

    Notes
    -----
    No restrictions on the use of this equation.

    .. math::
        Q = Sk(T_1 - T_2) \\ R_{\text{shape}}=\frac{1}{Sk}

    References
    ----------
    .. [1] Kreith, Frank, Raj Manglik, and Mark Bohn. Principles of Heat
       Transfer. Cengage, 2010.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    return 2*pi*D/(1. - D/(4.*Z))


def S_isothermal_pipe_to_plane(D, Z, L=1):
    r'''Returns the Shape factor `S` of a pipe of constant outer temperature
    and of outer diameter `D` which is `Z` distance from an infinite plane.
    Length `L` must be provided, but can be set to 1 to obtain a dimensionless
    shape factor used in some sources.

    .. math::
        S = \frac{2\pi L}{\cosh^{-1}(2z/D)}

    Parameters
    ----------
    D : float
        Diameter of the pipe, [m]
    Z : float
        Distance from the middle of the pipe to the infinite plane, [m]
    L : float, optional
        Length of the pipe, [m]

    Returns
    -------
    S : float
        Shape factor [m]

    Examples
    --------
    >>> S_isothermal_pipe_to_plane(1, 100, 3)
    3.146071454894645

    Notes
    -----
    L should be much larger than D.

    .. math::
        Q = Sk(T_1 - T_2) \\ R_{\text{shape}}=\frac{1}{Sk}

    References
    ----------
    .. [1] Kreith, Frank, Raj Manglik, and Mark Bohn. Principles of Heat
       Transfer. Cengage, 2010.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    return 2.*pi*L/acosh(2.*Z/D)


def S_isothermal_pipe_normal_to_plane(D, L):
    r'''Returns the Shape factor `S` of a pipe of constant outer temperature
    and of outer diameter `D` which extends into an infinite medium below an
    an infinite plane.

    .. math::
        S = \frac{2\pi L}{\ln(4L/D)}

    Parameters
    ----------
    D : float
        Diameter of the pipe, [m]
    L : float
        Length of the pipe, [m]

    Returns
    -------
    S : float
        Shape factor [m]

    Examples
    --------
    >>> S_isothermal_pipe_normal_to_plane(1, 100)
    104.86893910124888

    Notes
    -----
    L should be much larger than D.

    .. math::
        Q = Sk(T_1 - T_2) \\ R_{\text{shape}}=\frac{1}{Sk}

    References
    ----------
    .. [1] Kreith, Frank, Raj Manglik, and Mark Bohn. Principles of Heat
       Transfer. Cengage, 2010.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    return 2.*pi*L/log(4.*L/D)


def S_isothermal_pipe_to_isothermal_pipe(D1, D2, W, L=1.):
    r'''Returns the Shape factor `S` of a pipe of constant outer temperature
    and of outer diameter `D1` which is `w` distance from another infinite
    pipe of outer diameter`D2`. Length `L` must be provided, but can be set to
    1 to obtain a dimensionless shape factor used in some sources.

    .. math::
        S = \frac{2\pi L}{\cosh^{-1}\left(\frac{4w^2-D_1^2-D_2^2}{2D_1D_2}\right)}

    Parameters
    ----------
    D1 : float
        Diameter of one pipe, [m]
    D2 : float
        Diameter of the other pipe, [m]
    W : float
        Distance from the middle of one pipe to the middle of the other, [m]
    L : float, optional
        Length of the pipe, [m]

    Returns
    -------
    S : float
        Shape factor [m]

    Examples
    --------
    >>> S_isothermal_pipe_to_isothermal_pipe(.1, .2, 1, 1)
    1.188711034982268

    Notes
    -----
    L should be much larger than both diameters. L should be larger than W.

    .. math::
        Q = Sk(T_1 - T_2) \\ R_{\text{shape}}=\frac{1}{Sk}

    References
    ----------
    .. [1] Kreith, Frank, Raj Manglik, and Mark Bohn. Principles of Heat
       Transfer. Cengage, 2010.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    return 2.*pi*L/acosh((4*W**2 - D1**2 - D2**2)/(2.*D1*D2))


def S_isothermal_pipe_to_two_planes(D, Z, L=1.):
    r'''Returns the Shape factor `S` of a pipe of constant outer temperature
    and of outer diameter `D` which is `Z` distance from two infinite
    isothermal planes of equal temperatures, parallel to each other and
    enclosing the pipe. Length `L` must be provided, but can be set to
    1 to obtain a dimensionless shape factor used in some sources.

    .. math::
        S = \frac{2\pi L}{\ln\frac{8z}{\pi D}}

    Parameters
    ----------
    D : float
        Diameter of the pipe, [m]
    Z : float
        Distance from the middle of the pipe to either of the planes, [m]
    L : float, optional
        Length of the pipe, [m]

    Returns
    -------
    S : float
        Shape factor [m]

    Examples
    --------
    >>> S_isothermal_pipe_to_two_planes(.1, 5, 1)
    1.2963749299921428

    Notes
    -----
    L should be much larger than both diameters. L should be larger than W.

    .. math::
        Q = Sk(T_1 - T_2) \\ R_{\text{shape}}=\frac{1}{Sk}

    References
    ----------
    .. [1] Shape Factors for Heat Conduction Through Bodies with Isothermal or
       Convective Boundary Conditions, J. E. Sunderland, K. R. Johnson, ASHRAE
       Transactions, Vol. 70, 1964.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    return 2.*pi*L/log(8.*Z/(pi*D))


def S_isothermal_pipe_eccentric_to_isothermal_pipe(D1, D2, Z, L=1.):
    r'''Returns the Shape factor `S` of a pipe of constant outer temperature
    and of outer diameter `D1` which is `Z` distance from the center of another
    pipe of outer diameter`D2`. Length `L` must be provided, but can be set to
    1 to obtain a dimensionless shape factor used in some sources.

    .. math::
        S = \frac{2\pi L}{\cosh^{-1}
        \left(\frac{D_2^2 + D_1^2 - 4Z^2}{2D_1D_2}\right)}

    Parameters
    ----------
    D1 : float
        Diameter of inner pipe, [m]
    D2 : float
        Diameter of outer pipe, [m]
    Z : float
        Distance from the middle of inner pipe to the center of the other, [m]
    L : float, optional
        Length of the pipe, [m]

    Returns
    -------
    S : float
        Shape factor [m]

    Examples
    --------
    >>> S_isothermal_pipe_eccentric_to_isothermal_pipe(.1, .4, .05, 10)
    47.709841915608976

    Notes
    -----
    L should be much larger than both diameters. D2 should be larger than D1.

    .. math::
        Q = Sk(T_1 - T_2) \\ R_{\text{shape}}=\frac{1}{Sk}

    References
    ----------
    .. [1] Kreith, Frank, Raj Manglik, and Mark Bohn. Principles of Heat
       Transfer. Cengage, 2010.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    return 2.*pi*L/acosh((D2**2 + D1**2 - 4.*Z**2)/(2.*D1*D2))


# Specific heat transfer problems of conduction


def cylindrical_heat_transfer(Ti, To, hi, ho, Di, ts, ks):
    r'''Calculation for the heat transfer through a cylindrical wall,
    as occurs in pipes and cylindrical vessels. This is the core method
    which calculates the temperatures of each layer - and allows an outer
    layer to iterate on temperature or duty to meet a fixed specification,
    or include things like temperature dependent thermal conductivities
    or radiation.

    Parameters
    ----------
    Ti : float
        Temperature of the inside of the cylinder, [K]
    To : float
        External temperature outside the cylinder, away from the cylinder
        wall, [K]
    hi : float
        Inside heat transfer coefficient, [W/m^2/K]
    ho : float
        Outside heat transfer coefficient, [W/m^2/K]
    Di : float
        Inside diameter of cylinder, [m]
    ts : list[float]
        List of thicknesses of each layer of the cylinder, [m]
    ks : list[float]
        List of thermal conductivities of each layer of the cylinder, [w/m/K]

    Returns
    -------
    results : dict
        * Q : Heat exchanged through the cylinder (per meter of length), [W/m]
        * Rs : Thermal resistances of each of the layers, [m*K/W]
        * Ts : Temperatures of the outside of each of the layers, [K]
        * UA : Heat transfer coefficient times area (on a per-meter of
               cylinder) basis, [W/K/m]
        * U_inner : Heat transfer coefficient with respect to the inside
                    diameter, [W/K]
        * U_outer : Heat transfer coefficient with respect to the exterior
                    diameter, [W/K]
        * q : Specific heat exchanged (per square meter) through the cylinder
              (per meter of length), [W/m^3]

    Examples
    --------

    >>> pprint(cylindrical_heat_transfer(Ti=453.15, To=301.15, hi=1e12, ho=22.697193, Di=0.0779272, ts=[0.0054864, .05], ks=[56.045, 0.0598535265]))
    {'Q': 73.12000884069367,
     'Rs': [0.00022201030738405449, 1.189361782070256],
     'Ts': [453.15, 453.1226455779877, 306.578530147744],
     'UA': 0.48105268974140575,
     'U_inner': 1.9649599487726137,
     'U_outer': 0.8106078714663484,
     'q': 123.21239646288495}
    '''
    length = 1.0 # basis
    # Note - fouling is just another layer, should be converted to a thickness/thermal conductivity

    external_diameter = Di + 2.0*sum(ts)
    A_external = pi*external_diameter*length
    A_internal = pi*Di*length

    Rs = []
    Do_running = Di
    R_layers = 0.0
    for i in range(len(ts)):
        Do_running, Di_running = 2.0*ts[i]+Do_running, Do_running
        Ri = 0.5*external_diameter*log(Do_running/Di_running)/ks[i]
        R_layers += Ri
        Rs.append(Ri)

    D_ratio = external_diameter/Di
    inv_term = D_ratio/hi + R_layers + 1.0/ho

    U_external = 1.0/inv_term
    UA = A_external*U_external
    dT = Ti - To

    Q = UA*dT
    q = Q/A_external

    # Compute the temperature profile
    Ts = [Ti]
    for Ri in Rs:
        Ts.append(Ts[-1] - q*Ri)

    # Convert heat transfer coefficient area basis = U_i*A_i = U_o*A_o, divide
    ans = {'Q': Q, 'q': q, 'UA': UA, 'U_outer': U_external, 'U_inner': UA/A_internal, 'Ts': Ts,
          'Rs': Rs}
    return ans
