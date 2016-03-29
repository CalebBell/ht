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
from math import log, log10, exp, tanh

__all__ = ['laminar_T_const', 'laminar_Q_const',
'laminar_entry_thermal_Hausen', 'laminar_entry_Seider_Tate',
'laminar_entry_Baehr_Stephan', 'turbulent_Dittus_Boelter',
'turbulent_Sieder_Tate', 'turbulent_entry_Hausen', 'turbulent_Colburn',
'turbulent_Drexel_McAdams', 'turbulent_von_Karman', 'turbulent_Prandtl',
'turbulent_Friend_Metzner', 'turbulent_Petukhov_Kirillov_Popov',
'turbulent_Webb', 'turbulent_Sandall', 'turbulent_Gnielinski',
'turbulent_Gnielinski_smooth_1', 'turbulent_Gnielinski_smooth_2',
'turbulent_Churchill_Zajic', 'turbulent_ESDU', 'turbulent_Martinelli',
'turbulent_Nunner', 'turbulent_Dipprey_Sabersky', 'turbulent_Gowen_Smith',
'turbulent_Kawase_Ulbrecht', 'turbulent_Kawase_De', 'turbulent_Bhatti_Shah',
'Nu_conv_internal']

### Laminar

def laminar_T_const():
    r'''Returns internal convection Nusselt number for laminar flows
    in pipe according to [1]_, [2]_ and [3]_. Wall temperature is assumed
    constant.
    This is entirely theoretically derived and reproduced experimentally.

    .. math::
        Nu = 3.66

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    This applies only for fully thermally and hydraulically developed and flows.

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry’s Chemical Engineers’ Handbook,
       Eighth Edition. New York: McGraw-Hill Education, 2007.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ: Wiley, 2011.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    '''
    Nu = 3.66
    return Nu


def laminar_Q_const():
    r'''Returns internal convection Nusselt number for laminar flows
    in pipe according to [1]_, [2]_, and [3]_. Heat flux is assumed constant.
    This is entirely theoretically derived and reproduced experimentally.

    .. math::
        Nu = 4.354

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    This applies only for fully thermally and hydraulically developed and flows.
    Many sources round to 4.36, but [3]_ does not.

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry’s Chemical Engineers’ Handbook,
       Eighth Edition. New York: McGraw-Hill Education, 2007.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ: Wiley, 2011.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
        Berlin ; New York: Springer, 2010.
    '''
    Nu = 48/11.
    return Nu
### Laminar - entry region

def laminar_entry_thermal_Hausen(Re=None, Pr=None, L=None, Di=None):
    r'''Calculates average internal convection Nusselt number for laminar flows
    in pipe during the thermal entry region according to [1]_ as shown in
    [2]_ and cited by [3]_.

    .. math::
        Nu_D=3.66+\frac{0.0668\frac{D}{L}Re_{D}Pr}{1+0.04{(\frac{D}{L}
        Re_{D}Pr)}^{2/3}}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    L : float
        Length of pipe [m]
    Di : float
        Diameter of pipe [m]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    If Pr >> 1, (5 is a common requirement) this equation also applies to flows
    with developing velocity profile.
    As L gets larger, this equation  becomes the constant-temperature Nusselt
    number.

    Examples
    --------
    >>> laminar_entry_thermal_Hausen(Re=100000, Pr=1.1, L=5, Di=.5)
    39.01352358988535

    References
    ----------
    .. [1] Hausen, H. Darstellung des Warmeuberganges in Rohren durch
       verallgeminerte Potenzbeziehungen, Z. Ver deutsch. Ing Beih.
       Verfahrenstech., 4, 91-98, 1943
    .. [2] W. M. Kays. 1953. Numerical Solutions for Laminar Flow Heat Transfer
       in Circular Tubes.
    .. [3] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E.
       Hoboken, NJ: Wiley, 2011.
    '''
    Gz = Di/L*Re*Pr
    Nu = 3.66 + (0.0668*Gz)/(1+0.04*(Gz)**(2/3.))
    return Nu
#print [laminar_entry_thermal_Hausen(Re=100000, Pr=1.1, L=5, Di=.5)]

def laminar_entry_Seider_Tate(Re=None, Pr=None, L=None, Di=None, mu=None,
                              mu_w=None):
    r'''Calculates average internal convection Nusselt number for laminar flows
    in pipe during the thermal entry region as developed in [1]_, also
    shown in [2]_.

    .. math::
        Nu_D=1.86\left(\frac{D}{L}Re_DPr\right)^{1/3}\left(\frac{\mu_b}
        {\mu_s}\right)^{0.14}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    L : float
        Length of pipe [m]
    Di : float
        Diameter of pipe [m]
    mu : float
        Viscosity of fluid, [Pa*S or consistent with mu_w]
    mu_w : float
        Viscosity of fluid at wall temperature, [Pa*S or consistent with mu]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Reynolds number should be less than 10000. This should be calculated
    using pipe diameter.
    Prandlt number should be no less than air and no more than liquid metals;
    0.7 < Pr <  16700
    Viscosities should be the bulk and surface properties; they are optional.
    Outside the boundaries, this equation is provides very false results.

    Examples
    --------
    >>> laminar_entry_Seider_Tate(Re=100000, Pr=1.1, L=5, Di=.5)
    41.366029684589265

    References
    ----------
    .. [1] Sieder, E. N., and G. E. Tate. "Heat Transfer and Pressure Drop of
       Liquids in Tubes." Industrial & Engineering Chemistry 28, no. 12
       (December 1, 1936): 1429-35. doi:10.1021/ie50324a027.
    .. [2] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    '''
    Nu = 1.86*(Di/L*Re*Pr)**(1/3.0)
    if mu_w and mu:
        Nu *= (mu/mu_w)**0.14
    return Nu

#print [laminar_entry_Seider_Tate(Re=100000, Pr=1.1, L=5, Di=.5, mu=1E-3, mu_w=1.2E-3)]

def laminar_entry_Baehr_Stephan(Re=None, Pr=None, L=None, Di=None):
    r'''Calculates average internal convection Nusselt number for laminar flows
    in pipe during the thermal and velocity entry region according to [1]_ as
    shown in [2]_.

    .. math::
        Nu_D=\frac{\frac{3.657}{\tanh[2.264 Gz_D^{-1/3}+1.7Gz_D^{-2/3}]}
        +0.0499Gz_D\tanh(Gz_D^{-1})}{\tanh(2.432Pr^{1/6}Gz_D^{-1/6})}

        Gz = \frac{D}{L}Re_D Pr

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    L : float
        Length of pipe [m]
    Di : float
        Diameter of pipe [m]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    As L gets larger, this equation becomes the constant-temperature Nusselt
    number.

    Examples
    --------
    >>> laminar_entry_Baehr_Stephan(Re=100000, Pr=1.1, L=5, Di=.5)
    72.65402046550976

    References
    ----------
    .. [1] Baehr, Hans Dieter, and Karl Stephan. Heat and Mass Transfer.
       Springer, 2013.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E.
       Hoboken, NJ: Wiley, 2011.
    '''
    Gz = Di/L*Re*Pr
    Nu = (3.657/tanh(2.264*Gz**(-1/3.)+ 1.7*Gz**(-2/3.0)) + 0.0499*Gz*tanh(1./Gz))\
        /tanh(2.432*Pr**(1/6.0)*Gz**(-1/6.0))
    return Nu


#print [laminar_entry_Baehr_Stephan(Re=100000, Pr=1.1, L=5, Di=.5)]




### Turbulent - Equations with more comlicated options
def turbulent_Dittus_Boelter(Re=None, Pr=None, heating=True, revised=True):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [1]_, and [2]_, a reprint of [3]_.

    .. math::
        Nu = m*Re_D^{4/5}Pr^n

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    heating : bool
        Indicates if the process is heating or cooling, optional
    revised : bool
        Indicates if revised coefficients should be used or not

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    The revised coefficient is m = 0.023.
    The original form of Dittus-Boelter has a linear coefficient of 0.0243
    for heating and 0.0265 for cooling. These are sometimes rounded to 0.024
    and 0.026 respectively.
    The default, heating, provides n = 0.4. Cooling makes n 0.3.

    0.6 ≤ Pr ≤  160
    Re_{D} ≥ 10000
    L/D ≥ 10

    Examples
    --------
    >>> turbulent_Dittus_Boelter(Re=1E5, Pr=1.2)
    247.40036409449127
    >>> turbulent_Dittus_Boelter(Re=1E5, Pr=1.2, heating=False)
    242.9305927410295

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Dittus, F. W., and L. M. K. Boelter. "Heat Transfer in Automobile
       Radiators of the Tubular Type." International Communications in Heat
       and Mass Transfer 12, no. 1 (January 1985): 3-22.
       doi:10.1016/0735-1933(85)90003-X
    .. [3] Dittus, F. W., and L. M. K. Boelter, University of California
       Publications in Engineering, Vol. 2, No. 13, pp. 443-461, October 17,
       1930.
    '''
    m = 0.023
    if heating:
        power = 0.4
    else:
        power = 0.3

    if heating and not revised:
        m = 0.0243
    elif not heating and not revised:
        m = 0.0265
    else:
        m = 0.023

    Nu = m*Re**0.8*Pr**power
    return Nu


def turbulent_Sieder_Tate(Re=None, Pr=None, mu=None, mu_w=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [1]_ and supposedly [2]_.

    .. math::
        Nu = 0.027Re^{4/5}Pr^{1/3}\left(\frac{\mu}{\mu_s}\right)^{0.14}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    mu : float
        Viscosity of fluid, [Pa*S or consistent with mu_w]
    mu_w : float
        Viscosity of fluid at wall temperature, [Pa*S or consistent with mu]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    A linear coefficient of 0.023 is often listed with this equation. The
    source of the discrepancy is not known. The equation is not present in the
    original paper, but is nevertheless the source usually cited for it.

    Examples
    --------
    >>> turbulent_Sieder_Tate(Re=1E5, Pr=1.2)
    286.9178136793052
    >>> turbulent_Sieder_Tate(Re=1E5, Pr=1.2, mu=0.01, mu_w=0.067)
    219.84016455766044

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Sieder, E. N., and G. E. Tate. "Heat Transfer and Pressure Drop of
       Liquids in Tubes." Industrial & Engineering Chemistry 28, no. 12
       (December 1, 1936): 1429-35. doi:10.1021/ie50324a027.
    '''
    Nu = 0.027*Re**0.8*Pr**(1/3.)
    if mu_w and mu:
        Nu *= (mu/mu_w)**0.14
    return Nu


def turbulent_entry_Hausen(Re=None, Pr=None, Di=None, x=None):
    r'''Calculates internal convection Nusselt number for the entry region
    of a turbulent flow in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = 0.037(Re^{0.75} - 180)Pr^{0.42}[1+(x/D)^{-2/3}]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    Di : float
        Inside diameter of pipe, [m]
    x : float
        Length inside of pipe for calculation, [m]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.7 < Pr ≤ 3  and 10^4 ≤ Re ≤ 5*10^6.

    Examples
    --------
    >>> turbulent_entry_Hausen(Re=1E5, Pr=1.2, Di=0.154, x=0.05)
    677.7228275901755

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] H. Hausen, "Neue Gleichungen fÜr die Wärmeübertragung bei freier
       oder erzwungener Stromung,"Allg. Warmetchn., (9): 75-79, 1959.
    '''
    Nu = 0.037*(Re**0.75 - 180)*Pr**0.42*(1 + (x/Di)**(-2/3.))
    return Nu


### Regular correlations, Re, Pr and fd only


def turbulent_Colburn(Re=None, Pr=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = 0.023Re^{0.8}Pr^{1/3}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5 < Pr < 3  and 10^4 < Re < 10^5.

    Examples
    --------
    >>> turbulent_Colburn(Re=1E5, Pr=1.2)
    244.41147091200068

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Colburn, Allan P. "A Method of Correlating Forced Convection
       Heat-Transfer Data and a Comparison with Fluid Friction." International
       Journal of Heat and Mass Transfer 7, no. 12 (December 1964): 1359-84.
       doi:10.1016/0017-9310(64)90125-5.
    '''
    Nu = 0.023*Re**0.8*Pr**(1/3.)
    return Nu


def turbulent_Drexel_McAdams(Re=None, Pr=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = 0.021Re^{0.8}Pr^{0.4}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
            Prandtl number, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is Pr ≤ 0.7 and 10^4 ≤ Re ≤ 5*10^6.

    Examples
    --------
    >>> turbulent_Drexel_McAdams(Re=1E5, Pr=0.6)
    171.19055301724387

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Drexel, Rober E., and William H. Mcadams. “Heat-Transfer
       Coefficients for Air Flowing in Round Tubes, in Rectangular Ducts, and
       around Finned Cylinders,” February 1, 1945.
       http://ntrs.nasa.gov/search.jsp?R=19930090924.
    '''
    Nu = 0.021*Re**0.8*Pr**(0.4)
    return Nu


def turbulent_von_Karman(Re=None, Pr=None, fd=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = \frac{(f/8)Re Pr}{1 + 5(f/8)^{0.5}\left[Pr-1+\ln\left(\frac{5Pr+1}
        {6}\right)\right]}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5 ≤ Pr ≤ 3  and 10^4 ≤ Re ≤ 10^5.

    Examples
    --------
    >>> turbulent_von_Karman(Re=1E5, Pr=1.2, fd=0.0185)
    255.7243541243272

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] T. von Karman, "The Analogy Between Fluid Friction and Heat
       Transfer," Trans. ASME, (61):705-710,1939.
    '''
    Nu = fd/8.*Re*Pr/(1 + 5*(fd/8.)**0.5*(Pr-1+log((5*Pr+1)/6.)))
    return Nu


def turbulent_Prandtl(Re=None, Pr=None, fd=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = \frac{(f/8)RePr}{1 + 8.7(f/8)^{0.5}(Pr-1)}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ 0.5 ≤ Pr ≤ 5 and 10^4 ≤ Re ≤ 5*10^6

    Examples
    --------
    >>> turbulent_Prandtl(Re=1E5, Pr=1.2, fd=0.0185)
    256.073339689557

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] L. Prandt, Fuhrrer durch die Stomungslehre, Vieweg, Braunschweig,
       p. 359, 1944.
    '''
    Nu = (fd/8.)*Re*Pr/(1+8.7*(fd/8.)**0.5*(Pr-1) )
    return Nu


def turbulent_Friend_Metzner(Re=None, Pr=None, fd=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = \frac{(f/8)RePr}{1.2 + 11.8(f/8)^{0.5}(Pr-1)Pr^{-1/3}}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ 50 < Pr ≤ 600  and 5*10^4 ≤ Re ≤ 5*10^6.
    The extreme limits on range should be considered!

    Examples
    --------
    >>> turbulent_Friend_Metzner(Re=1E5, Pr=100., fd=0.0185)
    1738.3356262055322

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Friend, W. L., and A. B. Metzner. “Turbulent Heat Transfer inside
       Tubes and the Analogy among Heat, Mass, and Momentum Transfer.” AIChE
       Journal 4, no. 4 (December 1, 1958): 393-402. doi:10.1002/aic.690040404.
    '''
    Nu = (fd/8.)*Re*Pr/(1.2 + 11.8*(fd/8.)**0.5*(Pr-1)*Pr**(-1/3.) )
    return Nu


def turbulent_Petukhov_Kirillov_Popov(Re=None, Pr=None, fd=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ and [3]_ as in [1]_.

    .. math::
        Nu = \frac{(f/8)RePr}{C+12.7(f/8)^{1/2}(Pr^{2/3}-1)}\\
        C = 1.07 + 900/Re - [0.63/(1+10Pr)]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5 < Pr ≤ 10^6  and 4000 ≤ Re ≤ 5*10^6

    Examples
    --------
    >>> turbulent_Petukhov_Kirillov_Popov(Re=1E5, Pr=1.2, fd=0.0185)
    250.11935088905105

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] B. S. Petukhov, and V. V. Kirillov, "The Problem of Heat Exchange
       in the Turbulent Flow of Liquids in Tubes," (Russian) Teploenergetika,
       (4): 63-68, 1958
    .. [3] B. S. Petukhov and V. N. Popov, "Theoretical Calculation of Heat
       Exchange in Turbulent Flow in Tubes of an Incompressible Fluidwith
       Variable Physical Properties," High Temp., (111): 69-83, 1963.
    '''
    C = 1.07 + 900./Re - (0.63/(1+10*Pr))
    Nu = (fd/8.)*Re*Pr/(C + 12.7*(fd/8.)**0.5*(Pr**(2/3.)-1))
    return Nu


def turbulent_Webb(Re=None, Pr=None, fd=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = \frac{(f/8)RePr}{1.07 + 9(f/8)^{0.5}(Pr-1)Pr^{1/4}}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5 < Pr ≤ 100  and 10^4 ≤ Re ≤ 5*10^6

    Examples
    --------
    >>> turbulent_Webb(Re=1E5, Pr=1.2, fd=0.0185)
    239.10130376815872

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Webb, Dr R. L. “A Critical Evaluation of Analytical Solutions and
       Reynolds Analogy Equations for Turbulent Heat and Mass Transfer in
       Smooth Tubes.” Wärme - Und Stoffübertragung 4, no. 4
       (December 1, 1971): 197–204. doi:10.1007/BF01002474.
    '''
    Nu = (fd/8.)*Re*Pr/(1.07 + 9*(fd/8.)**0.5*(Pr-1)*Pr**0.25)
    return Nu


def turbulent_Sandall(Re=None, Pr=None, fd=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = \frac{(f/8)RePr}{12.48Pr^{2/3} - 7.853Pr^{1/3} + 3.613\ln Pr + 5.8 + C}\\
        C = 2.78\ln((f/8)^{0.5} Re/45)

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5< Pr ≤ 2000  and 10^4 ≤ Re ≤ 5*10^6.

    Examples
    --------
    >>> turbulent_Sandall(Re=1E5, Pr=1.2, fd=0.0185)
    229.0514352970239

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Sandall, O. C., O. T. Hanna, and P. R. Mazet. “A New Theoretical
       Formula for Turbulent Heat and Mass Transfer with Gases or Liquids in
       Tube Flow.” The Canadian Journal of Chemical Engineering 58, no. 4
       (August 1, 1980): 443–47. doi:10.1002/cjce.5450580404.
    '''
    C = 2.78*log((fd/8.)**0.5*Re/45.)
    Nu = (fd/8.)**0.5*Re*Pr/(12.48*Pr**(2/3.) - 7.853*Pr**(1/3.) + 3.613*log(Pr) + 5.8 + C)
    return Nu


def turbulent_Gnielinski(Re=None, Pr=None, fd=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_. This is the most recent general
    equation, and is strongly recommended.

    .. math::
        Nu = \frac{(f/8)(Re-1000)Pr}{1+12.7(f/8)^{1/2}(Pr^{2/3}-1)}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5 < Pr ≤ 2000  and 2300 ≤ Re ≤ 5*10^6.

    Examples
    --------
    >>> turbulent_Gnielinski(Re=1E5, Pr=1.2, fd=0.0185)
    254.62682749359632

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Gnielinski, V. (1976). New Equation for Heat and Mass Transfer in
       Turbulent Pipe and Channel Flow, International Chemical Engineering,
       Vol. 16, pp. 359–368.
    '''
    Nu = (fd/8.)*(Re-1000)*Pr/(1 + 12.7*(fd/8.)**0.5*(Pr**(2/3.)-1))
    return Nu


def turbulent_Gnielinski_smooth_1(Re=None, Pr=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_. This is a simplified case assuming
    smooth pipe.

    .. math::
        Nu = 0.0214(Re^{0.8}-100)Pr^{0.4}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5 < Pr ≤ 1.5  and 10^4 ≤ Re ≤ 5*10^6.

    Examples
    --------
    >>> turbulent_Gnielinski_smooth_1(Re=1E5, Pr=1.2)
    227.88800494373442

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Gnielinski, V. (1976). New Equation for Heat and Mass Transfer in
       Turbulent Pipe and Channel Flow, International Chemical Engineering,
       Vol. 16, pp. 359–368.
    '''
    Nu = 0.0214*(Re**0.8-100)*Pr**0.4
    return Nu


def turbulent_Gnielinski_smooth_2(Re=None, Pr=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_. This is a simplified case assuming
    smooth pipe.

    .. math::
        Nu = 0.012(Re^{0.87}-280)Pr^{0.4}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 1.5 < Pr ≤ 500 and 3*10^3 ≤ Re ≤ 10^6.

    Examples
    --------
    >>> turbulent_Gnielinski_smooth_2(Re=1E5, Pr=7.)
    577.7692524513449

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Gnielinski, V. (1976). New Equation for Heat and Mass Transfer in
       Turbulent Pipe and Channel Flow, International Chemical Engineering,
       Vol. 16, pp. 359–368.
    '''
    Nu = 0.012*(Re**0.87 - 280)*Pr**0.4
    return Nu


def turbulent_Churchill_Zajic(Re=None, Pr=None, fd=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as developed in [1]_. Has yet to obtain
    popularity.

    .. math::
        Nu = \left\{\left(\frac{Pr_T}{Pr}\right)\frac{1}{Nu_{di}} +
        \left[1-\left(\frac{Pr_T}{Pr}\right)^{2/3}\right]\frac{1}{Nu_{D\infty}}
        \right\}^{-1}

        Nu_{di} = \frac{Re(f/8)}{1 + 145(8/f)^{-5/4}}

        Nu_{D\infty} = 0.07343Re\left(\frac{Pr}{Pr_T}\right)^{1/3}(f/8)^{0.5}

        Pr_T = 0.85 + \frac{0.015}{Pr}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    No restrictions on range. This is equation is developed with more
    theoretical work than others.

    Examples
    --------
    >>> turbulent_Churchill_Zajic(Re=1E5, Pr=1.2, fd=0.0185)
    260.5564907817961

    References
    ----------
    .. [1] Churchill, Stuart W., and Stefan C. Zajic. “Prediction of Fully
       Developed Turbulent Convection with Minimal Explicit Empiricism.”
       AIChE Journal 48, no. 5 (May 1, 2002): 927–40. doi:10.1002/aic.690480503.
    .. [2] Plawsky, Joel L. Transport Phenomena Fundamentals, Third Edition.
       CRC Press, 2014.
    '''
    Pr_T = 0.85 + 0.015/Pr
    Nu_di = Re*(fd/8.)/(1 + 145*(8./fd)**(-1.25))
    Nu_dinf = 0.07343*Re*(Pr/Pr_T)**(1/3.0)*(fd/8.)**0.5
    Nu = (Pr_T/Pr/Nu_di + (1 - (Pr_T/Pr)**(2/3.))/Nu_dinf)**-1
    return Nu


def turbulent_ESDU(Re=None, Pr=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to the ESDU as shown in [1]_.

    .. math::
        Nu = 0.0225Re^{0.795}Pr^{0.495}\exp(-0.0225\ln(Pr)^2)

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    4000 < Re < 1E6, 0.3 < Pr < 3000 and L/D > 60.
    This equation has not been checked. It was developed by a commercial group.
    This function is a small part of a much larger series of expressions
    accounting for many factors.

    Examples
    --------
    >>> turbulent_ESDU(Re=1E5, Pr=1.2)
    232.3017143430645

    References
    ----------
    .. [1] Hewitt, G. L. Shires, T. Reg Bott G. F., George L. Shires, and
       T. R. Bott. Process Heat Transfer. 1E. Boca Raton: CRC Press, 1994.
    '''
    Nu = 0.0225*Re**0.795*Pr**0.495*exp(-0.0225*log(Pr)**2)
    return Nu

### Correlations for 'rough' turbulent pipe

def turbulent_Martinelli(Re=None, Pr=None, fd=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_.

    .. math::
        Nu  = \frac{RePr(f/8)^{0.5}}{5[Pr + \ln(1+5Pr) + 0.5\ln(Re(f/8)^{0.5}/60)]}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    No range is given for this equation. Liquid metals are probably its only
    applicability.

    Examples
    --------
    >>> turbulent_Martinelli(Re=1E5, Pr=100., fd=0.0185)
    887.1710686396347

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Martinelli, R. C. (1947). "Heat transfer to molten metals".
       Trans. ASME, 69, 947-959.
    '''
    Nu = Re*Pr*(fd/8.)**0.5/5/(Pr + log(1+5*Pr) + 0.5*log(Re*(fd/8.)**0.5/60.))
    return Nu


def turbulent_Nunner(Re=None, Pr=None, fd=None, fd_smooth=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_.

    .. math::
        Nu = \frac{RePr(f/8)}{1 + 1.5Re^{-1/8}Pr^{-1/6}[Pr(f/f_s)-1]}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]
    fd_smooth : float
        Darcy friction factor of a smooth pipe [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Valid for Pr ≅ 0.7; bad results for Pr > 1.

    Examples
    --------
    >>> turbulent_Nunner(Re=1E5, Pr=0.7, fd=0.0185, fd_smooth=0.005)
    101.15841010919947

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] W. Nunner, "Warmeiibergang und Druckabfall in Rauhen Rohren,"
       VDI-Forschungsheft 445, ser. B,(22): 5-39, 1956
    '''
    Nu = Re*Pr*fd/8./(1 + 1.5*Re**-0.125*Pr**(-1/6.)*(Pr*fd/fd_smooth -1))
    return Nu


def turbulent_Dipprey_Sabersky(Re=None, Pr=None, fd=None, eD=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_.

    .. math::
        Nu = \frac{RePr(f/8)}{1 + (f/8)^{0.5}[5.19Re_\epsilon^{0.2} Pr^{0.44} - 8.48]}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    According to [1]_, the limits are:
    1.2 ≤ Pr ≤ 5.94 and 1.4*10^4 ≤ Re ≤ 5E5 and 0.0024 ≤ eD ≤ 0.049.

    Examples
    --------
    >>> turbulent_Dipprey_Sabersky(Re=1E5, Pr=1.2, fd=0.0185, eD=1E-3)
    288.33365198566656

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Dipprey, D. F., and R. H. Sabersky. “Heat and Momentum Transfer in
       Smooth and Rough Tubes at Various Prandtl Numbers.” International
       Journal of Heat and Mass Transfer 6, no. 5 (May 1963): 329–53.
       doi:10.1016/0017-9310(63)90097-8
    '''
    Re_e = Re*eD*(fd/8.)**0.5
    Nu = Re*Pr*fd/8./(1 + (fd/8.)**0.5*(5.19*Re_e**0.2*Pr**0.44 - 8.48))
    return Nu


def turbulent_Gowen_Smith(Re=None, Pr=None, fd=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_.

    .. math::
        Nu = \frac{Re Pr (f/8)^{0.5}} {4.5 + [0.155(Re(f/8)^{0.5})^{0.54}
        + (8/f)^{0.5}]Pr^{0.5}}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    0.7 ≤ Pr ≤ 14.3 and 10^4 ≤ Re ≤ 5E4 and 0.0021 ≤ eD ≤ 0.095

    Examples
    --------
    >>> turbulent_Gowen_Smith(Re=1E5, Pr=1.2, fd=0.0185)
    131.72530453824106

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Gowen, R. A., and J. W. Smith. “Turbulent Heat Transfer from Smooth
       and Rough Surfaces.” International Journal of Heat and Mass Transfer 11,
       no. 11 (November 1968): 1657–74. doi:10.1016/0017-9310(68)90046-X.
    '''
    Nu = Re*Pr*(fd/8.)**0.5/(4.5 + (0.155*(Re*(fd/8.)**0.5)**0.54 + (8./fd)**0.5)*Pr**0.5)
    return Nu


def turbulent_Kawase_Ulbrecht(Re=None, Pr=None, fd=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_.

    .. math::
        Nu = 0.0523RePr^{0.5}(f/4)^{0.5}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    No limits are provided.

    Examples
    --------
    >>> turbulent_Kawase_Ulbrecht(Re=1E5, Pr=1.2, fd=0.0185)
    389.6262247333975

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Kawase, Yoshinori, and Jaromir J. Ulbrecht. “Turbulent Heat and Mass
       Transfer in Dilute Polymer Solutions.” Chemical Engineering Science 37,
       no. 7 (1982): 1039–46. doi:10.1016/0009-2509(82)80134-6.
    '''
    Nu = 0.0523*Re*Pr**0.5*(fd/4.)**0.5
    return Nu


def turbulent_Kawase_De(Re=None, Pr=None, fd=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_.

    .. math::
        Nu = 0.0471 RePr^{0.5}(f/4)^{0.5}(1.11 + 0.44Pr^{-1/3} - 0.7Pr^{-1/6})

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    5.1 ≤ Pr ≤ 390 and 5000 ≤ Re ≤ 5E5 and 0.0024 ≤ eD ≤ 0.165.

    Examples
    --------
    >>> turbulent_Kawase_De(Re=1E5, Pr=1.2, fd=0.0185)
    296.5019733271324

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Kawase, Yoshinori, and Addie De. “Turbulent Heat and Mass Transfer
       in Newtonian and Dilute Polymer Solutions Flowing through Rough Pipes.”
       International Journal of Heat and Mass Transfer 27, no. 1
       (January 1984): 140–42. doi:10.1016/0017-9310(84)90246-1.
    '''
    Nu = 0.0471*Re*Pr**0.5*(fd/4.)**0.5*(1.11 + 0.44*Pr**(-1/3.) - 0.7*Pr**(-1/6.))
    return Nu


def turbulent_Bhatti_Shah(Re=None, Pr=None, fd=None, eD=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_. The most widely used rough
    pipe turbulent correlation.

    .. math::
        Nu_D = \frac{(f/8)Re_DPr}{1+\sqrt{f/8}(4.5Re_{\epsilon}^{0.2}Pr^{0.5}-8.48)}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    According to [1]_, the limits are:
    0.5 ≤ Pr ≤  10
    0.002 ≤ ε/D ≤  0.05
    10,000 ≤ Re_{D}
    Another correlation is listed in this equation, with a wider variety
    of validity.

    Examples
    --------
    >>> turbulent_Bhatti_Shah(Re=1E5, Pr=1.2, fd=0.0185, eD=1E-3)
    302.7037617414273

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] M. S. Bhatti and R. K. Shah. Turbulent and transition flow
       convective heat transfer in ducts. In S. Kakaç, R. K. Shah, and W.
       Aung, editors, Handbook of Single-Phase Convective Heat Transfer,
       chapter 4. Wiley-Interscience, New York, 1987.
    '''
    Re_e = Re*eD*(fd/8.)**0.5
    Nu = Re*Pr*fd/8./(1 + (fd/8.)**0.5*(4.5*Re_e**0.2*Pr**0.5 - 8.48))
    return Nu

#print [turbulent_Bhatti_Shah(Re=1E5, Pr=1.2, fd=0.0185, eD=1E-3)]


def Nu_conv_internal(Re=None, Pr=None, fd=None, eD=None, Di=None, x=None,
                     fd_smooth=None, AvailableMethods=False, Method=None):
    r'''This function handles choosing which internal flow heat transfer
    correlation to use, depending on the provided information.
    Generally this is used by a helper class, but can be used directly. Will
    automatically select the correlation to use if none is provided'''
    def list_methods():
        methods = []
        if Re < 2100:
            # Laminar!
            if all((Re, Pr, x, Di)):
                methods.append('Baehr-Stephan laminar thermal/velocity entry')
                methods.append('Hausen laminar thermal entry')
                methods.append('Seider-Tate laminar thermal entry')

            methods.append('Laminar - constant T')
            methods.append('Laminar - constant Q')
        else:
            if all((Re, Pr, fd)) and Pr < 0.03:
                # Liquid metals
                methods.append('Martinelli')
            if all((Re, Pr, fd)):
                methods.append('Churchill-Zajic')
                methods.append('Petukhov-Kirillov-Popov')
                methods.append('Gnielinski')
                methods.append('Sandall')
                methods.append('Webb')
                methods.append('Friend-Metzner')
                methods.append('Prandtl')
                methods.append('von-Karman')
                methods.append('Gowen-Smith')
                methods.append('Kawase-Ulbrecht')
                methods.append('Kawase-De')

            if all((Re, Pr)):
                methods.append('Dittus-Boelter')
                methods.append('Sieder-Tate')
                methods.append('Drexel-McAdams')
                methods.append('Colburn')
                methods.append('ESDU')
                methods.append('Gnielinski smooth low Pr') # 1
                methods.append('Gnielinski smooth high Pr') # 2

            if all((Re, Pr, Di, x)):
                methods.append('Hausen')
            if all((Re, Pr, fd, eD)):
                methods.append('Bhatti-Shah')
                methods.append('Dipprey-Sabersky')
            if all((Re, Pr, fd, fd_smooth)):
                methods.append('Nunner')
        methods.append('None')
        return methods

    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == 'Laminar - constant T':
        Nu = laminar_T_const()
    elif Method == 'Laminar - constant Q':
        Nu = laminar_Q_const()
    elif Method == 'Baehr-Stephan laminar thermal/velocity entry':
        Nu = laminar_entry_thermal_Hausen(Re=Re, Pr=Pr, L=x, Di=Di)
    elif Method == 'Hausen laminar thermal entry':
        Nu = laminar_entry_Seider_Tate(Re=Re, Pr=Pr, L=x, Di=Di)
    elif Method == 'Seider-Tate laminar thermal entry':
        Nu = laminar_entry_Baehr_Stephan(Re=Re, Pr=Pr, L=x, Di=Di)

    elif Method == 'Churchill-Zajic':
        Nu = turbulent_Churchill_Zajic(Re=Re, Pr=Pr, fd=fd)
    elif Method == 'Petukhov-Kirillov-Popov':
        Nu = turbulent_Petukhov_Kirillov_Popov(Re=Re, Pr=Pr, fd=fd)
    elif Method == 'Gnielinski':
        Nu = turbulent_Gnielinski(Re=Re, Pr=Pr, fd=fd)
    elif Method == 'Sandall':
        Nu = turbulent_Sandall(Re=Re, Pr=Pr, fd=fd)
    elif Method == 'Webb':
        Nu = turbulent_Webb(Re=Re, Pr=Pr, fd=fd)
    elif Method == 'Friend-Metzner':
        Nu = turbulent_Friend_Metzner(Re=Re, Pr=Pr, fd=fd)
    elif Method == 'Prandtl':
        Nu = turbulent_Prandtl(Re=Re, Pr=Pr, fd=fd)
    elif Method == 'von-Karman':
        Nu = turbulent_von_Karman(Re=Re, Pr=Pr, fd=fd)
    elif Method == 'Martinelli':
        Nu = turbulent_Martinelli(Re=Re, Pr=Pr, fd=fd)
    elif Method == 'Gowen-Smith':
        Nu = turbulent_Gowen_Smith(Re=Re, Pr=Pr, fd=fd)
    elif Method == 'Kawase-Ulbrecht':
        Nu = turbulent_Kawase_Ulbrecht(Re=Re, Pr=Pr, fd=fd)
    elif Method == 'Kawase-De':
        Nu = turbulent_Kawase_De(Re=Re, Pr=Pr, fd=fd)

    elif Method == 'Dittus-Boelter':
        Nu = turbulent_Dittus_Boelter(Re=Re, Pr=Pr)
    elif Method == 'Sieder-Tate':
        Nu = turbulent_Sieder_Tate(Re=Re, Pr=Pr)
    elif Method == 'Drexel-McAdams':
        Nu = turbulent_Drexel_McAdams(Re=Re, Pr=Pr)
    elif Method == 'Colburn':
        Nu = turbulent_Colburn(Re=Re, Pr=Pr)
    elif Method == 'ESDU':
        Nu = turbulent_ESDU(Re=Re, Pr=Pr)
    elif Method == 'Gnielinski smooth low Pr':
        Nu = turbulent_Gnielinski_smooth_1(Re=Re, Pr=Pr)
    elif Method == 'Gnielinski smooth high Pr':
        Nu = turbulent_Gnielinski_smooth_2(Re=Re, Pr=Pr)

    elif Method == 'Hausen':
        Nu = turbulent_entry_Hausen(Re=Re, Pr=Pr, Di=Di, x=x)
    elif Method == 'Bhatti-Shah':
        Nu = turbulent_Bhatti_Shah(Re=Re, Pr=Pr, fd=fd, eD=eD)
    elif Method == 'Dipprey-Sabersky':
        Nu = turbulent_Dipprey_Sabersky(Re=Re, Pr=Pr, fd=fd, eD=eD)
    elif Method == 'Nunner':
        Nu = turbulent_Nunner(Re=Re, Pr=Pr, fd=fd, fd_smooth=fd_smooth)

    elif Method == 'None':
        Nu = None
    else:
        raise Exception('Failure in in function')
    return Nu


## Comparison
#import matplotlib.pyplot as plt
#import numpy as np
#from fluids.friction import friction_factor
#Pr = 0.3
#Di = 0.0254*4
#roughness = .00015
#
#methods = Nu_conv_internal(Re=10000, Pr=Pr, fd=1.8E-5, x=2.5, Di=0.5, AvailableMethods=True, Method=None)
#
#plt.figure()
#Res = np.logspace(4, 6, 300)
#for way in methods:
#    Nus = []
#    for Re in Res:
#        fd = friction_factor(Re=Re, eD=roughness/Di)
#        Nus.append(Nu_conv_internal(Re=Re, Pr=Pr, fd=fd, x=2.5, Di=0.5, Method=way))
#    plt.plot(Res, Nus, label=way)
#plt.xlabel(r'Res')
#plt.ylabel('Nus')
#plt.legend()
#
#plt.show()
