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
from math import exp

__all__ = ['Nu_cylinder_Zukauskas', 'Nu_cylinder_Churchill_Bernstein',
           'Nu_cylinder_Sanitjai_Goldstein', 'Nu_cylinder_Fand',
           'Nu_cylinder_Perkins_Leppert_1964',
           'Nu_cylinder_Perkins_Leppert_1962', 'Nu_cylinder_Whitaker',
           'Nu_cylinder_McAdams']

### Single Cylinders in Crossflow


def Nu_cylinder_Zukauskas(Re, Pr, Prw=None):
    r'''Calculates Nusselt number for crossflow across a single tube at a
    specified Re. Method from [1]_, also shown without modification in [2]_.

    .. math::
        Nu_{D}=CRe^{m}Pr^{n}\left(\frac{Pr}{Pr_s}\right)^{1/4}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to cylinder diameter, [-]
    Pr : float
        Prandtl number at free stream temperature [-]
    Prw : float, optional
        Prandtl number at wall temperature, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Notes
    -----
    If Prandtl number at wall are not provided, the Prandtl number correction
    is not used and left to an outside function.

    n is 0.37 if Pr <= 10; otherwise n is 0.36.

    C and m are from the following table. If Re is outside of the ranges shown,
    the nearest range is used blindly.

    +---------+-------+-----+
    | Re      | C     | m   |
    +=========+=======+=====+
    | 1-40    | 0.75  | 0.4 |
    +---------+-------+-----+
    | 40-1E3  | 0.51  | 0.5 |
    +---------+-------+-----+
    | 1E3-2E5 | 0.26  | 0.6 |
    +---------+-------+-----+
    | 2E5-1E6 | 0.076 | 0.7 |
    +---------+-------+-----+

    Examples
    --------
    Example 7.3 in [2]_, matches.

    >>> Nu_cylinder_Zukauskas(7992, 0.707, 0.69)
    50.523612661934386

    References
    ----------
    .. [1] Zukauskas, A. Heat transfer from tubes in crossflow. In T.F. Irvine,
       Jr. and J. P. Hartnett, editors, Advances in Heat Transfer, volume 8,
       pages 93-160. Academic Press, Inc., New York, 1972.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    if Re <= 40:
        c, m = 0.75, 0.4
    elif Re < 1E3:
        c, m = 0.51, 0.5
    elif Re < 2E5:
        c, m = 0.26, 0.6
    else:
        c, m = 0.076, 0.7
    if Pr <= 10:
        n = 0.37
    else:
        n = 0.36
    Nu = c*Re**m*Pr**n
    if Prw:
        Nu = Nu*(Pr/Prw)**0.25
    return Nu


def Nu_cylinder_Churchill_Bernstein(Re, Pr):
    r'''Calculates Nusselt number for crossflow across a single tube
    at a specified `Re` and `Pr`, both evaluated at the film temperature. No
    other wall correction is necessary for this formulation. Method is
    shown without modification in [2]_ and many other texts.

    .. math::
        Nu_D = 0.3 + \frac{0.62 Re_D^{0.5} Pr^{1/3}}{[1 + (0.4/Pr)^{2/3}
        ]^{0.25}}\left[1 + \left(\frac{Re_D}{282000}\right)^{5/8}\right]^{0.8}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to cylinder diameter, [-]
    Pr : float
        Prandtl number at film temperature, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Notes
    -----
    May underestimate heat transfer in some cases, as it the formula is
    described in [1]_ as "appears to provide a lower bound for RePr > 0.4".
    An alternate exponent for a smaller range is also presented in [1]_.

    Examples
    --------
    Example 7.3 in [2]_, matches.

    >>> Nu_cylinder_Churchill_Bernstein(6071, 0.7)
    40.63708594124974

    References
    ----------
    .. [1] Churchill, S. W., and M. Bernstein. "A Correlating Equation for
       Forced Convection From Gases and Liquids to a Circular Cylinder in
       Crossflow." Journal of Heat Transfer 99, no. 2 (May 1, 1977):
       300-306. doi:10.1115/1.3450685.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    Nu = 0.3 + (0.62*Re**0.5*Pr**(1/3.))/(1 + (0.4/Pr)**(2/3.))**0.25*(
    1 +(Re/282000.)**(0.625))**0.8
    return Nu

#print [Nu_cylinder_Churchill_Bernstein(6071, 0.7)]


def Nu_cylinder_Sanitjai_Goldstein(Re, Pr):
    r'''Calculates Nusselt number for crossflow across a single tube
    at a specified `Re` and `Pr`, both evaluated at the film temperature. No
    other wall correction is necessary for this formulation. Method is the
    most recent implemented here and believed to be more accurate than other
    formulations available.

    .. math::
        Nu = 0.446Re^{0.5} Pr^{0.35} + 0.528\left[(6.5\exp(Re/5000))^{-5}
        + (0.031Re^{0.8})^{-5}\right]^{-1/5}Pr^{0.42}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to cylinder diameter, [-]
    Pr : float
        Prandtl number at film temperature, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Notes
    -----
    Developed with test results for water, mixtures of ethylene glycol and
    water, and air (Pr = 0.7 to 176). Re range from 2E3 to 9E4. Also presents
    results for local heat transfer coefficients.

    Examples
    --------
    >>> Nu_cylinder_Sanitjai_Goldstein(6071, 0.7)
    40.38327083519522

    References
    ----------
    .. [1] Sanitjai, S., and R. J. Goldstein. "Forced Convection Heat Transfer
       from a Circular Cylinder in Crossflow to Air and Liquids." International
       Journal of Heat and Mass Transfer 47, no. 22 (October 2004): 4795-4805.
       doi:10.1016/j.ijheatmasstransfer.2004.05.012.
    '''
    Nu = 0.446*Re**0.5*Pr**0.35 + 0.528*((6.5*exp(Re/5000.))**-5
    + (0.031*Re**0.8)**-5)**-0.2*Pr**0.42
    return Nu


def Nu_cylinder_Fand(Re, Pr):
    r'''Calculates Nusselt number for crossflow across a single tube
    at a specified `Re` and `Pr`, both evaluated at the film temperature. No
    other wall correction is necessary for this formulation. Also shown in
    [2]_.

    .. math::
        Nu = (0.35 + 0.34Re^{0.5} + 0.15Re^{0.58})Pr^{0.3}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to cylinder diameter, [-]
    Pr : float
        Prandtl number at film temperature, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Notes
    -----
    Developed with test results for water, and Re from 1E4 to 1E5, but also
    compared with other data in the literature. Claimed validity of Re from
    1E-1 to 1E5.

    Examples
    --------
    >>> Nu_cylinder_Fand(6071, 0.7)
    45.19984325481126

    References
    ----------
    .. [1] Fand, R. M. "Heat Transfer by Forced Convection from a Cylinder to
       Water in Crossflow." International Journal of Heat and Mass Transfer 8,
       no. 7 (July 1, 1965): 995-1010. doi:10.1016/0017-9310(65)90084-0.
    .. [2] Sanitjai, S., and R. J. Goldstein. "Forced Convection Heat Transfer
       from a Circular Cylinder in Crossflow to Air and Liquids." International
       Journal of Heat and Mass Transfer 47, no. 22 (October 2004): 4795-4805.
       doi:10.1016/j.ijheatmasstransfer.2004.05.012.
    '''
    Nu = (0.35 + 0.34*Re**0.5 + 0.15*Re**0.58)*Pr**0.3
    return Nu


def Nu_cylinder_McAdams(Re, Pr):
    r'''Calculates Nusselt number for crossflow across a single tube
    at a specified `Re` and `Pr`, both evaluated at the film temperature. No
    other wall correction is necessary for this formulation. Also shown in
    [2]_.

    .. math::
        Nu = (0.35 + 0.56 Re^{0.52})Pr^{0.3}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to cylinder diameter, [-]
    Pr : float
        Prandtl number at film temperature, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Notes
    -----
    Developed with very limited test results for water only.

    Examples
    --------
    >>> Nu_cylinder_McAdams(6071, 0.7)
    46.98179235867934

    References
    ----------
    .. [1] McAdams, William Henry. Heat Transmission. 3E. Malabar, Fla:
       Krieger Pub Co, 1985.
    .. [2] Fand, R. M. "Heat Transfer by Forced Convection from a Cylinder to
       Water in Crossflow." International Journal of Heat and Mass Transfer 8,
       no. 7 (July 1, 1965): 995-1010. doi:10.1016/0017-9310(65)90084-0.
    '''
    Nu = (0.35 + 0.56*Re**0.52)*Pr**0.3
    return Nu


def Nu_cylinder_Whitaker(Re, Pr, mu=None, muw=None):
    r'''Calculates Nusselt number for crossflow across a single tube as shown
    in [1]_ at a specified `Re` and `Pr`, both evaluated at the free stream
    temperature. Recommends a viscosity exponent correction of 0.25, which is
    applied only if provided. Also shown in [2]_.

    .. math::
        Nu_D = (0.4 Re_D^{0.5} + 0.06Re_D^{2/3})Pr^{0.4}
        \left(\frac{\mu}{\mu_w}\right)^{0.25}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to cylinder diameter, [-]
    Pr : float
        Prandtl number at free stream temperature, [-]
    mu : float, optional
        Viscosity of fluid at the free stream temperature [Pa*s]
    muw : float, optional
        Viscosity of fluid at the wall temperature [Pa*s]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Notes
    -----
    Developed considering data from 1 to 1E5 Re, 0.67 to 300 Pr, and range of
    viscosity ratios from 0.25 to 5.2. Found experimental data to generally
    agree with it within 25%.

    Examples
    --------
    >>> Nu_cylinder_Whitaker(6071, 0.7)
    45.94527461589126

    References
    ----------
    .. [1] Whitaker, Stephen. "Forced Convection Heat Transfer Correlations for
       Flow in Pipes, Past Flat Plates, Single Cylinders, Single Spheres, and
       for Flow in Packed Beds and Tube Bundles." AIChE Journal 18, no. 2
       (March 1, 1972): 361-371. doi:10.1002/aic.690180219.
    .. [2] Sanitjai, S., and R. J. Goldstein. "Forced Convection Heat Transfer
       from a Circular Cylinder in Crossflow to Air and Liquids." International
       Journal of Heat and Mass Transfer 47, no. 22 (October 2004): 4795-4805.
       doi:10.1016/j.ijheatmasstransfer.2004.05.012.
    '''
    Nu = (0.4*Re**0.5 + 0.06*Re**(2/3.))*Pr**0.3
    if mu and muw:
        Nu *= (mu/muw)**0.25
    return Nu

#print [Nu_cylinder_Whitaker(6071, 0.7, 1E-3, 1.2E-3)]

def Nu_cylinder_Perkins_Leppert_1962(Re, Pr, mu=None, muw=None):
    r'''Calculates Nusselt number for crossflow across a single tube as shown
    in [1]_ at a specified `Re` and `Pr`, both evaluated at the free stream
    temperature. Recommends a viscosity exponent correction of 0.25, which is
    applied only if provided. Also shown in [2]_.

    .. math::
        Nu = \left[0.30Re^{0.5} + 0.10Re^{0.67}\right]Pr^{0.4}
        \left(\frac{\mu}{\mu_w}\right)^{0.25}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to cylinder diameter, [-]
    Pr : float
        Prandtl number at free stream temperature, [-]
    mu : float, optional
        Viscosity of fluid at the free stream temperature [Pa*s]
    muw : float, optional
        Viscosity of fluid at the wall temperature [Pa*s]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Notes
    -----
    Considered results with Re from 40 to 1E5, Pr from 1 to 300; and viscosity
    ratios of 0.25 to 4.

    Examples
    --------
    >>> Nu_cylinder_Perkins_Leppert_1962(6071, 0.7)
    49.97164291175499

    References
    ----------
    .. [1] Perkins, Jr., H. C., and G. Leppert. "Forced Convection Heat
       Transfer From a Uniformly Heated Cylinder." Journal of Heat Transfer 84,
       no. 3 (August 1, 1962): 257-261. doi:10.1115/1.3684359.
    .. [2] Sanitjai, S., and R. J. Goldstein. "Forced Convection Heat Transfer
       from a Circular Cylinder in Crossflow to Air and Liquids." International
       Journal of Heat and Mass Transfer 47, no. 22 (October 2004): 4795-4805.
       doi:10.1016/j.ijheatmasstransfer.2004.05.012.
    '''
    Nu = (0.30*Re**0.5 + 0.10*Re**0.67)*Pr**0.4
    if mu and muw:
        Nu *= (mu/muw)**0.25
    return Nu


def Nu_cylinder_Perkins_Leppert_1964(Re, Pr, mu=None, muw=None):
    r'''Calculates Nusselt number for crossflow across a single tube as shown
    in [1]_ at a specified `Re` and `Pr`, both evaluated at the free stream
    temperature. Recommends a viscosity exponent correction of 0.25, which is
    applied only if provided. Also shown in [2]_.

    .. math::
        Nu = \left[0.31Re^{0.5} + 0.11Re^{0.67}\right]Pr^{0.4}
        \left(\frac{\mu}{\mu_w}\right)^{0.25}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to cylinder diameter, [-]
    Pr : float
        Prandtl number at free stream temperature, [-]
    mu : float, optional
        Viscosity of fluid at the free stream temperature [Pa*s]
    muw : float, optional
        Viscosity of fluid at the wall temperature [Pa*s]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Notes
    -----
    Considers new data since `Nu_cylinder_Perkins_Leppert_1962`, Re from 2E3 to
    1.2E5, Pr from 1 to 7, and surface to bulk temperature differences of
    11 to 66.

    Examples
    --------
    >>> Nu_cylinder_Perkins_Leppert_1964(6071, 0.7)
    53.61767038619986

    References
    ----------
    .. [1] Perkins Jr., H. C., and G. Leppert. "Local Heat-Transfer
       Coefficients on a Uniformly Heated Cylinder." International Journal of
       Heat and Mass Transfer 7, no. 2 (February 1964): 143-158.
       doi:10.1016/0017-9310(64)90079-1.
    .. [2] Sanitjai, S., and R. J. Goldstein. "Forced Convection Heat Transfer
       from a Circular Cylinder in Crossflow to Air and Liquids." International
       Journal of Heat and Mass Transfer 47, no. 22 (October 2004): 4795-4805.
       doi:10.1016/j.ijheatmasstransfer.2004.05.012.
    '''
    Nu = (0.31*Re**0.5 + 0.11*Re**0.67)*Pr**0.4
    if mu and muw:
        Nu *= (mu/muw)**0.25
    return Nu

#print [Nu_cylinder_Perkins_Leppert_1964(6071, 0.7, 1E-3, 1.2E-3)]