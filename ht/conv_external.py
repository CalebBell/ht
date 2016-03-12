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

__all__ = ['Nu_cylinder_Zukauskas', 'Nu_cylinder_Churchill_Bernstein']

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
        Prandtl number at bulk temperature [-]
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

#print [Nu_cylinder_Zukauskas(7992, 0.707, 0.69)]

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