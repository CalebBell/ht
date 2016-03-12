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

__all__ = ['Nu_cylinder_Churchill_Bernstein']


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