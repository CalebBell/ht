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

__all__ = ['Nu_vertical_plate_Churchill', 'Nu_horizontal_cylinder_Churchill',
           'Nu_sphere_Churchill']

def Nu_vertical_plate_Churchill(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a vertical
    plate according to the Churchill-Chu [1]_ correlation, also presented in
    [2]_. Plate must be isothermal; an alternate expression exists for constant
    heat flux.

    .. math::
        Nu_{L}=\left[0.825+\frac{0.387Ra_{L}^{1/6}}
        {[1+(0.492/Pr)^{9/16}]^{8/27}}\right]^2

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Although transition from laminar to turbulent is discrete in reality, this
    equation provides a smooth transition in value from laminar to turbulent.
    Checked with the original source.

    Can be applied to vertical cylinders as well, subject to the criteria below:

    .. math::
        \frac{D}{L}\ge \frac{35}{Gr_L^{1/4}}

    Examples
    --------
    From [2]_, Example 9.2, matches:

    >>> Nu_vertical_plate_Churchill(0.69, 2.63E9)
    147.16185223770603

    References
    ----------
    .. [1] Churchill, Stuart W., and Humbert H. S. Chu. "Correlating Equations
       for Laminar and Turbulent Free Convection from a Vertical Plate."
       International Journal of Heat and Mass Transfer 18, no. 11
       (November 1, 1975): 1323-29. doi:10.1016/0017-9310(75)90243-4.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    Ra = Pr * Gr
    Nu = (0.825 + (0.387*Ra**(1/6.)/(1 + (0.492/Pr)**(9/16.))**(8/27.)))**2
    return Nu

#print [Nu_vertical_plate_Churchill(.69, 2.63E9)]

def Nu_horizontal_cylinder_Churchill(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a horizontal
    cylinder according to the Churchill-Chu [1]_ correlation, also presented in
    [2]_. Cylinder must be isothermal; an alternate expression exists for
    constant heat flux.

    .. math::
        Nu_{D}=\left[0.60+\frac{0.387Ra_{D}^{1/6}}
        {[1+(0.559/Pr)^{9/16}]^{8/27}}\right]^2

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Although transition from laminar to turbulent is discrete in reality, this
    equation provides a smooth transition in value from laminar to turbulent.
    Checked with the original source, which has its powers unsimplified but
    is equivalent.

    [1]_ recommends 1E-5 as the lower limit for Ra, but no upper limit. [2]_
    suggests an upper limit of 1E12.

    Examples
    --------
    From [2]_, Example 9.2, matches:

    >>> Nu_horizontal_cylinder_Churchill(0.69, 2.63E9)
    139.13493970073597

    References
    ----------
    .. [1] Churchill, Stuart W., and Humbert H. S. Chu. "Correlating Equations
       for Laminar and Turbulent Free Convection from a Horizontal Cylinder."
       International Journal of Heat and Mass Transfer 18, no. 9
       (September 1975): 1049-53. doi:10.1016/0017-9310(75)90222-7.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    Ra = Pr * Gr
    Nu = (0.6 + 0.387*Ra**(1/6.)/(1 + (0.559/Pr)**(9/16.))**(8/27.))**2
    return Nu


def Nu_sphere_Churchill(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a sphere
    according to the Churchill [1]_ correlation. Sphere must be isothermal.

    .. math::
        Nu_D=2+\frac{0.589Ra_D^{1/4}} {\left[1+(0.469/Pr)^{9/16}\right]^{4/9}}
        \cdot\left\{1 + \frac{7.44\times 10^{-8}Ra}
        {[1+(0.469/Pr)^{9/16}]^{16/9}}\right\}^{1/12}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Although transition from laminar to turbulent is discrete in reality, this
    equation provides a smooth transition in value from laminar to turbulent.
    Checked with the original source.

    Good for Ra < 1E13. Limit of Nu is 2 at low Grashof numbers.

    Examples
    --------
    >>> Nu_sphere_Churchill(.7, 1E1), Nu_sphere_Churchill(.7, 1E7)
    (2.738104002574638, 25.670869440317578)

    References
    ----------
    .. [1] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    '''
    Ra = Pr * Gr
    Nu = 2 + (0.589*Ra**0.25/(1 + (0.469/Pr)**(9/16.))**(4/9.)*(
    1 + 7.44E-8*Ra/(1 + (0.469/Pr)**(9/16.))**(16/9.))**(1/12.))
    return Nu

#print [Nu_sphere_Churchill(.7, 1E1), Nu_sphere_Churchill(.7, 1E7)]
