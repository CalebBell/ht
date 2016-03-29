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
from math import exp, log
import numpy as np

__all__ = ['Nu_vertical_plate_Churchill',
           'Nu_sphere_Churchill', 'Nu_vertical_cylinder_Griffiths_Davis_Morgan',
           'Nu_vertical_cylinder_Jakob_Linke_Morgan',
           'Nu_vertical_cylinder_Carne_Morgan',
           'Nu_vertical_cylinder_Eigenson_Morgan',
           'Nu_vertical_cylinder_Touloukian_Morgan',
           'Nu_vertical_cylinder_McAdams_Weiss_Saunders',
           'Nu_vertical_cylinder_Kreith_Eckert',
           'Nu_vertical_cylinder_Hanesian_Kalish_Morgan',
           'Nu_vertical_cylinder_Al_Arabi_Khamis',
           'Nu_vertical_cylinder_Popiel_Churchill',
           'Nu_vertical_cylinder',
           'Nu_horizontal_cylinder_Churchill_Chu',
           'Nu_horizontal_cylinder_Kuehn_Goldstein',
           'Nu_horizontal_cylinder_Morgan',
           'Nu_horizontal_cylinder']


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


### Vertical cylinders

def Nu_vertical_cylinder_Griffiths_Davis_Morgan(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ correlated by [2]_, as
    presented in [3]_ and [4]_.

    .. math::
        Nu_H = 0.67 Ra_H^{0.25},\; 10^{7} < Ra < 10^{9}

        Nu_H = 0.0782 Ra_H^{0.357}, \; 10^{9} < Ra < 10^{11}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Cylinder of diameter 17.43 cm, length from 4.65 to 263.5 cm. Air as fluid.
    Transition between ranges is not smooth.
    If outside of range, no warning is given.

    Examples
    --------
    >>> [Nu_vertical_cylinder_Griffiths_Davis_Morgan(.999, 1E9),
    ... Nu_vertical_cylinder_Griffiths_Davis_Morgan(.7, 1.43E9)]
    [119.11492311615193, 127.75023824044263]

    References
    ----------
    .. [1] Griffiths, Ezer, A. H. Davis, and Great Britain. The Transmission of
       Heat by Radiation and Convection. London: H. M. Stationery off., 1922.
    .. [2] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [3] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [4] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 1E9 and turbulent is None):
        Nu = 0.0782*Ra**0.357
    else:
        Nu = 0.67*Ra**0.25
    return Nu


def Nu_vertical_cylinder_Jakob_Linke_Morgan(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ correlated by [2]_, as
    presented in [3]_ and [4]_.

    .. math::
        Nu_H = 0.555 Ra_H^{0.25},\; 10^{4} < Ra < 10^{8}

        Nu_H = 0.129 Ra_H^{1/3},\; 10^{8} < Ra < 10^{12}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Cylinder of diameter 3.5 cm, length from L/D = 4.3. Air as fluid.
    Transition between ranges is not smooth.
    If outside of range, no warning is given. Results are presented rounded in
    [4]_, and the second range is not shown in [3]_.

    Examples
    --------
    >>> [Nu_vertical_cylinder_Jakob_Linke_Morgan(.7, 1.42E8),
    ... Nu_vertical_cylinder_Jakob_Linke_Morgan(.7, 1.43E8)]
    [55.4165620291897, 59.89644813633899]

    References
    ----------
    .. [1] Jakob, M., and Linke, W., Warmeubergang beim Verdampfen von
       Flussigkeiten an senkrechten und waagerechten Flaschen, Phys. Z.,
       vol. 36, pp. 267-280, 1935.
    .. [2] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [3] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [4] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 1E8 and turbulent is None):
        Nu = 0.129*Ra**(1/3.)
    else:
        Nu = 0.555*Ra**0.25
    return Nu


def Nu_vertical_cylinder_Carne_Morgan(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ correlated by [2]_, as
    presented in [3]_ and [4]_.

    .. math::
        Nu_H = 1.07 Ra_H^{0.28},\; 2\times 10^{6} < Ra < 2\times 10^{8}

        Nu_H = 0.152 Ra_H^{0.38},\; 2\times 10^{8} < Ra < 2\times 10^{11}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Cylinder of diameters 0.475 cm to 7.62 cm, L/D from 8 to 127. Isothermal
    boundary condition was assumed, but not verified. Transition between ranges
    is not smooth. If outside of range, no warning is given. The higher range
    of [1]_ is not shown in [3]_, and the formula for the first is actually for
    the second in [3]_.

    Examples
    --------
    >>> [Nu_vertical_cylinder_Carne_Morgan(.7, 2.85E8),
    ... Nu_vertical_cylinder_Carne_Morgan(.7, 2.86E8)]
    [225.6149061669447, 216.97012327590352]

    References
    ----------
    .. [1] J. B. Carne. "LIX. Heat Loss by Natural Convection from Vertical
       Cylinders." The London, Edinburgh, and Dublin Philosophical Magazine and
       Journal of Science 24, no. 162 (October 1, 1937): 634-53.
       doi:10.1080/14786443708565140.
    .. [2] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [3] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [4] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 2E8 and turbulent is None):
        Nu = 0.152*Ra**0.38
    else:
        Nu = 1.07*Ra**0.28
    return Nu

#print [[Nu_vertical_cylinder_Carne_Morgan(i, 2E8, j) for i in (0.999999, 1.000001)]
#        for j in (True, False, None)]


def Nu_vertical_cylinder_Eigenson_Morgan(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ correlated by [2]_,
    presented in [3]_ and in more detail in [4]_.

    .. math::
        Nu_H = 0.48 Ra_H^{0.25},\; 10^{9} < Ra

        Nu_H = 51.5 + 0.0000726 Ra_H^{0.63},\; 10^{9} < Ra < 1.69 \times 10^{10}

        Nu_H = 0.148 Ra_H^{1/3} - 127.6 ,\; 1.69 \times 10^{10} < Ra

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Author presents results as appropriate for both flat plates and cylinders.
    Height of 2.5 m with diameters of 2.4, 7.55, 15, 35, and 50 mm. Another
    experiment of diameter 58 mm and length of 6.5 m was considered.
    Cylinder of diameters 0.475 cm to 7.62 cm, L/D from 8 to 127.Transition
    between ranges is not smooth. If outside of range, no warning is given.
    Formulas are presented similarly in [3]_ and [4]_, but only [4]_ shows
    the transition formula.

    Examples
    --------
    >>> [Nu_vertical_cylinder_Eigenson_Morgan(0.7, 1.42E9),
    ... Nu_vertical_cylinder_Eigenson_Morgan(0.7, 1.43E9),
    ... Nu_vertical_cylinder_Eigenson_Morgan(0.7, 2.4E10),
    ... Nu_vertical_cylinder_Eigenson_Morgan(0.7, 2.5E10)]
    [85.22908647061865, 85.47896057139417, 252.35445465640387, 256.64456353698154]

    References
    ----------
    .. [1] Eigenson L (1940). Les lois gouvernant la transmission de la chaleur
       aux gaz biatomiques par les parois des cylindres verticaux dans le cas
       de convection naturelle. Dokl Akad Nauk SSSR 26:440-444
    .. [2] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [3] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [4] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 1.69E10 and turbulent is None):
        Nu = 0.148*Ra**(1/3.) - 127.6
    elif 1E9 < Ra < 1.69E10 and turbulent is not False:
        Nu = 51.5 + 0.0000726*Ra**0.63
    else:
        Nu = 0.48*Ra**0.25
    return Nu

#print [[Nu_vertical_cylinder_Eigenson_Morgan(i, 1.69E10, j) for i in (0.999999, 1.000001)] for j in (True, False, None)]


def Nu_vertical_cylinder_Touloukian_Morgan(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ correlated by [2]_, as
    presented in [3]_ and [4]_.

    .. math::
        Nu_H = 0.726 Ra_H^{0.25},\; 2\times 10^{8} < Ra < 4\times 10^{10}

        Nu_H = 0.0674 (Gr_H Pr^{1.29})^{1/3},\; 4\times 10^{10} < Ra < 9\times 10^{11}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Cylinder of diameters 2.75 inch, with heights of 6, 18, and 36.25 inch.
    Temperature was controlled via multiple seperately controlled heating
    sections. Fluids were water and ethylene-glycol. Transition between ranges
    is not smooth. If outside of range, no warning is given. [2]_, [3]_, and
    [4]_ are in complete agreement about this formulation.

    Examples
    --------
    >>> [Nu_vertical_cylinder_Touloukian_Morgan(.7, 5.7E10),
    ... Nu_vertical_cylinder_Touloukian_Morgan(.7, 5.8E10)]
    [324.47395664562873, 223.80067132541936]

    References
    ----------
    .. [1] Touloukian, Y. S, George A Hawkins, and Max Jakob. Heat Transfer by
       Free Convection from Heated Vertical Surfaces to Liquids.
       Trans. ASME 70, 13-18 (1948).
    .. [2] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [3] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [4] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 4E10 and turbulent is None):
        Nu = 0.0674*(Gr*Pr**1.29)**(1/3.)
    else:
        Nu = 0.726*Ra**0.25
    return Nu


def Nu_vertical_cylinder_McAdams_Weiss_Saunders(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ and [2]_ correlated by
    [3]_, as presented in [4]_, [5]_, and [6]_.

    .. math::
        Nu_H = 0.59 Ra_H^{0.25},\; 10^{4} < Ra < 10^{9}

        Nu_H = 0.13 Ra_H^{1/3.},\; 10^{9} < Ra < 10^{12}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Transition between ranges is not smooth. If outside of range, no warning is
    given. For ranges under 10^4, a graph is provided, not included here.

    Examples
    --------
    >>> [Nu_vertical_cylinder_McAdams_Weiss_Saunders(.7, 1.42E9),
    ... Nu_vertical_cylinder_McAdams_Weiss_Saunders(.7, 1.43E9)]
    [104.76075212013542, 130.04331889690818]

    References
    ----------
    .. [1] Weise, Rudolf. "Warmeubergang durch freie Konvektion an
       quadratischen Platten." Forschung auf dem Gebiet des Ingenieurwesens
       A 6, no. 6 (November 1935): 281-92. doi:10.1007/BF02592565.
    .. [2] Saunders, O. A. "The Effect of Pressure Upon Natural Convection in
       Air." Proceedings of the Royal Society of London A: Mathematical,
       Physical and Engineering Sciences 157, no. 891 (November 2, 1936):
       278-91. doi:10.1098/rspa.1936.0194.
    .. [3] McAdams, William Henry. Heat Transmission. 3E. Malabar, Fla:
       Krieger Pub Co, 1985.
    .. [4] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [5] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [6] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 1E9 and turbulent is None):
        Nu = 0.13*Ra**(1/3.)
    else:
        Nu = 0.59*Ra**0.25
    return Nu


def Nu_vertical_cylinder_Kreith_Eckert(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_  correlated by
    [2]_, also as presented in [3]_, [4]_, and [5]_.

    .. math::
        Nu_H = 0.555 Ra_H^{0.25},\; 10^{5} < Ra < 10^{9}

        Nu_H = 0.021 Ra_H^{0.4},\; 10^{9} < Ra < 10^{12}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Transition between ranges is not smooth. If outside of range, no warning is
    given.

    Examples
    --------
    >>> [Nu_vertical_cylinder_Kreith_Eckert(.7, 1.42E9),
    ... Nu_vertical_cylinder_Kreith_Eckert(.7, 1.43E9)]
    [98.54613123165282, 83.63593679160734]

    References
    ----------
    .. [1] Eckert, E. R. G., Thomas W. Jackson, and United States. Analysis of
       Turbulent Free-Convection Boundary Layer on Flat Plate. National
       Advisory Committee for Aeronautics, no. 2207. Washington, D.C.: National
       Advisoty Committee for Aeronautics, 1950.
    .. [2] Kreith, Frank, Raj Manglik, and Mark Bohn. Principles of Heat
       Transfer. Cengage, 2010.
    .. [3] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [4] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [5] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 1E9 and turbulent is None):
        Nu = 0.021*Ra**0.4
    else:
        Nu = 0.555*Ra**0.25
    return Nu


def Nu_vertical_cylinder_Hanesian_Kalish_Morgan(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ correlated by
    [2]_, also as presented in [3]_ and [4]_.

    .. math::
        Nu_H = 0.48 Ra_H^{0.23},\; 10^{6} < Ra < 10^{8}

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
    For air and fluoro-carbons. If outside of range, no warning is given.
    Laminar range only!

    Examples
    --------
    >>> Nu_vertical_cylinder_Hanesian_Kalish_Morgan(.7, 1E7)
    18.014150492696604

    References
    ----------
    .. [1] Hanesian, D. and Kalish, R. "Heat Transfer by Natural Convection
       with Fluorocarbon Gases." IEEE Transactions on Parts, Materials and
       Packaging 6, no. 4 (December 1970): 147-148.
       doi:10.1109/TPMP.1970.1136270.
    .. [2] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [3] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [4] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    Nu = 0.48*Ra**0.23
    return Nu


### Vertical cylinders, more complex correlations
def Nu_vertical_cylinder_Al_Arabi_Khamis(Pr, Gr, L, D, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to [1]_, also as presented in [2]_ and [3]_.

    .. math::
        Nu_H = 2.9Ra_H^{0.25}/Gr_D^{1/12},\; 9.88 \times 10^7 \le Ra_H \le 2.7\times10^{9}

        Nu_H = 0.47 Ra_H^{0.333}/Gr_D^{1/12},\; 2.7 \times 10^9 \le Ra_H \le 2.95\times10^{10}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number with respect to cylinder height [-]
    L : float
        Length of vertical cylinder, [m]
    D : float
        Diameter of cylinder, [m]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    For air. Local Nusselt number results also given in [1]_. D from 12.75 to
    51 mm; H from 300 to 2000 mm. Temperature kept constant by steam condensing.

    If outside of range, no warning is given. Applies for range of:

    .. math::
        1.08 \times 10^4 \le Gr_D \le 6.9 \times 10^5

    Examples
    --------
    >>> [Nu_vertical_cylinder_Al_Arabi_Khamis(.71, 3.6E9, 10, 1),
    ... Nu_vertical_cylinder_Al_Arabi_Khamis(.71, 3.7E9, 10, 1)]
    [185.32314790756703, 183.89407579255627]

    References
    ----------
    .. [1] Al-Arabi, M., and M. Khamis. "Natural Convection Heat Transfer from
       Inclined Cylinders." International Journal of Heat and Mass Transfer 25,
       no. 1 (January 1982): 3-15. doi:10.1016/0017-9310(82)90229-0.
    .. [2] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [3] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Gr_D = Gr/L**3*D**3
    Ra = Pr*Gr
    if turbulent or (Ra > 2.6E9 and turbulent is None):
        Nu = 0.47*Ra**(1/3.)*Gr_D**(-1/12.)
    else:
        Nu = 2.9*Ra**0.25*Gr_D**(-1/12.)
    return Nu


def Nu_vertical_cylinder_Popiel_Churchill(Pr, Gr, L, D,
                     Nu_vertical_plate_correlation=Nu_vertical_plate_Churchill):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to [1]_, also  presented in [2]_.

    .. math::
        \frac{Nu}{Nu_{L,fp}} = 1 + B\left[32^{0.5}Gr_L^{-0.25}\frac{L}{D}\right]^C

        B = 0.0571322 + 0.20305 Pr^{-0.43}

        C = 0.9165 - 0.0043Pr^{0.5} + 0.01333\ln Pr + 0.0004809/Pr

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number with respect to cylinder height [-]
    L : float
        Length of vertical cylinder, [m]
    D : float
        Diameter of cylinder, [m]
    Nu_vertical_plate_correlation : function, optional
        Correlation for vertical plate heat transfer

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    For 0.01 < Pr < 100. Requires a vertical flat plate correlation.
    Both [2], [3] present a power of 2 instead of 0.5 on the 32 in the equation,
    but the original has the correct form.

    Examples
    --------
    >>> Nu_vertical_cylinder_Popiel_Churchill(0.7, 1E10, 2.5, 1)
    228.8979005514989

    References
    ----------
    .. [1] Popiel, C. O., J. Wojtkowiak, and K. Bober. "Laminar Free Convective
       Heat Transfer from Isothermal Vertical Slender Cylinder." Experimental
       Thermal and Fluid Science 32, no. 2 (November 2007): 607-613.
       doi:10.1016/j.expthermflusci.2007.07.003.
    .. [2] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [3] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    B = 0.0571322 + 0.20305*Pr**-0.43
    C = 0.9165 - 0.0043*Pr**0.5 + 0.01333*log(Pr) + 0.0004809/Pr
    Nu_fp = Nu_vertical_plate_correlation(Pr, Gr)
    Nu = Nu_fp*(1 + B*(32**0.5*Gr**-0.25*L/D)**C)
    return Nu



# Nice Name : (function_call, does_turbulent, does_laminar, transition_Ra, is_only_Pr_Gr)
vertical_cylinder_correlations = {
'Churchill Vertical Plate': (Nu_vertical_plate_Churchill, True, True, None, True),
'Griffiths, Davis, & Morgan': (Nu_vertical_cylinder_Griffiths_Davis_Morgan, True, True, 1.00E+009, True),
'Jakob, Linke, & Morgan': (Nu_vertical_cylinder_Jakob_Linke_Morgan, True, True, 1.00E+008, True),
'Carne & Morgan': (Nu_vertical_cylinder_Carne_Morgan, True, True, 2.00E+008, True),
'Eigenson & Morgan': (Nu_vertical_cylinder_Eigenson_Morgan, True, True, 6.90E+011, True),
'Touloukian & Morgan': (Nu_vertical_cylinder_Touloukian_Morgan, True, True, 4.00E+010, True),
'McAdams, Weiss & Saunders': (Nu_vertical_cylinder_McAdams_Weiss_Saunders, True, True, 1.00E+009, True),
'Kreith & Eckert': (Nu_vertical_cylinder_Kreith_Eckert, True, True, 1.00E+009, True),
'Hanesian, Kalish & Morgan': (Nu_vertical_cylinder_Hanesian_Kalish_Morgan, False, True, 1.00E+008, True),
'Al-Arabi & Khamis': (Nu_vertical_cylinder_Al_Arabi_Khamis, True, True, 2.60E+009, False),
'Popiel & Churchill': (Nu_vertical_cylinder_Popiel_Churchill, False, True, 1.00E+009, False),
}


def Nu_vertical_cylinder(Pr=None, Gr=None, L=None, D=None,
                         AvailableMethods=False, Method=None):
    r'''This function handles choosing which vertical cylinder free convection
    correlation is used. Generally this is used by a helper class, but can be
    used directly. Will automatically select the correlation to use if none is
    provided; returns None if insufficient information is provided.

    Prefered functions are 'Popiel & Churchill' for fully defined geometries,
    and 'McAdams, Weiss & Saunders' otherwise.

    Examples
    --------
    >>> Nu_vertical_cylinder(0.72, 1E7)
    30.562236756513943

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
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to calculate Nu with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        vertical_cylinder_correlations
    AvailableMethods : bool, optional
        If True, function will consider which methods which can be used to
        calculate Nu with the given inputs
    '''
    def list_methods():
        methods = []
        if all((Pr, Gr)):
            for key, values in vertical_cylinder_correlations.items():
                if values[4] or all((L, D)):
                    methods.append(key)
        methods.append('None')
        if 'Popiel & Churchill' in methods:
            methods.remove('Popiel & Churchill')
            methods.insert(0, 'Popiel & Churchill')
        elif 'McAdams, Weiss & Saunders' in methods:
            methods.remove('McAdams, Weiss & Saunders')
            methods.insert(0, 'McAdams, Weiss & Saunders')
        return methods

    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method in vertical_cylinder_correlations:
        if vertical_cylinder_correlations[Method][4]:
            Nu = vertical_cylinder_correlations[Method][0](Pr=Pr, Gr=Gr)
        else:
            Nu = vertical_cylinder_correlations[Method][0](Pr=Pr, Gr=Gr, L=L, D=D)
    elif Method == 'None':
        Nu = None
    else:
        raise Exception('Failure in in function')
    return Nu

#print [ Nu_vertical_cylinder(0.72, 1E7)]

#import matplotlib.pyplot as plt
#import numpy as np
##L, D = 1.5, 0.1
#Pr, Gr = 0.72, 1E8
#methods = Nu_vertical_cylinder(Pr, Gr, AvailableMethods=True)
#Grs = np.logspace(2, 12, 10000)
#
#for method in methods:
#    Nus = [Nu_vertical_cylinder(Pr=Pr, Gr=i, Method=method) for i in Grs]
#    plt.loglog(Grs, Nus, label=method)
#plt.legend()
#plt.show()


### Horizontal Cylinders

def Nu_horizontal_cylinder_Churchill_Chu(Pr, Gr):
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

    >>> Nu_horizontal_cylinder_Churchill_Chu(0.69, 2.63E9)
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
    Ra = Pr*Gr
    Nu = (0.6 + 0.387*Ra**(1/6.)/(1 + (0.559/Pr)**(9/16.))**(8/27.))**2
    return Nu


def Nu_horizontal_cylinder_Kuehn_Goldstein(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a horizontal
    cylinder according to the Kuehn-Goldstein [1]_ correlation, also shown in
    [2]_. Cylinder must be isothermal.

    .. math::
        \frac{2}{Nu_D} = \ln\left[1 + \frac{2}{\left[\left\{0.518Ra_D^{0.25}
        \left[1 + \left(\frac{0.559}{Pr}\right)^{3/5}\right]^{-5/12}
        \right\}^{15} + (0.1Ra_D^{1/3})^{15}\right]^{1/15}}\right]

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
    [1]_ suggests this expression is valid for all cases except low-Pr fluids.
    [2]_ suggests no restrictions.

    Examples
    --------
    >>> Nu_horizontal_cylinder_Kuehn_Goldstein(0.69, 2.63E9)
    122.99323525628186

    References
    ----------
    .. [1] Kuehn, T. H., and R. J. Goldstein. "Correlating Equations for
       Natural Convection Heat Transfer between Horizontal Circular Cylinders."
       International Journal of Heat and Mass Transfer 19, no. 10
       (October 1976): 1127-34. doi:10.1016/0017-9310(76)90145-9
    .. [2] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    Nu = 2./log(1 + 2./((0.518*Ra**0.25*(1. + (0.559/Pr)**0.6)**(-5/12.))**15. + (0.1*Ra**(1/3.))**15)**(1/15.))
    return Nu


def Nu_horizontal_cylinder_Morgan(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a horizontal
    cylinder according to the Morgan [1]_ correlations, a product of a very
    large review of the literature. Sufficiently common as to be shown in [2]_.
    Cylinder must be isothermal.

    .. math::
        Nu_D = C Ra_D^n

    +----------+----------+-------+-------+
    |  Gr min  |  Gr max  |  C    |  n    |
    +==========+==========+=======+=======+
    | 10E-10   |  10E-2   | 0.675 | 0.058 |
    +----------+----------+-------+-------+
    | 10E-2    |  10E2    | 1.02  | 0.148 |
    +----------+----------+-------+-------+
    | 10E2     |  10E4    | 0.850 | 0.188 |
    +----------+----------+-------+-------+
    | 10E4     |  10E7    | 0.480 | 0.250 |
    +----------+----------+-------+-------+
    | 10E7     |  10E12   | 0.125 | 0.333 |
    +----------+----------+-------+-------+

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
    Most comprehensive review with a new proposed equation to date.
    Discontinuous among the jumps in range. Blindly runs outside if upper and
    lower limits without warning.

    Examples
    --------
    >>> [Nu_horizontal_cylinder_Morgan(.9, i) for i in (1E-2, 1E2, 1E4, 1E7, 1E10)]
    [0.5136293570857408, 1.9853087795801612, 4.707783879945983, 26.290682760247975, 258.0315247153301]
    >>> Nu_horizontal_cylinder_Morgan(0.69, 2.63E9)
    151.3881997228419

    References
    ----------
    .. [1] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [2] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if Ra < 1E-2:
        C, n = 0.675, 0.058
    elif Ra < 1E2:
        C, n = 1.02, 0.148
    elif Ra < 1E4:
        C, n = 0.850, 0.188
    elif Ra < 1E7:
        C, n = 0.480, 0.250
    else:
        # up to 1E12
        C, n = 0.125, 0.333
    Nu = C*Ra**n
    return Nu


horizontal_cylinder_correlations = {
'Churchill-Chu': (Nu_horizontal_cylinder_Churchill_Chu),
'Kuehn & Goldstein':  (Nu_horizontal_cylinder_Kuehn_Goldstein),
'Morgan': (Nu_horizontal_cylinder_Morgan)
}

def Nu_horizontal_cylinder(Pr=None, Gr=None,
                           AvailableMethods=False, Method=None):
    r'''This function handles choosing which horizontal cylinder free convection
    correlation is used. Generally this is used by a helper class, but can be
    used directly. Will automatically select the correlation to use if none is
    provided; returns None if insufficient information is provided.

    Prefered functions are 'Morgan' when discontinuous results are acceptable
    and 'Churchill-Chu' otherwise.

    Examples
    --------
    >>> Nu_horizontal_cylinder(0.72, 1E7)
    24.864192615468973

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
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to calculate Nu with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        horizontal_cylinder_correlations
    AvailableMethods : bool, optional
        If True, function will consider which methods which can be used to
        calculate Nu with the given inputs
    '''
    def list_methods():
        methods = []
        if all((Pr, Gr)):
            for key, values in horizontal_cylinder_correlations.items():
                    methods.append(key)
        methods.append('None')
        if 'Morgan' in methods:
            methods.remove('Morgan')
            methods.insert(0, 'Morgan')
        return methods

    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method in horizontal_cylinder_correlations:
        Nu = horizontal_cylinder_correlations[Method](Pr=Pr, Gr=Gr)
    elif Method == 'None':
        Nu = None
    else:
        raise Exception('Failure in in function')
    return Nu


#import matplotlib.pyplot as plt
#import numpy as np
#Pr, Gr = 0.72, 1E8
#methods = Nu_horizontal_cylinder(Pr, Gr, AvailableMethods=True)
#Grs = np.logspace(-2, 2.5, 10000)
#
#for method in methods:
#    Nus = [Nu_horizontal_cylinder(Pr=Pr, Gr=i, Method=method) for i in Grs]
#    plt.semilogx(Grs, Nus, label=method)
#plt.legend()
#plt.show()

