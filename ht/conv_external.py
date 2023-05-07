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

from math import exp

__all__ = ['Nu_cylinder_Zukauskas', 'Nu_cylinder_Churchill_Bernstein',
           'Nu_cylinder_Sanitjai_Goldstein', 'Nu_cylinder_Fand',
           'Nu_cylinder_Perkins_Leppert_1964',
           'Nu_cylinder_Perkins_Leppert_1962', 'Nu_cylinder_Whitaker',
           'Nu_cylinder_McAdams',
           'Nu_external_cylinder',
           'Nu_external_cylinder_methods',
           'Nu_horizontal_plate_laminar_Baehr',
           'Nu_horizontal_plate_laminar_Churchill_Ozoe',
           'Nu_horizontal_plate_turbulent_Schlichting',
           'Nu_horizontal_plate_turbulent_Kreith',
           'Nu_external_horizontal_plate',
           'Nu_external_horizontal_plate_methods',
           'LAMINAR_TRANSITION_HORIZONTAL_PLATE', 'conv_horizontal_plate_methods',
           ]

### Single Cylinders in Crossflow


def Nu_cylinder_Zukauskas(Re, Pr, Prw=None):
    r'''Calculates Nusselt number for crossflow across a single tube at a
    specified Re. Method from [1]_, also shown without modification in [2]_.
    This method applies to both the laminar and turbulent regimes.

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
    if Pr <= 10.0:
        n = 0.37
    else:
        n = 0.36
    Nu = c*Re**m*Pr**n
    if Prw is not None:
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

    This method applies to both the laminar and turbulent regimes.

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
    return 0.3 + (0.62*Re**0.5*Pr**(1/3.))/(1 + (0.4/Pr)**(2/3.))**0.25*(
    1 +(Re/282000.)**(0.625))**0.8


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

    This method applies to both the laminar and turbulent regimes.

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
    # Interesting numerical issue:
    # The power of the  -5 exp Re term is moved inside the exponential to
    # avoid overflow errors
    # This occurs easily with a large diameter cylinder (such as a vessel)
    return 0.446*Re**0.5*Pr**0.35 + 0.528*((6.5**-5*exp(-5*Re/5000.))
    + (0.031*Re**0.8)**-5)**-0.2*Pr**0.42


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

    This method applies to both the laminar and turbulent regimes.

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
    return (0.35 + 0.34*Re**0.5 + 0.15*Re**0.58)*Pr**0.3


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

    This method applies to both the laminar and turbulent regimes.

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
    return (0.35 + 0.56*Re**0.52)*Pr**0.3


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

    This method applies to both the laminar and turbulent regimes.

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
    if mu is not None and muw is not None:
        Nu *= (mu/muw)**0.25
    return Nu


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

    This method applies to both the laminar and turbulent regimes.

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
    if mu is not None and muw is not None:
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

    This method applies to both the laminar and turbulent regimes.

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
    if mu is not None and muw is not None:
        Nu *= (mu/muw)**0.25
    return Nu


conv_external_cylinder_turbulent_methods = {
    'Zukauskas': (Nu_cylinder_Zukauskas, ('Re', 'Pr', 'Prw')),
    'Churchill-Bernstein': (Nu_cylinder_Churchill_Bernstein, ('Re', 'Pr')),
    'Sanitjai-Goldstein': (Nu_cylinder_Sanitjai_Goldstein, ('Re', 'Pr')),
    'Fand': (Nu_cylinder_Fand, ('Re', 'Pr')),
    'McAdams': (Nu_cylinder_McAdams, ('Re', 'Pr')),
    'Whitaker': (Nu_cylinder_Whitaker, ('Re', 'Pr', 'mu', 'muw')),
    'Perkins-Leppert 1962': (Nu_cylinder_Perkins_Leppert_1962, ('Re', 'Pr', 'mu', 'muw')),
    'Perkins-Leppert 1964': (Nu_cylinder_Perkins_Leppert_1964, ('Re', 'Pr', 'mu', 'muw')),
}

conv_external_cylinder_turbulent_methods_ranked = ['Sanitjai-Goldstein',
                                                   'Churchill-Bernstein',
                                                   'Zukauskas', 'Whitaker',
                                                   'Perkins-Leppert 1964',
                                                   'McAdams',  'Fand',
                                                   'Perkins-Leppert 1962']

conv_external_cylinder_methods = conv_external_cylinder_turbulent_methods.copy()

_missing_external_cylinder_method = "Correlation name not recognized; the availble methods are %s." %(list(conv_external_cylinder_methods.keys()))


def Nu_external_cylinder_methods(Re, Pr, Prw=None, mu=None, muw=None, check_ranges=True):
    r'''This function returns a list of correlation names for forced convection
    over an external cylinder.

    The preferred method 'Sanitjai-Goldstein'.

    Parameters
    ----------
    Re : float
        Reynolds number of fluid with respect to cylinder diameter, [-]
    Pr : float
        Prandtl number at either the free stream or wall temperature
        depending on the method, [-]
    Prw : float, optional
        Prandtl number at wall temperature, [-]
    mu : float, optional
        Viscosity of fluid at the free stream temperature [Pa*s]
    muw : float, optional
        Viscosity of fluid at the wall temperature [Pa*s]
    check_ranges : bool, optional
        Whether or not to return only correlations suitable for the provided
        data, [-]

    Returns
    -------
    methods : list[str]
        List of methods which can be used to calculate `Nu` with the given
        inputs

    Examples
    --------
    >>> Nu_external_cylinder_methods(0.72, 1E7)[0]
    'Sanitjai-Goldstein'
    '''
    methods = ['Sanitjai-Goldstein', 'Churchill-Bernstein', 'Fand', 'McAdams']
    if Prw is not None:
        methods.append('Zukauskas')
    if mu is not None and muw is not None:
        methods.extend(['Whitaker', 'Perkins-Leppert 1964', 'Perkins-Leppert 1962'])
    return methods


def Nu_external_cylinder(Re, Pr, Prw=None, mu=None, muw=None, Method=None):
    r'''Calculates Nusselt number for crossflow across a single tube at a
    specified `Re` and `Pr` according to the specified method. Optional
    parameters are `Prw`, `mu`, and `muw`. This function has eight methods
    available. The 'Sanitjai-Goldstein' method is
    the default.

    The front of the cyliner is normally always in a laminar regime; whereas
    the back is turbulent. The proportions change with `Re`; all correlations
    take this into account. For this heat transfer case, there is no separation
    between laminar and turbulent methods.

    Parameters
    ----------
    Re : float
        Reynolds number of fluid with respect to cylinder diameter, [-]
    Pr : float
        Prandtl number at either the free stream or wall temperature
        depending on the method, [-]
    Prw : float, optional
        Prandtl number at wall temperature, [-]
    mu : float, optional
        Viscosity of fluid at the free stream temperature [Pa*s]
    muw : float, optional
        Viscosity of fluid at the wall temperature [Pa*s]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        conv_external_cylinder_methods.

    Notes
    -----
    A comparison of the methods for various Prandtl and Reynolds number ranges
    is plotted below.

    .. plot:: plots/Nu_external_cylinder.py

    Examples
    --------
    >>> Nu_external_cylinder(6071, 0.7)
    40.38327083519522
    '''
    Method2 = 'Sanitjai-Goldstein' if Method is None else Method

    if Method2 == 'Sanitjai-Goldstein':
        return Nu_cylinder_Sanitjai_Goldstein(Re=Re, Pr=Pr)
    elif Method2 == 'Churchill-Bernstein':
        return Nu_cylinder_Sanitjai_Goldstein(Re=Re, Pr=Pr)
    elif Method2 == 'Fand':
        return Nu_cylinder_Fand(Re=Re, Pr=Pr)
    elif Method2 == 'McAdams':
        return Nu_cylinder_McAdams(Re=Re, Pr=Pr)

    elif Method2 == 'Zukauskas':
        return Nu_cylinder_Zukauskas(Re=Re, Pr=Pr, Prw=Prw)
    elif Method2 == 'Whitaker':
        return Nu_cylinder_Whitaker(Re=Re, Pr=Pr, mu=mu, muw=muw)
    elif Method2 == 'Perkins-Leppert 1962':
        return Nu_cylinder_Perkins_Leppert_1962(Re=Re, Pr=Pr, mu=mu, muw=muw)
    elif Method2 == 'Perkins-Leppert 1964':
        return Nu_cylinder_Perkins_Leppert_1964(Re=Re, Pr=Pr, mu=mu, muw=muw)
    else:

        raise ValueError(_missing_external_cylinder_method)

# Horizontal Plate in crossflow

def Nu_horizontal_plate_laminar_Baehr(Re, Pr):
    r'''Calculates Nusselt number for laminar flow across an **isothermal**
    flat plate at a specified `Re` and `Pr`, both evaluated at the bulk
    temperature. No other wall correction is necessary for this formulation.
    Four different equations are used for different Prandtl number ranges.

    The equation for the common Prandtl number range is also recommended in
    [2]_ and [3]_.

    if :math:`\text{Pr} < 0.005`:

    .. math::
        \text{Nu}_L = 1.128\text{Re}^{0.5}\text{Pr}^{0.5}

    if :math:`0.005 < \text{Pr} < 0.05`:

    .. math::
        \text{Nu}_L = 1.0\text{Re}^{0.5}\text{Pr}^{0.5}

    if :math:`0.6 < \text{Pr} < 10`:

    .. math::
        \text{Nu}_L = 0.664\text{Re}^{0.5}\text{Pr}^{1/3}

    if :math:`\text{Pr} > 10`:

    .. math::
        \text{Nu}_L = 0.678\text{Re}^{0.5}\text{Pr}^{1/3}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to plate length and bulk fluid properties,
        [-]
    Pr : float
        Prandtl number at bulk temperature, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to plate length and bulk temperature, [-]

    Notes
    -----
    Does not take into account the impact of free convection, which can
    increase the convection substantially.

    Examples
    --------
    >>> Nu_horizontal_plate_laminar_Baehr(1e5, 0.7)
    186.4378528752262

    References
    ----------
    .. [1] Baehr, Hans Dieter, and Karl Stephan. Heat and Mass Transfer.
       Springer, 2013.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E.
       Hoboken, NJ: Wiley, 2011.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    '''
    if Pr < 0.005:
        return 1.128*(Re*Pr)**0.5
    elif Pr < 0.05:
        return (Re*Pr)**0.5
    elif Pr < 10.0:
        # Equation in VDI handbook, G4 as well
        return 0.664*Re**0.5*Pr**(1/3.)
    else:
        return 0.678*Re**0.5*Pr**(1/3.)


def Nu_horizontal_plate_laminar_Churchill_Ozoe(Re, Pr):
    r'''Calculates Nusselt number for laminar flow across an **isothermal**
    flat plate at a specified `Re` and `Pr`, both evaluated at the bulk
    temperature. No other wall correction is necessary for this formulation.
    A single equation covers all Prandtl number ranges.

    .. math::
        Nu_L = \frac{0.6774Re_L^{1/2}Pr^{1/3}}{[1+(0.0468/Pr)^{2/3}]^{1/4}}


    Parameters
    ----------
    Re : float
        Reynolds number with respect to plate length and bulk fluid properties,
        [-]
    Pr : float
        Prandtl number at bulk temperature, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to plate length and bulk temperature, [-]

    Notes
    -----
    Does not take into account the impact of free convection, which can
    increase the convection substantially.

    Examples
    --------
    >>> Nu_horizontal_plate_laminar_Churchill_Ozoe(1e5, 0.7)
    183.08600782591418

    References
    ----------
    .. [1] Churchill, Stuart W., and Hiroyuki Ozoe. "Correlations for Laminar
       Forced Convection in Flow Over an Isothermal Flat Plate and in
       Developing and Fully Developed Flow in an Isothermal Tube." Journal of
       Heat Transfer 95, no. 3 (August 1, 1973): 416
       https://doi.org/10.1115/1.3450078.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E.
       Hoboken, NJ: Wiley, 2011.
    '''
    return (0.6774*Re**(0.5)*Pr**(1/3.)
            *(1.0 + (0.0468/Pr)**(2.0/3.0))**-0.25 )


def Nu_horizontal_plate_turbulent_Schlichting(Re, Pr):
    r'''Calculates Nusselt number for turbulent flow across an **isothermal**
    flat plate at a specified `Re` and `Pr`, both evaluated at the bulk
    temperature. The formulation of Schlichting is used, which adds a
    surface friction term to a formulation from Petukhov and Popov.

    .. math::
        \text{Nu}_L = \frac{0.037\text{Re}_L^{0.8} \text{Pr}}
        {1 + 2.443\text{Re}_L^{-0.1}(\text{Pr}^{2/3} - 1)}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to plate length and bulk fluid properties,
        [-]
    Pr : float
        Prandtl number at bulk temperature, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to plate length and bulk temperature, [-]

    Notes
    -----
    Does not take into account the impact of free convection, which can
    increase the convection substantially.

    Examples
    --------
    >>> Nu_horizontal_plate_turbulent_Schlichting(1e5, 0.7)
    309.620048541267

    References
    ----------
    .. [1] Schlichting, H., and Klaus Gersten. Grenzschicht-Theorie. 9th ed.
       Berlin Heidelberg: Springer-Verlag, 1997.
       http://www.springer.com/de/book/9783662075548.
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    '''
    num = 0.037*Re**0.8*Pr
    den = (1.0 + 2.443*Re**-0.1*(Pr**(2.0/3.0) - 1.0))
    return num/den


def Nu_horizontal_plate_turbulent_Kreith(Re, Pr):
    r'''Calculates Nusselt number for turbulent flow across an **isothermal**
    flat plate at a specified `Re` and `Pr`, both evaluated at the bulk
    temperature. The formulation of Kreith is used.

    .. math::
        \text{Nu}_L = 0.036\text{Re}_L^{0.8} \text{Pr}^{2/3}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to plate length and bulk fluid properties,
        [-]
    Pr : float
        Prandtl number at bulk temperature, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to plate length and bulk temperature, [-]

    Notes
    -----
    Does not take into account the impact of free convection, which can
    increase the convection substantially. Applies for turbulent flow only.

    Examples
    --------
    >>> Nu_horizontal_plate_turbulent_Kreith(1.03e6, 0.71)
    2074.8740070411122

    References
    ----------
    .. [1] Kreith, Frank, Raj Manglik, and Mark Bohn. Principles of Heat
       Transfer. Cengage, 2010.
    '''
    return 0.036*Pr**(1.0/3.0)*Re**0.8


conv_horizontal_plate_laminar_methods = {
    'Baehr': (Nu_horizontal_plate_laminar_Baehr, ('Re', 'Pr')),
    'Churchill Ozoe': (Nu_horizontal_plate_laminar_Churchill_Ozoe, ('Re', 'Pr')),
}

conv_horizontal_plate_turbulent_methods = {
    'Schlichting': (Nu_horizontal_plate_turbulent_Schlichting, ('Re', 'Pr')),
    'Kreith': (Nu_horizontal_plate_turbulent_Kreith, ('Re', 'Pr')),
}

conv_horizontal_plate_methods = conv_horizontal_plate_laminar_methods.copy()
conv_horizontal_plate_methods.update(conv_horizontal_plate_turbulent_methods)

LAMINAR_TRANSITION_HORIZONTAL_PLATE = 5E5

def Nu_external_horizontal_plate_methods(Re, Pr, L=None, x=None,
                                   check_ranges=True):
    r'''Returns a list of correlation names for calculating Nusselt number for
    forced convection across a horizontal plate, supporting both laminar
    and turbulent regimes.

    Parameters
    ----------
    Re : float
        Reynolds number with respect to bulk properties and plate length, [-]
    Pr : float
        Prandtl number with respect to bulk properties, [-]
    L : float, optional
        Length of horizontal plate, [m]
    x : float, optional
        Length of horizontal plate for specific calculation distance, [m]
    check_ranges : bool, optional
        Whether or not to return only correlations suitable for the provided
        data, [-]

    Returns
    -------
    methods : list[str]
        List of methods which can be used to calculate `Nu` with the given
        inputs

    Examples
    --------
    >>> Nu_external_horizontal_plate_methods(Re=1e7, Pr=.7)[0]
    'Schlichting'
    '''
    turbulent = Re >= LAMINAR_TRANSITION_HORIZONTAL_PLATE
    if check_ranges:
        if turbulent:
            return ['Schlichting', 'Kreith']
        else:
            return ['Baehr', 'Churchill Ozoe']
    else:
        return ['Baehr', 'Churchill Ozoe', 'Schlichting', 'Kreith']

def Nu_external_horizontal_plate(Re, Pr, L=None, x=None, Method=None,
                                 laminar_method='Baehr',
                                 turbulent_method='Schlichting',
                                 Re_transition=LAMINAR_TRANSITION_HORIZONTAL_PLATE):
    r'''This function calculates the heat transfer coefficient for external
    forced convection along a horizontal plate.

    Requires at a minimum a flow's Reynolds and Prandtl numbers `Re` and `Pr`.
    `L` and `x` are not used by any correlations presently, but are included
    for future support.

    If no correlation's name is provided as `Method`, the most accurate
    applicable correlation is selected.

    Parameters
    ----------
    Re : float
        Reynolds number with respect to bulk properties and plate length, [-]
    Pr : float
        Prandtl number with respect to bulk properties, [-]
    L : float, optional
        Length of horizontal plate, [m]
    x : float, optional
        Length of horizontal plate for specific calculation distance, [m]

    Returns
    -------
    Nu : float
        Nusselt number with respect to plate length, [-]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        conv_horizontal_plate_methods
    laminar_method : str, optional
        The prefered method for laminar flow, [-]
    turbulent_method : str, optional
        The prefered method for turbulent flow, [-]
    Re_transition : float, optional
        The transition Reynolds number for laminar changing to turbulent flow,
        [-]

    Examples
    --------
    Turbulent example

    >>> Nu_external_horizontal_plate(Re=1E7, Pr=.7)
    11496.952599969829
    '''
    turbulent = not Re < Re_transition
    if Method is None:
        Method2 = turbulent_method if turbulent else laminar_method
    else:
        Method2 = Method

    if Method2 == 'Baehr':
        return Nu_horizontal_plate_laminar_Baehr(Re=Re, Pr=Pr)
    elif Method2 == 'Churchill Ozoe':
        return Nu_horizontal_plate_laminar_Churchill_Ozoe(Re=Re, Pr=Pr)
    elif Method2 == 'Schlichting':
        return Nu_horizontal_plate_turbulent_Schlichting(Re=Re, Pr=Pr)
    elif Method2 == 'Kreith':
        return Nu_horizontal_plate_turbulent_Kreith(Re=Re, Pr=Pr)
    else:
        raise ValueError("Correlation name not recognized; see the "
                        "documentation for the available options.")
