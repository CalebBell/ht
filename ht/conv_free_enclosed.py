'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from math import exp, log

from fluids.numerics import bisplev, horner, implementation_optimize_tck, secant

__all__ = ['Nu_Nusselt_Rayleigh_Holling_Herwig', 'Nu_Nusselt_Rayleigh_Probert',
           'Nu_Nusselt_Rayleigh_Hollands',
           'Rac_Nusselt_Rayleigh', 'Rac_Nusselt_Rayleigh_disk',
           'Nu_Nusselt_vertical_Thess',
           'Nu_vertical_helical_coil_Ali',
           'Nu_vertical_helical_coil_Prabhanjan_Rennie_Raghavan',
           ]

__numba_additional_funcs__ = ['Nu_Nusselt_Rayleigh_Holling_Herwig_err']


def Nu_Nusselt_Rayleigh_Holling_Herwig_err(Nu, Ra, Ra_third, D2):
    err = Ra_third*(0.1/2.0*log(1.0/16.0*Ra*Nu) + D2)**(-4.0/3.0) - Nu
    return err


def Nu_Nusselt_Rayleigh_Holling_Herwig(Pr, Gr, buoyancy=True):
    r'''Calculates the Nusselt number for natural convection between two
    theoretical flat horizontal plates. The height between the plates is infinite, and
    one of the other dimensions of the plates is much larger than the other.

    This correlation is for the horizontal plate Rayleigh-Benard classic heat
    transfer problem, not for real finite geometry plates.

    This model is a non-linear equation which is solved numerically.
    The model can calculate `Nu` for `Ra` ranges between 350 and larger
    numbers; [1]_ recommends :math:`10^{5} < Ra < 10^{15}`.

    .. math::
        \text{Nu} = \frac{{Ra}^{1/3}}{[0.05\ln(\frac{0.078}{16}{Ra}^{1.323})
        + 2D]^{4/3}}

    .. math::
        D = -\frac{14.94}{{Ra}^{0.25}} + 3.43

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - plate
        temperature difference [-]
    buoyancy : bool, optional
        Whether or not the plate's free convection is buoyancy assisted (hot
        plate) or not, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to height between the two plates, [-]

    Notes
    -----
    A range of calculated values are provided in [1]_; they all match the
    results of this function. This model is recommended in [2]_.

    For :math:`Ra < 1708`, `Nu` = 1; for cases not assited by `buoyancy`,
    `Nu` is also 1.

    No success has been found finding an analytical solution in the major CAS
    packages, but the nonlinear function is in fact a function of one variable;
    this means a pade or chebyshev expansion could be performed.


    Examples
    --------
    >>> Nu_Nusselt_Rayleigh_Holling_Herwig(5.54, 3.21e8, buoyancy=True)
    77.54656801896913

    References
    ----------
    .. [1] Hölling, M., and H. Herwig. "Asymptotic Analysis of Heat Transfer in
       Turbulent Rayleigh-Bénard Convection." International Journal of Heat and
       Mass Transfer 49, no. 5 (March 1, 2006): 1129-36.
       https://doi.org/10.1016/j.ijheatmasstransfer.2005.09.002.
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    '''
    if not buoyancy:
        return 1.0
    Rac = 1708 # Constant

    Ra = Gr*Pr
    if Ra < Rac:
        return 1.0

    Ra_third = Ra**(1.0/3.0)
    D2 = 2.0*(-14.94*Ra**-0.25 + 3.43)
    Nu_guess = Ra_third*(0.1/2.0*log(.078/16.0*Ra**1.323) + D2)**(-4.0/3.0)
    return secant(Nu_Nusselt_Rayleigh_Holling_Herwig_err, Nu_guess, args=(Ra, Ra_third, D2))


def Nu_Nusselt_Rayleigh_Probert(Pr, Gr, buoyancy=True):
    r'''Calculates the Nusselt number for natural convection between two
    theoretical flat plates. The height between the plates is infinite, and
    one of the other dimensions of the plates is much larger than the other.

    This correlation is for the horizontal plate Rayleigh-Benard classic heat
    transfer problem, not for real finite geometry plates.

    Two sets of equations are used.

    For the laminar regime :math:`1708 < \text{Ra} \le 2.2\times 10^{4}`:

    .. math::
        \text{Nu} = 0.208(\text{Ra})^{0.25}

    For the turbulent regime :math:`2.2\times 10^{4} < \text{Ra}`:

    .. math::
        \text{Nu} = 0.092(\text{Ra})^{1/3}

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - plate
        temperature difference [-]
    buoyancy : bool, optional
        Whether or not the plate's free convection is buoyancy assisted (hot
        plate) or not, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to height between the two plates, [-]

    Notes
    -----
    This model is recommended in [2]_ as a rough model.

    For :math:`Ra < 1708`, `Nu` = 1; for cases not assited by `buoyancy`,
    `Nu` is also 1.

    Examples
    --------
    >>> Nu_Nusselt_Rayleigh_Probert(5.54, 3.21e8, buoyancy=True)
    111.46181048289132

    References
    ----------
    .. [1] Probert, SD, RG Brooks, and M Dixon. "Heat Transfer across
       Rectangular Cavities." CHEMICAL AND PROCESS ENGINEERING, 1970, 35.
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    '''
    if not buoyancy:
        return 1.0
    Rac = 1708 # Constant

    Ra = Gr*Pr
    if Ra < Rac:
        return 1.0
    elif Ra < 2.2e4:
        return 0.208*Ra**0.25
    else:
        return 0.092*Ra**(1.0/3.0)


def Nu_Nusselt_Rayleigh_Hollands(Pr, Gr, buoyancy=True, Rac=1708):
    r'''Calculates the Nusselt number for natural convection between two
    theoretical flat horizontal plates using the Hollands [1]_ correlation recommended
    in [2]_. This correlation supports different aspect ratios,
    so the plates can be real, finite objects and have their heat transfer
    accurately modeled. The influence comes from the `Rac` term, which should
    be calculated separately, using `Rac_Nusselt_Rayleigh` or
    `Rac_Nusselt_Rayleigh_disk`.

    .. math::
        \text{Nu} = 1 + \left[1 - \frac{1708}{\text{Ra}} \right]^*
        \left[k_1 + 2 \left(\frac{\text{Ra}^{1/3}}{k_2} \right)^{1
        - \ln({\text{Ra}}^{1/5}/k_2)} \right]^*
        + \left[\left(\frac{\text{Ra}}{5803}\right)^{1/3} - 1\right]^*

    .. math::
        k_1 = \frac{1.44}{1 + 0.018/{Pr} + 0.00136/{Pr}^2}

    .. math::
        k_2 = 75\exp(1.5\text{Pr}^{-0.5})

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - plate
        temperature difference [-]
    buoyancy : bool, optional
        Whether or not the plate's free convection is buoyancy assisted (hot
        plate) or not, [-]
    Rac : float, optional
        Critical Rayleigh number, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to height between the two plates, [-]

    Notes
    -----
    For :math:`Ra < {Ra}_c`, `Nu` = 1; for cases not assited by `buoyancy`,
    `Nu` is also 1.

    Examples
    --------
    >>> Nu_Nusselt_Rayleigh_Hollands(5.54, 3.21e8, buoyancy=True)
    69.02668649510

    Plates - 1 m height, 2 m long, 0.2 m long vs a 1 m^3 cube

    >>> Nu_Nusselt_Rayleigh_Hollands(.7, 3.21e6, buoyancy=True, Rac=Rac_Nusselt_Rayleigh(H=1, L=2, W=.2, insulated=False))
    4.666249131876

    >>> Nu_Nusselt_Rayleigh_Hollands(.7, 3.21e6, buoyancy=True, Rac=Rac_Nusselt_Rayleigh(H=1, L=1, W=1, insulated=False))
    8.786362614129

    References
    ----------
    .. [1] Hollands, K. G. T. "Multi-Prandtl Number Correlation Equations for
       Natural Convection in Layers and Enclosures." International Journal of
       Heat and Mass Transfer 27, no. 3 (March 1, 1984): 466-68.
       https://doi.org/10.1016/0017-9310(84)90295-3.
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    '''
    if not buoyancy:
        return 1.0
    Ra = Gr*Pr
    if Ra < Rac:
        return 1.0

    k1 = 1.44/(1.0 + 0.018/Pr + 0.00136/(Pr*Pr))
    k2 = 75*exp(1.5*Pr**-0.5)

    t1 = (1.0 - Rac/Ra)
    t2 = k1 + 2.0*(Ra**(1.0/3.0)/k2)**(1.0 - log(Ra**(1.0/3.0)/k2))
    t3 = (Ra/5803.0)**(1.0/3.0) - 1.0

    if Rac != 1708:
        t4 = max(0.0, (Ra/Rac)**(1.0/3.0) - 1.0)
        t5 = (1.0 - exp(-0.95*t4))
    else:
        t5 = 1.0

    Nu = 1.0 + max(0.0, t1)*max(0.0, t2) + max(0.0, t3)*t5
    return Nu


def Nu_Nusselt_vertical_Thess(Pr, Gr, H=None, L=None):
    r'''Calculates the Nusselt number for natural convection between two
    theoretical vertical flat plates using the correlation by Thess [1]
    in [1]_. This is a variant on the horizontal Rayleigh-Benard classic heat
    transfer problem.
    This correlation supports different aspect ratios,
    so the plates can be real, finite objects and have their heat transfer
    accurately modeled. The recommended range of the correlation is H/L < 80.

    For 1e4 < Ra < 1e7:

    .. math::
        \text{Nu} = 0.42{Pr}^{0.012} {Ra}^{0.25} \left(\frac{H}{L}\right)^{-0.25}

    For 1e7 < Ra > 1e9 (or when geometry is unknown):

    .. math::
         \text{Nu} = 0.049{Ra}^{0.33}

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - plate
        temperature difference [-]
    H : float, optional
        Height of vertical plate, [m]
    L : float, optional
        Length of vertical plate, [m]

    Returns
    -------
    Nu : float
        Nusselt number with respect to distance between the two plates, [-]

    Examples
    --------
    >>> Nu_Nusselt_vertical_Thess(.7, 3.21e6)
    6.112587569602785

    >>> Nu_Nusselt_vertical_Thess(.7, 3.21e6, L=10, H=1)
    28.79328626041646

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    '''
    Ra = Gr*Pr
    if Ra < 1e7 and H is not None and L is not None:
        return 0.42*Pr**0.012*Ra**0.25*(L/H)**0.25
    return 0.049*Ra**0.33



ratios_uninsulated_Catton = [0.125, 0.25, 0.5, 1, 2, 3, 4, 5, 6]
Racs_uninstulated_Catton = [[9802960, 1554480, 606001, 469377, 444995, 444363, 457007, 473725, 494741],
[1554480, 638754, 115596, 64270.8, 53529.7, 50816.4, 50136.1, 50088.7, 50410.1],
[606001, 115596, 48178.9, 14615.3, 11374.5, 9831.6, 9312, 9099.4, 8980.2],
[469377, 64270.8, 14615.3, 6974, 5138.2, 3906, 3633.6, 3446.2, 3358],
[444995, 53529.7, 11374.5, 5137.9, 3773.6, 2753.6, 2530.5, 2359.5, 2285.7],
[444363, 50816.4, 9831.6, 3906, 2753, 2557.4, 2337.2, 2174.44, 2101],
[457007, 50136.1, 9311.9, 3633.6, 2530.5, 2337.2, 2270.2, 2110.9, 2037.2],
[473725, 50088.6, 9099.4, 3446.2, 2359.5, 2174.4, 2110.9, 2081.7, 2007.8],
[494742, 50410.1, 8980.2, 3357.9, 2285.7, 2100.9, 2037.2, 2007.8, 1991.9]]

tck_uninstulated_Catton = implementation_optimize_tck([[0.125, 0.125, 0.125, 0.125, 0.41375910864088195,
                              0.5819413331927507, 1.9885569998423345, 2.8009586482973834,
                              3.922852887459219, 6.0, 6.0, 6.0, 6.0],
                             [0.125, 0.125, 0.125, 0.125, 0.4180739258304788,
                              0.6521218159098487, 1.4270223336187269,
                              2.89426640315332, 3.9239774081390215,
                              6.0, 6.0, 6.0, 6.0],
 [16.098194938851986, 14.026983058722742, 13.35866942808268, 13.043296359953983,
  13.008470795621905, 12.991279831677808, 13.040841344665466, 13.07803101947673,
  13.111789672293794, 14.074352449019207, 14.878522936155216, 11.151352953023258,
  11.096394321545977, 10.813773781060574, 10.796217122120712, 10.78189560829848,
  10.774336865714089, 10.78004622910552, 13.400086198278455, 11.369928815173187,
  11.82067779495709, 9.6860949637944, 9.686120336218499, 9.50952376562826,
  9.444619552074945, 9.452058024482865, 9.441608909473647, 12.933722760010111,
  10.873615956186896, 8.971126166473885, 8.520162104980807, 8.317346176887659,
  7.837750498437191, 7.78951404473208, 7.690715685713949, 7.695209247397283,
  13.025815591825872, 10.75723159025179, 9.734653433466208, 8.569056561731081,
  8.77031704228521, 7.853798846698488, 7.939088236475908, 7.748880239519593,
  7.785611785518214, 12.992898431724237, 10.728320934519346, 9.37520794405935,
  8.247995842200584, 7.753730020752022, 7.937553314495094, 7.6598493250444255,
  7.673199977054488, 7.63790748099515, 13.041869920313422, 10.713059500923494,
  9.364505568407685, 8.18000143764639, 7.927764179244221, 7.660718938605501,
  7.85174473958641, 7.5354388646400965, 7.614740168201775, 13.077057211283323,
  10.706667262420716, 9.341451094646674, 8.122270764822368, 7.671593316397699,
  7.697000470802994, 7.530680469875164, 7.720180133976149, 7.59900173760075,
  13.111791693551362, 10.711047679739433, 9.339770955175847, 8.117021359757253,
  7.727537757463738, 7.654072928976537, 7.607359118625173, 7.602197791148399,
  7.596844236081228], 3, 3])

ratios_insulated_Catton = [0.125, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 12]
Racs_instulated_Catton = [[3011718, 333013, 70040, 37689, 39798, 36262, 37058, 35875, 36209, 35664, 35794, 35486, 35556, 35380, 35451, 35193],
[333013, 203163, 28452, 11962, 12540, 11020, 11251, 10757, 10858, 10635, 10666, 10544, 10571, 10499, 10518, 10426],
[70040, 28452, 17307, 5262, 5341, 4524, 4567, 4330, 4355, 4245, 4261, 4186, 4196, 4158, 4165, 4118],
[37689, 11962, 5262, 3446, 3270, 2789, 2754, 2622, 2609, 2552, 2545, 2502, 2498, 2480, 2447, 2453],
[39798, 12540, 5341, 3270, None, None, None, None, None, None, None, None, None, None, None, None],
[36262, 11020, 4524, 2789, None, 2276, 2222, 2121, 2098, 2057, 2044, 2009, 2001, 1989, 1984, 1967],
[37058, 11251, 4567, 2754, None, 2222, None, None, None, None, None, None, None, None, None, None],
[35875, 10757, 4330, 2622, None, 2121, None, 2004, 1978, 1941, 1927, 1897, 1888, 1879, 1871, 1855],
[36209, 10858, 4355, 2609, None, 2098, None, 1978, None, None, None, None, None, None, None, None],
[35664, 10635, 4245, 2552, None, 2057, None, 1941, None, 1894, 1878, 1852, 1842, 1833, 1826, 1808],
[35794, 10666, 4261, 2545, None, 2044, None, 1927, None, 1878, None, None, None, None, None, None],
[35486, 10544, 4186, 2502, None, 2009, None, 1897, None, 1852, None, None, None, 1810, 1803, 1783],
[35556, 10571, 4196, 2498, None, 2001, None, 1888, None, 1842, None, None, None, None, None, None],
[35380, 10499, 4158, 2480, None, 1989, None, 1879, None, 1833, None, 1810, None, 1797, 1789, 1768],
[35451, 10518, 4165, 2447, None, 1984, None, 1871, None, 1826, None, 1803, None, 1789, None, None],
[35193, 10426, 4118, 2453, None, 1967, None, 1855, None, 1808, None, 1783, None, 1768, None, 1741]]


tck_insulated_Catton = implementation_optimize_tck([[0.125, 0.125, 0.2165763979498294, 0.25, 0.4948545767149843,
                                                     0.8432690088415454, 2.297018168305444, 5.324310151069744, 12.0, 12.0],
 [0.125, 0.125, 0.125, 0.37135574365684176, 0.8160817162671293, 1.1103105500488575,
  1.9000136398530074, 3.521092600950009, 12.0, 12.0, 12.0],
  [14.917942380813974, 12.196391449028951, 10.665084931671647, 10.531834082947338,
   10.57637568816619, 10.486173564722383, 10.471864979770599, 10.468190753935556,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.715841947316376, 12.462417612931137,
   9.174421085152083, 9.411191211042704, 9.409695481542864, 9.28122664900159,
   9.249608368005552, 9.251639244971427, 11.165512470689693, 10.01308504970903,
   9.75292707527754, 8.509349912597454, 8.566854764542974, 8.372517445356857,
   8.32618713246236, 8.329704835832104, 10.56848779064929, 9.163970117017675,
   8.369187019066972, 8.19799054440329, 8.087508877612247, 7.896372367041187,
   7.806891615973793, 7.835687464634469, 10.509836235182163, 9.041210689705586,
   8.118960504225761, 7.909354018896528, 7.735269232380504, 7.614379036546508,
   7.4775491512154515, 7.529024952770015, 10.474423221467699, 8.98482837851057,
   8.036532362247245, 7.822308882170893, 7.6362269726600065, 7.539826337638537,
   7.459554042916101, 7.480930154132415, 10.469149134470264, 8.978694786931275,
   8.024134988827441, 7.811393154091167, 7.627457342156321, 7.521833838146938,
   7.4376750879045455, 7.462202956737165], 1, 2])


def Rac_Nusselt_Rayleigh(H, L, W, insulated=True):
    r'''Calculates the critical Rayleigh number for free convection to begin
    in the Nusselt-Rayleigh parallel horizontal plate scenario. There are
    actually two cases - one for the top plate to be insulated (adiabatic) and
    the other where it has infinite thermal conductivity/is infinitely thin or
    not present (perfectly conducting). All real cases will lie between the
    two.

    Parameters
    ----------
    H : float
        Distance between the two plates, [m]
    L : float
        Length of the plates, [m]
    W : float
        Width of the plates, [m]
    insulated : bool, optional
        Whether the top plate is insulated or uninsulated, [-]

    Returns
    -------
    Rac : float
        Critical Rayleigh number, [-]

    Examples
    --------
    >>> Rac_Nusselt_Rayleigh(1, .5, 2, False)
    2530.500000000005
    >>> Rac_Nusselt_Rayleigh(1, .5, 2, True)
    2071.0089443385655

    Notes
    -----
    Splines have been fit to data in [1]_ for the uninsulated case and [2]_
    for the insulated case. The data is presented in the original papers and
    in [3]_.

    References
    ----------
    .. [1] Catton, Ivan. "Effect of Wall Conduction on the Stability of a Fluid
       in a Rectangular Region Heated from Below." Journal of Heat Transfer 94,
       no. 4 (November 1, 1972): 446-52. https://doi.org/10.1115/1.3449966.
    .. [2] Catton, Ivan. "Convection in a Closed Rectangular Region: The Onset
       of Motion." Journal of Heat Transfer 92, no. 1 (February 1, 1970):
       186-88. https://doi.org/10.1115/1.3449626.
    .. [3] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    H_L_ratio = min(max(H/L, 0.125), 12.0)
    W_L_ratio = min(max(W/L, 0.125), 12.0)

    if insulated:
        Rac = exp(bisplev(W_L_ratio, H_L_ratio, tck_insulated_Catton))
    else:
        Rac = exp(bisplev(W_L_ratio, H_L_ratio, tck_uninstulated_Catton))
    return Rac


uninsulated_disk_coeffs = [1.3624571738082523, -0.24301326192178863, -6.152310426160362,
                           1.1950540229805053, 11.401090141352329, -2.405543860763877,
                           -11.091871509655324, 2.519761389270987, 5.992609902331248,
                           -1.4345227368881952, -1.7445130176764998, 0.42892571421446996,
                           0.22897205478499438, -0.042179780698649895, -0.01904413256783342,
                           0.006771075600246057, 0.13171026423861615]


insulated_disk_coeffs = [0.2173851248644496, 0.09672312658254612, -1.0800494968302843,
                         -0.3323452633903514, 2.1789014174652115, 0.43391756058946473,
                         -2.275756526433769, -0.29309565826688255, 1.3153930583762103,
                         0.14707146242791974, -0.44891166228441826, -0.045070571352735386,
                         0.08693822836596571, 0.010343944709216, -0.01325209778273359,
                         0.0035707992137628142, 0.13258956599554672]


def Rac_Nusselt_Rayleigh_disk(H, D, insulated=True):
    r'''Calculates the critical Rayleigh number for free convection to begin
    in the parallel horizontal disk scenario. There are
    actually two cases - one for the top plate to be insulated (adiabatic) and
    the other where it has infinite thermal conductivity/is infinitely thin or
    not present (perfectly conducting). All real cases will lie between the
    two.

    Parameters
    ----------
    H : float
        Distance between the two disks, [m]
    D : float
        Diameter of the two disks, [m]
    insulated : bool, optional
        Whether the top plate is insulated or uninsulated, [-]

    Returns
    -------
    Rac : float
        Critical Rayleigh number, [-]

    Examples
    --------
    >>> Rac_Nusselt_Rayleigh_disk(H=1, D=.4, insulated=False)
    151199.9999999945

    >>> Rac_Nusselt_Rayleigh_disk(H=1, D=4, insulated=False)
    1891.520931853363

    >>> Rac_Nusselt_Rayleigh_disk(2, 1, True)
    24347.31479211917

    Notes
    -----
    The range of data covered by this function is `D`/`H` from 0.4 to infinity.
    As inifinity is not well suited to polynomial form, the upper limit is
    6 in actuality. Values outside that range are rounded to the limits.

    This function provides 17-coefficient polynomial fits to interpolate in the
    table of values in [1]_. The source of the coefficients is cited as being
    from [2]_.

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Buell, J. C., and I. Catton. "The Effect of Wall Conduction on the
       Stability of a Fluid in a Right Circular Cylinder Heated From Below."
       Journal of Heat Transfer 105, no. 2 (May 1, 1983): 255-60.
       https://doi.org/10.1115/1.3245571.
    '''
    x = min(max(D/H, 0.4), 6.0)
    if insulated:
        coeffs = insulated_disk_coeffs
    else:
        coeffs = uninsulated_disk_coeffs
    return exp(1.0/horner(coeffs, 0.357142857142857151*(x - 3.2)))


### Free convection vertical helical coil

def Nu_vertical_helical_coil_Ali(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a vertical
    helical coil inside a tank or other vessel according to the Ali [1]_
    correlation.

    .. math::
        Nu_L = 0.555Gr_L^{0.301} Pr^{0.314}

    Parameters
    ----------
    Pr : float
        Prandtl number of the fluid surrounding the coil with properties
        evaluated at bulk conditions or as described in the notes [-]
    Gr : float
        Prandtl number of the fluid surrounding the coil with properties
        evaluated at bulk conditions or as described in the notes
        (for the two temperatures, use the average coil fluid temperature and
        the temperature of the fluid outside the coil) [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to the total length of the helical coil
        (and bulk thermal conductivity), [-]

    Notes
    -----
    In [1]_, the temperature at which the fluid surrounding the coil's
    properties were evaluated at was calculated in an unusual fashion. The
    average temperature of the fluid inside the coil
    :math:`(T_{in} + T_{out})/2` is averaged with the fluid outside the coil's
    temperature.

    The correlation is valid for Prandtl numbers between 4.4 and 345,
    and tank diameter/coil outer diameter ratios between 10 and 30.

    Examples
    --------
    >>> Nu_vertical_helical_coil_Ali(4.4, 1E11)
    1808.5774997297106

    References
    ----------
    .. [1] Ali, Mohamed E. "Natural Convection Heat Transfer from Vertical
       Helical Coils in Oil." Heat Transfer Engineering 27, no. 3 (April 1,
       2006): 79-85.
    '''
    return 0.555*Gr**0.301*Pr**0.314


def Nu_vertical_helical_coil_Prabhanjan_Rennie_Raghavan(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a vertical
    helical coil inside a tank or other vessel according to the Prabhanjan,
    Rennie, and Raghavan [1]_ correlation.

    .. math::
        Nu_H = 0.0749\text{Ra}_H^{0.3421}

    The range of Rayleigh numbers is as follows:

    .. math::
        9 \times 10^{9} < \text{Ra} < 4 \times 10^{11}

    Parameters
    ----------
    Pr : float
        Prandtl number calculated with the film temperature -
        wall and temperature very far from the coil average, [-]
    Gr : float
        Grashof number calculated with the film temperature -
        wall and temperature very far from the coil average,
        and using the total height of the coil [-]

    Returns
    -------
    Nu : float
        Nusselt number using the total height of the coil
        and the film temperature, [-]

    Notes
    -----
    [1]_ also has several other equations using different characteristic
    lengths.

    Examples
    --------
    >>> Nu_vertical_helical_coil_Prabhanjan_Rennie_Raghavan(4.4, 1E11)
    720.6211067718227

    References
    ----------
    .. [1] Prabhanjan, Devanahalli G., Timothy J. Rennie, and G. S. Vijaya
       Raghavan. "Natural Convection Heat Transfer from Helical Coiled Tubes."
       International Journal of Thermal Sciences 43, no. 4 (April 1, 2004):
       359-65.
    '''
    Ra = Pr*Gr
    return 0.0749*Ra**0.3421
