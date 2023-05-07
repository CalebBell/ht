'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from math import exp, pi, radians, sin

from fluids.numerics import bisplev, horner, implementation_optimize_tck, splev
from fluids.numerics import numpy as np

from ht.core import WALL_FACTOR_PRANDTL, wall_factor

__all__ = ['dP_Kern', 'dP_Zukauskas',
           'Nu_ESDU_73031', 'Nu_Zukauskas_Bejan','Nu_HEDH_tube_bank',
           'Nu_Grimison_tube_bank',
           'Zukauskas_tube_row_correction',
           'ESDU_tube_row_correction',
           'ESDU_tube_angle_correction',
           'baffle_correction_Bell', 'baffle_leakage_Bell',
           'bundle_bypassing_Bell', 'unequal_baffle_spacing_Bell',
           'laminar_correction_Bell']

__numba_additional_funcs__ = ['Grimison_C1_aligned_interp', 'Grimison_m_aligned_interp',
                              'Grimson_C1_staggered_interp', 'Grimson_m_staggered_interp',
                              'Kern_f_Re', 'Bell_baffle_configuration_obj', 'Bell_baffle_leakage_obj',
                              'Bell_bundle_bypass_low_obj', 'Bell_bundle_bypass_high_obj']

try:
    IS_NUMBA # type: ignore # noqa: F821
except:
    IS_NUMBA = False
# Applies for row 1-9.
Grimson_Nl_aligned = [0.64, 0.8, 0.87, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]
Grimson_Nl_staggered = [0.68, 0.75, 0.83, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99]


Grimison_SL_aligned = [1.25, 1.5, 2, 3]
Grimison_ST_aligned = Grimison_SL_aligned
Grimison_C1_aligned = [[0.348, 0.275, 0.1, 0.0633],
                                [0.367, 0.25, 0.101, 0.0678],
                                [0.418, 0.299, 0.229, 0.198],
                                [0.29, 0.357, 0.374, 0.286]]
Grimison_m_aligned = [[0.592, 0.608, 0.704, 0.752],
                               [0.586, 0.62, 0.702, 0.744],
                               [0.57, 0.602, 0.632, 0.648],
                               [0.601, 0.584, 0.581, 0.608]]

Grimison_C1_aligned_tck = implementation_optimize_tck([[1.25, 1.25, 1.25, 1.25, 3.0, 3.0, 3.0, 3.0],
                           [1.25, 1.25, 1.25, 1.25, 3.0, 3.0, 3.0, 3.0],
                           [0.34800000000000003, 0.20683194444444492, -0.18023055555555617,
                            0.06330000000000001, 0.3755277777777776, -0.28351037808642043,
                            0.24365763888889008, -0.0007166666666667326, 0.5481111111111114,
                            0.2925767746913588, 0.8622214506172828, 0.5207777777777779, 0.29,
                            0.5062500000000002, 0.26944444444444426, 0.286],
                            3, 3], force_numpy=IS_NUMBA)

Grimison_C1_aligned_interp = lambda x, y : float(bisplev(x, y, Grimison_C1_aligned_tck))


Grimison_m_aligned_tck = implementation_optimize_tck([[1.25, 1.25, 1.25, 1.25, 3.0, 3.0, 3.0, 3.0],
                          [1.25, 1.25, 1.25, 1.25, 3.0, 3.0, 3.0, 3.0],
                          [0.5920000000000001, 0.5877777777777775, 0.9133333333333344,
                           0.752, 0.5828472222222219, 0.7998613040123475,
                           0.7413584104938251, 0.7841111111111112, 0.5320833333333332,
                           0.5504147376543196, 0.30315663580247154, 0.4148888888888891,
                           0.601, 0.5454861111111109, 0.6097500000000002, 0.608],
                           3, 3], force_numpy=IS_NUMBA)
Grimison_m_aligned_interp = lambda x, y : float(bisplev(x, y, Grimison_m_aligned_tck))


Grimson_SL_staggered = [1.25, 1.5, 2, 3, 1, 1.25, 1.5, 2, 3, 0.9,
                                 1.125, 1.25, 1.5, 2, 3, 0.6, 0.9, 1.125, 1.25,
                                 1.5, 2, 3]

Grimson_ST_staggered = [1.25, 1.25, 1.25, 1.25, 1.5, 1.5, 1.5, 1.5,
                                 1.5, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]

Grimson_m_staggered = [0.556, 0.568, 0.572, 0.592, 0.558, 0.554,
                                0.562, 0.568, 0.58, 0.571, 0.565, 0.556, 0.568,
                                0.556, 0.562, 0.636, 0.581, 0.56, 0.562, 0.568,
                                0.57, 0.574]

Grimson_C1_staggered = [0.518, 0.451, 0.404, 0.31, 0.497, 0.505, 0.46,
                                 0.416, 0.356, 0.446, 0.478, 0.519, 0.452,
                                 0.482, 0.44, 0.213, 0.401, 0.518, 0.522,
                                 0.488, 0.449, 0.428]

"""`interp2d` creates warnings when used on these. They are avoided by
pre-generating the splines, and interfacing with fitpack at a lower level.
"""
tck_Grimson_m_staggered = implementation_optimize_tck([[1.25, 1.25, 1.8667584356619125, 2.0, 2.8366905775206916, 3.0, 3.0],
     [0.6, 0.6, 1.0085084989709654, 1.340729148958038, 1.5154196399508033, 3.0, 3.0],
     [1.731351706314169, 0.3675823638826614, 0.6267891238439347, 0.5623083927989683, 0.5920000000000982, 1.180171700201992,
               0.7874995409316767, 0.4622370503994375, 0.562004066622535, 0.5623955950882191, 0.5680620929528815, 0.5720626262793304,
               0.5510099520872309, 0.5641771077227365, 0.5597975310692721, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6361653765016168,
               0.5601991640778442, 0.5621224100266599, 0.5684014375982079, 0.573932491076899],
    1, 1], force_numpy=IS_NUMBA)

tck_Grimson_C1_staggered = implementation_optimize_tck([[1.25, 1.25, 1.936293121624252, 2.0, 2.094408820089069, 3.0, 3.0],
    [0.6, 0.6, 1.1841422334268308, 1.3897531616318943, 1.6483901017748916, 3.0, 3.0],
    [0.534042720665836, 0.5446897215451869, 0.4613632028066018, 0.4370513304331604, 0.31000000000000005, 0.3060114256888106,
              0.4719357486311919, 0.5043332405690643, 0.4371755864391464, 0.4362779343788622, 0.364660449991649, 0.5144234623651529,
              0.4513822953351327, 0.4852710459180796, 0.4420724694173403, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21898644381978172,
              0.5500312131715677, 0.4969529176876636, 0.46150347905703587, 0.4270770845430577],
    1, 1], force_numpy=IS_NUMBA)

Grimson_m_staggered_interp = lambda x, y: float(bisplev(x, y, tck_Grimson_m_staggered))
Grimson_C1_staggered_interp = lambda x, y: float(bisplev(x, y, tck_Grimson_C1_staggered))



def Nu_Grimison_tube_bank(Re, Pr, Do, tube_rows, pitch_parallel, pitch_normal):
    r'''Calculates Nusselt number for crossflow across a tube bank
    of tube rows at a specified `Re`, `Pr`, and `D` using the Grimison
    methodology as described in [1]_.

    .. math::
        \bar{Nu_D} = 1.13C_1Re_{D,max}^m Pr^{1/3}C_2

    Parameters
    ----------
    Re : float
        Reynolds number with respect to average (bulk) fluid properties and
        tube outside diameter, [-]
    Pr : float
        Prandtl number with respect to average (bulk) fluid properties, [-]
    Do : float
        Tube outer diameter, [m]
    tube_rows : int
        Number of tube rows per bundle, [-]
    pitch_parallel : float
        Distance between tube center along a line parallel to the flow;
        has been called `longitudinal` pitch, `pp`, `s2`, `SL`, and `p2`, [m]
    pitch_normal : float
        Distance between tube centers in a line 90° to the line of flow;
        has been called the `transverse` pitch, `pn`, `s1`, `ST`, and `p1`, [m]

    Returns
    -------
    Nu : float
        Nusselt number with respect to tube outside diameter, [-]

    Notes
    -----
    Tube row correction factors are applied for tube row counts less than 10,
    also published in [1]_.

    Examples
    --------
    >>> Nu_Grimison_tube_bank(Re=10263.37, Pr=.708, tube_rows=11,
    ... pitch_normal=.05, pitch_parallel=.05, Do=.025)
    79.07883866010

    >>> Nu_Grimison_tube_bank(Re=10263.37, Pr=.708, tube_rows=11,
    ... pitch_normal=.07, pitch_parallel=.05, Do=.025)
    79.92721078571

    References
    ----------
    .. [1] Grimson, E. D. (1937) Correlation and Utilisation of New Data on
       Flow Resistance and Heat Transfer for Cross Flow of Gases over Tube
       Banks. Trans. ASME. 59 583-594
    '''
    staggered = abs(1 - pitch_normal/pitch_parallel) > 0.05
    a = pitch_normal/Do # sT
    b = pitch_parallel/Do
    if not staggered:
        C1 = float(bisplev(b, a, Grimison_C1_aligned_tck))
        m = float(bisplev(b, a, Grimison_m_aligned_tck))
    else:
        C1 = float(bisplev(b, a, tck_Grimson_C1_staggered))
        m = float(bisplev(b, a, tck_Grimson_m_staggered))

    tube_rows = int(tube_rows)
    if tube_rows < 10:
        if tube_rows < 1:
            tube_rows = 1
        if staggered:
            C2 = Grimson_Nl_staggered[tube_rows]
        else:
            C2 = Grimson_Nl_aligned[tube_rows]
    else:
        C2 = 1.0
    Nu = 1.13*Re**m*Pr**(1.0/3.0)*C2*C1
    return Nu


Zukauskas_Czs_low_Re_staggered = [0.8295, 0.8792, 0.9151, 0.9402, 0.957, 0.9677,
    0.9745, 0.9785, 0.9808, 0.9823, 0.9838, 0.9855, 0.9873, 0.9891, 0.991,
    0.9929, 0.9948, 0.9967, 0.9987]
Zukauskas_Czs_high_Re_staggered = [0.6273, 0.7689, 0.8473, 0.8942, 0.9254,
    0.945, 0.957, 0.9652, 0.9716, 0.9765, 0.9803, 0.9834, 0.9862, 0.989,
    0.9918, 0.9943, 0.9965, 0.998, 0.9986]
Zukauskas_Czs_inline = [0.6768, 0.8089, 0.8687, 0.9054, 0.9303, 0.9465, 0.9569,
    0.9647, 0.9712, 0.9766, 0.9811, 0.9847, 0.9877, 0.99, 0.992, 0.9937,
    0.9953, 0.9969, 0.9986]

def Zukauskas_tube_row_correction(tube_rows, staggered=True, Re=1E4):
    r'''Calculates the tube row correction factor according to a graph
    digitized from [1] for heat transfer across
    a tube bundle. The correction factors are slightly different for
    staggered vs. inline configurations; for the staggered configuration,
    factors are available separately for `Re` larger or smaller than 1000.

    This method is a tabular lookup, with values of 1 when the tube row count
    is 20 or more.

    Parameters
    ----------
    tube_rows : int
        Number of tube rows per bundle, [-]
    staggered : bool, optional
        Whether in the in-line or staggered configuration, [-]
    Re : float, optional
        The Reynolds number of flow through the tube bank using the bare tube
        outer diameter and the minimum flow area through the bundle, [-]

    Returns
    -------
    F : float
        Tube row count correction factor, [-]

    Notes
    -----
    The basis for this method is that an infinitely long tube bank has a
    factor of 1; in practice the factor is reached at 20 rows.

    Examples
    --------
    >>> Zukauskas_tube_row_correction(4, staggered=True)
    0.8942
    >>> Zukauskas_tube_row_correction(6, staggered=False)
    0.9465

    References
    ----------
    .. [1] Zukauskas, A. Heat transfer from tubes in crossflow. In T.F. Irvine,
       Jr. and J. P. Hartnett, editors, Advances in Heat Transfer, volume 8,
       pages 93-160. Academic Press, Inc., New York, 1972.
    '''
    tube_rows = int(tube_rows) # sanity for indexing
    if tube_rows < 1:
        tube_rows = 1
    if staggered: # in-line, with a tolerance of 0.05 proximity
        if tube_rows <= 19:
            factors = Zukauskas_Czs_low_Re_staggered if Re < 1000 else Zukauskas_Czs_high_Re_staggered
            correction = factors[tube_rows-1]
        else:
            correction = 1.0
    else:
        if tube_rows <= 19:
            correction = Zukauskas_Czs_inline[tube_rows-1]
        else:
            correction = 1.0
    return correction


def Nu_Zukauskas_Bejan(Re, Pr, tube_rows, pitch_parallel, pitch_normal,
                       Pr_wall=None):
    r'''Calculates Nusselt number for crossflow across a tube bank
    of tube number n at a specified `Re` according to the method of Zukauskas
    [1]_. A fit to graphs from [1]_ published in [2]_ is used for the
    correlation. The tube row correction factor is obtained from digitized
    graphs from [1]_, and a lookup table was created and is used for speed.

    The formulas are as follows:

    Aligned tube banks:

    .. math::
        \bar Nu_D = 0.9 C_nRe_D^{0.4}Pr^{0.36}\left(\frac{Pr}{Pr_w}\right)^{0.25}
        \text{ for } 1 < Re < 100

    .. math::
        \bar Nu_D = 0.52 C_nRe_D^{0.5}Pr^{0.36}\left(\frac{Pr}{Pr_w}\right)^{0.25}
        \text{ for } 100 < Re < 1000

    .. math::
        \bar Nu_D = 0.27 C_nRe_D^{0.63}Pr^{0.36}\left(\frac{Pr}{Pr_w}\right)^{0.25}
        \text{ for } 1000 < Re < 20000

    .. math::
        \bar Nu_D = 0.033 C_nRe_D^{0.8}Pr^{0.36}\left(\frac{Pr}{Pr_w}\right)^{0.25}
        \text{ for } 20000 < Re < 200000

    Staggered tube banks:

    .. math::
        \bar Nu_D = 1.04C_nRe_D^{0.4}Pr^{0.36}\left(\frac{Pr}{Pr_w}\right)^{0.25}
        \text{ for } 1 < Re < 500

    .. math::
        \bar Nu_D = 0.71C_nRe_D^{0.5}Pr^{0.36}\left(\frac{Pr}{Pr_w}\right)^{0.25}
        \text{ for } 500 < Re < 1000

    .. math::
        \bar Nu_D = 0.35 C_nRe_D^{0.6}Pr^{0.36}\left(\frac{Pr}{Pr_w}\right)^{0.25}
        \left(\frac{X_t}{X_l}\right)^{0.2}
        \text{ for } 1000 < Re < 20000

    .. math::
        \bar Nu_D = 0.031 C_nRe_D^{0.8}Pr^{0.36}\left(\frac{Pr}{Pr_w}\right)^{0.25}
        \left(\frac{X_t}{X_l}\right)^{0.2}
        \text{ for } 20000 < Re < 200000

    Parameters
    ----------
    Re : float
        Reynolds number with respect to average (bulk) fluid properties and
        tube outside diameter, [-]
    Pr : float
        Prandtl number with respect to average (bulk) fluid properties, [-]
    tube_rows : int
        Number of tube rows per bundle, [-]
    pitch_parallel : float
        Distance between tube center along a line parallel to the flow;
        has been called `longitudinal` pitch, `pp`, `s2`, `SL`, and `p2`, [m]
    pitch_normal : float
        Distance between tube centers in a line 90° to the line of flow;
        has been called the `transverse` pitch, `pn`, `s1`, `ST`, and `p1`, [m]
    Pr_wall : float, optional
        Prandtl number at the wall temperature; provide if a correction with
        the defaults parameters is desired; otherwise apply the correction
        elsewhere, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to tube outside diameter, [-]

    Notes
    -----
    If `Pr_wall` is not provided, the Prandtl number correction
    is not used and left to an outside function.  A Prandtl number exponent of
    0.25 is recommended in [1]_ for heating and cooling for both liquids and
    gases.

    Examples
    --------
    >>> Nu_Zukauskas_Bejan(Re=1E4, Pr=7., tube_rows=10, pitch_parallel=.05, pitch_normal=.05)
    175.9202277145248

    References
    ----------
    .. [1] Zukauskas, A. Heat transfer from tubes in crossflow. In T.F. Irvine,
       Jr. and J. P. Hartnett, editors, Advances in Heat Transfer, volume 8,
       pages 93-160. Academic Press, Inc., New York, 1972.
    .. [2] Bejan, Adrian. "Convection Heat Transfer", 4E. Hoboken,
       New Jersey: Wiley, 2013.
    '''
    staggered = abs(1 - pitch_normal/pitch_parallel) > 0.05

    f = 1.0
    if not staggered:
        if Re < 100:
            c, m = 0.9, 0.4
        elif Re < 1000:
            c, m = 0.52, 0.05
        elif Re < 2E5:
            c, m = 0.27, 0.63
        else:
            c, m = 0.033, 0.8
    else:
        if Re < 500:
            c, m = 1.04, 0.4
        elif Re < 1000:
            c, m = 0.71, 0.5
        elif Re < 2E5:
            c, m = 0.35, 0.6
            f = (pitch_normal/pitch_parallel)**0.2
        else:
            c, m = 0.031, 0.8
            f = (pitch_normal/pitch_parallel)**0.2

    Nu = c*Re**m*Pr**0.36*f
    if Pr_wall is not None:
        Nu*= (Pr/Pr_wall)**0.25
    Cn = Zukauskas_tube_row_correction(tube_rows, staggered=staggered, Re=Re)
    Nu *= Cn
    return Nu


# For row counts 3 to 9, inclusive. Lower tube counts shouldn't be considered
# tube banks. 10 is 1.
ESDU_73031_F2_inline = [0.8479, 0.8957, 0.9306, 0.9551, 0.9724, 0.9839, 0.9902]
ESDU_73031_F2_staggered = [0.8593, 0.8984, 0.9268, 0.9482, 0.965, 0.9777, 0.9868]

def ESDU_tube_row_correction(tube_rows, staggered=True, Re=3000.0, method='Hewitt'):
    r'''Calculates the tube row correction factor according to [1]_ as shown in
    [2]_ for heat transfer across a tube bundle. This is also used for finned
    bundles. The correction factors are slightly different for staggered vs.
    inline configurations.

    This method is a tabular lookup, with values of 1 when the tube row count
    is 10 or more.

    Parameters
    ----------
    tube_rows : int
        Number of tube rows per bundle, [-]
    staggered : bool, optional
        Whether in the in-line or staggered configuration, [-]
    Re : float, optional
        The Reynolds number of flow through the tube bank using the bare tube
        outer diameter and the minimum flow area through the bundle, [-]
    method : str, optional
        'Hewitt'; this may have another option in the future, [-]

    Returns
    -------
    F2 : float
        ESDU tube row count correction factor, [-]

    Notes
    -----
    In [1]_, for line data, there are two curves given for different Reynolds
    number ranges. This is not included in [2]_ and only an average curve is
    given. This is not implemented here; `Re` is an argument but does not
    impact the result of this function.

    For tube counts 1-7, [3]_ claims the factors from [1]_ are on average:
    [0.65, 0.77, 0.84, 0.9, 0.94, 0.97, 0.99].

    Examples
    --------
    >>> ESDU_tube_row_correction(4, staggered=True)
    0.8984
    >>> ESDU_tube_row_correction(6, staggered=False)
    0.9551

    References
    ----------
    .. [1] "Convective Heat Transfer During Crossflow of Fluids Over Plain Tube
       Banks." ESDU 73031 (November 1, 1973).
    .. [2] Hewitt, G. L. Shires, T. Reg Bott G. F., George L. Shires, and T.
       R. Bott. Process Heat Transfer. 1st edition. Boca Raton: CRC Press,
       1994.
    .. [3] Rabas, T. J., and J. Taborek. "Survey of Turbulent Forced-Convection
       Heat Transfer and Pressure Drop Characteristics of Low-Finned Tube Banks
       in Cross Flow."  Heat Transfer Engineering 8, no. 2 (January 1987):
       49-62.
    '''
    if method == 'Hewitt':
        if staggered: # in-line, with a tolerance of 0.05 proximity
            if tube_rows <= 2:
                correction = ESDU_73031_F2_staggered[0]
            elif tube_rows >= 10:
                correction = 1.0
            else:
                correction = ESDU_73031_F2_staggered[tube_rows-3]
        else:
            if tube_rows <= 2:
                correction = ESDU_73031_F2_inline[0]
            elif tube_rows >= 10:
                correction = 1.0
            else:
                correction = ESDU_73031_F2_inline[tube_rows-3]
        return correction


def ESDU_tube_angle_correction(angle):
    r'''Calculates the tube bank inclination correction factor according to
    [1]_ for heat transfer across a tube bundle.

    .. math::
        F_3 = \frac{Nu_{\theta}}{Nu_{\theta=90^{\circ}}} = (\sin(\theta))^{0.6}

    Parameters
    ----------
    angle : float
        The angle of inclination of the tuba bank with respect to the
        longitudinal axis (90° for a straight tube bank)

    Returns
    -------
    F3 : float
        ESDU tube inclination correction factor, [-]

    Notes
    -----
    A curve is given in [1]_ but it is so close the function, it is likely the
    function is all that is used. [1]_ claims this correction is valid for
    :math:`100 < Re < 10^{6}`.

    For angles less than 10°, the problem should be considered internal
    flow, not flow across a tube bank.

    Examples
    --------
    >>> ESDU_tube_angle_correction(75)
    0.9794139080247666

    References
    ----------
    .. [1] "Convective Heat Transfer During Crossflow of Fluids Over Plain Tube
       Banks." ESDU 73031 (November 1, 1973).
    '''
    return sin(radians(angle))**0.6


def Nu_ESDU_73031(Re, Pr, tube_rows, pitch_parallel, pitch_normal,
                  Pr_wall=None, angle=90.0):
    r'''Calculates the Nusselt number for crossflow across a tube bank
    with a specified number of tube rows, at a specified `Re` according to
    [1]_, also shown in [2]_.

    .. math::
        \text{Nu} = a \text{Re}^m\text{Pr}^{0.34}F_1 F_2

    The constants `a` and `m` come from the following tables:

    In-line tube banks:

    +---------+-------+-------+
    | Re      | a     | m     |
    +=========+=======+=======+
    | 10-300  | 0.742 | 0.431 |
    +---------+-------+-------+
    | 300-2E5 | 0.211 | 0.651 |
    +---------+-------+-------+
    | 2E5-2E6 | 0.116 | 0.700 |
    +---------+-------+-------+

    Staggered tube banks:

    +---------+-------+-------+
    | Re      | a     | m     |
    +=========+=======+=======+
    | 10-300  | 1.309 | 0.360 |
    +---------+-------+-------+
    | 300-2E5 | 0.273 | 0.635 |
    +---------+-------+-------+
    | 2E5-2E6 | 0.124 | 0.700 |
    +---------+-------+-------+

    Parameters
    ----------
    Re : float
        Reynolds number with respect to average (bulk) fluid properties and
        tube outside diameter, [-]
    Pr : float
        Prandtl number with respect to average (bulk) fluid properties, [-]
    tube_rows : int
        Number of tube rows per bundle, [-]
    pitch_parallel : float
        Distance between tube center along a line parallel to the flow;
        has been called `longitudinal` pitch, `pp`, `s2`, `SL`, and `p2`, [m]
    pitch_normal : float
        Distance between tube centers in a line 90° to the line of flow;
        has been called the `transverse` pitch, `pn`, `s1`, `ST`, and `p1`, [m]
    Pr_wall : float, optional
        Prandtl number at the wall temperature; provide if a correction with
        the defaults parameters is desired; otherwise apply the correction
        elsewhere, [-]
    angle : float, optional
        The angle of inclination of the tuba bank with respect to the
        longitudinal axis (90° for a straight tube bank)

    Returns
    -------
    Nu : float
        Nusselt number with respect to tube outside diameter, [-]

    Notes
    -----
    The tube-row count correction factor `F2` can be disabled by setting `tube_rows`
    to 10. The property correction factor `F1` can be disabled by not specifying
    `Pr_wall`. A Prandtl number exponent of 0.26 is recommended in [1]_ for
    heating and cooling for both liquids and gases.

    The pitches are used to determine whhether or not to use data for staggered
    or inline tube banks.

    The inline coefficients are valid for a normal pitch to tube diameter ratio
    from 1.2 to 4; and the staggered ones from 1 to 4.
    The overall accuracy of this method is claimed to be 15%.

    See Also
    --------
    ESDU_tube_angle_correction
    ESDU_tube_row_correction

    Examples
    --------
    >>> Nu_ESDU_73031(Re=1.32E4, Pr=0.71, tube_rows=8, pitch_parallel=.09,
    ... pitch_normal=.05)
    98.2563319140594

    References
    ----------
    .. [1] "High-Fin Staggered Tube Banks: Heat Transfer and Pressure Drop for
       Turbulent Single Phase Gas Flow." ESDU 86022 (October 1, 1986).
    .. [2] Hewitt, G. L. Shires, T. Reg Bott G. F., George L. Shires, and T.
       R. Bott. Process Heat Transfer. 1st edition. Boca Raton: CRC Press,
       1994.
    '''
    staggered = abs(1 - pitch_normal/pitch_parallel) > 0.05
    if staggered:
        if Re <= 300:
            a, m = 1.309, 0.360
        elif Re <= 2E5:
            a, m = 0.273, 0.635
        else:
            a, m = 0.124, 0.700
    else:
        if Re <= 300:
            a, m = 0.742, 0.431
        elif Re <= 2E5:
            a, m = 0.211, 0.651
        else:
            a, m = 0.116, 0.700

    F2 = ESDU_tube_row_correction(tube_rows=tube_rows, staggered=staggered)
    F3 = ESDU_tube_angle_correction(angle)
    if Pr_wall is not None:
        F1 = wall_factor(Pr=Pr, Pr_wall=Pr_wall, Pr_heating_coeff=0.26,
                         Pr_cooling_coeff=0.26,
                         property_option=WALL_FACTOR_PRANDTL)
    else:
        F1 = 1.0
    return a*Re**m*Pr**0.34*F1*F2*F3


def Nu_HEDH_tube_bank(Re, Pr, Do, tube_rows, pitch_parallel, pitch_normal):
    r'''Calculates Nusselt number for crossflow across a tube bank
    of tube rows at a specified `Re`, `Pr`, and `D` using the Heat Exchanger
    Design Handbook (HEDH) methodology, presented in [1]_.

    .. math::
        Nu = Nu_m   f_N

    .. math::
        Nu_m = 0.3 + \sqrt{Nu_{m,lam}^2 + Nu_{m,turb}^2}

    .. math::
        Nu_{m,turb} = \frac{0.037Re^{0.8} Pr}{1 + 2.443Re^{-0.1}(Pr^{2/3} -1)}

    .. math::
        Nu_{m,lam} = 0.664Re^{0.5} Pr^{1/3}

    .. math::
        \psi = 1 - \frac{\pi}{4a} \text{ if b >= 1}

    .. math::
        \psi = 1 - \frac{\pi}{4ab} \text{if b < 1}

    .. math::
        f_A = 1 + \frac{0.7}{\psi^{1.5}}\frac{b/a-0.3}{(b/a) + 0.7)^2} \text{if inline}

    .. math::
        f_A = 1 + \frac{2}{3b} \text{elif partly staggered}

    .. math::
        f_N = \frac{1 + (n-1)f_A}{n}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to average (bulk) fluid properties and
        tube outside diameter, [-]
    Pr : float
        Prandtl number with respect to average (bulk) fluid properties, [-]
    Do : float
        Tube outer diameter, [m]
    tube_rows : int
        Number of tube rows per bundle, [-]
    pitch_parallel : float
        Distance between tube center along a line parallel to the flow;
        has been called `longitudinal` pitch, `pp`, `s2`, `SL`, and `p2`, [m]
    pitch_normal : float
        Distance between tube centers in a line 90° to the line of flow;
        has been called the `transverse` pitch, `pn`, `s1`, `ST`, and `p1`, [m]

    Returns
    -------
    Nu : float
        Nusselt number with respect to tube outside diameter, [-]

    Notes
    -----
    Prandtl number correction left to an outside function, although a set
    of coefficients were specified in [1]_ because they depent on whether
    heating or cooling is happening, and for gases, use a temperature ratio
    instaed of Prandtl number.

    The claimed range of validity of these expressions is :math:`10 < Re < 1E5`
    and :math:`0.6 < Pr < 1000`.

    Examples
    --------
    >>> Nu_HEDH_tube_bank(Re=1E4, Pr=7., tube_rows=10, pitch_normal=.05,
    ... pitch_parallel=.05, Do=.03)
    382.4636554404698

    Example 3.11 in [2]_:

    >>> Nu_HEDH_tube_bank(Re=10263.37, Pr=.708, tube_rows=11, pitch_normal=.05,
    ... pitch_parallel=.05, Do=.025)
    149.18735251017594

    References
    ----------
    .. [1] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    .. [2] Baehr, Hans Dieter, and Karl Stephan. Heat and Mass Transfer.
       Springer, 2013.
    '''
    staggered = abs(1 - pitch_normal/pitch_parallel) > 0.05
    a = pitch_normal/Do
    b = pitch_parallel/Do
    if b >= 1:
        voidage = 1. - pi/(4.0*a)
    else:
        voidage = 1. - pi/(4.0*a*b)
    Re = Re/voidage
    Nu_laminar = 0.664*Re**0.5*Pr**(1.0/3.)
    Nu_turbulent = 0.037*Re**0.8*Pr/(1. + 2.443*Re**-0.1*(Pr**(2/3.) - 1.0))
    Nu = 0.3 + (Nu_laminar*Nu_laminar + Nu_turbulent*Nu_turbulent)**0.5
    if not staggered:
        fA = 1.0 + 0.7/voidage**1.5*(b/a - 0.3)/(b/a + 0.7)**2
    else:
        fA = 1.0 + 2./(3.0*b)
        # a further partly staggered tube bank correlation exists, using another pitch
    if tube_rows < 10:
        fn = (1.0 + (tube_rows - 1.0)*fA)/tube_rows
    else:
        fn = fA
    Nu = Nu*fn
    return Nu


"""
Graph presented in Peters and Timmerhaus uses fanning friction factor.
This uses Darcy's friction factor.
These coefficients were generated to speed up loading of this module.
They are regenerated and checked in the tests.

"""
Kern_f_Re_tck = implementation_optimize_tck([[9.9524, 9.9524, 9.9524, 9.9524, 17.9105, 27.7862, 47.2083, 83.9573,
                           281.996, 1122.76, 42999.9, 1012440.0, 1012440.0, 1012440.0, 1012440.0],
                 [6.040435949178239, 4.64973456285782, 2.95274850806163, 1.9569061885042,
                           1.1663069946420412, 0.6830549536215098, 0.4588680265447762, 0.22387792331971723,
                           0.12721190975530583, 0.1395456548881242, 0.12888895743468684, 0.0, 0.0, 0.0, 0.0],
                 3], force_numpy=IS_NUMBA)
Kern_f_Re = lambda x: float(splev(x, Kern_f_Re_tck))


def dP_Kern(m, rho, mu, DShell, LSpacing, pitch, Do, NBaffles, mu_w=None):
    r'''Calculates pressure drop for crossflow across a tube bank
    according to the equivalent-diameter method developed by Kern [1]_,
    presented in [2]_.

    .. math::
        \Delta P = \frac{f (m/S_s)^2 D_s(N_B+1)}{2\rho D_e(\mu/\mu_w)^{0.14}}

    .. math::
        S_S = \frac{D_S (P_T-D_o) L_B}{P_T}

    .. math::
        D_e = \frac{4(P_T^2 - \pi D_o^2/4)}{\pi D_o}

    Parameters
    ----------
    m : float
        Mass flow rate, [kg/s]
    rho : float
        Fluid density, [kg/m^3]
    mu : float
        Fluid viscosity, [Pa*s]
    DShell : float
        Diameter of exchanger shell, [m]
    LSpacing : float
        Baffle spacing, [m]
    pitch : float
        Tube pitch, [m]
    Do : float
        Tube outer diameter, [m]
    NBaffles : float
        Baffle count, []
    mu_w : float
        Fluid viscosity at wall temperature, [Pa*s]

    Returns
    -------
    dP : float
        Pressure drop across bundle, [Pa]

    Notes
    -----
    Adjustment for viscosity left out of this function.
    Example is from [2]_. Roughly 10% difference due to reading of graph.
    Graph scanned from [1]_, and interpolation is used to read it.

    Examples
    --------
    >>> dP_Kern(m=11., rho=995., mu=0.000803, mu_w=0.000657, DShell=0.584,
    ... LSpacing=0.1524, pitch=0.0254, Do=.019, NBaffles=22)
    18980.58768759033

    References
    ----------
    .. [1] Kern, Donald Quentin. Process Heat Transfer. McGraw-Hill, 1950.
    .. [2] Peters, Max, Klaus Timmerhaus, and Ronald West. Plant Design and
       Economics for Chemical Engineers. 5E. New York: McGraw-Hill, 2002.
    '''
    # Adjustment for viscosity performed if given
    Ss = DShell*(pitch-Do)*LSpacing/pitch
    De = 4*(pitch*pitch - pi*Do*Do/4.)/pi/Do
    Vs = m/Ss/rho
    Re = rho*De*Vs/mu
    f = Kern_f_Re(Re)
    if mu_w:
        return f*(Vs*rho)**2*DShell*(NBaffles+1)/(2*rho*De*(mu/mu_w)**0.14)
    else:
        return f*(Vs*rho)**2*DShell*(NBaffles+1)/(2*rho*De)

_Zukauskas_correlations_loaded = False
def load_Zukauskas_correlations():
    global _Zukauskas_correlations_loaded, dP_staggered_f, dP_staggered_correction, dP_inline_f, dP_inline_correction
    from scipy.interpolate import RectBivariateSpline
    _Zukauskas_correlations_loaded = True

    _dP_staggered_Res = np.array([10, 10.9129, 11.6733, 13.1024, 14.0153, 14.9918, 17.1536, 18.5267, 19.8182, 20.7261, 22.243, 23.7936, 26.7057, 28.5663, 32.2732,
        34.858, 37.2879, 41.0554, 44.4722, 47.8949, 51.2337, 55.3369, 65.1821, 70.4025, 76.0437, 82.1368, 88.7182, 95.1284, 100.553, 103.386, 108.398,
        116.441, 118.455, 127.808, 129.188, 139.389, 140.899, 153.665, 155.444, 167.595, 168.914, 182.793, 197.771, 201.613, 217.768, 223.559, 241.759,
        246.457, 268.516, 278.915, 292.866, 304.208, 322.535, 335.015, 351.772, 366.482, 402.412, 415.414, 451.79, 465.314, 497.559, 512.453, 542.68,
        570.321, 609.312, 610.163, 671.039, 671.953, 731.917, 732.915, 813.886, 839.919, 896.808, 977.69, 1016.19, 1119.14, 1221.31, 1244.48, 1346.07,
        1455.66, 1482.44, 1603.12, 1616.93, 1748.56, 1780.79, 1925.77, 1961.27, 2056.71, 2060.37, 2266.81, 2308.27, 2474.96, 2542.2, 2723.03, 2799.84,
        2996.9, 3053.95, 3274.27, 3363.57, 3606.09, 4001.84, 4005.75, 4367.03, 4411.71, 4809.6, 4854.24, 5297.21, 5346.19, 5777.99, 5836.5, 6184.44,
        6739.62, 6817.15, 7422.65, 7435.62, 8188.61, 8256.81, 9005.89, 9089.79, 9914.09, 9931.42, 10832, 11357.6, 11913.2, 12508.2, 13011.2, 13642.4,
        14309.8, 15024.5, 15759.5, 16387, 17188.6, 18046.5, 18772.3, 19683.7, 20458.2, 22313.4, 22950.8, 24573.9, 26311.7, 27049.2, 28976.2, 29516.6, 31605,
        32505.6, 34805.6, 35453.4, 37961.9, 39045, 39838.4, 40171.7, 43802.4, 43836, 47853, 48253.3, 52629.1, 57429.8, 57958.7, 60823.7, 63808, 66429.9,
        72454.1, 76644.8, 79791.3, 86914.7, 87727.5, 94796.5, 95846.9, 102543, 103393, 112734, 123172, 124193, 134342, 136770, 147946, 149173, 161368,
        162701, 177710, 179183, 193825, 197329, 203406, 205093, 224028, 225878, 246499, 248787, 268891, 271756, 296172, 299307, 323098, 329652, 355768,
        363073, 388139, 399883, 411321, 411637, 453053, 453370, 494224, 499159, 539099, 549766, 593776, 617117, 617548, 679896, 741914, 748826, 816818,
        899347, 899975, 991217, 1029890, 1039630, 1134310, 1145030, 1249310, 1261120, 1375630, 1388740, 1515150, 1529530, 1668760, 1684660, 1837940,
        1855450, 2063320, 2064190, 2251140, 2273460, 2479450, 2502990, 2730830, 2756750
    ])
    _dP_staggered_Re_125 = np.array([23.9929, 22.6513, 21.1808, 19.0604, 17.8231, 16.6661, 14.5725, 13.6264, 12.8644, 12.1931, 11.3569, 10.7219, 9.55649, 8.93611,
        7.91304, 7.32822, 6.89654, 6.28568, 5.80434, 5.44301, 5.08949, 4.72306, 4.06698, 3.79555, 3.5683, 3.30447, 3.1177, 2.91006, 2.77913, 2.71412,
        2.60635, 2.4487, 2.41753, 2.2802, 2.25939, 2.12672, 2.11005, 1.98054, 1.96397, 1.85661, 1.84576, 1.74274, 1.66846, 1.63677, 1.56011, 1.53763,
        1.47248, 1.45689, 1.38943, 1.36053, 1.32959, 1.30743, 1.27402, 1.2528, 1.22604, 1.20401, 1.15477, 1.13664, 1.10541, 1.09271, 1.06394, 1.05209,
        1.02957, 1.01043, 0.985509, 0.984989, 0.950966, 0.950537, 0.92446, 0.924083, 0.894818, 0.885516, 0.868347, 0.848317, 0.840024, 0.819658, 0.801646,
        0.797824, 0.782058, 0.766644, 0.763863, 0.752037, 0.75061, 0.737713, 0.736366, 0.730623, 0.728723, 0.723802, 0.723618, 0.709974, 0.707146, 0.696311,
        0.694446, 0.689689, 0.685538, 0.675409, 0.672874, 0.663594, 0.66181, 0.657217, 0.636046, 0.63585, 0.619904, 0.619273, 0.613337, 0.612083, 0.601667,
        0.601114, 0.595116, 0.592882, 0.580202, 0.570252, 0.568954, 0.558333, 0.558117, 0.542262, 0.541366, 0.532074, 0.530674, 0.517089, 0.516819,
        0.502141, 0.497421, 0.492707, 0.484889, 0.478584, 0.471858, 0.465173, 0.458449, 0.451954, 0.448019, 0.443305, 0.436261, 0.430589, 0.424819,
        0.420179, 0.409927, 0.406655, 0.398825, 0.391145, 0.387928, 0.380033, 0.378482, 0.372795, 0.369679, 0.362205, 0.359995, 0.351918, 0.34995, 0.348549,
        0.347907, 0.341093, 0.341015, 0.332198, 0.331281, 0.322228, 0.316669, 0.315569, 0.310077, 0.30713, 0.304674, 0.296022, 0.29109, 0.287612, 0.282751,
        0.282227, 0.277435, 0.276759, 0.271491, 0.270748, 0.263364, 0.258755, 0.258047, 0.251406, 0.250064, 0.244264, 0.243818, 0.239612, 0.239024,
        0.232805, 0.232168, 0.226194, 0.225387, 0.224028, 0.224027, 0.224011, 0.22401, 0.223994, 0.223993, 0.223979, 0.223977, 0.223962, 0.22396, 0.223947,
        0.223943, 0.22393, 0.223926, 0.223915, 0.223909, 0.223904, 0.223904, 0.223887, 0.223887, 0.226011, 0.225818, 0.224086, 0.223853, 0.22384, 0.225949,
        0.225988, 0.225971, 0.225955, 0.225954, 0.225938, 0.225921, 0.225921, 0.225904, 0.223951, 0.224158, 0.22588, 0.225878, 0.225863, 0.225861, 0.225846,
        0.225844, 0.225829, 0.225827, 0.225812, 0.22581, 0.225794, 0.225793, 0.225774, 0.225774, 0.225759, 0.225757, 0.227901, 0.227913, 0.227897, 0.227896
    ])
    _dP_staggered_Re_15 = np.array([9.34201, 8.81965, 8.28809, 7.42806, 6.97391, 6.57517, 5.84093, 5.50985, 5.16014, 4.93488, 4.68126, 4.42254, 3.99955, 3.773,
        3.39505, 3.20519, 3.02598, 2.77577, 2.61488, 2.474, 2.33566, 2.20505, 1.96531, 1.8554, 1.76851, 1.68568, 1.60674, 1.54592, 1.47385, 1.45584,
        1.42566, 1.36641, 1.35191, 1.29626, 1.28859, 1.24598, 1.24005, 1.18197, 1.17596, 1.13745, 1.13449, 1.10514, 1.04611, 1.03299, 1.01551, 1.00639,
        0.975508, 0.956979, 0.921361, 0.906001, 0.886645, 0.876509, 0.861323, 0.848885, 0.833067, 0.820018, 0.790987, 0.781353, 0.757982, 0.750599, 0.73523,
        0.72878, 0.715526, 0.703825, 0.690704, 0.69043, 0.671089, 0.670816, 0.658238, 0.65804, 0.642607, 0.638042, 0.628642, 0.616468, 0.611099, 0.59789,
        0.592593, 0.59088, 0.5807, 0.571709, 0.569635, 0.559848, 0.558786, 0.554428, 0.553416, 0.5491, 0.548097, 0.54293, 0.542793, 0.537633, 0.53548,
        0.52734, 0.525928, 0.522325, 0.519436, 0.51244, 0.510312, 0.502563, 0.501212, 0.497646, 0.483767, 0.483639, 0.479479, 0.478991, 0.469919, 0.469457,
        0.465373, 0.464541, 0.457403, 0.456485, 0.452112, 0.44443, 0.443408, 0.435131, 0.434907, 0.419981, 0.418722, 0.415054, 0.414669, 0.405475, 0.405291,
        0.396251, 0.391403, 0.387694, 0.383945, 0.378953, 0.373041, 0.369506, 0.365933, 0.360178, 0.355541, 0.350503, 0.34544, 0.342437, 0.338861, 0.33562,
        0.326088, 0.324262, 0.319875, 0.312702, 0.30994, 0.303633, 0.301961, 0.295857, 0.293384, 0.286801, 0.285051, 0.281193, 0.27962, 0.27688, 0.276192,
        0.269144, 0.269082, 0.261395, 0.260961, 0.256484, 0.249175, 0.248418, 0.244472, 0.240616, 0.237428, 0.23135, 0.228333, 0.226286, 0.219953, 0.21934,
        0.21432, 0.213615, 0.209306, 0.208785, 0.203406, 0.198042, 0.197549, 0.193322, 0.192633, 0.189133, 0.188615, 0.183883, 0.183431, 0.178539, 0.17805,
        0.173635, 0.172752, 0.171298, 0.171157, 0.169685, 0.169823, 0.171289, 0.171462, 0.172926, 0.173289, 0.176258, 0.176871, 0.181386, 0.182469,
        0.186641, 0.188307, 0.193913, 0.196789, 0.199552, 0.199594, 0.201486, 0.2015, 0.203218, 0.203417, 0.203404, 0.203401, 0.204688, 0.205335, 0.205334,
        0.203367, 0.203354, 0.203352, 0.203338, 0.205273, 0.20528, 0.205265, 0.205258, 0.205257, 0.205243, 0.205241, 0.205227, 0.205226, 0.205212, 0.20521,
        0.205196, 0.205195, 0.205181, 0.205179, 0.205165, 0.205164, 0.205146, 0.205146, 0.205132, 0.205131, 0.205117, 0.205115, 0.205101, 0.2051
    ])
    _dP_staggered_Re_2 = np.array([3.3699, 3.25874, 3.1513, 2.97524, 2.87715, 2.78229, 2.60185, 2.504, 2.4214, 2.36801, 2.2862, 2.21078, 2.08731, 2.01849, 1.89955,
        1.82808, 1.76778, 1.68508, 1.61934, 1.56066, 1.50918, 1.4524, 1.33872, 1.28835, 1.23986, 1.19319, 1.14827, 1.10908, 1.07889, 1.06407, 1.03929,
        1.00291, 0.994386, 0.957472, 0.951802, 0.912623, 0.908283, 0.874086, 0.869647, 0.841073, 0.838095, 0.808676, 0.780364, 0.773598, 0.747536, 0.738905,
        0.721828, 0.717582, 0.690875, 0.682216, 0.671254, 0.666188, 0.658464, 0.647496, 0.633663, 0.625691, 0.607864, 0.60192, 0.586506, 0.581184, 0.573473,
        0.57011, 0.559368, 0.551449, 0.543432, 0.543274, 0.533071, 0.532917, 0.522924, 0.522784, 0.512946, 0.509741, 0.503124, 0.493595, 0.490661, 0.483683,
        0.479474, 0.477483, 0.469851, 0.466187, 0.465338, 0.461708, 0.461273, 0.453348, 0.452088, 0.448562, 0.447435, 0.443915, 0.443836, 0.439616, 0.43882,
        0.43577, 0.434512, 0.431212, 0.430014, 0.427098, 0.426293, 0.423332, 0.422987, 0.422964, 0.422929, 0.422883, 0.414874, 0.414451, 0.410887, 0.410884,
        0.410855, 0.410436, 0.40691, 0.406003, 0.40083, 0.393272, 0.392277, 0.385608, 0.385474, 0.374217, 0.373188, 0.36608, 0.365067, 0.355721, 0.355584,
        0.349483, 0.344411, 0.338995, 0.335717, 0.333088, 0.328254, 0.323091, 0.319967, 0.316935, 0.312854, 0.307934, 0.303486, 0.299932, 0.296289,
        0.293462, 0.288427, 0.286092, 0.279681, 0.274029, 0.271776, 0.265638, 0.264031, 0.260457, 0.258987, 0.253176, 0.251647, 0.24824, 0.2468, 0.243526,
        0.242183, 0.237587, 0.237538, 0.231397, 0.230821, 0.224266, 0.218493, 0.217895, 0.215809, 0.213044, 0.210747, 0.20588, 0.202787, 0.200602, 0.196037,
        0.195433, 0.19047, 0.189774, 0.185568, 0.18506, 0.181897, 0.176863, 0.176379, 0.172288, 0.171368, 0.166958, 0.166501, 0.162215, 0.161773, 0.158952,
        0.15869, 0.15501, 0.154182, 0.153022, 0.152707, 0.151364, 0.15124, 0.152546, 0.152684, 0.155311, 0.155668, 0.158336, 0.158692, 0.162743, 0.164018,
        0.169607, 0.171511, 0.177917, 0.179674, 0.181351, 0.181379, 0.184846, 0.184874, 0.188409, 0.188408, 0.188396, 0.18876, 0.190196, 0.190315, 0.190318,
        0.190617, 0.190888, 0.190917, 0.191188, 0.191489, 0.191491, 0.191793, 0.191913, 0.191942, 0.191929, 0.191928, 0.191915, 0.191913, 0.1919, 0.192079,
        0.193733, 0.193731, 0.193718, 0.193717, 0.193703, 0.193702, 0.193686, 0.193686, 0.193673, 0.193861, 0.195522, 0.195521, 0.195508, 0.195506
    ])
    _dP_staggered_Re_25 = np.array([1.79994, 1.76013, 1.72122, 1.65648, 1.61986, 1.58405, 1.51479, 1.47657, 1.44391, 1.4226, 1.38964, 1.3589, 1.30781, 1.2789,
        1.22814, 1.19714, 1.17066, 1.13385, 1.10416, 1.07732, 1.05349, 1.02689, 0.972573, 0.948019, 0.924073, 0.900732, 0.877981, 0.857886, 0.842238,
        0.834508, 0.821498, 0.802211, 0.797489, 0.771119, 0.767464, 0.742087, 0.738758, 0.71986, 0.717012, 0.693528, 0.691126, 0.673248, 0.655161, 0.650796,
        0.633605, 0.627855, 0.611017, 0.606947, 0.589142, 0.581418, 0.572075, 0.564906, 0.555118, 0.548858, 0.542958, 0.537932, 0.523109, 0.517617,
        0.503395, 0.500444, 0.493804, 0.488993, 0.479779, 0.472711, 0.470631, 0.4705, 0.461663, 0.461524, 0.45287, 0.452758, 0.444238, 0.442841, 0.439921,
        0.431589, 0.431576, 0.423352, 0.4167, 0.415283, 0.415257, 0.412759, 0.412007, 0.408794, 0.408443, 0.405032, 0.404211, 0.400713, 0.399901, 0.39662,
        0.396488, 0.389473, 0.388156, 0.385458, 0.384426, 0.381792, 0.380731, 0.377866, 0.377075, 0.377054, 0.377046, 0.374429, 0.36984, 0.369804, 0.366623,
        0.366285, 0.36626, 0.366258, 0.363072, 0.362738, 0.359622, 0.35915, 0.352384, 0.349036, 0.348637, 0.345681, 0.345518, 0.336581, 0.335833, 0.330069,
        0.329459, 0.320103, 0.320041, 0.316923, 0.313944, 0.310956, 0.305966, 0.302055, 0.299216, 0.296371, 0.292087, 0.287957, 0.285478, 0.282464,
        0.277935, 0.274359, 0.271138, 0.268545, 0.263228, 0.261591, 0.256305, 0.250791, 0.248501, 0.244499, 0.2436, 0.239026, 0.236807, 0.231877, 0.230603,
        0.225941, 0.22405, 0.222707, 0.222159, 0.217943, 0.217893, 0.21226, 0.211731, 0.206127, 0.20064, 0.200073, 0.197111, 0.194215, 0.192662, 0.188591,
        0.185104, 0.182162, 0.178493, 0.178125, 0.174048, 0.173476, 0.170156, 0.169754, 0.16495, 0.160643, 0.160246, 0.156122, 0.155402, 0.152988, 0.15273,
        0.148798, 0.148394, 0.144533, 0.144169, 0.139245, 0.138483, 0.137648, 0.137427, 0.136218, 0.136117, 0.137425, 0.137564, 0.138759, 0.139196,
        0.142785, 0.143115, 0.145559, 0.14672, 0.151214, 0.152571, 0.157129, 0.160246, 0.163232, 0.163273, 0.168458, 0.168493, 0.172862, 0.173428, 0.177879,
        0.178569, 0.181323, 0.18658, 0.18658, 0.186566, 0.186553, 0.186552, 0.186539, 0.186525, 0.186508, 0.184125, 0.183189, 0.182964, 0.182952, 0.18295,
        0.182938, 0.182936, 0.182924, 0.182922, 0.18291, 0.182909, 0.184483, 0.184655, 0.184643, 0.184641, 0.182866, 0.182873, 0.18444, 0.184612, 0.184599,
        0.184598, 0.184585, 0.184584
    ])
    _dP_staggered_Re_parameters = np.array([_dP_staggered_Re_125, _dP_staggered_Re_15, _dP_staggered_Re_2, _dP_staggered_Re_25]).T
    dP_staggered_f = RectBivariateSpline(_dP_staggered_Res, np.array([1.25, 1.5, 2, 2.5]), _dP_staggered_Re_parameters, kx=3, ky=3, s=0.002)

    # Excellent plot, though it does linear extrapolation on some lines
    #import matplotlib.pyplot as plt
    #dP_staggered_f_zs = np.array([1.25, 1.5, 2, 2.5])
    #low, high = min(_dP_staggered_Res), max(_dP_staggered_Res)
    #xs = np.linspace(low, high, 50000)
    #for i in range(4):
    #    plt.loglog(_dP_staggered_Res, _dP_staggered_Re_parameters.T[i, :], '.')
    #    plt.loglog(xs, dP_staggered_f(xs, dP_staggered_f_zs[i]), '--')
    #plt.show()


    _dP_staggered_correction_parameters = np.array([0.4387, 0.470647, 0.494366, 0.52085, 0.542787, 0.583019, 0.609319, 0.659047, 0.685413, 0.729582, 0.800982,
        0.84214, 0.892449, 0.947309, 1.00903, 1.07052, 1.16389, 1.22243, 1.26584, 1.32314, 1.37597, 1.40437, 1.45385, 1.51093, 1.55814, 1.61775, 1.68647,
        1.74589, 1.79853, 1.86586, 1.92335, 1.97322, 2.12053, 2.22751, 2.34521, 2.45793, 2.58193, 2.71226, 2.84909, 2.99282, 3.14389, 3.22668, 3.32915,
        3.54351
    ])
    _dP_staggered_correction_Re_100 = np.array([0.996741, 0.996986, 0.997157, 0.997339, 0.997482, 0.997731, 0.997885, 0.998158, 0.998294, 0.998512, 0.998836,
        0.999011, 0.999213, 0.99942, 0.99964, 0.999846, 1.00241, 1.02216, 1.0392, 1.06545, 1.08705, 1.0995, 1.1206, 1.14708, 1.16583, 1.18871, 1.21407,
        1.23518, 1.25628, 1.27868, 1.29996, 1.31593, 1.36025, 1.39055, 1.42224, 1.45114, 1.48144, 1.51175, 1.54205, 1.57235, 1.60267, 1.62032, 1.64208,
        1.68552
    ])
    _dP_staggered_correction_Re_1000 = np.array([1.03576, 1.02714, 1.02111, 1.01712, 1.01206, 1.00798, 1.00547, 1.001, 0.999839, 0.999378, 0.998689, 0.998319,
        0.997891, 0.997451, 0.996985, 0.999249, 1.00245, 1.0135, 1.02415, 1.03618, 1.04682, 1.0534, 1.06478, 1.07524, 1.0836, 1.09539, 1.10811, 1.11825,
        1.12833, 1.13858, 1.1481, 1.15678, 1.17941, 1.19487, 1.21106, 1.22398, 1.24068, 1.25657, 1.27109, 1.28706, 1.30317, 1.31111, 1.3196, 1.33956
    ])
    _dP_staggered_correction_Re_10000 = np.array([1.20211, 1.18293, 1.16951, 1.15527, 1.14308, 1.12148, 1.10821, 1.09069, 1.08213, 1.06633, 1.04824, 1.04041,
        1.03015, 1.02269, 1.01509, 1.00905, 1.00302, 1.00302, 1.00304, 1.00623, 1.00905, 1.0103, 1.01246, 1.01508, 1.01696, 1.01926, 1.0225, 1.02674,
        1.03074, 1.03432, 1.03618, 1.03931, 1.04813, 1.05451, 1.05855, 1.0674, 1.07355, 1.08006, 1.08719, 1.09572, 1.10324, 1.10854, 1.11428, 1.12663
    ])
    _dP_staggered_correction_Re_100000 = np.array([1.45829, 1.42587, 1.40486, 1.38291, 1.36389, 1.32864, 1.30754, 1.27136, 1.25327, 1.22447, 1.18203, 1.15678,
        1.12845, 1.10251, 1.07182, 1.04763, 1.00824, 0.984925, 0.975402, 0.965711, 0.960152, 0.957646, 0.9534, 0.948334, 0.945015, 0.942714, 0.940164,
        0.937857, 0.936683, 0.936683, 0.934823, 0.933668, 0.933668, 0.933668, 0.933668, 0.933668, 0.933668, 0.936683, 0.936683, 0.936683, 0.939698,
        0.939698, 0.939698, 0.939698
    ])
    _dP_staggered_correction_Re_parameters = np.array([_dP_staggered_correction_Re_100, _dP_staggered_correction_Re_1000, _dP_staggered_correction_Re_10000, _dP_staggered_correction_Re_100000]).T
    dP_staggered_correction = RectBivariateSpline(_dP_staggered_correction_parameters, np.array([1E2, 1E3, 1E4, 1E5]), _dP_staggered_correction_Re_parameters, kx=1, ky=3, s=0.002)

    # Maybe good plot - bad around the middle
    #dP_staggered_correction_zs = np.array([1E2, 1E3, 1E4, 1E5])
    #low, high = min(_dP_staggered_correction_parameters), max(_dP_staggered_correction_parameters)
    #xs = np.linspace(low, high, 50000)
    #for i in range(4):
    #    plt.loglog(_dP_staggered_correction_parameters, _dP_staggered_correction_Re_parameters.T[i, :], '.')
    #    plt.loglog(xs, dP_staggered_correction(xs, dP_staggered_correction_zs[i]), '--')
    #plt.show()


    _dP_inline_Res = np.array([28.5094, 30.8092, 32.9727, 35.3563, 41.2101, 45.9365, 49.1622, 52.6143, 56.3102, 59.107, 63.7533, 68.3605, 73.1607, 82.9896, 91.2679,
        107.829, 116.528, 124.713, 134.774, 144.237, 157.106, 169.784, 183.484, 202.173, 218.488, 241.163, 278.938, 301.447, 325.772, 352.069, 402.667,
        439.431, 479.551, 528.457, 576.706, 600.39, 654.321, 666.665, 722.026, 795.679, 802.401, 883.594, 965.211, 973.774, 1022.26, 1107.38, 1126.59,
        1220.48, 1343.51, 1368.32, 1468.16, 1616.19, 1646.72, 1764.04, 1814.79, 1944.21, 1998.93, 2038.12, 2041.06, 2246.18, 2249.48, 2455.2, 2476.81,
        2705.84, 2729.59, 2982.07, 3008.17, 3257.9, 3313.34, 3590.4, 3618.29, 3946.71, 4030.55, 4063.47, 4434.98, 4446.05, 4852.32, 4895.14, 5347.3,
        5394.74, 5830.48, 5994.16, 6003.24, 6545.85, 6615.94, 7143.99, 7226.2, 7873.1, 8101.49, 8113.39, 8928.33, 8941.23, 9765.31, 9845.06, 10343.9,
        10430.3, 11407.3, 11956.6, 12562.5, 13176.9, 13719.7, 14521.4, 15236.6, 16651, 17465.4, 18505, 20393.2, 20419.3, 22474.5, 22503.3, 24559, 25546.2,
        27064.9, 29789.7, 30724.6, 32829.2, 34810.9, 36179.8, 38362.8, 39871.4, 40721.2, 41061.4, 44854.2, 45239.5, 48975.7, 49855.5, 53971.7, 54426.4,
        59979.7, 60058.1, 66101.3, 66184.5, 72230.6, 72907, 81043.8, 81128.8, 89317.2, 89406.8, 97574.2, 98430.6, 103433, 104341, 112924, 114990, 123239,
        126726, 135811, 139659, 149668, 153913, 163348, 169621, 180015, 186933, 206011, 206189, 227042, 227233, 247788, 250418, 273078, 275976, 300948,
        304142, 331663, 335183, 365513, 369392, 406751, 407092, 448264, 448640, 494013, 494428, 544433, 544890, 605857, 606365, 667691, 668251, 735835,
        736453, 803766, 810935, 877478, 893699, 967033, 984910, 1044050, 1044920, 1150600, 1151570, 1268030, 1269100, 1397450, 1398620, 1540070, 1541370,
        1697250, 1698680, 1854500, 1871040
    ])
    _dP_inline_Re_125 = np.array([5.93109, 5.54354, 5.22463, 4.91025, 4.2207, 3.80075, 3.54753, 3.31117, 3.12106, 2.97108, 2.75394, 2.58829, 2.41584, 2.14646,
        1.9677, 1.6944, 1.58146, 1.49066, 1.3913, 1.29861, 1.20507, 1.12508, 1.05285, 0.971815, 0.909568, 0.833438, 0.747763, 0.704808, 0.66432, 0.632336,
        0.581074, 0.549915, 0.52122, 0.493379, 0.470594, 0.461052, 0.443442, 0.440821, 0.430173, 0.421676, 0.421664, 0.422183, 0.429662, 0.431075, 0.438954,
        0.446788, 0.449097, 0.46, 0.468865, 0.471657, 0.482859, 0.492167, 0.493224, 0.496938, 0.499697, 0.506586, 0.50325, 0.502485, 0.502646, 0.512335,
        0.512408, 0.516812, 0.517255, 0.52129, 0.521276, 0.521127, 0.521113, 0.520979, 0.520951, 0.520816, 0.520351, 0.51557, 0.515536, 0.515148, 0.51047,
        0.510337, 0.505209, 0.504298, 0.49523, 0.493744, 0.480865, 0.480679, 0.480676, 0.476399, 0.47587, 0.467664, 0.466446, 0.453034, 0.456863, 0.457087,
        0.448375, 0.448242, 0.439335, 0.438121, 0.430813, 0.430369, 0.426369, 0.421825, 0.417567, 0.415493, 0.413748, 0.40895, 0.404931, 0.397616, 0.393735,
        0.389087, 0.3814, 0.3813, 0.373863, 0.373765, 0.366718, 0.36344, 0.361258, 0.355173, 0.352681, 0.347915, 0.343752, 0.341039, 0.33675, 0.333804,
        0.332203, 0.331636, 0.325673, 0.325338, 0.322369, 0.321193, 0.316002, 0.315506, 0.309826, 0.30975, 0.303711, 0.303632, 0.297644, 0.297168, 0.291819,
        0.291767, 0.288848, 0.288818, 0.283136, 0.281514, 0.272212, 0.272406, 0.274764, 0.273631, 0.269345, 0.267807, 0.264025, 0.263257, 0.261364, 0.26134,
        0.26129, 0.260266, 0.258656, 0.257969, 0.256205, 0.25619, 0.255938, 0.255937, 0.253893, 0.253614, 0.253287, 0.253278, 0.253208, 0.253199, 0.251176,
        0.2509, 0.250577, 0.250568, 0.250491, 0.25049, 0.250412, 0.250412, 0.250334, 0.250333, 0.250256, 0.250255, 0.25017, 0.250169, 0.250092, 0.250091,
        0.250013, 0.250013, 0.249942, 0.250177, 0.252337, 0.252322, 0.252258, 0.252244, 0.252196, 0.252196, 0.252117, 0.252117, 0.252039, 0.252038, 0.25196,
        0.251959, 0.251881, 0.25188, 0.251802, 0.251802, 0.254214, 0.25446
    ])
    _dP_inline_Re_15 = np.array([2.51237, 2.49499, 2.32876, 2.1908, 1.87501, 1.68786, 1.5828, 1.484, 1.38705, 1.31326, 1.23965, 1.16623, 1.08879, 0.973353, 0.88678,
        0.773105, 0.72388, 0.681815, 0.636838, 0.600558, 0.55801, 0.525955, 0.495741, 0.458149, 0.43183, 0.403164, 0.367208, 0.350209, 0.334532, 0.32017,
        0.299699, 0.288076, 0.276903, 0.268782, 0.258357, 0.25326, 0.250756, 0.249796, 0.245772, 0.243513, 0.243318, 0.245622, 0.252914, 0.25366, 0.257801,
        0.264766, 0.266548, 0.276173, 0.284237, 0.286562, 0.295607, 0.301306, 0.303191, 0.310225, 0.312873, 0.319399, 0.321166, 0.322408, 0.3225, 0.328697,
        0.328793, 0.335219, 0.335506, 0.338421, 0.33871, 0.341653, 0.341978, 0.344926, 0.344907, 0.344818, 0.344809, 0.344713, 0.34469, 0.344681, 0.344584,
        0.344549, 0.341421, 0.341109, 0.341012, 0.341002, 0.331021, 0.337212, 0.337554, 0.33746, 0.337449, 0.334469, 0.334034, 0.33154, 0.330712, 0.33067,
        0.327386, 0.327336, 0.324341, 0.324066, 0.320821, 0.320812, 0.320676, 0.317593, 0.314385, 0.312656, 0.311204, 0.309367, 0.30782, 0.304983, 0.303469,
        0.301435, 0.296008, 0.295937, 0.295845, 0.295844, 0.29001, 0.28882, 0.287086, 0.284229, 0.28322, 0.280847, 0.277486, 0.275535, 0.273856, 0.272992,
        0.272948, 0.272482, 0.267582, 0.267318, 0.264881, 0.264389, 0.262202, 0.261791, 0.257077, 0.257024, 0.254467, 0.254434, 0.252125, 0.25188, 0.246946,
        0.246897, 0.244434, 0.244409, 0.239588, 0.239582, 0.239543, 0.239114, 0.235263, 0.234574, 0.232704, 0.232439, 0.232387, 0.231931, 0.230263,
        0.230024, 0.22998, 0.229521, 0.228102, 0.227634, 0.227563, 0.227562, 0.227491, 0.227491, 0.225446, 0.225198, 0.225135, 0.225127, 0.225065, 0.225057,
        0.224994, 0.224987, 0.224924, 0.224916, 0.224846, 0.224846, 0.224776, 0.224776, 0.224706, 0.224705, 0.224636, 0.224635, 0.224558, 0.224558,
        0.224488, 0.224488, 0.224418, 0.224417, 0.224354, 0.224348, 0.224291, 0.224278, 0.224221, 0.224208, 0.224166, 0.224165, 0.224095, 0.224095,
        0.224025, 0.224025, 0.223955, 0.223955, 0.223885, 0.223885, 0.223815, 0.223815, 0.223752, 0.223745
    ])
    _dP_inline_Re_2 = np.array([0.225144, 0.225088, 0.225039, 0.224988, 0.224877, 0.224799, 0.22475, 0.224701, 0.224652, 0.224617, 0.224562, 0.224511, 0.224462,
        0.224371, 0.224303, 0.224182, 0.224127, 0.224078, 0.224022, 0.223973, 0.223911, 0.223855, 0.223799, 0.22373, 0.223674, 0.223603, 0.223498, 0.223442,
        0.223386, 0.223331, 0.223234, 0.223171, 0.223109, 0.223039, 0.222976, 0.222947, 0.222886, 0.222872, 0.222815, 0.222745, 0.222739, 0.22267, 0.222607,
        0.222601, 0.222566, 0.222509, 0.222496, 0.222439, 0.22237, 0.222357, 0.222307, 0.222238, 0.222225, 0.222176, 0.222155, 0.222106, 0.222086, 0.222072,
        0.222091, 0.224181, 0.224192, 0.224129, 0.224123, 0.224059, 0.224053, 0.223989, 0.223983, 0.225938, 0.226122, 0.226064, 0.226058, 0.225995, 0.22598,
        0.225974, 0.22591, 0.225909, 0.225845, 0.225839, 0.225774, 0.225768, 0.225712, 0.225692, 0.225715, 0.227854, 0.227574, 0.225564, 0.225556, 0.225494,
        0.225473, 0.225472, 0.225402, 0.225401, 0.225337, 0.225331, 0.224173, 0.223979, 0.221897, 0.220812, 0.220777, 0.220743, 0.219816, 0.218518,
        0.217425, 0.215326, 0.214141, 0.212854, 0.209889, 0.209854, 0.207766, 0.207721, 0.204025, 0.20238, 0.199994, 0.196092, 0.194852, 0.192218, 0.189918,
        0.189156, 0.188004, 0.185898, 0.184756, 0.184308, 0.182455, 0.18245, 0.182403, 0.182219, 0.180718, 0.18056, 0.17874, 0.178739, 0.178684, 0.178683,
        0.178633, 0.178614, 0.176826, 0.176836, 0.178506, 0.17851, 0.17846, 0.178455, 0.178427, 0.178422, 0.178376, 0.178366, 0.178326, 0.17831, 0.17827,
        0.178254, 0.178215, 0.178199, 0.179233, 0.179895, 0.179866, 0.179844, 0.179788, 0.179788, 0.179732, 0.179731, 0.179681, 0.179675, 0.179625,
        0.179619, 0.179569, 0.179563, 0.179513, 0.179507, 0.179457, 0.179451, 0.179395, 0.179395, 0.179339, 0.179339, 0.179283, 0.179282, 0.179227,
        0.179226, 0.179165, 0.179165, 0.179109, 0.179109, 0.179053, 0.179053, 0.179002, 0.178997, 0.178952, 0.178941, 0.178896, 0.178885, 0.178852,
        0.178851, 0.178796, 0.178795, 0.17874, 0.178739, 0.178684, 0.178684, 0.178628, 0.178628, 0.178572, 0.178572, 0.178521, 0.178516
    ])
    _dP_inline_Re_25 = np.array([0.349884, 0.344353, 0.339587, 0.334753, 0.324384, 0.31723, 0.31284, 0.308509, 0.304238, 0.301224, 0.296579, 0.292359, 0.288312,
        0.280944, 0.275511, 0.266235, 0.262027, 0.258398, 0.254314, 0.250794, 0.24643, 0.242534, 0.238699, 0.233991, 0.230291, 0.225667, 0.219023, 0.21556,
        0.212151, 0.208795, 0.203116, 0.199504, 0.195956, 0.192086, 0.18867, 0.187117, 0.18384, 0.183136, 0.179837, 0.176255, 0.175964, 0.174204, 0.174155,
        0.17415, 0.174122, 0.174078, 0.174068, 0.175436, 0.175686, 0.175676, 0.175636, 0.175582, 0.175571, 0.175532, 0.175516, 0.176657, 0.177193, 0.175451,
        0.175475, 0.177126, 0.177125, 0.177076, 0.177071, 0.17702, 0.177015, 0.176965, 0.17696, 0.176915, 0.176905, 0.176859, 0.176855, 0.176805, 0.176793,
        0.176789, 0.178483, 0.178481, 0.178431, 0.178426, 0.178375, 0.17837, 0.178326, 0.17831, 0.178309, 0.178259, 0.178253, 0.178209, 0.178203, 0.178154,
        0.178137, 0.178136, 0.178082, 0.178081, 0.17803, 0.178026, 0.177997, 0.177992, 0.177941, 0.177208, 0.176296, 0.175343, 0.174528, 0.175213, 0.176039,
        0.175988, 0.175263, 0.17421, 0.172454, 0.172453, 0.1724, 0.172399, 0.17235, 0.171731, 0.170592, 0.168894, 0.168351, 0.167192, 0.16716, 0.167139,
        0.166121, 0.165455, 0.165443, 0.165435, 0.163918, 0.163771, 0.162422, 0.162121, 0.160642, 0.1605, 0.16361, 0.163528, 0.158824, 0.158823, 0.158779,
        0.158774, 0.15872, 0.158736, 0.160236, 0.160219, 0.158765, 0.15862, 0.158595, 0.158591, 0.15855, 0.158541, 0.158506, 0.158492, 0.158456, 0.158442,
        0.158407, 0.158392, 0.158362, 0.158343, 0.158313, 0.158293, 0.158244, 0.158257, 0.159755, 0.15974, 0.15815, 0.158145, 0.158101, 0.158095, 0.158051,
        0.158046, 0.158002, 0.157996, 0.157952, 0.157947, 0.157898, 0.157898, 0.157849, 0.157848, 0.157799, 0.157799, 0.15775, 0.15775, 0.157696, 0.157695,
        0.157646, 0.157646, 0.157597, 0.157597, 0.157552, 0.157548, 0.157508, 0.157499, 0.157459, 0.157449, 0.15742, 0.157419, 0.157371, 0.15737, 0.157321,
        0.157321, 0.157272, 0.157272, 0.157223, 0.157223, 0.157174, 0.157173, 0.157129, 0.157125
    ])
    _dP_inline_Re_parameters = np.array([_dP_inline_Re_125, _dP_inline_Re_15, _dP_inline_Re_2, _dP_inline_Re_25]).T
    dP_inline_f = RectBivariateSpline(_dP_inline_Res, np.array([1.25, 1.5, 2, 2.5]), _dP_inline_Re_parameters, kx = 3, ky = 3, s = 0.002)


    _dP_inline_correction_parameters = np.array([0.0661637, 0.0767956, 0.0811521, 0.091014, 0.0965946, 0.102863, 0.114663, 0.117455, 0.132109, 0.135196, 0.152089,
        0.168558, 0.19133, 0.192037, 0.21534, 0.217736, 0.244667, 0.247747, 0.324839, 0.392087, 0.446129, 2.2286, 2.3885, 2.63783, 2.92864, 3.00382,
        4.05259, 4.2551, 4.54434, 4.84314, 5.09577, 5.59171, 5.71411
    ])
    _dP_inline_correction_Re_1000 = np.array([7.53832, 6.86113, 6.54006, 6.09616, 5.93568, 5.34629, 5.0612, 4.9696, 4.55428, 4.48266, 4.13474, 3.85306, 3.53216,
        3.52323, 3.22988, 3.19898, 2.89667, 2.86799, 2.31194, 1.99054, 1.798, 0.557156, 0.529536, 0.491093, 0.453615, 0.444813, 0.351914, 0.339127,
        0.322613, 0.30739, 0.295752, 0.27562, 0.271127
    ])
    _dP_inline_correction_Re_10000 = np.array([6.19059, 5.63447, 5.44146, 5.0612, 4.86597, 4.66786, 4.34453, 4.27598, 3.95623, 3.88747, 3.57369, 3.37337, 3.09718,
        3.08911, 2.83271, 2.81518, 2.63689, 2.61495, 2.18225, 1.92462, 1.76564, 0.603218, 0.575945, 0.534133, 0.499018, 0.491093, 0.401321, 0.388344,
        0.370649, 0.353159, 0.339788, 0.316659, 0.311496
    ])
    _dP_inline_correction_Re_100000 = np.array([4.50727, 4.13004, 3.99851, 3.73838, 3.61014, 3.47942, 3.31256, 3.27702, 3.10877, 3.0728, 2.87638, 2.71515, 2.52473,
        2.52055, 2.39441, 2.38256, 2.23167, 2.21606, 1.89994, 1.70733, 1.58802, 0.668818, 0.644362, 0.610869, 0.577472, 0.569658, 0.484948, 0.4719,
        0.454805, 0.438843, 0.426501, 0.404846, 0.399958
    ])
    _dP_inline_correction_Re_1000000 = np.array([3.14214, 2.9391, 2.8673, 2.72361, 2.64416, 2.56157, 2.46985, 2.45024, 2.36473, 2.34829, 2.22756, 2.1327, 2.02212,
        2.01899, 1.92414, 1.91509, 1.81755, 1.80738, 1.63471, 1.50647, 1.43004, 0.74756, 0.730366, 0.704554, 0.675458, 0.668194, 0.588052, 0.575945,
        0.563366, 0.551447, 0.540255, 0.520396, 0.515871
    ])

    _dP_inline_correction_zs = np.array([1E3, 1E4, 1E5, 1E6])
    _dP_inline_correction_Re_parameters = np.array([_dP_inline_correction_Re_1000, _dP_inline_correction_Re_10000, _dP_inline_correction_Re_100000, _dP_inline_correction_Re_1000000]).T
    dP_inline_correction = RectBivariateSpline(_dP_inline_correction_parameters, _dP_inline_correction_zs, _dP_inline_correction_Re_parameters, kx=1, ky=3, s=0.002) # s=0.002
    # RectBivariateSpline does a terrible job

    #import matplotlib.pyplot as plt
    #low, high = min(_dP_inline_correction_parameters), max(_dP_inline_correction_parameters)
    #xs = np.logspace(np.log10(low), np.log10(high), 300000)
    #for i in range(4):
    #    plt.loglog(_dP_inline_correction_parameters, _dP_inline_correction_Re_parameters.T[i, :], '.')
    #    plt.loglog(xs, dP_inline_correction(xs, _dP_inline_correction_zs[i]), '--')
    #plt.show()

def dP_Zukauskas(Re, n, ST, SL, D, rho, Vmax):
    r'''Calculates pressure drop for crossflow across a tube bank
    of tube number n at a specified Re. Method presented in [1]_.
    Also presented in [2]_.

    .. math::
        \Delta P = N_L \chi \left(\frac{\rho V_{max}^2}{2}\right)f

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    n : float
        Number of tube rows, [-]
    ST : float
        Transverse pitch, used only by some conditions, [m]
    SL : float
        Longitudal pitch, used only by some conditions, [m]
    D : float
        Tube outer diameter, [m]
    rho : float
        Fluid density, [kg/m^3]
    Vmax : float
        Maximum velocity, [m/s]

    Returns
    -------
    dP : float
        Pressure drop, [Pa]

    Notes
    -----
    Does not account for effects in a heat exchanger.
    Example 2 is from [2]_. Matches to 0.3%; figures are very approximate.
    Interpolation used with 4 graphs to obtain friction factor and a
    correction factor.

    Examples
    --------
    >>> dP_Zukauskas(Re=13943., n=7, ST=0.0313, SL=0.0343, D=0.0164, rho=1.217, Vmax=12.6)
    235.22916169
    >>> dP_Zukauskas(Re=13943., n=7, ST=0.0313, SL=0.0313, D=0.0164, rho=1.217, Vmax=12.6)
    217.0750033

    References
    ----------
    .. [1] Zukauskas, A. Heat transfer from tubes in crossflow. In T.F. Irvine,
       Jr. and J. P. Hartnett, editors, Advances in Heat Transfer, volume 8,
       pages 93-160. Academic Press, Inc., New York, 1972.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    if not _Zukauskas_correlations_loaded:
        load_Zukauskas_correlations()
    a = ST/D
    b = SL/D
    if a == b:
        parameter = (a-1.)/(b-1.)
        f = float(dP_inline_f(Re, b))
        x = float(dP_inline_correction(parameter, Re))
    else:
        parameter = a/b
        f = float(dP_staggered_f(Re, a))
        x = float(dP_staggered_correction(parameter, Re))

    return n*x*f*rho/2*Vmax**2


"""Note: the smoothing factor was tunned to keep only 7 knots/9 coeffs while
getting near to requiring more knots. The fitting for a digitized graph is
likely to be at the maximum possible accuracy. Any speed increasing fit
function should fit the smoothed function, not the raw data.
"""
Bell_baffle_configuration_tck = implementation_optimize_tck([[0.0, 0.0, 0.0, 0.0, 0.517361, 0.802083, 0.866319,
                                           0.934028, 0.977431, 1.0, 1.0, 1.0, 1.0],
                                 [0.5328447885827443, 0.6821475548927218, 0.9074424740361304,
                                           1.0828783604984582, 1.1485665329698214, 1.1612486065399008,
                                           1.1216591944456349, 1.0762015137576528, 1.0314244120288227,
                                           0.0, 0.0, 0.0, 0.0],
                                3], force_numpy=IS_NUMBA)

Bell_baffle_configuration_obj = lambda x : float(splev(x, Bell_baffle_configuration_tck))

"""Derived with:

fit = Chebfun.from_function(lambda x: Bell_baffle_configuration_obj(0.5*(x+1)), domain=[-1,1], N=8)
cheb2poly(fit.coefficients())[::-1].tolist()

xs = np.linspace(0, 1, 3000)
f = Bell_baffle_configuration_obj
print(max([(f(i)-fit(i*2-1))/f(i) for i in xs]), 'MAX ERR')
print(np.mean([abs(f(i)-fit(i*2-1))/f(i) for i in xs]), 'MEAN ERR')
"""
Bell_baffle_configuration_coeffs = [-17.267087530974095, -17.341072676377735,
    60.38380262590988, 60.78202803861199, -83.86556326987701, -84.74024411236306, 58.66461844872558,
    59.56146082596216, -21.786957547130935, -22.229378707598116, 4.1167302227508, 4.226246012504343,
    -0.3349723004600481, -0.3685826653263089, -0.0629839069257099, 0.35883309630976157,
    0.9345478582873352]

def baffle_correction_Bell(crossflow_tube_fraction, method='spline'):
    r'''Calculate the baffle correction factor `Jc` which accounts for
    the fact that all tubes are not in crossflow to the fluid - some
    have fluid flowing parallel to them because they are situated in
    the "window", where the baffle is cut, instead of between the tips
    of adjacent baffles.

    Equal to 1 for no tubes in the window, increases to 1.15 when the
    windows are small and velocity there is high; decreases to about 0.52
    for very large baffle cuts. Well designed exchangers should typically
    have a value near 1.0.

    Cubic spline interpolation is the default method of retrieving a value
    from the graph, which was digitized with Engauge-Digitizer.

    The interpolation can be slightly slow, so a Chebyshev polynomial was fit
    to a maximum error of 0.142%, average error 0.04% - well within the margin
    of error of the digitization of the graph; this is approximately 10 times
    faster, accessible via the 'chebyshev' method.

    The Heat Exchanger Design Handbook [4]_, [5]_ provides the linear curve
    fit, which covers the "practical" range of baffle cuts 15-45% but not the
    last dip in the graph. This method is not recommended, but can be used via
    the method "HEDH".

    .. math::
        J_c = 0.55 + 0.72Fc

    Parameters
    ----------
    crossflow_tube_fraction : float
        Fraction of tubes which are between baffle tips and not
        in the window, [-]
    method : str, optional
        One of 'chebyshev', 'spline', or 'HEDH'

    Returns
    -------
    Jc : float
        Baffle correction factor in the Bell-Delaware method, [-]

    Notes
    -----
    max: ~1.1536 at ~0.9066
    min: ~0.5328 at 0
    value at 1: ~1.0314

    For the 'spline' method, this function takes ~13 us per call.
    The other two methods are approximately 10x faster.

    Examples
    --------
    For a HX with four groups of tube bundles; the top and bottom being 9
    tubes each, in the window, and the two middle bundles having 41 tubes
    each, for a total of 100 tubes, the fraction between baffle tubes and
    not in the window is 0.82. The correction factor is then:

    >>> baffle_correction_Bell(0.82)
    1.1258554691854046

    References
    ----------
    .. [1] Bell, Kenneth J. Final Report of the Cooperative Research Program on
       Shell and Tube Heat Exchangers. University of Delaware, Engineering
       Experimental Station, 1963.
    .. [2] Bell, Kenneth J. Delaware Method for Shell-Side Design. In Heat
       Transfer Equipment Design, by Shah, R.  K., Eleswarapu Chinna Subbarao,
       and R. A. Mashelkar. CRC Press, 1988.
    .. [3] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [4] Schlünder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    .. [5] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    '''
    if method == 'spline':
        Jc = Bell_baffle_configuration_obj(crossflow_tube_fraction)
    elif method == 'chebyshev':
        return horner(Bell_baffle_configuration_coeffs, 2.0*crossflow_tube_fraction - 1.0)
    elif method == 'HEDH':
        Jc = 0.55 + 0.72*crossflow_tube_fraction
    return Jc


"""Note: The smoothing factor was hand tuned to not overfit from points which
were clearly wrong in the digitization. It will predict values above 1 however
for some values; this must be checked!
"""
Bell_baffle_leakage_x_max = 0.743614

Bell_baffle_leakage_tck = implementation_optimize_tck([[0.0, 0.0, 0.0, 0.0, 0.0213694, 0.0552542, 0.144818,
                                     0.347109, 0.743614, 0.743614, 0.743614, 0.743614],
                                    [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0],
                                    [1.0001228445490002, 0.9988161050974387, 0.9987070557919563, 0.9979385859402731,
                                     0.9970983069823832, 0.96602540121758, 0.955136014969614, 0.9476842472211648,
                                     0.9351143114374392, 0.9059649602818451, 0.9218915266550902, 0.9086000082864022,
                                     0.8934758292610783, 0.8737960765592091, 0.83185251064324, 0.8664296734965998,
                                     0.8349705397843921, 0.809133298969704, 0.7752206120745123, 0.7344035693011536,
                                     0.817047920445813, 0.7694560150930563, 0.7250979336267909, 0.6766754605968431,
                                     0.629304180420512, 0.7137237030611423, 0.6408238328161417, 0.5772000233279148,
                                     0.504889627280836, 0.440579886434288, 0.6239736474980684, 0.5273646894226224,
                                     0.43995388722059986, 0.34359277007615313, 0.26986439252143746, 0.5640689738382749,
                                     0.4540959882735219, 0.35278120580740957, 0.24364672351604122, 0.1606942128340308],
                           3, 1], force_numpy=IS_NUMBA)
Bell_baffle_leakage_obj = lambda x, z : float(bisplev(x, z, Bell_baffle_leakage_tck))


def baffle_leakage_Bell(Ssb, Stb, Sm, method='spline'):
    r'''Calculate the baffle leakage factor `Jl` which accounts for
    leakage between each baffle.
    Cubic spline interpolation is the default method of retrieving a value
    from the graph, which was digitized with Engauge-Digitizer.

    The Heat Exchanger Design Handbook [4]_, [5]_ provides a curve
    fit as well. This method is not recommended, but can be used via
    the method "HEDH".

    .. math::
        J_L = 0.44(1-r_s) + [1 - 0.44(1-r_s)]\exp(-2.2r_{lm})

    .. math::
        r_s = \frac{S_{sb}}{S_{sb} + S_{tb}}

    .. math::
        r_{lm} = \frac{S_{sb} + S_{tb}}{S_m}

    Parameters
    ----------
    Ssb : float
        Shell to baffle leakage area, [m^2]
    Stb : float
        Total baffle leakage area, [m^2]
    Sm : float
        Crossflow area, [m^2]
    method : str, optional
        One of 'spline', or 'HEDH'

    Returns
    -------
    Jl : float
        Baffle leakage factor in the Bell-Delaware method, [-]

    Notes
    -----
    Takes ~5 us per call.
    If the `x` parameter is larger than 0.743614, it is clipped to it.

    The HEDH curve fits are rather poor and only 6x faster to evaluate.
    The HEDH example in [6]_'s spreadsheet has an error and uses 0.044 instead
    of 0.44 in the equation.

    Examples
    --------
    >>> baffle_leakage_Bell(1, 3, 8)
    0.5906621282470
    >>> baffle_leakage_Bell(1, 3, 8, 'HEDH')
    0.5530236260777

    References
    ----------
    .. [1] Bell, Kenneth J. Final Report of the Cooperative Research Program on
       Shell and Tube Heat Exchangers. University of Delaware, Engineering
       Experimental Station, 1963.
    .. [2] Bell, Kenneth J. Delaware Method for Shell-Side Design. In Heat
       Transfer Equipment Design, by Shah, R.  K., Eleswarapu Chinna Subbarao,
       and R. A. Mashelkar. CRC Press, 1988.
    .. [3] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [4] Schlünder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    .. [5] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    .. [6] Hall, Stephen. Rules of Thumb for Chemical Engineers, Fifth Edition.
       5th edition. Oxford ; Waltham , MA: Butterworth-Heinemann, 2012.
    '''
    x = (Ssb + Stb)/Sm
    if x > Bell_baffle_leakage_x_max:
        x = Bell_baffle_leakage_x_max
    z = Ssb/(Ssb + Stb)
    if z > 1.0 or z < 0.0:
        raise ValueError('Ssb/(Ssb + Stb) must be between 0 and 1')
    if method == 'spline':
        Jl = Bell_baffle_leakage_obj(x, z)
        Jl = min(float(Jl), 1.0)
    elif method == 'HEDH':
        # Hemisphere uses 0.44 as coefficient, rules of thumb uses 0.044 in spreadsheet
        Jl = 0.44*(1.0 - z) + (1.0 - 0.44*(1.0 - z))*exp(-2.2*x)
    return Jl


Bell_bundle_bypass_x_max = 0.69532
Bell_bundle_bypass_high_spl = implementation_optimize_tck([[0.0, 0.0, 0.0, 0.0, 0.434967, 0.69532, 0.69532, 0.69532, 0.69532],
                               [0.0, 0.0, 0.0, 0.0, 0.1, 0.16666666666666666, 0.5, 0.5, 0.5, 0.5],
                               [0.9992518012440722, 0.9989007625058475, 1.0018411070735471, 0.9941457497302127,
                                         1.0054152224744488, 1.0000002120327414, 0.8193710201718651, 0.8906557463728106,
                                         0.9236476444228989, 0.9466472125718047, 1.002564972451326, 1.0000001328221189,
                                         0.6099796629837915, 0.7779198818216049, 0.8128716798013131, 0.935864247770527,
                                         0.932707600057425, 0.9999978349038892, 0.46653330555544065, 0.6543895994806808,
                                         0.7244471950409509, 0.8599376452211228, 0.9622021460141503, 0.9999989177211911,
                                         0.42206076955873406, 0.6230810793228677, 0.6903177740858685, 0.8544752061829647,
                                         0.9373953303873518, 0.9999983130568033],
                               3, 3], force_numpy=IS_NUMBA)
Bell_bundle_bypass_high_obj = lambda x, y: float(bisplev(x, y, Bell_bundle_bypass_high_spl))


Bell_bundle_bypass_low_spl = implementation_optimize_tck([[0.0, 0.0, 0.0, 0.0, 0.434967, 0.69532, 0.69532, 0.69532, 0.69532],
                              [0.0, 0.0, 0.0, 0.0, 0.1, 0.16666666666666666, 0.5, 0.5, 0.5, 0.5],
                              [1.0015970586968514, 0.9976793473578099, 1.0037098839305505, 0.9953304170745584,
                                        1.0031587186511541, 1.00000028406872, 0.8027498596582175, 0.9050562101782131,
                                        0.9133675590990569, 0.9611563766991582, 0.9879481797594364, 0.9999988983171519,
                                        0.5813496854191834, 0.7520908533825839, 0.7927234268976187, 0.9090698658126287,
                                        0.9857133220039945, 0.9999986096716597, 0.43493461007512263, 0.6478801160783917,
                                        0.6961255921403956, 0.861432071791341, 0.9243020549338703, 0.999997894037133,
                                        0.39110224578093694, 0.606829928454368, 0.6600680810505178, 0.8482579667665061,
                                        0.9223728343461776, 0.9999978298360785],
                                   3, 3], force_numpy=IS_NUMBA)
Bell_bundle_bypass_low_obj = lambda x, y : float(bisplev(x, y, Bell_bundle_bypass_low_spl))


def bundle_bypassing_Bell(bypass_area_fraction, seal_strips, crossflow_rows,
                          laminar=False, method='spline'):
    r'''Calculate the bundle bypassing effect `Jb` according to the
    Bell-Delaware method for heat exchanger design.
    Cubic spline interpolation is the default method of retrieving a value
    from the graph, which was digitized with Engauge-Digitizer.

    The Heat Exchanger Design Handbook [4]_ provides a curve
    fit as well. This method is not recommended, but can be used via
    the method "HEDH":

    .. math::
        J_b = \exp\left[-1.25 F_{sbp} (1 -  {2r_{ss}}^{1/3} )\right]

    For laminar flows, replace 1.25 with 1.35.

    Parameters
    ----------
    bypass_area_fraction : float
        Fraction of the crossflow area which is not blocked by a baffle or
        anything else and available for bypassing, [-]
    seal_strips : int
        Number of seal strips per side of a baffle added to prevent bypassing,
        [-]
    crossflow_rows : int
        The number of tube rows in the crosslfow of the baffle, [-]
    laminar : bool
        Whether to use the turbulent correction values or the laminar ones;
        the Bell-Delaware method uses a Re criteria of 100 for this, [-]
    method : str, optional
        One of 'spline', or 'HEDH'

    Returns
    -------
    Jb : float
        Bundle bypassing effect correction factor in the Bell-Delaware method,
        [-]

    Notes
    -----
    Takes ~5 us per call.
    If the `bypass_area_fraction` parameter is larger than 0.695, it is clipped
    to it.

    Examples
    --------
    >>> bundle_bypassing_Bell(0.5, 5, 25)
    0.8469611760884599

    >>> bundle_bypassing_Bell(0.5, 5, 25, method='HEDH')
    0.8483210970579099

    References
    ----------
    .. [1] Bell, Kenneth J. Final Report of the Cooperative Research Program on
       Shell and Tube Heat Exchangers. University of Delaware, Engineering
       Experimental Station, 1963.
    .. [2] Bell, Kenneth J. Delaware Method for Shell-Side Design. In Heat
       Transfer Equipment Design, by Shah, R.  K., Eleswarapu Chinna Subbarao,
       and R. A. Mashelkar. CRC Press, 1988.
    .. [3] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [4] Schlünder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    '''
    z = seal_strips/crossflow_rows
    x = bypass_area_fraction
    if method == 'spline':
        if x > Bell_bundle_bypass_x_max:
            x = Bell_bundle_bypass_x_max

        if laminar:
            Jb = Bell_bundle_bypass_low_obj(x, z)
        else:
            Jb = Bell_bundle_bypass_high_obj(x, z)
        Jb = min(Jb, 1.0)
    elif method == 'HEDH':
        c = 1.35 if laminar else 1.25
        Jb = exp(-c*x*(1.0 - (2.0*z)**(1/3.)))
    return Jb


def unequal_baffle_spacing_Bell(baffles, baffle_spacing,
                                baffle_spacing_in=None,
                                baffle_spacing_out=None,
                                laminar=False):
    r'''Calculate the correction factor for unequal baffle spacing `Js`,
    which accounts for higher velocity of fluid flow and greater heat transfer
    coefficients when the in and/or out baffle spacing is less than the
    standard spacing.

    .. math::
        J_s = \frac{(n_b - 1) + (B_{in}/B)^{(1-n_b)} + (B_{out}/B)^{(1-n_b)}}
        {(n_b - 1) + (B_{in}/B) + (B_{out}/B)}

    Parameters
    ----------
    baffles : int
        Number of baffles, [-]
    baffle_spacing : float
        Average spacing between one end of one baffle to the start of
        the next baffle for non-exit baffles, [m]
    baffle_spacing_in : float, optional
        Spacing between entrace to first baffle, [m]
    baffle_spacing_out : float, optional
        Spacing between last baffle and exit, [m]
    laminar : bool, optional
        Whether to use the turbulent exponent or the laminar one;
        the Bell-Delaware method uses a Re criteria of 100 for this, [-]

    Returns
    -------
    Js : float
        Unequal baffle spacing correction factor, [-]

    Notes
    -----

    Examples
    --------
    >>> unequal_baffle_spacing_Bell(16, .1, .15, 0.15)
    0.9640087802805195

    References
    ----------
    .. [1] Bell, Kenneth J. Final Report of the Cooperative Research Program on
       Shell and Tube Heat Exchangers. University of Delaware, Engineering
       Experimental Station, 1963.
    .. [2] Bell, Kenneth J. Delaware Method for Shell-Side Design. In Heat
       Transfer Equipment Design, by Shah, R.  K., Eleswarapu Chinna Subbarao,
       and R. A. Mashelkar. CRC Press, 1988.
    .. [3] Schlünder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    .. [4] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    .. [5] Hall, Stephen. Rules of Thumb for Chemical Engineers, Fifth Edition.
       5th edition. Oxford ; Waltham , MA: Butterworth-Heinemann, 2012.
    '''
    if baffle_spacing_in is None:
        baffle_spacing_in = baffle_spacing
    if baffle_spacing_out is None:
        baffle_spacing_out = baffle_spacing
    n = 1.0/3.0 if laminar else 0.6
    Js = ((baffles - 1.0) + (baffle_spacing_in/baffle_spacing)**(1.0 - n)
          + (baffle_spacing_out/baffle_spacing)**(1.0 - n))/((baffles - 1.0)
          + (baffle_spacing_in/baffle_spacing)
          + (baffle_spacing_out/baffle_spacing))
    return Js


def laminar_correction_Bell(Re, total_row_passes):
    r'''Calculate the correction factor for adverse temperature gradient built
    up in laminar flow `Jr`.

    This correction begins at Re = 100, and is interpolated between the value
    of the formula until Re = 20, when it is the value of the formula. It is
    1 for Re >= 100. The value of the formula is not allowed to be less than
    0.4.

    .. math::
        Jr^* = \left(\frac{10}{N_{row,passes,tot}}\right)^{0.18}

    Parameters
    ----------
    Re : float
        Shell Reynolds number in the Bell-Delaware method, [-]
    total_row_passes : int
        The total number of rows passed by the fluid, including those in
        windows and counting repeat passes of tube rows, [-]

    Returns
    -------
    Jr : float
        Correction factor for adverse temperature gradient built up in laminar
        flow, [-]

    Notes
    -----
    [5]_ incorrectly uses the number of tube rows per crosslfow section, not
    total.

    Examples
    --------
    >>> laminar_correction_Bell(30, 80)
    0.7267995454361379

    References
    ----------
    .. [1] Bell, Kenneth J. Final Report of the Cooperative Research Program on
       Shell and Tube Heat Exchangers. University of Delaware, Engineering
       Experimental Station, 1963.
    .. [2] Bell, Kenneth J. Delaware Method for Shell-Side Design. In Heat
       Transfer Equipment Design, by Shah, R.  K., Eleswarapu Chinna Subbarao,
       and R. A. Mashelkar. CRC Press, 1988.
    .. [3] Schlünder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    .. [4] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    .. [5] Hall, Stephen. Rules of Thumb for Chemical Engineers, Fifth Edition.
       5th edition. Oxford ; Waltham , MA: Butterworth-Heinemann, 2012.
    '''
    if Re > 100.0:
        return 1.0
    Jrr = (10.0/total_row_passes)**0.18
    if Re < 20.0:
        Jr = Jrr
    else:
        Jr = Jrr + ((20.0-Re)/80.0)*(Jrr - 1.0)
    if Jr < 0.4:
        Jr = 0.4
    return Jr
