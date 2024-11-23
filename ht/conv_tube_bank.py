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

dP_staggered_f_tck = implementation_optimize_tck([
    [1.00000e+01, 1.00000e+01, 1.00000e+01, 1.00000e+01, 1.16733e+01,
 1.31024e+01, 1.40153e+01, 1.49918e+01, 1.71536e+01, 1.85267e+01,
 1.98182e+01, 2.07261e+01, 2.22430e+01, 2.37936e+01, 2.67057e+01,
 3.22732e+01, 3.48580e+01, 3.72879e+01, 4.10554e+01, 4.44722e+01,
 4.78949e+01, 5.12337e+01, 5.53369e+01, 6.51821e+01, 7.04025e+01,
 7.60437e+01, 8.21368e+01, 8.87182e+01, 9.51284e+01, 1.03386e+02,
 1.08398e+02, 1.29188e+02, 1.55444e+02, 1.68914e+02, 1.82793e+02,
 1.97771e+02, 2.23559e+02, 2.78915e+02, 3.35015e+02, 4.97559e+02,
 7.31917e+02, 1.11914e+03, 1.74856e+03, 2.30827e+03, 3.36357e+03,
 4.36703e+03, 4.85424e+03, 6.81715e+03, 9.91409e+03, 1.96837e+04,
 3.98384e+04, 8.69147e+04, 1.77710e+05, 3.29652e+05, 4.53370e+05,
 6.17548e+05, 1.38874e+06, 2.75675e+06, 2.75675e+06, 2.75675e+06,
 2.75675e+06],
    [1.25, 1.25, 1.25, 1.25, 2.5 , 2.5 , 2.5 , 2.5 ],
    [ 23.993727564949694 , -10.244567028024317 ,  10.175863843810976 ,
   1.7999490702171597,  23.522204247227894 , -10.206973826192078 ,
  10.10518140310304  ,   1.780787357025903 ,  21.056398673217306 ,
  -8.809585811775245 ,   9.191305635699116 ,   1.7221653902388994,
  19.30502693908655  ,  -8.454343171481037 ,   8.87084592501417  ,
   1.6628307168281609,  17.765109289150598 ,  -7.7133771799002115,
   8.35255872722289  ,   1.6185259877007414,  16.183640933328732 ,
  -6.833873600289119 ,   7.722187414020649 ,   1.5690827476330849,
  14.694535763926543 ,  -6.150405226666587 ,   7.22393729697023  ,
   1.5213836637875222,  13.605573060235757 ,  -5.4221574812289735,
   6.657567596977061 ,   1.4767279611552022,  13.00457363939894  ,
  -5.517651045065633 ,   6.649512891288891 ,   1.4466870011383937,
  11.995125279422092 ,  -4.905206010958158 ,   6.23685497204518  ,
   1.4176362748134403,  11.320923863947378 ,  -4.3609663889341554,
   5.823387965136728 ,   1.3889295522585867,  10.55081833434363  ,
  -4.2246761575331195,   5.6424642699146474,   1.3494878611293746,
   9.136637777542301 ,  -3.3738486997113624,   4.988770673854508 ,
   1.2912945512808613,   8.142082809925073 ,  -3.0640564076701335,
   4.673896497087432 ,   1.2394666235276033,   7.307435480320903 ,
  -2.4507512853218927,   4.184651311906238 ,   1.197270318810632 ,
   6.823162531391426 ,  -2.341520042292131 ,   4.031868925359543 ,
   1.1654503211708163,   6.292609559336966 ,  -2.172105165179806 ,
   3.851181307698692 ,   1.1342864317994863,   5.7736639984989955,
  -1.8024125693369684,   3.540068552249239 ,   1.103663937920097 ,
   5.456996186731616 ,  -1.6944442210715904,   3.394074222435801 ,
   1.0771356178101834,   5.04760166750843  ,  -1.5241640716734404,
   3.2252084065116606,   1.051345959799993 ,   4.547228150208529 ,
  -1.2330993185368226,   2.940049991699233 ,   1.0141134328178203,
   4.14534270298608  ,  -1.0164114172228296,   2.7045876551545796,
   0.9793370063645501,   3.771066238885074 ,  -0.8748901176380479,
   2.528229788862222 ,   0.9470018923394338,   3.578714777343189 ,
  -0.7900362603927293,   2.4000658218893482,   0.9229840539840486,
   3.2657684683098536,  -0.5425913450233749,   2.185378113310861 ,
   0.8999825646391477,   3.1380949473394075,  -0.5491995911762386,
   2.1192169444451854,   0.8771813561150372,   2.8710190479957056,
  -0.3178235005664693,   1.9163896313226092,   0.8571821847817371,
   2.7427220096515885,  -0.350653244945934 ,   1.8829269684603023,
   0.8348323725060139,   2.4836006182031682,  -0.1352407446908991,
   1.6647896666111899,   0.8142139723548999,   2.2164004811891416,
  -0.0371909044098638,   1.492301567301747 ,   0.7533707758356281,
   1.9945634411125996,   0.1123996806357918,   1.2956302127875685,
   0.7249191411158085,   1.842686059394845 ,   0.1607181649910975,
   1.2153169523034464,   0.6897021922030252,   1.7377903418611442,
   0.289292795896553 ,   1.08344601062934  ,   0.6729372197342073,
   1.6396914034896823,   0.1989615127535343,   1.0850029234159282,
   0.6499001080984353,   1.487502515283381 ,   0.3730846731246894,
   0.9013971733000221,   0.6182632804995061,   1.3522211422419148,
   0.2911476890926835,   0.8964977394892637,   0.5767990957491218,
   1.1810712591370587,   0.3394790407828216,   0.7652655011740844,
   0.5323341532737186,   1.02811191715534  ,   0.2887724828326413,
   0.7066399775545615,   0.4778931341762794,   0.8895112183187074,
   0.3210418642443666,   0.6143186197557094,   0.4467010633652874,
   0.7932715410752142,   0.3211662514007235,   0.5619909984438982,
   0.4128889836673581,   0.7375513367809549,   0.3144259758348928,
   0.528973351907581 ,   0.4094681633270362,   0.6972754846297644,
   0.3205333114809629,   0.4918522964719353,   0.379830474083385 ,
   0.6641316600566393,   0.281121553527687 ,   0.509197543189297 ,
   0.3783415752249853,   0.6244559596650743,   0.2837572552667433,
   0.4993191757674828,   0.3669254758741835,   0.6023735792473888,
   0.2761933758206954,   0.4987865496444748,   0.364477871468622 ,
   0.5590674813131451,   0.2621282514144033,   0.4629707633705547,
   0.344500324531334 ,   0.4782618774782103,   0.2485411042324743,
   0.3784315811630596,   0.3033924096858183,   0.3937668979943847,
   0.2127684318169733,   0.3154828002796004,   0.2539053524737366,
   0.31876438844008  ,   0.158795177701543 ,   0.2564298818590743,
   0.2026052448806063,   0.2666571057762706,   0.125595092472467 ,
   0.2213179726704931,   0.1679289011957269,   0.2145850940558933,
   0.0821608103084532,   0.1746802190336271,   0.1296792214950967,
   0.226583087040426 ,   0.1193363052691195,   0.1939070597138282,
   0.1415527224622063,   0.2236870028120291,   0.187323377693516 ,
   0.1956693312744808,   0.1768800813854576,   0.2267018978658211,
   0.1724455167753975,   0.1962302349704176,   0.1908917559338361,
   0.2244440017710325,   0.1881858244890039,   0.2045576574847943,
   0.1746985249870369,   0.2264047973889584,   0.1728059947546293,
   0.2044769358453351,   0.1891991544177786,   0.2281141554162358,
   0.1740746507220272,   0.2115805378214652,   0.1837416784138279],
    3, 3
])

dP_staggered_correction_tck = implementation_optimize_tck([
    [0.4387  , 0.4387  , 0.609319, 0.84214 , 1.22243 , 1.45385 , 2.22751 ,
 3.54351 , 3.54351 ],
    [   100.,    100.,    100.,    100., 100000., 100000., 100000., 100000.],
    [  0.9974058596752864,   2.2479507236683025,  -2.927557461079741 ,
   1.4506256824457617,   0.9973256008060759,   1.2891120004637224,
   2.57756043014528  ,   1.308194055102965 ,   0.99328828308559  ,
   0.9777071534489982,   2.690730180307897 ,   1.1511575741885283,
   1.0225516740093321,   0.654636148703268 ,   3.5328468308711605,
   0.9841004806410495,   1.1242321684355248,  -1.285317900605766 ,
  18.760679189927053 ,   0.9465800254548173,   1.4029517756874132,
  -6.782908749288587 ,  62.67683464108982  ,   0.9308263957725467,
   1.6896512655274156, -12.264942447053453 , 107.33977758986381  ,
   0.9417015493955058],
    1, 3
])

dP_inline_f_tck = implementation_optimize_tck([
    [2.85094e+01, 2.85094e+01, 2.85094e+01, 2.85094e+01, 3.29727e+01,
 3.53563e+01, 4.12101e+01, 5.26143e+01, 5.91070e+01, 6.37533e+01,
 8.29896e+01, 1.24713e+02, 1.57106e+02, 2.78938e+02, 5.28457e+02,
 7.95679e+02, 1.10738e+03, 1.61619e+03, 4.85232e+03, 6.54585e+03,
 8.11339e+03, 1.45214e+04, 5.39717e+04, 9.84306e+04, 1.69621e+05,
 6.05857e+05, 1.87104e+06, 1.87104e+06, 1.87104e+06, 1.87104e+06],
    [1.25, 1.25, 1.25, 1.25, 2.5 , 2.5 , 2.5 , 2.5 ],
    [ 5.930973938528779 , -1.4817265127271453,  0.455379701866362 ,
  0.3498836798224479,  5.658593208961423 , -0.864342651441936 ,
  0.0859302669335574,  0.3461356038232322,  5.313966014275998 ,
 -0.9488816374714495,  0.1958290207282649,  0.3409206025216998,
  4.7479017538057215, -0.9103110524004313,  0.2580592449510698,
  0.3323409371000172,  3.985380347891111 , -0.8082048062363013,
  0.3083826543793066,  0.3209043535273658,  3.386764509595187 ,
 -0.6210063941958163,  0.2773912522639271,  0.3101279173763394,
  2.997259057262402 , -0.6436424918427293,  0.3541133458508599,
  0.301791078024438 ,  2.5313481368123476, -0.4636698687516479,
  0.3080129633362942,  0.291588283194631 ,  1.9067355729954962,
 -0.4062699968822227,  0.3701523002949395,  0.2746968125881167,
  1.4860009590268453, -0.3047213533591268,  0.3724777524638413,
  0.2586442595556128,  0.9575648609085928, -0.204166339279796 ,
  0.3941734724654094,  0.2357333057087107,  0.617062433119386 ,
 -0.1088086799459149,  0.3940863377254021,  0.2089273471639561,
  0.4722964733105959, -0.0558097050630406,  0.3870405603703184,
  0.1913524081468901,  0.3995262320386994, -0.025364894783629 ,
  0.3882892706432594,  0.1722324338381349,  0.4645326824064042,
  0.00678125270346  ,  0.3521348515608552,  0.1753669353885564,
  0.5446976745961594,  0.152419457668676 ,  0.2469868959242308,
  0.176815573751498 ,  0.5104609851427283,  0.1262432245605664,
  0.2778791531974582,  0.1778458681575732,  0.4747778372905396,
  0.1841815013323758,  0.2390262532478637,  0.1784924161355043,
  0.4331707217581522,  0.1966086663135327,  0.2420381252055944,
  0.1781297356716778,  0.3463614484492485,  0.2262216378962929,
  0.1657411941914758,  0.1704140168055773,  0.3137425743888637,
  0.2228368744465938,  0.1287382966547652,  0.1588140647661027,
  0.2666982751158755,  0.2063058104326992,  0.1587051313760535,
  0.158741122371858 ,  0.2469799643017819,  0.2074867426640942,
  0.1619911852835217,  0.1580881696361456,  0.2548927358191072,
  0.2072520441235713,  0.1588874948041774,  0.1572373415378531,
  0.2498062734215278,  0.2019373850118195,  0.1635492298499229,
  0.1573584818834395,  0.2539191494035861,  0.2008897736617199,
  0.1630751863377278,  0.1570998393113034],
    3, 3
])

dP_inline_correction_tck = implementation_optimize_tck([
    [0.02    , 0.02    , 0.02    , 0.02    , 0.066164, 0.08    , 0.16    ,
 0.32    , 0.64    , 1.28    , 2.56    , 5.7141  , 5.7141  , 5.7141  ,
 5.7141  ],
    [   1000.,    1000.,    1000.,    1000., 1000000., 1000000., 1000000.,
 1000000.],
    [ 1.6050000000000011e+01, -9.9185525084848649e+01,  8.0188908903030449e+02,
  5.3722000000000012e+00,  9.7642684825332609e+00, -5.9212586722045586e+01,
  4.8681375915163028e+02,  3.8371626170338349e+00,  8.2631729047131390e+00,
 -5.3624502687061614e+01,  4.4328465663719857e+02,  3.3370326010388487e+00,
  5.0902027741289038e+00, -2.5039923342821140e+01,  2.0959688954308831e+02,
  2.4638592963292831e+00,  3.3252028425473772e+00, -1.2251528629540998e+01,
  1.1583792320215544e+02,  2.0227774411216131e+00,  1.7798328575941409e+00,
  2.7213811631652502e+00, -1.5319618243273222e+01,  1.4299822635240957e+00,
  1.1740829684913225e+00,  9.4029637834424262e-01,  3.9974603020412691e+00,
  1.1420933087379526e+00,  6.3974151735102358e-01,  3.5438685909104040e+00,
 -2.2199543860124965e+01,  8.1942218695180635e-01,  4.0902459885654202e-01,
  1.0486758802627454e+00, -2.1456210505006823e-01,  6.4235546429644219e-01,
  3.0522194412697784e-01,  2.4239289999078180e+00, -1.3982480984625624e+01,
  5.5177093110224618e-01,  2.7113000000000009e-01,  1.8811070427272716e+00,
 -9.5202273363636269e+00,  5.1587000000000016e-01],
    3, 3
])


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
    235.2291
    >>> dP_Zukauskas(Re=13943., n=7, ST=0.0313, SL=0.0313, D=0.0164, rho=1.217, Vmax=12.6)
    161.147

    References
    ----------
    .. [1] Zukauskas, A. Heat transfer from tubes in crossflow. In T.F. Irvine,
       Jr. and J. P. Hartnett, editors, Advances in Heat Transfer, volume 8,
       pages 93-160. Academic Press, Inc., New York, 1972.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    a = ST/D
    b = SL/D
    if a == b:
        parameter = (a-1.)/(b-1.)
        f = float(bisplev(Re, b, dP_inline_f_tck))
        x = float(bisplev(parameter, Re, dP_inline_correction_tck))

    else:
        parameter = a/b
        f = float(bisplev(Re, a, dP_staggered_f_tck))
        x = float(bisplev(parameter, Re, dP_staggered_correction_tck))
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
    0.84696117608

    >>> bundle_bypassing_Bell(0.5, 5, 25, method='HEDH')
    0.84832109705

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
