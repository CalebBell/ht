'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from math import pi, radians, sin

from fluids.friction import Kumar_beta_list, friction_plate_Martin_1999, friction_plate_Martin_VDI

__all__ = ['Nu_plate_Kumar', 'Nu_plate_Martin', 'Nu_plate_Muley_Manglik',
           'Nu_plate_Khan_Khan']


Kumar_ms = [[0.349, 0.663, 0.663],
      [0.349, 0.598, 0.663],
      [0.333, 0.591, 0.732],
      [0.326, 0.529, 0.703],
      [0.326, 0.503, 0.718]]

Kumar_C1s = [[0.718, 0.348, 0.348],
       [0.718, 0.400, 0.300],
       [0.630, 0.291, 0.130],
       [0.562, 0.306, 0.108],
       [0.562, 0.331, 0.087]]

Kumar_Nu_Res = [[10.0, 10.0],
          [10.0, 100.0],
          [20.0, 300.0],
          [20.0, 400.0],
          [20.0, 500.0]]


def Nu_plate_Kumar(Re, Pr, chevron_angle, mu=None, mu_wall=None):
    r'''Calculates Nusselt number for single-phase flow in a
    **well-designed** Chevron-style plate heat exchanger according to [1]_.
    The data is believed to have been developed by APV International Limited,
    since acquired by SPX Corporation. This uses a curve fit of that data
    published in [2]_.

    .. math::
        Nu = C_1 Re^m Pr^{0.33}\left(\frac{\mu}{\mu_{wall}}\right)^{0.17}

    `C1` and `m` are coefficients looked up in a table, with varying ranges
    of Re validity and chevron angle validity. See the source for their
    exact values. The wall fluid property correction is included only if the
    viscosity values are provided.

    Parameters
    ----------
    Re : float
        Reynolds number with respect to the hydraulic diameter of the channels,
        [-]
    Pr : float
        Prandtl number calculated with bulk fluid properties, [-]
    chevron_angle : float
        Angle of the plate corrugations with respect to the vertical axis
        (the direction of flow if the plates were straight), between 0 and
        90. Many plate exchangers use two alternating patterns; use their
        average angle for that situation [degrees]
    mu : float, optional
        Viscosity of the fluid at the bulk (inlet and outlet average)
        temperature, [Pa*s]
    mu_wall : float, optional
        Viscosity of fluid at wall temperature, [Pa*s]

    Returns
    -------
    Nu : float
        Nusselt number with respect to `Dh`, [-]

    Notes
    -----
    Data on graph from Re=0.1 to Re=10000, with chevron angles 30 to 65 degrees.
    See `PlateExchanger` for further clarification on the definitions.

    It is believed the constants used in this correlation were curve-fit to
    the actual graph in [1]_ by the author of [2]_ as there is no

    As the coefficients change, there are numerous small discontinuities,
    although the data on the graphs is continuous with sharp transitions
    of the slope.

    The author of [1]_ states clearly this correlation is "applicable only to
    well designed Chevron PHEs".

    Examples
    --------
    >>> Nu_plate_Kumar(Re=2000, Pr=0.7, chevron_angle=30)
    47.757818892853955

    With the wall-correction factor included:

    >>> Nu_plate_Kumar(Re=2000, Pr=0.7, chevron_angle=30, mu=1E-3, mu_wall=8E-4)
    49.604284135097544

    References
    ----------
    .. [1] Kumar, H. "The plate heat exchanger: construction and design." In
       First U.K. National Conference on Heat Transfer: Held at the University
       of Leeds, 3-5 July 1984, Institute of Chemical Engineering Symposium
       Series, vol. 86, pp. 1275-1288. 1984.
    .. [2] Ayub, Zahid H. "Plate Heat Exchanger Literature Survey and New Heat
       Transfer and Pressure Drop Correlations for Refrigerant Evaporators."
       Heat Transfer Engineering 24, no. 5 (September 1, 2003): 3-16.
       doi:10.1080/01457630304056.
    '''
    # Uses the standard diameter as characteristic diameter
    beta_list_len = len(Kumar_beta_list)

    for i in range(beta_list_len):
        if chevron_angle <= Kumar_beta_list[i]:
            C1_options, m_options, Re_ranges = Kumar_C1s[i], Kumar_ms[i], Kumar_Nu_Res[i]
            break
        elif i == beta_list_len-1:
            C1_options, m_options, Re_ranges = Kumar_C1s[-1], Kumar_ms[-1], Kumar_Nu_Res[-1]

    Re_len = len(Re_ranges)

    for j in range(Re_len):
        if Re <= Re_ranges[j]:
            C1, m = C1_options[j], m_options[j]
            break
        elif j == Re_len-1:
            C1, m = C1_options[-1], m_options[-1]

    Nu = C1*Re**m*Pr**0.33
    if mu_wall is not None and mu is not None:
        Nu *= (mu/mu_wall)**0.17
    return Nu


def Nu_plate_Martin(Re, Pr, chevron_angle, variant='1999'):
    r'''Calculates Nusselt number for single-phase flow in a
    Chevron-style plate heat exchanger according to [1]_, also shown in [2]_
    and [3]_.

    .. math::
        Nu = 0.122 Pr^{1/3} \left[f_d Re^2 \sin (2\phi)\right]^{0.374}

    The Darcy friction factor should be calculated with the Martin (1999)
    friction factor correlation, as that is what the power of 0.374 was
    regressed with. It can be altered to a later formulation by Martin in the
    VDI Heat Atlas 2E, which increases the calculated heat transfer friction
    slightly.

    Parameters
    ----------
    Re : float
        Reynolds number with respect to the hydraulic diameter of the channels,
        [-]
    Pr : float
        Prandtl number calculated with bulk fluid properties, [-]
    chevron_angle : float
        Angle of the plate corrugations with respect to the vertical axis
        (the direction of flow if the plates were straight), between 0 and
        90. Many plate exchangers use two alternating patterns; use their
        average angle for that situation [degrees]
    variant : str
        One of '1999' or 'VDI'; chooses between the two Martin friction
        factor correlations, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to `Dh`, [-]

    Notes
    -----
    Based on experimental data from Re from 200 - 10000 and enhancement
    factors calculated with chevron angles of 0 to 80 degrees. See
    `PlateExchanger` for further clarification on the definitions.

    Note there is a discontinuity at Re = 2000 for the transition from
    laminar to turbulent flow, arising from the friction factor correlation's
    transition ONLY, although the literature suggests the transition
    is actually smooth.

    A viscosity correction power for liquid flows of (1/6) is suggested, and
    for gases, no correction factor.

    Examples
    --------
    >>> Nu_plate_Martin(Re=2000.0, Pr=.7, chevron_angle=45.0)
    30.427601053757

    References
    ----------
    .. [1] Martin, Holger. "A Theoretical Approach to Predict the Performance
       of Chevron-Type Plate Heat Exchangers." Chemical Engineering and
       Processing: Process Intensification 35, no. 4 (January 1, 1996): 301-10.
       https://doi.org/10.1016/0255-2701(95)04129-X.
    .. [2] Martin, Holger. "Economic optimization of compact heat exchangers."
       EF-Conference on Compact Heat Exchangers and Enhancement Technology for
       the Process Industries, Banff, Canada, July 18-23, 1999, 1999.
       https://publikationen.bibliothek.kit.edu/1000034866.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    if variant == '1999':
        fd = friction_plate_Martin_1999(Re, chevron_angle)
    elif variant == 'VDI':
        fd = friction_plate_Martin_VDI(Re, chevron_angle)
    else:
        raise ValueError("Supported friction factor correlations are Martin's"
                        " '1999' correlation or his 'VDI' correlation only")

    # VDI, original, and Björn Palm and Joachim Claesson recommend 0.122 leading coeff
    # The 0.205 in some publications is what happens when the friction factor
    # is in a fanning basis; = 4^0.374*1.22 = 2.048944
    Nu = 0.122*Pr**(1/3.)*(fd*Re*Re*sin(2.0*radians(chevron_angle)))**0.374
    return Nu


def Nu_plate_Muley_Manglik(Re, Pr, chevron_angle, plate_enlargement_factor):
    r'''Calculates Nusselt number for single-phase flow in a
    Chevron-style plate heat exchanger according to [1]_, also shown in [2]_.

    .. math::
        Nu = [0.2668 - 0.006967(\beta) + 7.244\times 10^{-5}(\beta)^2]
        \times[20.7803 - 50.9372\phi + 41.1585\phi^2 - 10.1507\phi^3]
        \times Re^{[0.728 + 0.0543\sin[(2\pi\beta/90) + 3.7]]} Pr^{1/3}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to the hydraulic diameter of the channels,
        [-]
    Pr : float
        Prandtl number calculated with bulk fluid properties, [-]
    chevron_angle : float
        Angle of the plate corrugations with respect to the vertical axis
        (the direction of flow if the plates were straight), between 0 and
        90. Many plate exchangers use two alternating patterns; use their
        average angle for that situation [degrees]
    plate_enlargement_factor : float
        The extra surface area multiplier as compared to a flat plate
        caused the corrugations, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to `Dh`, [-]

    Notes
    -----
    The correlation as presented in [1]_ suffers from a typo, with a
    coefficient of 10.51 instead of 10.15. Several more decimal places were
    published along with the corrected typo in [2]_. This has a *very large*
    difference if not implemented.

    The viscosity correction power is recommended to be the blanket
    Sieder and Tate (1936) value of 0.14.

    The correlation is recommended in the range of Reynolds numbers above
    1000, chevron angles between 30 and 60 degrees, and enlargement factors
    from 1 to 1.5. Due to its cubic nature it is not likely to give good
    results if the chevron angle or enlargement factors are out of those
    ranges.

    Examples
    --------
    >>> Nu_plate_Muley_Manglik(Re=2000, Pr=.7, chevron_angle=45,
    ... plate_enlargement_factor=1.18)
    36.49087100602062

    References
    ----------
    .. [1] Muley, A., and R. M. Manglik. "Experimental Study of Turbulent Flow
       Heat Transfer and Pressure Drop in a Plate Heat Exchanger With Chevron
       Plates." Journal of Heat Transfer 121, no. 1 (February 1, 1999): 110-17.
       doi:10.1115/1.2825923.
    .. [2] Palm, Björn, and Joachim Claesson. "Plate Heat Exchangers:
       Calculation Methods for Single- and Two-Phase Flow (Keynote)," January
       1, 2005, 103-13. https://doi.org/10.1115/ICMM2005-75092.
    '''
    beta, phi = chevron_angle, plate_enlargement_factor
    t1 = (0.2668 - 0.006967*beta + 7.244E-5*beta**2)
    #t2 = (20.78 - 50.94*phi + 41.16*phi**2 - 10.51*phi**3)
    # It was the extra decimals which were needed
    t2 = (20.7803 - 50.9372*phi + 41.1585*phi**2 - 10.1507*phi**3)
    t3 = (0.728 + 0.0543*sin((2*pi*beta/90) + 3.7))
    return t1*t2*Re**t3*Pr**(1/3.)


def Nu_plate_Khan_Khan(Re, Pr, chevron_angle):
    r'''Calculates Nusselt number for single-phase flow in a
    Chevron-style plate heat exchanger according to [1]_.

    .. math::
        Nu = \left(0.0161\frac{\beta}{\beta_{max}} + 0.1298\right)
        Re^{\left(0.198 \frac{\beta}{\beta_{max}} + 0.6398\right)}
        Pr^{0.35}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to the hydraulic diameter of the channels,
        [-]
    Pr : float
        Prandtl number calculated with bulk fluid properties, [-]
    chevron_angle : float
        Angle of the plate corrugations with respect to the vertical axis
        (the direction of flow if the plates were straight), between 0 and
        90. Many plate exchangers use two alternating patterns; use their
        average angle for that situation [degrees]

    Returns
    -------
    Nu : float
        Nusselt number with respect to `Dh`, [-]

    Notes
    -----
    The viscosity correction power is recommended to be the blanket
    Sieder and Tate (1936) value of 0.14.

    The correlation is recommended in the range of Reynolds numbers from
    500 to 2500, chevron angles between 30 and 60 degrees, and Prandtl
    numbers between 3.5 and 6.

    Examples
    --------
    >>> Nu_plate_Khan_Khan(Re=1000, Pr=4.5, chevron_angle=30)
    38.40883639103741

    References
    ----------
    .. [1] Khan, T. S., M. S. Khan, Ming-C. Chyu, and Z. H. Ayub. "Experimental
       Investigation of Single Phase Convective Heat Transfer Coefficient in a
       Corrugated Plate Heat Exchanger for Multiple Plate Configurations."
       Applied Thermal Engineering 30, no. 8 (June 1, 2010): 1058-65.
       https://doi.org/10.1016/j.applthermaleng.2010.01.021.
    '''
    beta_max = 60.
    beta_ratio = chevron_angle/beta_max
    Nu = (0.0161*beta_ratio + 0.1298)*Re**(0.198*beta_ratio + 0.6398)*Pr**0.35
    return Nu

