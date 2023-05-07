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

import os
from math import exp, floor, log, sqrt, tanh  # tanh= 1/coth

from fluids.constants import Btu, degree_Fahrenheit, foot, hour, inch
from fluids.numerics import bisect, brenth, factorial, gamma, horner, iv, quad, secant
from fluids.numerics import numpy as np
from fluids.piping import BWG_SI, BWG_integers

__all__ = ['effectiveness_from_NTU', 'NTU_from_effectiveness', 'calc_Cmin',
'calc_Cmax', 'calc_Cr', 'Pp', 'Pc',
'NTU_from_UA', 'UA_from_NTU', 'effectiveness_NTU_method', 'F_LMTD_Fakheri',
'temperature_effectiveness_basic', 'temperature_effectiveness_TEMA_J',
'temperature_effectiveness_TEMA_H', 'temperature_effectiveness_TEMA_G',
'temperature_effectiveness_TEMA_E', 'temperature_effectiveness_plate',
'temperature_effectiveness_air_cooler',
'P_NTU_method',  'NTU_from_P_basic',
'NTU_from_P_J', 'NTU_from_P_G', 'NTU_from_P_E', 'NTU_from_P_H',
'NTU_from_P_plate', 'check_tubing_TEMA', 'get_tube_TEMA',
'DBundle_min', 'shell_clearance', 'baffle_thickness', 'D_baffle_holes',
'L_unsupported_max', 'Ntubes', 'size_bundle_from_tubecount',
'Ntubes_Perrys', 'Ntubes_VDI', 'Ntubes_Phadkeb',
'DBundle_for_Ntubes_Phadkeb',
'Ntubes_HEDH', 'DBundle_for_Ntubes_HEDH',  'D_for_Ntubes_VDI',
'TEMA_heads', 'TEMA_shells',
'TEMA_rears', 'TEMA_services', 'baffle_types', 'triangular_Ns',
'triangular_C1s', 'square_Ns', 'square_C1s', 'R_value']

R_value = foot*foot*degree_Fahrenheit*hour/Btu

__numba_additional_funcs__ = ['crossflow_effectiveness_to_int', 'to_solve_Ntubes_Phadkeb',
                              '_tubecount_objf_Perry', '_NTU_max_for_P_solver',
                              '_NTU_from_P_solver', '_NTU_from_P_objective', '_NTU_from_P_erf']
try:
    if IS_NUMBA: # type: ignore # noqa: F821
        __numba_additional_funcs__.append('factorial')
        def factorial(n): # noqa: F811
            return gamma(n + 1.0)

except:
    pass


def crossflow_effectiveness_to_int(v, NTU, t0):
    x0 = v*v*t0
    return (1. + NTU - x0)*exp(-x0)*v*float(iv(0.0, v))

def effectiveness_from_NTU(NTU, Cr, subtype='counterflow', n_shell_tube=None):
    r'''Returns the effectiveness of a heat exchanger at a specified heat
    capacity rate, number of transfer units, and configuration. The following
    configurations are supported:

        * Counterflow (ex. double-pipe)
        * Parallel (ex. double pipe inefficient configuration)
        * Shell and tube exchangers with even numbers of tube passes,
          one or more shells in series
        * Crossflow, single pass, fluids unmixed
        * Crossflow, single pass, Cmax mixed, Cmin unmixed
        * Crossflow, single pass, Cmin mixed, Cmax unmixed
        * Boiler or condenser

    These situations are normally not those which occur in real heat exchangers,
    but are useful for academic purposes. More complicated expressions are
    available for other methods. These equations are confirmed in [1]_,
    [2]_, and [3]_.

    For parallel flow heat exchangers:

    .. math::
        \epsilon = \frac{1 - \exp[-NTU(1+C_r)]}{1+C_r}

    For counterflow heat exchangers:

    .. math::
        \epsilon = \frac{1 - \exp[-NTU(1-C_r)]}{1-C_r\exp[-NTU(1-C_r)]},\; C_r < 1

    .. math::
        \epsilon = \frac{NTU}{1+NTU},\; C_r = 1

    For TEMA E shell-and-tube heat exchangers with one shell pass, 2n tube
    passes:

    .. math::
        \epsilon_1 = 2\left\{1 + C_r + \sqrt{1+C_r^2}\times\frac{1+\exp
        [-(NTU)_1\sqrt{1+C_r^2}]}{1-\exp[-(NTU)_1\sqrt{1+C_r^2}]}\right\}^{-1}

    For TEMA E shell-and-tube heat exchangers with more than one shell pass, 2n
    tube passes (this model assumes each exchanger has an equal share of the
    overall NTU or said more plainly, the same UA):

    .. math::
        \epsilon = \left[\left(\frac{1-\epsilon_1 C_r}{1-\epsilon_1}\right)^2
        -1\right]\left[\left(\frac{1-\epsilon_1 C_r}{1-\epsilon_1}\right)^n
        - C_r\right]^{-1}

    For cross-flow (single-pass) heat exchangers with both fluids unmixed, there
    is an approximate and an exact formula. The approximate one is:

    .. math::
        \epsilon = 1 - \exp\left[\left(\frac{1}{C_r}\right)
        (NTU)^{0.22}\left\{\exp\left[C_r(NTU)^{0.78}\right]-1\right\}\right]

    The exact solution for crossflow (fluids unmixed) uses SciPy's quad
    to perform an integral (there is no analytical integral solution available).
    :math:`I_0(v)` is the modified Bessel function of the first kind. This formula
    was developed in [4]_.

    .. math::
        \epsilon = \frac{1}{C_r} - \frac{\exp(-C_r \cdot NTU)}{2(C_r NTU)^2}
        \int_0^{2 NTU\sqrt{C_r}} \left(1 + NTU - \frac{v^2}{4C_r NTU}\right)
        \exp\left(-\frac{v^2}{4C_r NTU}\right)v I_0(v) dv

    For cross-flow (single-pass) heat exchangers with Cmax mixed, Cmin unmixed:

    .. math::
        \epsilon = \left(\frac{1}{C_r}\right)(1 - \exp\left\{-C_r[1-\exp(-NTU)]\right\})

    For cross-flow (single-pass) heat exchangers with Cmin mixed, Cmax unmixed:

    .. math::
        \epsilon = 1 - \exp(-C_r^{-1}\{1 - \exp[-C_r(NTU)]\})

    For cases where `Cr` = 0, as in an exchanger with latent heat exchange,
    flow arrangement does not matter:

    .. math::
        \epsilon = 1 - \exp(-NTU)

    Parameters
    ----------
    NTU : float
        Thermal Number of Transfer Units [-]
    Cr : float
        The heat capacity rate ratio, of the smaller fluid to the larger
        fluid, [-]
    subtype : str, optional
        The subtype of exchanger; one of 'counterflow', 'parallel', 'crossflow'
        'crossflow approximate', 'crossflow, mixed Cmin',
        'crossflow, mixed Cmax', 'boiler', 'condenser', 'S&T'
    n_shell_tube : None or int, optional
        The number of shell and tube exchangers in a row, [-]

    Returns
    -------
    effectiveness : float
        The thermal effectiveness of the heat exchanger, [-]

    Notes
    -----
    Once the effectiveness of the exchanger has been calculated, the total
    heat transferred can be calculated according to the following formulas,
    depending on which stream temperatures are known:

    If the inlet temperatures for both sides are known:

    .. math::
        Q=\epsilon C_{min}(T_{h,i}-T_{c,i})

    If the outlet temperatures for both sides are known:

    .. math::
        Q = \frac{\epsilon C_{min}C_{hot}C_{cold}(T_{c,o}-T_{h,o})}
        {\epsilon  C_{min}(C_{hot} +C_{cold}) - (C_{hot}C_{cold}) }

    If the hot inlet and cold outlet are known:

    .. math::
        Q = \frac{\epsilon C_{min}C_c(T_{co}-T_{hi})}{\epsilon C_{min}-C_c}

    If the hot outlet stream and cold inlet stream are known:

    .. math::
        Q = \frac{\epsilon C_{min}C_h(T_{ci}-T_{ho})}{\epsilon C_{min}-C_h}


    If the inlet and outlet conditions for a single side are known, the
    effectiveness wasn't needed for it to be calculated. For completeness,
    the formulas are as follows:

    .. math::
        Q = C_c(T_{c,o} - T_{c,i}) = C_h(T_{h,i} - T_{h,o})

    There is also a term called :math:`Q_{max}`, which is the heat which would
    have been transferred if the effectiveness was 1. It is calculated as
    follows:

    .. math::
        Q_{max} = \frac{Q}{\text{effectiveness}}

    Examples
    --------
    Worst case, parallel flow:

    >>> effectiveness_from_NTU(NTU=5, Cr=0.7, subtype='parallel')
    0.5881156068417585

    Crossflow, somewhat higher effectiveness:

    >>> effectiveness_from_NTU(NTU=5, Cr=0.7, subtype='crossflow')
    0.84448217997

    Counterflow, better than either crossflow or parallel flow:

    >>> effectiveness_from_NTU(NTU=5, Cr=0.7, subtype='counterflow')
    0.920670368605

    One shell and tube heat exchanger gives worse performance than counterflow,
    but they are designed to be economical and compact which a counterflow
    exchanger would not be. As the number of shells approaches infinity,
    the counterflow result is obtained exactly.

    >>> effectiveness_from_NTU(NTU=5, Cr=0.7, subtype='S&T')
    0.683497704431
    >>> effectiveness_from_NTU(NTU=5, Cr=0.7, subtype='S&T', n_shell_tube=50)
    0.920505870278


    Overall case of rating an existing heat exchanger where a known flowrate
    of steam and oil are contacted in crossflow, with the steam side mixed
    (example 10-9 in [3]_):

    >>> U = 275 # W/m^2/K
    >>> A = 10.82 # m^2
    >>> Cp_oil = 1900 # J/kg/K
    >>> Cp_steam = 1860 # J/kg/K
    >>> m_steam = 5.2 # kg/s
    >>> m_oil = 0.725 # kg/s
    >>> Thi = 130 # °C
    >>> Tci = 15 # °C
    >>> Cmin = calc_Cmin(mh=m_steam, mc=m_oil, Cph=Cp_steam, Cpc=Cp_oil)
    >>> Cmax = calc_Cmax(mh=m_steam, mc=m_oil, Cph=Cp_steam, Cpc=Cp_oil)
    >>> Cr = calc_Cr(mh=m_steam, mc=m_oil, Cph=Cp_steam, Cpc=Cp_oil)
    >>> NTU = NTU_from_UA(UA=U*A, Cmin=Cmin)
    >>> eff = effectiveness_from_NTU(NTU=NTU, Cr=Cr, subtype='crossflow, mixed Cmax')
    >>> Q = eff*Cmin*(Thi - Tci)
    >>> Tco = Tci + Q/(m_oil*Cp_oil)
    >>> Tho = Thi - Q/(m_steam*Cp_steam)
    >>> Cmin, Cmax, Cr
    (1377.5, 9672.0, 0.1424214226633)
    >>> NTU, eff, Q
    (2.16007259528, 0.831218036142, 131675.3271504)
    >>> Tco, Tho
    (110.5900741563, 116.3859256461)

    Alternatively, if only the outlet temperatures had been known:

    >>> Tco = 110.59007415639887
    >>> Tho = 116.38592564614977
    >>> Cc, Ch = Cmin, Cmax # In this case but not always
    >>> Q = eff*Cmin*Cc*Ch*(Tco - Tho)/(eff*Cmin*(Cc+Ch) - Ch*Cc)
    >>> Thi = Tho + Q/Ch
    >>> Tci = Tco - Q/Cc
    >>> Q, Tci, Thi
    (131675.3271504, 14.99999999999, 130.0000000000)

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    .. [2] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [3] Holman, Jack. Heat Transfer. 10th edition. Boston: McGraw-Hill
       Education, 2009.
    .. [4] Triboix, Alain. "Exact and Approximate Formulas for Cross Flow Heat
       Exchangers with Unmixed Fluids." International Communications in Heat
       and Mass Transfer 36, no. 2 (February 1, 2009): 121-24.
       doi:10.1016/j.icheatmasstransfer.2008.10.012.
    '''
    if Cr > 1:
        raise ValueError('Heat capacity rate must be less than 1 by definition.')

    if subtype == 'counterflow':
        if Cr < 1:
            return (1. - exp(-NTU*(1. - Cr)))/(1. - Cr*exp(-NTU*(1. - Cr)))
        elif Cr == 1:
            return NTU/(1. + NTU)
    elif subtype == 'parallel':
            return (1. - exp(-NTU*(1. + Cr)))/(1. + Cr)
    elif 'S&T' == subtype:
        # str_shells = subtype.split('S&T')[0]
        shells = n_shell_tube if n_shell_tube is not None else 1
        NTU = NTU/shells

        x0 = sqrt(1. + Cr*Cr)
        x1 = exp(-NTU*x0)
        top = 1. + x1
        bottom = 1. - x1
        effectiveness = 2./(1. + Cr + x0*top/bottom)
        if shells > 1:
            # this applies to crossflow also according to Efficiency and Effectiveness of Heat Exchanger Series equation 21
            term = ((1. - effectiveness*Cr)/(1. - effectiveness))**shells
            effectiveness = (term - 1.)/(term - Cr)
        return effectiveness
    elif subtype == 'crossflow':
        t0 = 1.0/(4.*Cr*NTU)
        res, err = quad(crossflow_effectiveness_to_int, 0, 2.*NTU*sqrt(Cr), args=(NTU, t0,))
        int_term = res
        CrNTU = Cr*NTU
        return 1./Cr - exp(-CrNTU)/(2.*CrNTU*CrNTU)*int_term
    elif subtype == 'crossflow approximate':
        return 1. - exp(1./Cr*NTU**0.22*(exp(-Cr*NTU**0.78) - 1.))
    elif subtype == 'crossflow, mixed Cmin':
        return 1. -exp(-1.0/Cr*(1. - exp(-Cr*NTU)))
    elif subtype ==  'crossflow, mixed Cmax':
        return (1./Cr)*(1. - exp(-Cr*(1. - exp(-NTU))))
    elif subtype == 'boiler' or subtype == 'condenser':
        return  1. - exp(-NTU)
    else:
        raise ValueError('Input heat exchanger type not recognized')


def NTU_from_effectiveness(effectiveness, Cr, subtype='counterflow', n_shell_tube=None):
    r'''Returns the Number of Transfer Units of a heat exchanger at a specified
    heat capacity rate, effectiveness, and configuration. The following
    configurations are supported:

        * Counterflow (ex. double-pipe)
        * Parallel (ex. double pipe inefficient configuration)
        * Shell and tube exchangers with even numbers of tube passes,
          one or more shells in series (TEMA E (one pass shell) only)
        * Crossflow, single pass, fluids unmixed
        * Crossflow, single pass, Cmax mixed, Cmin unmixed
        * Crossflow, single pass, Cmin mixed, Cmax unmixed
        * Boiler or condenser

    These situations are normally not those which occur in real heat exchangers,
    but are useful for academic purposes. More complicated expressions are
    available for other methods. These equations are confirmed in [1]_, [2]_,
    and [3]_.

    For parallel flow heat exchangers:

    .. math::
        NTU = -\frac{\ln[1 - \epsilon(1+C_r)]}{1+C_r}

    For counterflow heat exchangers:

    .. math::
        NTU = \frac{1}{C_r-1}\ln\left(\frac{\epsilon-1}{\epsilon C_r-1}\right)

    .. math::
        NTU = \frac{\epsilon}{1-\epsilon} \text{ if } C_r = 1

    For TEMA E shell-and-tube heat exchangers with one shell pass, 2n tube
    passes:

    .. math::
        (NTU)_1 = -(1 + C_r^2)^{-0.5}\ln\left(\frac{E-1}{E+1}\right)

    .. math::
        E = \frac{2/\epsilon_1 - (1 + C_r)}{(1 + C_r^2)^{0.5}}

    For TEMA E shell-and-tube heat exchangers with more than one shell pass, 2n
    tube passes (this model assumes each exchanger has an equal share of the
    overall NTU or said more plainly, the same UA):

    .. math::
        \epsilon_1 = \frac{F-1}{F-C_r}

    .. math::
        F = \left(\frac{\epsilon C_r-1}{\epsilon-1}\right)^{1/n}

    .. math::
        NTU = n(NTU)_1

    For cross-flow (single-pass) heat exchangers with both fluids unmixed,
    there is no analytical solution. However, the function is monotonically
    increasing, and a closed-form solver is implemented as 'crossflow approximate',
    guaranteed to solve for :math:`10^{-7} < NTU < 10^5`. The exact solution
    for 'crossflow' uses the approximate solution's initial guess as a starting
    point for Newton's method. Some issues are noted at effectivenesses higher
    than 0.9 and very high NTUs, because the numerical integral term approaches
    1 too quickly.

    For cross-flow (single-pass) heat exchangers with Cmax mixed, Cmin unmixed:

    .. math::
        NTU = -\ln\left[1 + \frac{1}{C_r}\ln(1 - \epsilon C_r)\right]

    For cross-flow (single-pass) heat exchangers with Cmin mixed, Cmax unmixed:

    .. math::
        NTU = -\frac{1}{C_r}\ln[C_r\ln(1-\epsilon)+1]

    For cases where `Cr` = 0, as in an exchanger with latent heat exchange,
    flow arrangement does not matter:

    .. math::
        NTU = -\ln(1-\epsilon)

    Parameters
    ----------
    effectiveness : float
        The thermal effectiveness of the heat exchanger, [-]
    Cr : float
        The heat capacity rate ratio, of the smaller fluid to the larger
        fluid, [-]
    subtype : str, optional
        The subtype of exchanger; one of 'counterflow', 'parallel', 'crossflow'
        'crossflow approximate', 'crossflow, mixed Cmin',
        'crossflow, mixed Cmax', 'boiler', 'condenser', 'S&T'.
    n_shell_tube : None or int, optional
        The number of shell and tube exchangers in a row, [-]


    Returns
    -------
    NTU : float
        Thermal Number of Transfer Units [-]

    Notes
    -----
    Unlike :obj:`ht.hx.effectiveness_from_NTU`, not all inputs can
    calculate the NTU - many exchanger types have effectiveness limits
    below 1 which depend on `Cr` and the number of shells in the case of
    heat exchangers. If an impossible input is given, an error will be raised
    and the maximum possible effectiveness will be printed.

    >>> NTU_from_effectiveness(.99, Cr=.7, subtype='5S&T') # doctest: +SKIP
    Traceback (most recent call last):
    Exception: The specified effectiveness is not physically possible for this configuration; the maximum effectiveness possible is 0.974122977755.

    Examples
    --------
    Worst case, parallel flow:

    >>> NTU_from_effectiveness(effectiveness=0.5881156068417585, Cr=0.7, subtype='parallel')
    5.000000000000

    Crossflow, somewhat higher effectiveness:

    >>> NTU_from_effectiveness(effectiveness=0.8444821799748551, Cr=0.7, subtype='crossflow')
    5.000000000000

    Counterflow, better than either crossflow or parallel flow:

    >>> NTU_from_effectiveness(effectiveness=0.9206703686051108, Cr=0.7, subtype='counterflow')
    5.0

    Shell and tube exchangers:

    >>> NTU_from_effectiveness(effectiveness=0.6834977044311439, Cr=0.7, subtype='S&T')
    5.000000000000
    >>> NTU_from_effectiveness(effectiveness=0.9205058702789254, Cr=0.7, n_shell_tube=50, subtype='S&T')
    4.999999999999


    Overall case of rating an existing heat exchanger where a known flowrate
    of steam and oil are contacted in crossflow, with the steam side mixed,
    known inlet and outlet temperatures, and unknown UA
    (based on example 10-8 in [3]_):

    >>> Cp_oil = 1900 # J/kg/K
    >>> Cp_steam = 1860 # J/kg/K
    >>> m_steam = 5.2 # kg/s
    >>> m_oil = 1.45 # kg/s
    >>> Thi = 130 # °C
    >>> Tci = 15 # °C
    >>> Tco = 85 # °C # Design specification
    >>> Q = Cp_oil*m_oil*(Tci-Tco)
    >>> dTh = Q/(m_steam*Cp_steam)
    >>> Tho = Thi + dTh
    >>> Cmin = calc_Cmin(mh=m_steam, mc=m_oil, Cph=Cp_steam, Cpc=Cp_oil)
    >>> Cmax = calc_Cmax(mh=m_steam, mc=m_oil, Cph=Cp_steam, Cpc=Cp_oil)
    >>> Cr = calc_Cr(mh=m_steam, mc=m_oil, Cph=Cp_steam, Cpc=Cp_oil)
    >>> effectiveness = -Q/Cmin/(Thi-Tci)
    >>> NTU = NTU_from_effectiveness(effectiveness, Cr, subtype='crossflow, mixed Cmax')
    >>> UA = UA_from_NTU(NTU, Cmin)
    >>> U = 200 # Assume this was calculated; would actually need to be obtained iteratively as U depends on the exchanger geometry
    >>> A = UA/U
    >>> Tho, Cmin, Cmax, Cr
    (110.06100082712986, 2755.0, 9672.0, 0.2848428453267163)
    >>> effectiveness, NTU, UA, A
    (0.608695652173, 1.1040839095, 3041.75117083, 15.2087558541)

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    .. [2] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [3] Holman, Jack. Heat Transfer. 10th edition. Boston: McGraw-Hill
       Education, 2009.
    '''
    if Cr > 1:
        raise ValueError('Heat capacity rate must be less than 1 by definition.')

    if subtype == 'counterflow':
        # [2]_ gives the expression 1./(1-Cr)*log((1-Cr*eff)/(1-eff)), but
        # this is just the same equation rearranged differently.
        if Cr < 1:
            return 1./(Cr - 1.)*log((effectiveness - 1.)/(effectiveness*Cr - 1.))
        elif Cr == 1:
            return effectiveness/(1. - effectiveness)
    elif subtype == 'parallel':
        if effectiveness*(1. + Cr) > 1:
            raise ValueError('The specified effectiveness is not physically '
                             'possible for this configuration; the maximum effectiveness '
                             'possible is %s.' % (1./(Cr + 1.))) # numba: delete
#                             ) # numba: uncomment
        return -log(1. - effectiveness*(1. + Cr))/(1. + Cr)
    elif 'S&T' == subtype:
        # [2]_ gives the expression
        # D = (1+Cr**2)**0.5
        # 1/D*log((2-eff*(1+Cr-D))/(2-eff*(1+Cr + D)))
        # This is confirmed numerically to be the same equation rearranged
        # differently
        shells = n_shell_tube if n_shell_tube is not None else 1

        F = ((effectiveness*Cr - 1.)/(effectiveness - 1.))**(1./shells)
        e1 = (F - 1.)/(F - Cr)
        E = (2./e1 - (1. + Cr))/(1. + Cr**2)**0.5

        if (E - 1.)/(E + 1.) <= 0:
            # Derived with SymPy
            max_effectiveness = (-((-Cr + sqrt(Cr**2 + 1) + 1)/(Cr + sqrt(Cr**2 + 1) - 1))**shells + 1)/(Cr - ((-Cr + sqrt(Cr**2 + 1) + 1)/(Cr + sqrt(Cr**2 + 1) - 1))**shells)
            raise ValueError('The specified effectiveness is not physically ' # numba: delete
'possible for this configuration; the maximum effectiveness possible is %s.' % (max_effectiveness)) # numba: delete
#            raise ValueError("Fail") # numba: uncomment

        NTU = -(1. + Cr*Cr)**-0.5*log((E - 1.)/(E + 1.))
        return shells*NTU
    elif subtype == 'crossflow':
        # Can't use a bisect solver here because at high NTU there's a derivative of 0
        # due to the integral term not changing when it's very near one
        guess = NTU_from_effectiveness(effectiveness, Cr, 'crossflow approximate')
        def to_solve(NTU, Cr, effectiveness):
            return effectiveness_from_NTU(NTU, Cr, subtype='crossflow') - effectiveness
        return secant(to_solve, guess, args=(Cr, effectiveness))
    elif subtype == 'crossflow approximate':
        # This will fail if NTU is more than 10,000 or less than 1E-7, but
        # this is extremely unlikely to occur in normal usage.
        # Maple and SymPy and Wolfram Alpha all failed to obtain an exact
        # analytical expression even with coefficients for 0.22 and 0.78 or
        # with an explicit value for Cr. The function has been plotted,
        # and appears to be monotonic - there is only one solution.
        def to_solve(NTU, Cr, effectiveness):
            return (1. - exp(1./Cr*NTU**0.22*(exp(-Cr*NTU**0.78) - 1.))) - effectiveness
        return brenth(to_solve, 1E-7, 1E5, args=(Cr, effectiveness))

    elif subtype == 'crossflow, mixed Cmin':
        if Cr*log(1. - effectiveness) < -1:
            raise ValueError('The specified effectiveness is not physically \
possible for this configuration; the maximum effectiveness possible is %s.' % (1. - exp(-1./Cr)))
        return -1./Cr*log(Cr*log(1. - effectiveness) + 1.)

    elif subtype ==  'crossflow, mixed Cmax':
        if 1./Cr*log(1. - effectiveness*Cr) < -1:
            raise ValueError('The specified effectiveness is not physically \
possible for this configuration; the maximum effectiveness possible is %s.' % ((exp(Cr) - 1.0)*exp(-Cr)/Cr))
        return -log(1. + 1./Cr*log(1. - effectiveness*Cr))

    elif subtype in ['boiler', 'condenser']:
        return -log(1. - effectiveness)
    else:
        raise ValueError('Input heat exchanger type not recognized')


def calc_Cmin(mh, mc, Cph, Cpc):
    r'''Returns the heat capacity rate for the minimum stream
    having flows `mh` and `mc`, with averaged heat capacities `Cph` and `Cpc`.

    .. math::
        C_c = m_cC_{p,c}

        C_h = m_h C_{p,h}

        C_{min} = \min(C_c, C_h)

    Parameters
    ----------
    mh : float
        Mass flow rate of hot stream, [kg/s]
    mc : float
        Mass flow rate of cold stream, [kg/s]
    Cph : float
        Averaged heat capacity of hot stream, [J/kg/K]
    Cpc : float
        Averaged heat capacity of cold stream, [J/kg/K]

    Returns
    -------
    Cmin : float
        The heat capacity rate of the smaller fluid, [W/K]

    Notes
    -----
    Used with the effectiveness method for heat exchanger design.
    Technically, it doesn't matter if the hot and cold streams are in the right
    order for the input, but it is easiest to use this function when the order
    is specified.

    Examples
    --------
    >>> calc_Cmin(mh=22., mc=5.5, Cph=2200, Cpc=4400.)
    24200.0

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    Ch = mh*Cph
    Cc = mc*Cpc
    return min(Ch, Cc)


def calc_Cmax(mh, mc, Cph, Cpc):
    r'''Returns the heat capacity rate for the maximum stream
    having flows `mh` and `mc`, with averaged heat capacities `Cph` and `Cpc`.

    .. math::
        C_c = m_cC_{p,c}

    .. math::
        C_h = m_h C_{p,h}

    .. math::
        C_{max} = \max(C_c, C_h)

    Parameters
    ----------
    mh : float
        Mass flow rate of hot stream, [kg/s]
    mc : float
        Mass flow rate of cold stream, [kg/s]
    Cph : float
        Averaged heat capacity of hot stream, [J/kg/K]
    Cpc : float
        Averaged heat capacity of cold stream, [J/kg/K]

    Returns
    -------
    Cmax : float
        The heat capacity rate of the larger fluid, [W/K]

    Notes
    -----
    Used with the effectiveness method for heat exchanger design.
    Technically, it doesn't matter if the hot and cold streams are in the right
    order for the input, but it is easiest to use this function when the order
    is specified.

    Examples
    --------
    >>> calc_Cmax(mh=22., mc=5.5, Cph=2200, Cpc=4400.)
    48400.0

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    Ch = mh*Cph
    Cc = mc*Cpc
    return max(Ch, Cc)


def calc_Cr(mh, mc, Cph, Cpc):
    r'''Returns the heat capacity rate ratio for a heat exchanger
    having flows `mh` and `mc`, with averaged heat capacities `Cph` and `Cpc`.

    .. math::
        C_r=C^*=\frac{C_{min}}{C_{max}}

    Parameters
    ----------
    mh : float
        Mass flow rate of hot stream, [kg/s]
    mc : float
        Mass flow rate of cold stream, [kg/s]
    Cph : float
        Averaged heat capacity of hot stream, [J/kg/K]
    Cpc : float
        Averaged heat capacity of cold stream, [J/kg/K]

    Returns
    -------
    Cr : float
        The heat capacity rate ratio, of the smaller fluid to the larger
        fluid, [W/K]

    Notes
    -----
    Used with the effectiveness method for heat exchanger design.
    Technically, it doesn't matter if the hot and cold streams are in the right
    order for the input, but it is easiest to use this function when the order
    is specified.

    Examples
    --------
    >>> calc_Cr(mh=22., mc=5.5, Cph=2200, Cpc=4400.)
    0.5

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    Ch = mh*Cph
    Cc = mc*Cpc
    Cmin = min(Ch, Cc)
    Cmax = max(Ch, Cc)
    return Cmin/Cmax


def NTU_from_UA(UA, Cmin):
    r'''Returns the Number of Transfer Units for a heat exchanger having
    `UA`, and with `Cmin` heat capacity rate.

    .. math::
        NTU = \frac{UA}{C_{min}}

    Parameters
    ----------
    UA : float
        Combined Area-heat transfer coefficient term, [W/K]
    Cmin : float
        The heat capacity rate of the smaller fluid, [W/K]

    Returns
    -------
    NTU : float
        Thermal Number of Transfer Units [-]

    Notes
    -----
    Used with the effectiveness method for heat exchanger design.

    Examples
    --------
    >>> NTU_from_UA(4400., 22.)
    200.0

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    return UA/Cmin


def UA_from_NTU(NTU, Cmin):
    r'''Returns the combined area-heat transfer term for a heat exchanger
    having a specified `NTU`, and with `Cmin` heat capacity rate.

    .. math::
        UA = NTU C_{min}

    Parameters
    ----------
    NTU : float
        Thermal Number of Transfer Units [-]
    Cmin : float
        The heat capacity rate of the smaller fluid, [W/K]

    Returns
    -------
    UA : float
        Combined area-heat transfer coefficient term, [W/K]

    Notes
    -----
    Used with the effectiveness method for heat exchanger design.

    Examples
    --------
    >>> UA_from_NTU(200., 22.)
    4400.0

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    return NTU*Cmin


def Pp(x, y):
    r'''Basic helper calculator which accepts a transformed R1 and NTU1 as
    inputs for a common term used in the calculation of the P-NTU method for
    plate exchangers.

    Returns a value which is normally used in other calculations before the
    actual P1 is calculated.

    .. math::
        P_p(x, y) = \frac{1 - \exp[-x(1 + y)]}{1 + y}

    Parameters
    ----------
    x : float
        A modification of NTU1, the Thermal Number of Transfer Units [-]
    y : float
        A modification of R1, the thermal effectiveness [-]

    Returns
    -------
    z : float
        Just another term in the calculation, [-]

    Notes
    -----
    Used with the P-NTU plate method for heat exchanger design. At y = -1,
    this function has a ZeroDivisionError but can be evaluated at the limit
    to be z = x

    Examples
    --------
    >>> Pp(5, .4)
    0.713634370024

    References
    ----------
    .. [1] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [2] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    if y == -1.0:
        return x
    return (1. - exp(-x*(1. + y)))/(1. + y)


def Pc(x, y):
    r'''Basic helper calculator which accepts a transformed R1 and NTU1 as
    inputs for a common term used in the calculation of the P-NTU method for
    plate exchangers.

    Returns a value which is normally used in other calculations before the
    actual P1 is calculated. Nominally used in counterflow calculations

    .. math::
        P_c(x, y) = \frac{1 - \exp[-x(1 - y)]}{1 - y\exp[-x(1 - y)]}

    Parameters
    ----------
    x : float
        A modification of NTU1, the Thermal Number of Transfer Units [-]
    y : float
        A modification of R1, the thermal effectiveness [-]

    Returns
    -------
    z : float
        Just another term in the calculation, [-]

    Notes
    -----
    Used with the P-NTU plate method for heat exchanger design. At y =-1,
    this function has a ZeroDivisionError but can be evaluated at the limit
    to be :math:`z = \frac{x}{1+x}`.

    Examples
    --------
    >>> Pc(5, .7)
    0.920670368605

    References
    ----------
    .. [1] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [2] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    term = exp(-x*(1. - y))
    if (1. - y*term) == 0.0:
        return x/(1. + x)
    return (1. - term)/(1. - y*term)


def effectiveness_NTU_method(mh, mc, Cph, Cpc, subtype='counterflow', Thi=None,
                             Tho=None, Tci=None, Tco=None, UA=None,
                             n_shell_tube=None):
    r'''Wrapper for the various effectiveness-NTU method function calls,
    which can solve a heat exchanger. The heat capacities and mass flows
    of each stream and the type of the heat exchanger are always required.
    As additional inputs, one combination of the following inputs is required:

    * Three of the four inlet and outlet stream temperatures.
    * Temperatures for the cold outlet and hot inlet and UA
    * Temperatures for the cold outlet and hot outlet and UA
    * Temperatures for the cold inlet and hot inlet and UA
    * Temperatures for the cold inlet and hot outlet and UA

    Parameters
    ----------
    mh : float
        Mass flow rate of hot stream, [kg/s]
    mc : float
        Mass flow rate of cold stream, [kg/s]
    Cph : float
        Averaged heat capacity of hot stream, [J/kg/K]
    Cpc : float
        Averaged heat capacity of cold stream, [J/kg/K]
    subtype : str, optional
        The subtype of exchanger; one of 'counterflow', 'parallel', 'crossflow'
        'crossflow, mixed Cmin', 'crossflow, mixed Cmax', 'boiler', 'condenser',
        'S&T', or 'nS&T' where n is the number of shell and tube exchangers in
        a row
    Thi : float, optional
        Inlet temperature of hot fluid, [K]
    Tho : float, optional
        Outlet temperature of hot fluid, [K]
    Tci : float, optional
        Inlet temperature of cold fluid, [K]
    Tco : float, optional
        Outlet temperature of cold fluid, [K]
    UA : float, optional
        Combined Area-heat transfer coefficient term, [W/K]
    n_shell_tube : None or int, optional
        The number of shell and tube exchangers in a row, [-]

    Returns
    -------
    results : dict
        * Q : Heat exchanged in the heat exchanger, [W]
        * UA : Combined area-heat transfer coefficient term, [W/K]
        * Cr : The heat capacity rate ratio, of the smaller fluid to the larger
          fluid, [W/K]
        * Cmin : The heat capacity rate of the smaller fluid, [W/K]
        * Cmax : The heat capacity rate of the larger fluid, [W/K]
        * effectiveness : The thermal effectiveness of the heat exchanger, [-]
        * NTU : Thermal Number of Transfer Units [-]
        * Thi : Inlet temperature of hot fluid, [K]
        * Tho : Outlet temperature of hot fluid, [K]
        * Tci : Inlet temperature of cold fluid, [K]
        * Tco : Outlet temperature of cold fluid, [K]

    See Also
    --------
    effectiveness_from_NTU
    NTU_from_effectiveness

    Examples
    --------
    Solve a heat exchanger to determine UA and effectiveness given the
    configuration, flows, subtype, the cold inlet/outlet temperatures, and the
    hot stream inlet temperature.

    >>> from pprint import pprint
    >>> pprint(effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900,
    ... subtype='crossflow, mixed Cmax', Tci=15, Tco=85, Thi=130))
    {'Cmax': 9672.0,
     'Cmin': 2755.0,
     'Cr': 0.284842845326,
     'NTU': 1.104083909,
     'Q': 192850.0,
     'Tci': 15,
     'Tco': 85,
     'Thi': 130,
     'Tho': 110.0610008271,
     'UA': 3041.75117083,
     'effectiveness': 0.608695652173}

    Solve the same heat exchanger with the UA specified, and known inlet
    temperatures:

    >>> pprint(effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900,
    ... subtype='crossflow, mixed Cmax', Tci=15, Thi=130, UA=3041.75))
    {'Cmax': 9672.0,
     'Cmin': 2755.0,
     'Cr': 0.284842845326,
     'NTU': 1.104083484573,
     'Q': 192849.9631022,
     'Tci': 15,
     'Tco': 84.9999866069,
     'Thi': 130,
     'Tho': 110.0610046420,
     'UA': 3041.75,
     'effectiveness': 0.608695535712}
    '''
    Cmin = calc_Cmin(mh=mh, mc=mc, Cph=Cph, Cpc=Cpc)
    Cmax = calc_Cmax(mh=mh, mc=mc, Cph=Cph, Cpc=Cpc)
    Cr = calc_Cr(mh=mh, mc=mc, Cph=Cph, Cpc=Cpc)
    Cc = mc*Cpc
    Ch = mh*Cph
    if UA is not None:
        NTU = NTU_from_UA(UA=UA, Cmin=Cmin)
        effectiveness = eff = effectiveness_from_NTU(NTU=NTU, Cr=Cr, n_shell_tube=n_shell_tube, subtype=subtype)

        possible_inputs = [(Tci, Thi), (Tci, Tho), (Tco, Thi), (Tco, Tho)]
        if not any(i for i in possible_inputs if None not in i):
            raise ValueError('One set of (Tci, Thi), (Tci, Tho), (Tco, Thi), or (Tco, Tho) are required along with UA.')

        if Thi is not None and Tci is not None:
            Q = eff*Cmin*(Thi - Tci)
        elif Tho is not None and Tco is not None:
            Q = eff*Cmin*Cc*Ch*(Tco - Tho)/(eff*Cmin*(Cc+Ch) - Ch*Cc)
        elif Thi is not None and Tco is not None:
            Q = Cmin*Cc*eff*(Tco-Thi)/(eff*Cmin - Cc)
        elif Tho is not None and Tci is not None:
            Q = Cmin*Ch*eff*(Tci-Tho)/(eff*Cmin - Ch)
        # The following is not used as it was decided to require complete temperature information
#        elif Tci and Tco:
#            Q = Cc*(Tco - Tci)
#        elif Tho and Thi:
#            Q = Ch*(Thi-Tho)
        # Compute the remaining temperatures with the fewest lines of code
        if Tci is not None and Tco is None:
            Tco = Tci + Q/(Cc)
        else:
            Tci = Tco - Q/(Cc)
        if Thi is not None and Tho is None:
            Tho = Thi - Q/(Ch)
        else:
            Thi = Tho + Q/(Ch)

    elif UA is None:
        # Case where we're solving for UA
        # Three temperatures are required
        # Ensures all four temperatures are set and Q is calculated
        if Thi is not None and Tho is not None:
            Q = mh*Cph*(Thi-Tho)
            if Tci is not None and Tco is None:
                Tco = Tci + Q/(mc*Cpc)
            elif Tco is not None and Tci is None:
                Tci = Tco - Q/(mc*Cpc)
            elif Tco is not None and Tci is not None:
                Q2 = mc*Cpc*(Tco-Tci)
                if abs((Q-Q2)/Q) > 0.01:
                    raise ValueError('The specified heat capacities, mass flows, and temperatures are inconsistent')
            else:
                raise ValueError('At least one temperature is required to be specified on the cold side.')

        elif Tci is not None and Tco is not None:
            Q = mc*Cpc*(Tco-Tci)
            if Thi is not None and Tho is None:
                Tho = Thi - Q/(mh*Cph)
            elif Tho is not None and Thi is None:
                Thi = Tho + Q/(mh*Cph)
            else:
                raise ValueError('At least one temperature is required to be specified on the cold side.')
        else:
            raise ValueError('Three temperatures are required to be specified '
                            'when solving for UA')

        effectiveness = Q/Cmin/(Thi-Tci)
        NTU = NTU_from_effectiveness(effectiveness, Cr, n_shell_tube=n_shell_tube, subtype=subtype)
        UA = UA_from_NTU(NTU, Cmin)
    return {'Q': Q, 'UA': UA, 'Cr':Cr, 'Cmin': Cmin, 'Cmax':Cmax,
            'effectiveness': effectiveness, 'NTU': NTU, 'Thi': Thi, 'Tho': Tho,
            'Tci': Tci, 'Tco': Tco}


def temperature_effectiveness_air_cooler(R1, NTU1, rows, passes, coerce=True):
    r'''Returns temperature effectiveness `P1` of an air cooler with
    a specified heat capacity ratio, number of transfer units `NTU1`,
    number of rows `rows`, and number of passes `passes`. The supported cases
    are as follows:

    * N rows 1 pass
    * N row N pass (up to N = 5)
    * 4 rows 2 passes

    For N rows 1 passes ([2]_, shown in [1]_ and [3]_):

    .. math::
        P = \frac{1}{R} \left\{1 - \left[\frac{N\exp(NKR)}
        {1 + \sum_{i=1}^{N-1}\sum_{j=0}^i  {{i}\choose{j}}K^j \exp(-(i-j)NTU/N)
        \sum_{k=0}^j \frac{(NKR)^k}{k!}}\right]^{-1}\right\}

    For 2 rows 2 passes (cited as from [4]_ in [1]_):

    .. math::
        P_1 = \frac{1}{R}\left(1 -\frac{1}{\xi}\right)

    .. math::
        \xi = \frac{K}{2} + \left(1 - \frac{K}{2}\right)\exp(2KR)

    .. math::
        K = 1 - \exp\left(\frac{-NTU}{2}\right)

    For 3 rows / 3 passes (cited as from [4]_ in [1]_):

    .. math::
        \xi = K\left[1 - \frac{K}{4} - RK\left(1 - \frac{K}{2}\right)\right]
        \exp(KR) + \exp(3KR)\left(1 - \frac{K}{2}\right)^2

    .. math::
        K = 1 - \exp\left(\frac{-NTU}{3}\right)

    For 4 rows / 4 passes (cited as from [4]_ in [1]_):

    .. math::
        \xi = \frac{K}{2}\left(1 - \frac{K}{2} + \frac{K^2}{4}\right)
        + K\left(1 - \frac{K}{2}\right)
        \left[1 - \frac{R}{8}K\left(1 - \frac{K}{2}\right)\exp(2KR)\right]
        + \exp(4KR)\left(1 - \frac{K}{2}\right)^3

    .. math::
        K = 1 - \exp\left(\frac{-NTU}{4}\right)

    For 5 rows / 5 passes (cited as from [4]_ in [1]_):

    .. math::
        \xi = \left\{K \left(1 - \frac{3}{4}K + \frac{K^2}{2}- \frac{K^3}{8}
        \right) - RK^2\left[1 -K + \frac{3}{4}K^2 - \frac{1}{4}K^3
        - \frac{R}{2}K^2\left(1 - \frac{K}{2}\right)^2\right]\right\}\exp(KR)
        + \left[K\left(1 - \frac{3}{4}K + \frac{1}{16}K^3\right) - 3RK^2\left(1
        - \frac{K}{2}\right)^3\right]\exp(3KR)+ \left(1 - \frac{K}{2}\right)^4
        \exp(5KR)

    For 4 rows / 2 passes (cited as from [4]_ in [1]_):

    .. math::
        P_1 = \frac{1}{R}\left(1 -\frac{1}{\xi}\right)

    .. math::
        \xi = \left\{\frac{R}{2}K^3[4 - K + 2RK^2] + \exp(4KR)
        + K\left[1 - \frac{K}{2} + \frac{K^2}{8}\right]
        \left[1 - \exp(4KR)\right]
        \right\}\frac{1}{(1+RK^2)^2}

    .. math::
        K = 1 - \exp\left(\frac{-NTU}{4}\right)

    Parameters
    ----------
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 (process fluid side) [-]
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 (process fluid side) [-]
    rows : int
        Number of rows of tubes in the air cooler [-]
    passes : int
        Number of passes the process fluid undergoes [-]
    coerce : bool
        If True, the number of passes or rows, if otherwise unsupported, will
        be replaced with a similar number to allow the calculation to proceed,
        [-]

    Returns
    -------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 (process fluid side) [-]

    Notes
    -----
    For the 1-pass case, the exact formula used can take a while to compute for
    large numbers of tube rows; 100 us for 20 rows, 1 ms for 50 rows.
    Floating point rounding behavior can also be an issue for large numbers of
    tube passes, leading to thermal effectivenesses larger than one being
    returned:

    >>> temperature_effectiveness_air_cooler(1e-10, 100, rows=150, passes=1.0)
    1.000026728092962

    Furthermore, as a factorial of the number of tube counts is used, there
    comes a point where standard floats are not able to hold the intermediate
    calculations values and an error will occur:

    >>> temperature_effectiveness_air_cooler(.5, 1.1, rows=200, passes=1.0)
    Traceback (most recent call last):
    ...
    OverflowError: int too large to convert to float


    Examples
    --------
    >>> temperature_effectiveness_air_cooler(.5, 2, rows=2, passes=2)
    0.7523072855817072

    References
    ----------
    .. [1] Thulukkanam, Kuppan. Heat Exchanger Design Handbook, Second Edition.
       CRC Press, 2013.
    .. [2] Schedwill, H., "Thermische Auslegung von Kreuzstromwarmeaustauschern,
       Fortschr-Ber." VDI Reihe 6 19, VDI, Germany, 1968.
    .. [3] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1983.
    .. [4]  Nicole, F. J. L.. "Mean temperature difference for heat exchanger
       design." Council for Scientific and Industrial Research, Special Report
       Chem. 223, Pretoria, South Africa (1972).
    '''
    if passes == 1:
        N = rows
        K = 1. - exp(-NTU1/N)
        NKR1 = N*K*R1
        NTU1_N = NTU1/N
        top = N*exp(N*K*R1)
        # TODO speed this up
        # Precalculate integer factorials up to N
        # Factorial fits in 64 bit int only up to N = 20
        # https://stackoverflow.com/questions/62056035/how-to-call-math-factorial-from-numba-with-nopython-mode
        Np1 = N+1
        factorials = [factorial(i) for i in range(N)]
        K_powers = [K**j for j in range(0, N+1)]
        NKR1_powers = [NKR1**k for k in range(0, N+1)]
        exp_terms = [exp(i*NTU1_N) for i in range(-N+1, 1)] # Only need to compute one exp, then multiply
        NKR1_powers_over_factorials = [NKR1_powers[k]/factorials[k]
                                       for k in range(N)]

        # Precompute even more...
        NKR1_pows_div_factorials = [0]
        for k in NKR1_powers_over_factorials:
            NKR1_pows_div_factorials.append(NKR1_pows_div_factorials[-1]+k)
        NKR1_pows_div_factorials.pop(0)

        final_speed = [0.0]*N
        for i in range(N):
            final_speed[i] = K_powers[i]*NKR1_pows_div_factorials[i]
#        final_speed = [i*j for i, j in zip(K_powers, NKR1_pows_div_factorials)]

        tot = 0.
        for i in range(1, N):
            for j in range(0, i+1):
                # can't optimize the factorial
                prod = factorials[i]/(factorials[i-j]*factorials[j])
                tot1 = prod*exp_terms[j-i-1]
                tot += tot1*final_speed[j]

        return 1./R1*(1. - 1./(top/(1.+tot)))
    elif rows == passes == 2:
        K = 1. - exp(-0.5*NTU1)
        xi = 0.5*K + (1. - 0.5*K)*exp(2.*K*R1)
        return 1./R1*(1. - 1./xi)
    elif rows == passes == 3:
        K = 1. - exp(-NTU1/3.)
        xi = (K*(1. - 0.25*K - R1*K*(1. - 0.5*K))*exp(K*R1)
              + exp(3.*K*R1)*(1. - 0.5*K)**2)
        return 1./R1*(1. - 1./xi)
    elif rows == passes == 4:
        K = 1. - exp(-0.25*NTU1)
        xi = (0.5*K*(1. - 0.5*K + 0.25*K**2)
              + K*(1. - 0.5*K)*(1. - 0.125*R1*K*(1. - 0.5*K)*exp(2.*K*R1))
              + exp(4.*K*R1)*(1. - 0.5*K)**3)
        return 1./R1*(1. - 1./xi)
    elif rows == passes == 5:
        K = 1. - exp(-0.2*NTU1)
        K2 = K*K
        K3 = K2*K
        xi = (K*(1. - .75*K + .5*K2 - .125*K3)
              - R1*K2*(1. - K + .75*K2 - .25*K3
              - .5*R1*K2*(1. - .5*K)**2))*exp(K*R1)
        xi += ((K*(1. - .75*K + 1/16.*K3) - 3*R1*K2*(1. - .5*K)**3)
              *exp(3*K*R1) + (1. - .5*K)**4*exp(5*K*R1))
        return 1./R1*(1. - 1./xi)
    elif rows == 4 and passes == 2:
        K = 1. - exp(-0.25*NTU1)
        xi = (0.5*R1*K**3*(4. - K + 2.*R1*K**2) + exp(4.*K*R1) + K*(1. - 0.5*K
              + 0.125*K**2)*(1 - exp(4.*K*R1)))*(1. + R1*K**2)**-2
        return 1./R1*(1. - 1./xi)
    else:
        if coerce:
            if passes > rows:
                passes = rows # bad user input - replace with an exception?
            new_passes, new_rows = passes, rows
            # Domain reduction
            if passes > 5:
                new_passes = passes = 5
            if rows > 5:
                new_rows = rows = 5
            if rows -1 == passes:
                new_rows, new_passes = rows -1, passes
            elif (passes == 2 or passes == 3 or passes == 5) and rows >= 4:
                new_rows, new_passes = 4, 2

            return temperature_effectiveness_air_cooler(R1=R1, NTU1=NTU1, rows=new_rows, passes=new_passes)

        else:
            raise ValueError('Number of passes and rows not supported.')


def temperature_effectiveness_basic(R1, NTU1, subtype='crossflow'):
    r'''Returns temperature effectiveness `P1` of a heat exchanger with
    a specified heat capacity ratio, number of transfer units `NTU1`,
    and of type `subtype`. This function performs the calculations for the
    basic cases, not actual shell-and-tube exchangers. The supported cases
    are as follows:

    * Counterflow (ex. double-pipe)
    * Parallel (ex. double pipe inefficient configuration)
    * Crossflow, single pass, fluids unmixed
    * Crossflow, single pass, fluid 1 mixed, fluid 2 unmixed
    * Crossflow, single pass, fluid 2 mixed, fluid 1 unmixed
    * Crossflow, single pass, both fluids mixed

    For parallel flow heat exchangers (this configuration is symmetric):

    .. math::
        P_1 = \frac{1 - \exp[-NTU_1(1+R_1)]}{1 + R_1}

    For counterflow heat exchangers (this configuration is symmetric):

    .. math::
        P_1 = \frac{1 - \exp[-NTU_1(1-R_1)]}{1 - R_1 \exp[-NTU_1(1-R_1)]}

    For cross-flow (single-pass) heat exchangers with both fluids unmixed
    (this configuration is symmetric), there are two solutions available;
    a frequently cited approximation and an exact solution which uses
    a numerical integration developed in [4]_. The approximate solution is:

    .. math::
        P_1 \approx 1 - \exp\left[\frac{NTU_1^{0.22}}{R_1}
        (\exp(-R_1 NTU_1^{0.78})-1)\right]

    The exact solution for crossflow (single pass, fluids unmixed) is:

    .. math::
        \epsilon = \frac{1}{R_1} - \frac{\exp(-R_1 \cdot NTU_1)}{2(R_1 NTU_1)^2}
        \int_0^{2 NTU_1\sqrt{R_1}} \left(1 + NTU_1 - \frac{v^2}{4R_1 NTU_1}
        \right)\exp\left(-\frac{v^2}{4R_1 NTU_1}\right)v I_0(v) dv

    For cross-flow (single-pass) heat exchangers with fluid 1 mixed, fluid 2
    unmixed:

    .. math::
        P_1 = 1 - \exp\left(-\frac{K}{R_1}\right)

    .. math::
        K = 1 - \exp(-R_1 NTU_1)

    For cross-flow (single-pass) heat exchangers with fluid 2 mixed, fluid 1
    unmixed:

    .. math::
        P_1 = \frac{1 - \exp(-K R_1)}{R_1}

    .. math::
        K = 1 - \exp(-NTU_1)

    For cross-flow (single-pass) heat exchangers with both fluids mixed
    (this configuration is symmetric):

    .. math::
        P_1 = \left(\frac{1}{K_1} + \frac{R_1}{K_2} - \frac{1}{NTU_1}\right)^{-1}

    .. math::
        K_1 = 1 - \exp(-NTU_1)

    .. math::
        K_2 = 1 - \exp(-R_1 NTU_1)

    Parameters
    ----------
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 [-]
    subtype : float
        The type of heat exchanger; one of 'counterflow', 'parallel',
        'crossflow', 'crossflow approximate', 'crossflow, mixed 1',
        'crossflow, mixed 2', 'crossflow, mixed 1&2'.

    Returns
    -------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]

    Notes
    -----
    The crossflow case is an approximation only. There is an actual
    solution involving an infinite sum. This was implemented, but found to
    differ substantially so the approximation is used instead.

    Examples
    --------
    >>> temperature_effectiveness_basic(R1=.1, NTU1=4, subtype='counterflow')
    0.9753412729761263

    References
    ----------
    .. [1] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [2] Thulukkanam, Kuppan. Heat Exchanger Design Handbook, Second Edition.
       CRC Press, 2013.
    .. [3] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [4] Triboix, Alain. "Exact and Approximate Formulas for Cross Flow Heat
       Exchangers with Unmixed Fluids." International Communications in Heat
       and Mass Transfer 36, no. 2 (February 1, 2009): 121-24.
       doi:10.1016/j.icheatmasstransfer.2008.10.012.
    '''
    if subtype == 'counterflow':
        # Same as TEMA 1 pass
        if R1 == 1.0:
            """from sympy import *
            R1, NTU1 = symbols('R1, NTU1')
            P1 = (1 - exp(-NTU1*(1 - R1)))/(1 - R1*exp(-NTU1*(1-R1)))
            limit(P1, R1, 1)
            """
            P1 = -NTU1/(-NTU1 - 1.0)
        else:
            P1 = (1.0 - exp(-NTU1*(1 - R1)))/(1.0 - R1*exp(-NTU1*(1-R1)))
    elif subtype == 'parallel':
        P1 = (1.0 - exp(-NTU1*(1 + R1)))/(1.0 + R1)
    elif subtype == 'crossflow approximate':
        # This isn't technically accurate, an infinite sum is required
        # It has been computed from two different sources
        # but is found not to be within the 1% claimed of this equation
        P1 = 1.0 - exp(NTU1**0.22/R1*(exp(-R1*NTU1**0.78) - 1.))
    elif subtype == 'crossflow':
        # TODO attempt chebyshev approximation of P1 as a function of R1, NTU1 (for stability)
        R1_NTU1_4_inv = 1.0/(4.*R1*NTU1)
        int_term = quad(crossflow_effectiveness_to_int, 0.0, 2.*NTU1*R1**0.5, args=(NTU1, R1_NTU1_4_inv))[0]
        P1 = 1./R1 - exp(-R1*NTU1)/(2.*(R1*NTU1)**2)*int_term
    elif subtype == 'crossflow, mixed 1':
        # Not symmetric
        K = 1 - exp(-R1*NTU1)
        P1 = 1 - exp(-K/R1)
    elif subtype == 'crossflow, mixed 2':
        # Not symmetric
        K = 1 - exp(-NTU1)
        P1 = (1 - exp(-K*R1))/R1
    elif subtype == 'crossflow, mixed 1&2':
        K1 = 1. - exp(-NTU1)
        K2 = 1. - exp(-R1*NTU1)
        P1 = (1./K1 + R1/K2 - 1./NTU1)**-1
    else:
        raise ValueError('Subtype not recognized.')
    return P1


def temperature_effectiveness_TEMA_J(R1, NTU1, Ntp):
    r'''Returns temperature effectiveness `P1` of a TEMA J type heat exchanger
    with a specified heat capacity ratio, number of transfer units `NTU1`,
    and of number of tube passes `Ntp`. The supported cases are as follows:

    * One tube pass (shell fluid mixed)
    * Two tube passes (shell fluid mixed, tube pass mixed between passes)
    * Four tube passes (shell fluid mixed, tube pass mixed between passes)

    For 1-1 TEMA J shell and tube exchangers, shell and tube fluids mixed:

    .. math::
        P_1 = \frac{1}{R_1}\left[1- \frac{(2-R_1)(2E + R_1 B)}{(2+R_1)
        (2E - R_1/B)}\right]

    For 1-2 TEMA J, shell and tube fluids mixed. There are two possible
    arrangements for the flow and the number of tube passes, but the equation
    is the same in both:

    .. math::
        P_1 = \left[1 + \frac{R_1}{2} + \lambda B - 2\lambda C D\right]^{-1}

    .. math::
        B = \frac{(A^\lambda +1)}{A^\lambda -1}

    .. math::
        C = \frac{A^{(1 + \lambda)/2}}{\lambda - 1 + (1 + \lambda)A^\lambda}

    .. math::
        D = 1 + \frac{\lambda A^{(\lambda-1)/2}}{A^\lambda -1}

    .. math::
        A = \exp(NTU_1)

    .. math::
        \lambda = (1 + R_1^2/4)^{0.5}

    For 1-4 TEMA J, shell and tube exchanger with both sides mixed:

    .. math::
        P_1 = \left[1 + \frac{R_1}{4}\left(\frac{1+3E}{1+E}\right) + \lambda B
        - 2 \lambda C D\right]^{-1}

    .. math::
        B = \frac{A^\lambda +1}{A^\lambda -1}

    .. math::
        C = \frac{A^{(1+\lambda)/2}}{\lambda - 1 + (1 + \lambda)A^\lambda}

    .. math::
        D = 1 + \frac{\lambda A^{(\lambda-1)/2}}{A^\lambda -1}

    .. math::
        A = \exp(NTU_1)

    .. math::
        E = \exp(R_1 NTU_1/2)

    .. math::
        \lambda = (1 + R_1^2/16)^{0.5}

    Parameters
    ----------
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 (shell side = 1, tube side = 2) [-]
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 (shell side = 1, tube side
        = 2) [-]
    Ntp : int
        Number of tube passes, 1, 2, or 4, [-]

    Returns
    -------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]

    Notes
    -----
    For numbers of tube passes that are not 1, 2, or 4, an exception is raised.
    The convention for the formulas in [1]_ and [3]_ are with the shell side
    as side 1, and the tube side as side 2. [2]_ has formulas with the
    opposite convention.

    Examples
    --------
    >>> temperature_effectiveness_TEMA_J(R1=1/3., NTU1=1., Ntp=1)
    0.5699085193651295

    References
    ----------
    .. [1] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [2] Thulukkanam, Kuppan. Heat Exchanger Design Handbook, Second Edition.
       CRC Press, 2013.
    .. [3] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    if Ntp == 1:
        A = exp(NTU1)
        B = exp(-NTU1*R1/2.)
        if R1 != 2:
            P1 = 1./R1*(1. - (2. - R1)*(2.*A + R1*B)/(2. + R1)/(2.*A - R1/B))
        else:
            P1 = 0.5*(1. - (1. + A**-2)/2./(1. + NTU1))
    elif Ntp == 2:
        lambda1 = (1. + R1*R1/4.)**0.5
        A = exp(NTU1)
        D = 1. + lambda1*A**((lambda1 - 1.)/2.)/(A**lambda1 - 1.)
        C = A**((1+lambda1)/2.)/(lambda1 - 1. + (1. + lambda1)*A**lambda1)
        B = (A**lambda1 + 1.)/(A**lambda1 - 1.)
        P1 = 1./(1. + R1/2. + lambda1*B - 2.*lambda1*C*D)
    elif Ntp == 4:
        lambda1 = (1. + R1**2/16.)**0.5
        E = exp(R1*NTU1/2.)
        A = exp(NTU1)
        D = 1. + lambda1*A**((lambda1-1)/2.)/(A**lambda1-1.)
        C = A**((1+lambda1)/2.)/(lambda1 - 1. + (1. + lambda1)*A**lambda1)
        B = (A**lambda1 + 1.)/(A**lambda1-1)
        P1 = 1./(1. + R1/4.*(1. + 3.*E)/(1. + E) + lambda1*B - 2.*lambda1*C*D)
    else:
        raise ValueError('Supported numbers of tube passes are 1, 2, and 4.')
    return P1


def temperature_effectiveness_TEMA_H(R1, NTU1, Ntp, optimal=True):
    r'''Returns temperature effectiveness `P1` of a TEMA H type heat exchanger
    with a specified heat capacity ratio, number of transfer units `NTU1`,
    and of number of tube passes `Ntp`. For the two tube pass case, there are
    two possible orientations, one inefficient and one efficient controlled
    by the `optimal` option. The supported cases are as follows:

    * One tube pass (tube fluid split into two streams individually mixed,
      shell fluid mixed)
    * Two tube passes (shell fluid mixed, tube pass mixed between passes)
    * Two tube passes (shell fluid mixed, tube pass mixed between passes, inlet
      tube side next to inlet shell-side)

    1-1 TEMA H, tube fluid split into two streams individually mixed, shell
    fluid mixed:

    .. math::
        P_1 = E[1 + (1 - BR_1/2)(1 - A R_1/2 + ABR_1)] - AB(1 - BR_1/2)

    .. math::
        A = \frac{1}{1 + R_1/2}\{1 - \exp[-NTU_1(1 + R_1/2)/2]\}

    .. math::
        B = \frac{1-D}{1-R_1 D/2}

    .. math::
        D = \exp[-NTU_1(1-R_1/2)/2]

    .. math::
        E = (A + B - ABR_1/2)/2

    1-2 TEMA H, shell and tube fluids mixed in each pass at the cross section:

    .. math::
        P_1 = \frac{1}{R_1}\left[1 - \frac{(1-D)^4}{B - 4G/R_1}\right]

    .. math::
        B = (1+H)(1+E)^2

    .. math::
        G = (1-D)^2(D^2 + E^2) + D^2(1 + E)^2

    .. math::
        H = [1 - \exp(-2\beta)]/(4/R_1 -1)

    .. math::
        E = [1 - \exp(-\beta)]/(4/R_1 - 1)

    .. math::
        D = [1 - \exp(-\alpha)]/(4/R_1 + 1)

    .. math::
        \alpha = NTU_1(4 + R_1)/8

    .. math::
        \beta = NTU_1(4-R_1)/8

    1-2 TEMA H, shell and tube fluids mixed in each pass at the cross section
    but with the inlet tube stream coming in next to the shell fluid inlet
    in an inefficient way (this is only shown in [2]_, and the stream 1/2
    convention in it is different but converted here; P1 is still returned):

    .. math::
        P_2 = \left[1 - \frac{B + 4GR_2}{(1-D)^4}\right]

    .. math::
        B = (1 + H)(1 + E)^2

    .. math::
        G = (1-D)^2(D^2 + E^2) + D^2(1 + E)^2

    .. math::
        D = \frac{1 - \exp(-\alpha)}{1 - 4R_2}

    .. math::
        E = \frac{\exp(-\beta) - 1}{4R_2 +1}

    .. math::
        H = \frac{\exp(-2\beta) - 1}{4R_2 +1}

    .. math::
        \alpha = \frac{NTU_2}{8}(4R_2 -1)

    .. math::
        \beta = \frac{NTU_2}{8}(4R_2 +1)

    Parameters
    ----------
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 (shell side = 1, tube side = 2) [-]
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 (shell side = 1, tube side
        = 2) [-]
    Ntp : int
        Number of tube passes, 1, or 2, [-]
    optimal : bool, optional
        Whether or not the arrangement is configured to give more of a
        countercurrent and efficient (True) case or an inefficient parallel
        case, [-]

    Returns
    -------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]

    Notes
    -----
    For numbers of tube passes greater than 1 or 2, an exception is raised.
    The convention for the formulas in [1]_ and [3]_ are with the shell side
    as side 1, and the tube side as side 2. [2]_ has formulas with the
    opposite convention.

    Examples
    --------
    >>> temperature_effectiveness_TEMA_H(R1=1/3., NTU1=1., Ntp=1)
    0.5730728284905833

    References
    ----------
    .. [1] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [2] Thulukkanam, Kuppan. Heat Exchanger Design Handbook, Second Edition.
       CRC Press, 2013.
    .. [3] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    if Ntp == 1:
        A = 1./(1 + R1/2.)*(1. - exp(-NTU1*(1. + R1/2.)/2.))
        D = exp(-NTU1*(1. - R1/2.)/2.)
        if R1 != 2:
            B = (1. - D)/(1. - R1*D/2.)
        else:
            B = NTU1/(2. + NTU1)
        E = (A + B - A*B*R1/2.)/2.
        P1 = E*(1. + (1. - B*R1/2.)*(1. - A*R1/2. + A*B*R1)) - A*B*(1. - B*R1/2.)
    elif Ntp == 2 and optimal:
        alpha = NTU1*(4. + R1)/8.
        beta = NTU1*(4. - R1)/8.
        D = (1. - exp(-alpha))/(4./R1 + 1)
        if R1 != 4:
            E = (1. - exp(-beta))/(4./R1 - 1.)
            H = (1. - exp(-2.*beta))/(4./R1 - 1.)
        else:
            E = NTU1/2.
            H = NTU1
        G = (1-D)**2*(D**2 + E**2) + D**2*(1+E)**2
        B = (1. + H)*(1. + E)**2
        P1 = 1./R1*(1. - (1. - D)**4/(B - 4.*G/R1))
    elif Ntp == 2 and not optimal:
        R1_orig = R1
        #NTU2 = NTU1*R1_orig but we want to treat it as NTU1 in this case
        NTU1 = NTU1*R1_orig # switch 1
        # R2 = 1/R1 but we want to treat it as R1 in this case
        R1 = 1./R1_orig # switch 2

        beta = NTU1*(4.*R1 + 1)/8.
        alpha = NTU1/8.*(4.*R1 - 1.)
        H = (exp(-2.*beta) - 1.)/(4.*R1 + 1.)
        E = (exp(-beta) - 1.)/(4.*R1 + 1.)
        B = (1. + H)*(1. + E)**2
        if R1 != 0.25:
            D = (1. - exp(-alpha))/(1. - 4.*R1)
            G = (1. - D)**2*(D**2 + E**2) + D**2*(1. + E)**2
            P1 = (1. - (B + 4.*G*R1)/(1. - D)**4)
        else:
            D = -NTU1/8.
            G = (1. - D)**2*(D**2 + E**2) + D**2*(1. + E)**2
            P1 = (1. - (B + 4.*G*R1)/(1. - D)**4)
        P1 = P1/R1_orig # switch 3, confirmed
    else:
        raise ValueError('Supported numbers of tube passes are 1 and 2.')
    return P1


def temperature_effectiveness_TEMA_G(R1, NTU1, Ntp, optimal=True):
    r'''Returns temperature effectiveness `P1` of a TEMA G type heat exchanger
    with a specified heat capacity ratio, number of transfer units `NTU1`,
    and of number of tube passes `Ntp`. For the two tube pass case, there are
    two possible orientations, one inefficient and one efficient controlled
    by the `optimal` option. The supported cases are as follows:

    * One tube pass (tube fluid split into two streams individually mixed,
      shell fluid mixed)
    * Two tube passes (shell and tube exchanger with shell and tube fluids
      mixed in each pass at the cross section), counterflow arrangement
    * Two tube passes (shell and tube exchanger with shell and tube fluids
      mixed in each pass at the cross section), parallelflow arrangement

    1-1 TEMA G, tube fluid split into two streams individually mixed, shell
    fluid mixed (this configuration is symmetric):

    .. math::
        P_1 = A + B - AB(1 + R_1) + R_1 AB^2

    .. math::
        A = \frac{1}{1 + R_1}\{1 - \exp(-NTU_1(1+R_1)/2)\}

    .. math::
        B = \frac{1 - D}{1 - R_1 D}

    .. math::
        D = \exp[-NTU_1(1-R_1)/2]

    1-2 TEMA G, shell and tube exchanger with shell and tube fluids mixed in
    each pass at the cross section:

    .. math::
        P_1 = (B - \alpha^2)/(A + 2 + R_1 B)

    .. math::
        A = -2 R_1(1-\alpha)^2/(2 + R_1)

    .. math::
        B = [4 - \beta(2+R_1)]/(2 - R_1)

    .. math::
        \alpha = \exp[-NTU_1(2+R_1)/4]

    .. math::
        \beta = \exp[-NTU_1(2 - R_1)/2]

    1-2 TEMA G, shell and tube exchanger in overall parallelflow arrangement
    with shell and tube fluids mixed in each pass at the cross section
    (this is only shown in [2]_, and the stream convention in it is different
    but converted here; P1 is still returned):

    .. math::
        P_2 = \frac{(B-\alpha^2)}{R_2(A - \alpha^2/R_2 + 2)}

    .. math::
        A = \frac{(1-\alpha)^2}{(R_2-0.5)}

    .. math::
        B = \frac{4R_2 - \beta(2R_2 - 1)}{2R_2 + 1}

    .. math::
        \alpha = \exp\left(\frac{-NTU_2(2R_2-1)}{4}\right)

    .. math::
        \beta = \exp\left(\frac{-NTU_2(2R_2+1)}{2}\right)

    Parameters
    ----------
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 (shell side = 1, tube side = 2) [-]
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 (shell side = 1, tube side
        = 2) [-]
    Ntp : int
        Number of tube passes, 1 or 2, [-]
    optimal : bool, optional
        Whether or not the arrangement is configured to give more of a
        countercurrent and efficient (True) case or an inefficient parallel
        case (only applies for two passes), [-]

    Returns
    -------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]

    Notes
    -----
    For numbers of tube passes greater than 1 or 2, an exception is raised.
    The convention for the formulas in [1]_ and [3]_ are with the shell side
    as side 1, and the tube side as side 2. [2]_ has formulas with the
    opposite convention.

    Examples
    --------
    >>> temperature_effectiveness_TEMA_G(R1=1/3., NTU1=1., Ntp=1)
    0.5730149350867675

    References
    ----------
    .. [1] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [2] Thulukkanam, Kuppan. Heat Exchanger Design Handbook, Second Edition.
       CRC Press, 2013.
    .. [3] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    if Ntp == 1:
        D = exp(-NTU1*(1. - R1)/2.)
        if R1 != 1:
            B = (1. - D)/(1. - R1*D)
        else:
            B = NTU1/(2. + NTU1)
        A = 1./(1. + R1)*(1. - exp(-NTU1*(1. + R1)/2.))
        P1 = A + B - A*B*(1. + R1) + R1*A*B**2
    elif Ntp == 2 and optimal:
        if R1 != 2:
            beta = exp(-NTU1*(2. - R1)/2.)
            alpha = exp(-NTU1*(2. + R1)/4.)
            B = (4. - beta*(2. + R1))/(2. - R1)
            A = -2.*R1*(1-alpha)**2/(2. + R1)
            P1 = (B - alpha**2)/(A + 2. + R1*B)
        else:
            alpha = exp(-NTU1)
            P1 = (1. + 2.*NTU1 - alpha**2)/(4. + 4.*NTU1 - (1. - alpha)**2)
    elif Ntp == 2 and not optimal:
        R1_orig = R1
        #NTU2 = NTU1*R1_orig but we want to treat it as NTU1 in this case
        NTU1 = NTU1*R1_orig # switch 1
        # R2 = 1/R1 but we want to treat it as R1 in this case
        R1 = 1./R1_orig # switch 2
        if R1 != 0.5:
            beta = exp(-NTU1*(2.*R1 + 1.)/2.)
            alpha = exp(-NTU1*(2.*R1 - 1.)/4.)
            B = (4.*R1 - beta*(2.*R1 - 1.))/(2.*R1 + 1.)
            A = (1. - alpha)**2/(R1 - 0.5)
            P1 = (B - alpha**2)/(R1*(A - alpha**2/R1 + 2.))
        else:
            beta = exp(-2.*R1*NTU1)
            P1 = (1. + 2.*R1*NTU1 - beta)/R1/(4. + 4.*R1*NTU1 + R1**2*NTU1**2)
        P1 = P1/R1_orig # switch 3, confirmed
    else:
        raise ValueError('Supported numbers of tube passes are 1 and 2.')
    return P1


def temperature_effectiveness_TEMA_E(R1, NTU1, Ntp=1, optimal=True):
    r'''Returns temperature effectiveness `P1` of a TEMA E type heat exchanger
    with a specified heat capacity ratio, number of transfer units `NTU1`,
    number of tube passes `Ntp`, and whether or not it is arranged in a more
    countercurrent (optimal configuration) way or a more parallel (optimal=False)
    case. The supported cases are as follows:

    * 1-1 TEMA E, shell fluid mixed
    * 1-2 TEMA E, shell fluid mixed (this configuration is symmetric)
    * 1-2 TEMA E, shell fluid split into two steams individually mixed
    * 1-3 TEMA E, shell and tube fluids mixed, one parallel pass and two
      counterflow passes (efficient)
    * 1-3 TEMA E, shell and tube fluids mixed, two parallel passes and one
      counteflow pass (inefficient)
    * 1-N TEMA E, shall and tube fluids mixed, efficient counterflow orientation,
      N an even number

    1-1 TEMA E, shell fluid mixed:

    .. math::
        P_1 = \frac{1 - \exp[-NTU_1(1-R_1)]}{1 - R_1 \exp[-NTU_1(1-R_1)]}

    1-2 TEMA E, shell fluid mixed (this configuration is symmetric):

    .. math::
        P_1 = \frac{2}{1 + R_1 + E\coth(E\cdot NTU_1/2)}

    .. math::
        E = [1 + R_1^2]^{1/2}

    1-2 TEMA E, shell fluid split into two steams individually mixed:

    .. math::
        P_1 = \frac{1}{R_1}\left[1 - \frac{(2-R_1)(2E+R_1B)}{(2+R_1)(2E-R_1/B)}
        \right]

    .. math::
        E = \exp(NTU_1)

    .. math::
        B = \exp(-NTU_1 R_1/2)

    1-3 TEMA E, shell and tube fluids mixed, one parallel pass and two
    counterflow passes (efficient):

    .. math::
        P_1 = \frac{1}{R_1} \left[1 - \frac{C}{AC + B^2}\right]

    .. math::
        A = X_1(R_1 + \lambda_1)(R_1 - \lambda_2)/(2\lambda_1) - X_3 \delta
        - X_2(R_1 + \lambda_2)(R_1-\lambda_1)/(2\lambda_2) + 1/(1-R_1)

    .. math::
        B = X_1(R_1-\lambda_2) - X_2(R_1-\lambda_1) + X_3\delta

    .. math::
        C = X_2(3R_1 + \lambda_1) - X_1(3R_1 + \lambda_2) + X_3 \delta

    .. math::
        X_i = \exp(\lambda_i NTU_1/3)/(2\delta),\;\; i = 1,2,3

    .. math::
        \delta = \lambda_1 - \lambda_2

    .. math::
        \lambda_1 = -\frac{3}{2} + \left[\frac{9}{4} + R_1(R_1-1)\right]^{1/2}

    .. math::
        \lambda_2 = -\frac{3}{2} - \left[\frac{9}{4} + R_1(R_1-1)\right]^{1/2}

    .. math::
        \lambda_3 = R_1

    1-3 TEMA E, shell and tube fluids mixed, two parallel passes and one
    counteflow pass (inefficient):

    .. math::
        P_2 = \left[1 - \frac{C}{(AC + B^2)}\right]

    .. math::
        A = \chi_1(1 + R_2 \lambda_1)(1 - R_2\lambda_2)/(2R_2^2\lambda_1) - E
        -\chi_2(1 + R_2\lambda_2)(1 - R_2\lambda_1)/(2R^2\lambda_2) + R/(R-1)

    .. math::
        B = \chi_1(1 - R_2\lambda_2)/R_2 - \chi_2(1 - R_2 \lambda_1)/R_2 + E

    .. math::
        C = -\chi_1(3 + R_2\lambda_2)/R_2 + \chi_2(3 + R_2\lambda_1)/R_2 + E

    .. math::
        E = 0.5\exp(NTU_2/3)

    .. math::
        \lambda_1 = (-3 + \delta)/2

    .. math::
        \lambda_2 = (-3 - \delta)/2

    .. math::
        \delta = \frac{[9R_2^2 + 4(1-R_2))]^{0.5}}{R_2}

    .. math::
        \chi_1 = \frac{\exp(\lambda_1 R_2 NTU_2/3)}{2\delta}

    .. math::
        \chi_2 = \frac{\exp(\lambda_2 R_2 NTU_2/3)}{2\delta}

    1-N TEMA E, shall and tube fluids mixed, efficient counterflow orientation,
    N an even number:

    .. math::
        P_2 = \frac{2}{A + B + C}

    .. math::
        A = 1 + R_2 + \coth(NTU_2/2)

    .. math::
        B = \frac{-1}{N_1}\coth\left(\frac{NTU_2}{2N_1}\right)

    .. math::
        C = \frac{1}{N_1}\sqrt{1 + N_1^2 R_2^2}
        \coth\left(\frac{NTU_2}{2N_1}\sqrt{1 + N_1^2 R_2^2}\right)

    .. math::
        N_1 = \frac{N_{tp}}{2}

    Parameters
    ----------
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 (shell side = 1, tube side = 2) [-]
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 (shell side = 1, tube side
        = 2) [-]
    Ntp : int
        Number of tube passes, 1, 2, 3, 4, or an even number[-]
    optimal : bool, optional
        Whether or not the arrangement is configured to give more of a
        countercurrent and efficient (True) case or an inefficient parallel
        case, [-]

    Returns
    -------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]

    Notes
    -----
    For odd numbers of tube passes greater than 3, an exception is raised.
    [2]_ actually has a formula for 5 tube passes, but it is an entire page
    long.
    The convention for the formulas in [1]_ and [3]_ are with the shell side
    as side 1, and the tube side as side 2. [2]_ has formulas with the
    opposite convention.

    Examples
    --------
    >>> temperature_effectiveness_TEMA_E(R1=1/3., NTU1=1., Ntp=1)
    0.5870500654031314

    References
    ----------
    .. [1] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [2] Thulukkanam, Kuppan. Heat Exchanger Design Handbook, Second Edition.
       CRC Press, 2013.
    .. [3] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    if Ntp == 1:
        # Just the basic counterflow case
        if R1 != 1:
            P1 = (1-exp(-NTU1*(1-R1))) / (1 - R1*exp(-NTU1*(1-R1)))
        else:
            P1 = NTU1/(1. + NTU1)
    elif Ntp == 2 and optimal:
        if R1 != 1:
            E = (1. + R1**2)**0.5
            P1 = 2./(1 + R1 + E/tanh(E*NTU1/2.))
        else:
            P1 = 1/(1 + 1/tanh(NTU1*2**-0.5)*2**-0.5)
    elif Ntp == 2 and not optimal:
        # Shah, reverse flow but with divider; without divider would be parallel.
        # Same as J-1, but E = A and B = B.
        A = exp(NTU1)
        B = exp(-NTU1*R1/2.)
        if R1 != 2:
            P1 = 1/R1*(1 - (2-R1)*(2*A + R1*B)/(2+R1)/(2*A - R1/B))
        else:
            P1 = 0.5*(1 - (1 + A**-2)/2./(1+NTU1))
    elif Ntp == 3 and optimal:
        # This gives slightly different results than in Thulukkanam!
        lambda3 = R1 # in Rosehnhow, this is minus. makes a small diff though
        lambda2 = -1.5 - (2.25 + R1*(R1-1))**0.5
        lambda1 = -1.5 + (2.25 + R1*(R1-1))**0.5
        delta = lambda1 - lambda2
        X1 = exp(lambda1*NTU1/3.)/2/delta
        X2 = exp(lambda2*NTU1/3.)/2/delta
        X3 = exp(lambda3*NTU1/3.)/2/delta
        C = X2*(3*R1 + lambda1) - X1*(3*R1 + lambda2) + X3*delta
        B = X1*(R1 - lambda2) - X2*(R1 - lambda1) + X3*delta
        if R1 != 1:
            A = X1*(R1 + lambda1)*(R1 - lambda2)/2/lambda1 - X3*delta - X2*(R1 + lambda2)*(R1 - lambda1)/2/lambda2 + 1./(1-R1)
        else:
            A = -exp(-NTU1)/18 - exp(NTU1/3.)/2 + (NTU1 + 5)/9.
        P1 = 1./R1*(1. - C/(A*C + B*B))
    elif Ntp == 3 and not optimal:
        # Thulukkanam, Parallel instead of direct.
        R1_orig = R1
        #NTU2 = NTU1*R1_orig but we want to treat it as NTU1 in this case
        NTU1 = NTU1*R1_orig # switch 1
        # R2 = 1/R1 but we want to treat it as R1 in this case
        R1 = 1./R1_orig # switch 2

        delta = (9*R1**2 + 4*(1 - R1))**0.5/R1
        l1 = (-3 + delta)/2.
        l2 = (-3 - delta)/2.
        chi1 = exp(l1*R1*NTU1/3.)/2/delta
        chi2 = exp(l2*R1*NTU1/3.)/2/delta
        E = 0.5*exp(NTU1/3.)
        C = -chi1*(3 + R1*l2)/R1 + chi2*(3 + R1*l1)/R1 + E
        B = chi1*(1 - R1*l2)/R1 - chi2*(1 - R1*l1)/R1 + E
#        if R1 != 1:
        A = (chi1*(1 + R1*l1)*(1 - R1*l2)/(2*R1**2*l1) - E
             - chi2*(1 + R1*l2)*(1 - R1*l1)/(2*R1**2*l2) + R1*(R1 -1))
        # The below change is NOT CONSISTENT with the main expression and is disabled
#        else:
#            A = -exp(-NTU1)/18. - exp(NTU1/3)/2. + (5 + NTU1)/9.
        P1 = (1 - C/(A*C + B**2))

        P1 = P1/R1_orig # switch 3, confirmed

    elif Ntp == 4 or Ntp %2 == 0:
        # The 4 pass case is present in all three sources, and is confirmed to
        # give the same results for all three for Ntp = 4. However,
        # what is awesome about the Thulukkanam version is that it supports
        # n tube passes so long as n is even.
        R1_orig = R1
        #NTU2 = NTU1*R1_orig but we want to treat it as NTU1 in this case
        NTU1 = NTU1*R1_orig # switch 1
        # R2 = 1/R1 but we want to treat it as R1 in this case
        R1 = 1./R1_orig # switch 2

        N1 = Ntp/2.
        C = 1/N1*(1 + N1**2*R1**2)**0.5/tanh(NTU1/(2*N1)*(1 + N1**2*R1**2)**0.5)
        B = -1/N1/tanh(NTU1/(2*N1))
        A = 1 + R1 + 1/tanh(NTU1/2.)
        P1 = 2/(A + B + C)

        P1 = P1/R1_orig # switch 3, confirmed
    else:
        raise ValueError('For TEMA E shells with an odd number of tube passes more than 3, no solution is implemented.')
    return P1


def temperature_effectiveness_plate(R1, NTU1, Np1, Np2, counterflow=True,
                                    passes_counterflow=True, reverse=False):
    r'''Returns the temperature effectiveness `P1` of side 1 of a plate heat
    exchanger with a specified side 1 heat capacity ratio `R1`, side 1 number
    of transfer units `NTU1`, number of passes on sides 1 and 2 (respectively
    `Np1` and `Np2`).

    For all cases, the function also takes as arguments whether the exchanger
    is setup in an overall counter or parallel orientation `counterflow`, and
    whether or not individual stream passes are themselves counterflow or
    parallel.

    The 20 supported cases are as follows. (the first number of sides listed
    refers to side 1, and the second number refers to side 2):

    * 1 pass/1 pass parallelflow
    * 1 pass/1 pass counterflow
    * 1 pass/2 pass
    * 1 pass/3 pass or 3 pass/1 pass (with the two end passes in parallel)
    * 1 pass/3 pass or 3 pass/1 pass (with the two end passes in counterflow)
    * 1 pass/4 pass
    * 2 pass/2 pass, overall parallelflow, individual passes in parallel
    * 2 pass/2 pass, overall parallelflow, individual passes counterflow
    * 2 pass/2 pass, overall counterflow, individual passes parallelflow
    * 2 pass/2 pass, overall counterflow, individual passes counterflow
    * 2 pass/3 pass or 3 pass/2 pass, overall parallelflow
    * 2 pass/3 pass or 3 pass/2 pass, overall counterflow
    * 2 pass/4 pass or 4 pass/2 pass, overall parallel flow
    * 2 pass/4 pass or 4 pass/2 pass, overall counterflow flow

    For all plate heat exchangers, there are two common formulas used by most
    of the expressions.

    .. math::
        P_p(x, y) = \frac{1 - \exp[-x(1 + y)]}{1 + y}

        P_c(x, y) = \frac{1 - \exp[-x(1 - y)]}{1 - y\exp[-x(1 - y)]}

    The main formulas used are as follows. Note that for some cases such as
    4 pass/2 pass, the formula is not shown because it is that of 2 pass/4
    pass, but with R1, NTU1, and P1 conversions.

    For 1 pass/1 pass paralleflow (streams symmetric):

    .. math::
        P_1 = P_p(NTU_1, R_1)

    For 1 pass/1 pass counterflow (streams symmetric):

    .. math::
        P_1 = P_c(NTU_1, R_1)

    For 1 pass/2 pass (any of the four possible configurations):

    .. math::
        P_1 = 0.5(A + B - 0.5ABR_1)

    .. math::
        A = P_p(NTU_1, 0.5R_1)

    .. math::
        B = P_c(NTU_1, 0.5R_1)

    For 1 pass/3 pass (with the two end passes in parallel):

    .. math::
        P_1 = \frac{1}{3}\left[B + A\left(1 - \frac{R_1 B}{3}\right)\left(2
        - \frac{R_1 A}{3}\right)\right]

    .. math::
        A = P_p\left(NTU_1, \frac{R_1}{3}\right)

    .. math::
        B = P_c\left(NTU_1, \frac{R_1}{3}\right)

    For 1 pass/3 pass (with the two end passes in counterflow):

    .. math::
        P_1 = \frac{1}{3}\left[A + B\left(1 - \frac{R_1 A}{3}\right)\left(2
        - \frac{R_1 B}{3}\right)\right]

    .. math::
        A = P_p\left(NTU_1, \frac{R_1}{3}\right)

    .. math::
        B = P_c\left(NTU_1, \frac{R_1}{3}\right)

    For 1 pass/4 pass (any of the four possible configurations):

    .. math::
        P_1 = \frac{1-Q}{R_1}

    .. math::
        Q = \left(1 - \frac{AR_1}{4}\right)^2\left(1 - \frac{BR_1}{4}\right)^2

    .. math::
        A = P_p\left(NTU_1, \frac{R_1}{4}\right)

    .. math::
        B = P_c\left(NTU_1, \frac{R_1}{4}\right)

    For 2 pass/2 pass, overall parallelflow, individual passes in parallel
    (stream symmetric):

    .. math::
        P_1 = P_p(NTU_1, R_1)

    For 2 pass/2 pass, overall parallelflow, individual passes counterflow
    (stream symmetric):

    .. math::
        P_1 = B[2 - B(1 + R_1)]

    .. math::
        B = P_c\left(\frac{NTU_1}{2}, R_1\right)

    For 2 pass/2 pass, overall counterflow, individual passes parallelflow
    (stream symmetric):

    .. math::
        P_1 = \frac{2A - A^2(1 + R_1)}{1 - R_1 A^2}

    .. math::
        A = P_p\left(\frac{NTU_1}{2}, R_1\right)

    For 2 pass/2 pass, overall counterflow and individual passes counterflow
    (stream symmetric):

    .. math::
        P_1 = P_c(NTU_1, R_1)

    For 2 pass/3 pass, overall parallelflow:

    .. math::
        P_1 = A + B - \left(\frac{2}{9} + \frac{D}{3}\right)
        (A^2 + B^2) - \left(\frac{5}{9} + \frac{4D}{3}\right)AB
        + \frac{D(1+D)AB(A+B)}{3} - \frac{D^2A^2B^2}{9}

    .. math::
        A = P_p\left(\frac{NTU_1}{2}, D\right)

    .. math::
        B = P_c\left(\frac{NTU_1}{2}, D\right)

    .. math::
        D = \frac{2R_1}{3}

    For 2 pass/3 pass, overall counterflow:

    .. math::
        P_1 = \frac{A + 0.5B + 0.5C + D}{R_1}

    .. math::
        A = \frac{2R_1 EF^2 - 2EF + F - F^2}
        {2R_1 E^2 F^2 - E^2 - F^2 - 2EF + E + F}

    .. math::
        B = \frac{A(E-1)}{F}

    .. math::
        C = \frac{1 - A}{E}

    .. math::
        D = R_1 E^2 C - R_1 E + R_1 - \frac{C}{2}

    .. math::
        E = \frac{3}{2R_1 G}

    .. math::
        F = \frac{3}{2R_1 H}

    .. math::
        G = P_c\left(\frac{NTU_1}{2}, \frac{2R_1}{3}\right)

    .. math::
        H = P_p\left(\frac{NTU_1}{2}, \frac{2R_1}{3}\right)

    For 2 pass/4 pass, overall parallel flow:

    .. math::
        P_1 = 2D - (1 + R_1)D^2

    .. math::
        D = \frac{A + B - 0.5ABR_1}{2}

    .. math::
        A = P_p\left(\frac{NTU_1}{2}, \frac{R_1}{2}\right)

    .. math::
        B = P_c\left(\frac{NTU_1}{2}, \frac{R_1}{2}\right)

    For 2 pass/4 pass, overall counterflow flow:

    .. math::
        P_1 = \frac{2D - (1+R_1)D^2}{1 - D^2 R_1}

    .. math::
        D = \frac{A + B - 0.5ABR_1}{2}

    .. math::
        A = P_p\left(\frac{NTU_1}{2}, \frac{R_1}{2}\right)

    .. math::
        B = P_c\left(\frac{NTU_1}{2}, \frac{R_1}{2}\right)

    Parameters
    ----------
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 [-]
    Np1 : int
        Number of passes on side 1 [-]
    Np2 : int
        Number of passes on side 2 [-]
    counterflow : bool
        Whether or not the overall flow through the heat exchanger is in
        counterflow or parallel flow, [-]
    passes_counterflow : bool
        In addition to the overall flow direction, in some cases individual
        passes may be in counter or parallel flow; this controls that [-]
    reverse : bool
        Used **internally only** to allow cases like the 1-4 formula to work
        for the 4-1 flow case, without having to duplicate the code [-]

    Returns
    -------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]

    Notes
    -----
    For diagrams of these heat exchangers, see [3]_.
    In all cases, each pass is assumed to be made up of an infinite number
    of plates. The fluid velocities must be uniform across the plate channels,
    and the flow must be uniformly distributed between the channels. The heat
    transfer coefficient is also assumed constant.

    The defaults of counterflow=True and passes_counterflow=True will always
    result in the most efficient heat exchanger option, normally what is
    desired.

    If a number of passes which is not supported is provided, an exception is
    raised.

    Examples
    --------
    Three passes on side 1; one pass on side 2; two end passes in counterflow
    orientation.

    >>> temperature_effectiveness_plate(R1=1/3., NTU1=1., Np1=3, Np2=1)
    0.5743514352720835

    If the same heat exchanger (in terms of NTU1 and R1) were operating with
    sides 1 and 2 switched, a slightly less efficient design results.

    >>> temperature_effectiveness_plate(R1=1/3., NTU1=1., Np1=1, Np2=3)
    0.5718726757657066

    References
    ----------
    .. [1] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [2] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [3] Kandlikar, S. G., and R. K. Shah. "Asymptotic Effectiveness-NTU
       Formulas for Multipass Plate Heat Exchangers." Journal of Heat Transfer
       111, no. 2 (May 1, 1989): 314-21. doi:10.1115/1.3250679.
    .. [4] Kandlikar, S. G., and R. K. Shah. "Multipass Plate Heat Exchangers
       Effectiveness-NTU Results and Guidelines for Selecting Pass
       Arrangements." Journal of Heat Transfer 111, no. 2 (May 1, 1989):
       300-313. doi:10.1115/1.3250678.
    '''
    if Np1 == 1 and Np2 == 1 and counterflow:
        return Pc(NTU1, R1)
    elif Np1 == 1 and Np2 == 1 and not counterflow:
        return Pp(NTU1, R1)
    elif Np1 == 1 and Np2 == 2:
        # There are four configurations but all have the same formula
        # They do behave different depending on the number of available plates
        # but this model assues infinity
        # There are four more arrangements that are equivalent as well
        A = Pp(NTU1, 0.5*R1)
        B = Pc(NTU1, 0.5*R1)
        return 0.5*(A + B - 0.5*A*B*R1)
    elif Np1 == 1 and Np2 == 3 and counterflow:
        # There are six configurations, two formulas
        # Each behaves differently though as a function of number of plates
        A = Pp(NTU1, R1/3.)
        B = Pc(NTU1, R1/3.)
        return 1/3.*(A + B*(1. - R1*A/3.)*(2. - R1*B/3.))
    elif Np1 == 1 and Np2 == 3 and not counterflow:
        A = Pp(NTU1, R1/3.)
        B = Pc(NTU1, R1/3.)
        return 1/3.*(B + A*(1. - R1*B/3.)*(2. - R1*A/3.))
    elif Np1 == 1 and Np2 == 4:
        # four configurations
        # Again a function of number of plates, but because expressions assume
        # infinity it gets ignored and they're the same
        A = Pp(NTU1, 0.25*R1)
        B = Pc(NTU1, 0.25*R1)
        t1 = (1. - 0.25*A*R1)
        t2 = (1. - 0.25*B*R1)
        t3 = t1*t2 # minor optimization
        return (1. - t3*t3)/R1
    elif Np1 == 2 and Np2 == 2:
        if counterflow and passes_counterflow:
            return Pc(NTU1, R1)
        elif counterflow and not passes_counterflow:
            A = Pp(0.5*NTU1, R1)
            return (2.*A - A*A*(1. + R1))/(1. - R1*A*A)
        elif not counterflow and passes_counterflow:
            B = Pc(0.5*NTU1, R1)
            return B*(2. - B*(1. + R1))
        elif not counterflow and not passes_counterflow:
            return temperature_effectiveness_plate(R1, NTU1, Np1=1, Np2=1,
                                                   passes_counterflow=True,
                                                   counterflow=False,
                                                   reverse=False)
    elif Np1 == 2 and Np2 == 3:
        # One place says there are four configurations; no other discussion is
        # presented
        if counterflow:
            H = Pp(0.5*NTU1, 2./3.*R1)
            G = Pc(0.5*NTU1, 2./3.*R1)
            E = 1./(2./3.*R1*G)
            F = 1./(2./3.*R1*H)
            E2 = E*E
            F2 = F*F
            A = (2.*R1*E*F2 - 2.*E*F + F - F2)/(2.*R1*E2*F2 - E2 - F2 - 2.*E*F + E + F)
            C = (1. - A)/E
            D = R1*E*E*C - R1*E + R1 - 0.5*C
            B = A*(E - 1.)/F
            return (A + 0.5*B + 0.5*C + D)/R1
        elif not counterflow:
            D = 2*R1/3.
            A = Pp(NTU1/2, D)
            B = Pc(NTU1/2, D)
            return (A + B - (2/9. + D/3.)*(A*A + B*B)
                    -(5./9. + 4./3.*D)*A*B
                    + D*(1. + D)*A*B*(A + B)/3.
                    - D*D*A*A*B*B/9.)
    elif Np1 == 2 and Np2 == 4:
        # Both cases are correct for passes_counterflow=True or False
        if counterflow:
            A = Pp(0.5*NTU1, 0.5*R1)
            B = Pc(0.5*NTU1, 0.5*R1)
            D = 0.5*(A + B - 0.5*A*B*R1)
            return (2.*D - (1. + R1)*D*D)/(1. - D*D*R1)
        elif not counterflow:
            A = Pp(0.5*NTU1, 0.5*R1)
            B = Pc(0.5*NTU1, 0.5*R1)
            D = 0.5*(A + B - 0.5*A*B*R1)
            return 2.*D - ((1. + R1)*D*D)
    if not reverse:
        # only the asymmetric cases will be able to solve by flipping things
        # Note that asymmetric performs differently depending on the arguments
        # The user still needs to input R1, NTU1 for side 1
        # so if they want to do a 3-1 instead of a 1-3 as is implemented here
        # They give R1 and NTU1 for "3 pass" side instead of the "1 pass" side
        # and will get back P1 for the "3 pass" side.

        # numba is dying on this recursion when caching is on, disable caching for now
        R2 = 1./R1
        NTU2 = NTU1*R1
        P2 = temperature_effectiveness_plate(R1=R2, NTU1=NTU2, Np1=Np2, Np2=Np1,
                                             counterflow=counterflow,
                                             passes_counterflow=passes_counterflow,
                                             reverse=True)
        P1 = P2*R2
        return P1

    raise ValueError('Supported number of passes does not have a formula available')


NTU_from_plate_2_3_parallel_offset = [7.5e-09, 1.4249999999999999e-08, 2.7074999999999996e-08, 5.144249999999999e-08, 9.774074999999998e-08, 1.8570742499999996e-07,
        3.528441074999999e-07, 6.704038042499998e-07, 1.2737672280749996e-06, 2.420157733342499e-06, 4.598299693350748e-06, 8.73676941736642e-06,
        1.6599861892996197e-05, 3.153973759669277e-05, 5.9925501433716265e-05, 0.0001138584527240609, 0.0002163310601757157, 0.0004110290143338598,
        0.0007809551272343336, 0.0014838147417452338, 0.002819248009315944, 0.005356571217700294, 0.010177485313630557, 0.019337222095898058,
        0.036740721982206306, 0.06980737176619198, 0.13263400635576475, 0.25200461207595304, 0.47880876294431074, 0.9097366495941903,
        1.7284996342289616, 3.2841493050350268, 6.23988367956655, 11.855778991176445, 22.525980083235243, 42.79936215814696, 81.31878810047922,
        154.5056973909105, 293.56082504272996, 557.7655675811869
    ]
NTU_from_plate_2_3_parallel_p = [
        [6.462119185839311e+42, -7.89735987601022e+35, -9.087083856062007e+27, 1.7050113422248866e+19, -1688535720.902906, 39.613950137696285],
        [2.318252075647079e+40, 4.220218073315233e+33, 1.8285573384676073e+26, 3.084411438713216e+18, 21021750157.222004, 38.330242461631514],
        [-2.725246602458712e+39, 2.3068644128776514e+32, 4.627829472767517e+25, 1.1310001900283219e+18, 966837452.7610933, 37.046534751334335],
        [1.2235529142820281e+37, 8.353473028988188e+30, 1.1577363564247304e+24, 5.26111342386283e+16, 1261185446.350375, 35.762826989453515],
        [-2.268909561806343e+38, -1.3039314202429206e+32, -1.0060952358606218e+25, 2.825039921898014e+17, -4479452951.060747, 34.47911926523136],
        [3.1317729161848482e+35, 1.2588127743217981e+30, 3.276383218600776e+23, 5869028606256265.0, 1575663397.1343608, 33.19541167935255],
        [1.269695711103147e+34, 4.290756179192165e+28, 2.9775379948312816e+22, 5786034109781830.0, 164435306.647866, 31.911704317934102],
        [2.3505913930420825e+33, 1.1812928249795345e+28, 9.295788631785714e+21, 646240998532863.4, 152570202.7077346, 30.627996765657006],
        [9.372652014253886e+30, 1.3865784239035505e+26, 4.189500257555692e+20, 382474529586341.4, 87229551.47782452, 29.34428979550431],
        [2.4394239228002745e+29, 8.48439767361896e+24, 5.6565841650672206e+19, 104617438878045.67, 31457806.46330641, 28.060583495837115],
        [1.1756565638642434e+28, 8.043965680172975e+23, 9.83892031482124e+18, 28459853223759.457, -9973441.564916672, 26.776878200784502],
        [-7.434125480121518e+25, -9.162097000642035e+21, -1.8461512379952253e+17, -265723153860.95157, 9996257.22456268, 25.493174743647415],
        [2.0776904219507715e+25, 5.321566516109094e+21, 2.830031172775278e+17, 5145809833029.548, 30832206.351568446, 24.20947444204136],
        [-2.0513335420466532e+24, -2.289228286826745e+20, -929209616723210.4, 76660920130.18863, -1702371.8751699005, 22.925779468541926],
        [1.0921158668621774e+22, 1.0481031815969962e+19, 1918841223032205.0, 108793646298.43532, 1871101.8043717288, 21.642093202587958],
        [4.225007197010862e+20, 1.0174799962646577e+18, 418193922144622.8, 50623892747.72083, 1687944.9487507509, 20.35842139409845],
        [5.3571938699724366e+19, 1.9585365429957734e+17, 127061183144718.2, 22936953503.569767, 702355.1752422716, 19.074772810895016],
        [-4.732337226080112e+19, -1.1230910989054965e+17, -30110540624204.457, 6413264110.352206, -490016.8688730465, 17.791160116387005],
        [7.152554048415291e+17, 4327530795294686.5, 3402335929741.3604, -374943769.5945296, 826.0224931539136, 16.507601903407284],
        [984051013256121.4, 24396685224717.965, 97948169286.30563, 109018337.85835141, 26368.75705403507, 15.22412348284966],
        [50680993613887.11, 879940841713.1937, 3547863447.171348, 8460582.67871622, 12799.65592799938, 13.940760190166468],
        [-875507484434.7006, 25808107236.834812, 1010170510.2032485, 5345388.737994069, 4433.937793425797, 12.65756872526503],
        [-2545609349.155158, 1744025195.285, 104920108.05542274, 1393742.0881427797, 5165.737303799409, 11.374674386567444],
        [-44107270179.70516, -9966996021.270891, -227618846.07818735, 221021.67601654973, -3146.660020222547, 10.092433966069105],
        [-17223688.786470495, -23864400.404286776, -3058744.705574641, -92465.34160573207, 79.26693119855088, 8.811928082820112],
        [207089.07738276388, 1227730.4574734285, 66877.00675493496, -61.49057944248921, -50.69640655670642, 7.536263692403265],
        [-119095.47949526514, -896903.6380506152, -426050.23346265673, -50907.75212691221, -869.3604740196895, 6.27351591893888],
        [938.3304032079799, 15625.662817632307, 14723.43865453997, 3567.9928498984227, 173.109130231987, 5.042051163423371],
        [10.211511823367426, 282.8051136783963, 451.2780669544769, 189.16900463066924, 20.74121725239963, 3.8767657222797793],
        [1.1464843558305287, 40.27461672000967, 59.16225274077587, 10.379056077245352, 5.7929254308893565, 2.829369860323157],
        [-0.01276147756024723, -0.48777843281385097, 0.6032501474532808, 4.559017517350226, 1.8620463339982214, 1.9541776601728913],
        [-0.00014226514748006712, -0.023808643483844285, -0.289121674152071, -1.114005659180411, -1.1972235528397606, 1.2832212535010843],
        [-2.581120907700105e-05, -0.00418426633038938, 0.009339781397016987, 0.45600757529303554, -0.4034481381751588, 0.8099512337803161],
        [9.273821864840757e-07, 0.0007547318108085425, 0.03245749164236026, 0.2965758497751961, -0.27934160268782265, 0.4972383631571961],
        [-6.233458586767228e-06, -0.015884785719281296, -0.7538527292952916, 0.3269735695585127, -1.7829706967875358, 0.2994738457522529],
        [-5.781637596658974e-06, -0.0035915273327539812, -0.01157399390997103, 0.029559240961787817, 0.07388163435077193, 0.17786369186050757],
        [3.961559826678127e-10, 1.4543254012176232e-06, 0.0001957616811595701, 0.0005362176178959326, 0.013938290268178794, 0.10448348895756158],
        [-3.2803452031636568e-09, -5.9441710409871e-06, -2.9118957190454924e-05, 0.0009942748181445327, -0.004462241345741868, 0.06081951964311699],
        [-4.279563643962056e-12, -6.561380309179023e-08, -3.081429869708402e-05, 0.0006859774272069931, -0.005747071000328576, 0.035126028098573805],
        [-2.8686462733352614e-14, -7.790696462481343e-10, -6.653666019778786e-07, 1.23027958890778e-05, -0.0010604984777168322, 0.020147982521643307]
    ]

NTU_from_plate_2_3_parallel_q = [
        [1.3683227596264214e+41, -2.165884158620195e+34, -2.2644202669570306e+26, 4.297163948003187e+17, -35893193.88590451, 1.0],
        [7.421678196787487e+38, 1.2085634160881847e+32, 5.007192734165482e+24, 8.236249667102427e+16, 552099325.8742635, 1.0],
        [-7.873002863853876e+37, 7.713934423114926e+30, 1.3099803429483467e+24, 3.054836281163621e+16, 28091870.519562516, 1.0],
        [4.291647340124204e+35, 2.5740797788209477e+29, 3.375618651322059e+22, 1500065233601050.2, 36352377.943491936, 1.0],
        [-7.882176775266176e+36, -3.9793411938031736e+30, -2.8656985978364168e+23, 8113694498408136.0, -129324377.97821534, 1.0],
        [1.41563514882479e+34, 4.110117035735913e+28, 9.893491716866435e+21, 191433581840910.7, 47790732.509775005, 1.0],
        [4.999607502786247e+32, 1.4716031312619367e+27, 9.64517819667706e+20, 182008960314679.44, 5330443.45680183, 1.0],
        [9.614824718113606e+31, 4.141511196223844e+26, 3.0530586050969154e+20, 21521729107166.945, 5078800.175545379, 1.0],
        [4.1734091264684445e+29, 5.27412019351319e+24, 1.4929447878932255e+19, 13174953396585.623, 3026132.05876614, 1.0],
        [1.1769712254415933e+28, 3.433836747260235e+23, 2.1211399654690634e+18, 3756068921305.945, 1150517.6889615546, 1.0],
        [6.078422345745446e+26, 3.430506170699249e+22, 3.854675824830577e+17, 1055299400508.5115, -356221.4839331609, 1.0],
        [-4.089621096755519e+24, -4.1038010771907025e+20, -7474524051100801.0, -7335562207.356166, 401094.5632574399, 1.0],
        [1.2324411281606296e+24, 2.5711536866148785e+20, 1.2592856537866202e+16, 218766477060.82593, 1278536.155959092, 1.0],
        [-1.1176764700506538e+23, -1.0287618349794472e+19, -27891844673339.754, 3102282562.5889225, -71489.86007645915, 1.0],
        [7.664428901664619e+20, 5.762787874997372e+17, 95614758117398.92, 5149781499.376999, 87998.69012738229, 1.0],
        [3.4119512881216352e+19, 6.137984726024919e+16, 22450868109060.75, 2555122134.822306, 83774.17840871602, 1.0],
        [4.6001726715440077e+18, 1.251467305711515e+16, 7213875441880.186, 1219435827.5993042, 37305.80357283532, 1.0],
        [-3.7939650022637384e+18, -6879494995144079.0, -1586382848606.9028, 352684674.27383363, -27269.243916658892, 1.0],
        [6.7216868954665256e+16, 295784765041063.1, 202637936045.26175, -22780898.93638217, 205.15524877583223, 1.0],
        [123022914844252.03, 2025598600989.9355, 7038321451.723578, 7292211.533895971, 1820.5540872255187, 1.0],
        [5798659105371.397, 73063980599.12009, 280808963.08350337, 647166.1744273758, 969.0169714952737, 1.0],
        [-78112612992.76324, 3642113943.605489, 91804780.43020414, 430751.9137746645, 379.78281402152874, 1.0],
        [386280523.6835199, 257220540.02040184, 11120485.271634161, 129819.12602039713, 471.40597550979186, 1.0],
        [-10693211009.303871, -1223510009.089181, -22274545.978089698, 18549.30307838344, -301.55242573732323, 1.0],
        [-9964085.923738554, -4365150.737138249, -411386.6438696054, -10484.174450932624, 15.149670076979616, 1.0],
        [541842.864264581, 193482.276072197, 9029.947067528488, -46.62458209582185, -2.9624038642089237, 1.0],
        [-220621.23296789522, -274557.9340203262, -86401.22446411186, -8443.070539356306, -136.23322675049926, 1.0],
        [3251.5504349303765, 6784.343990577191, 3934.5214064481625, 757.415724007334, 35.811562912463835, 1.0],
        [53.522734456906555, 170.40437538950374, 161.01012955496134, 53.54537645357537, 6.286762810448105, 1.0],
        [7.431444192909625, 26.301988720695135, 22.960637706381636, 4.801125750057049, 2.6341536761332267, 1.0],
        [-0.09119759477325269, -0.1801061475533146, 1.1300524605333688, 2.654994783545935, 1.3104918958676977, 1.0],
        [-0.003219836192304881, -0.06183937303877033, -0.40176496853684673, -1.0694441157327017, -0.7230743001450916, 1.0],
        [-0.0006010237525033756, -0.004761184653841278, 0.0793179816483663, 0.5022239858108873, -0.3793171422255701, 1.0],
        [8.145183997513813e-05, 0.005514321701324045, 0.10462537982334606, 0.5591837393502016, -0.4962661882194018, 1.0],
        [-0.0015949168754582125, -0.14279546004455532, -2.4776545238210446, 0.8797754471386745, -5.918081385718453, 1.0],
        [-0.0004175562163856941, -0.021446016127794405, -0.06189815758682979, 0.1741242853443794, 0.4345628604094691, 1.0],
        [1.3086618344372457e-07, 3.314955362016943e-05, 0.0019252809212552016, 0.00649537976330168, 0.14368642580610227, 1.0],
        [-5.903426856053935e-07, -0.00010040636378527861, -0.00038871197761919683, 0.015942278794350923, -0.06787066591282914, 1.0],
        [-5.058158205238362e-09, -4.452304825119217e-06, -0.0008198822672829164, 0.01904864306291245, -0.16068039934665684, 1.0],
        [-5.653709672617434e-11, -9.033364681018711e-08, -3.206216480769893e-05, 0.0005282786596535016, -0.05107404858407456, 1.0]
    ]

NTU_from_plate_2_4_parallel_offset = [7.5e-08, 1.5e-07, 3e-07, 6e-07, 1.2e-06, 2.4e-06, 4.8e-06, 9.6e-06, 1.92e-05, 3.84e-05, 7.68e-05, 0.0001536, 0.0003072, 0.0006144, 0.0012288,
        0.0024576, 0.0049152, 0.0098304, 0.0196608, 0.0393216, 0.0786432, 0.1572864, 0.3145728, 0.6291456, 1.2582912, 2.5165824, 5.0331648, 10.0663296,
        20.1326592, 40.2653184, 80.5306368
    ]

NTU_from_plate_2_4_parallel_p = [
        [9.311147683727706e+30, 4.5399729910913565e+24, 3.1701820445412416e+17, -3255667075.4059496, 34.19784973443562],
        [-8.462341214355869e+28, -1.0512639657566505e+23, -2.397873666023334e+16, -1034750601.3220837, 32.81155554496288],
        [-3.6927863016874774e+26, -5.302214079709395e+20, 234857662396561.97, 234020488.33100477, 31.425261515365865],
        [1.4107278670661704e+28, 1.0940739712642787e+22, -9489860868172880.0, -687734269.4135523, 30.03896731810403],
        [1.1202073850258956e+26, 9.003750344938989e+20, 1018203877208583.4, -168299066.7668561, 28.65267341622791],
        [1.8786218497132513e+24, 2.175783072978128e+19, -23527291714453.8, -214131092.84929407, 27.266380416202626],
        [1.594257427685343e+23, 2.5690709590606633e+18, 5268834190667.634, -5146824.418519467, 25.880088256899946],
        [-1.288122210563083e+24, -2.519981017465274e+19, 71292241683275.88, 50019381.18860227, 24.493798790383742],
        [4.550486151425468e+20, 6.198563773137455e+16, 1012994755481.759, -6103023.336117795, 23.107514134304758],
        [1.4422649047868342e+19, 4076064182439184.0, 179797171632.60355, 949172.8606007934, 21.721238910630372],
        [-1.9069286759409193e+19, -1.2582415698734126e+16, -1417288085609.7737, -31351459.7223076, 20.334983003863815],
        [1.7928329210392084e+18, 381796280677969.25, -92227167011.14949, -1569077.3721812628, 18.94876593787189],
        [760697593557199.6, 2716878092435.118, 1858731756.7367892, 384593.9690124189, 17.562626424902703],
        [-2980345873795.288, 16009202094.020222, 90691363.25017029, 80729.4890879997, 16.17664340548921],
        [131451648291400.28, 725493611523.8276, 609748841.6264181, 108426.56091552545, 14.790977175191966],
        [-6486852327.346251, -700388548.9015493, -5924391.104567888, -7146.787711492918, 13.40595533578488],
        [3015535151.0983796, 286704001.5011526, 4236710.766282784, 17437.87545106799, 12.022255562960611],
        [-10942972.229197213, -1899779.4140737972, -10761.968304033877, 1039.3968445688706, 10.641289014994197],
        [8220953.631429097, 2918560.727065227, 148692.00451266085, 2287.789927663398, 9.266006978920416],
        [-1464338.9963959353, 124017.62034205094, 30117.83144319644, 483.6759667244992, 7.902526411526671],
        [66462.98718285977, 130896.08947621491, 17250.88254238583, 284.03597528444567, 6.5631079987885395],
        [10553.524946159525, 11986.93762850903, 1221.0626417135122, -30.985812105731874, 5.270578375104107],
        [10.082588418517084, 303.14370012586807, 202.6175740387262, 25.89698508509613, 4.062131572817784],
        [0.5499329874641293, 17.521023550239388, 43.72139301663008, 27.37645885999178, 2.986658250212071],
        [-0.010913878122056958, 0.45840298571677157, 2.2251798856421043, 2.423889635140142, 2.090206859688884],
        [0.0006495212907688513, 0.32658889500269045, 4.465059447323286, 11.849683742435108, 1.3958641696720409],
        [-0.00010585785411094622, -0.036741246609915734, -0.7420406683674805, -3.1387647586257446, 0.8947297519985067],
        [0.0002762890128259569, 0.07593463651942846, 1.0334886344644887, -1.3947478189369613, 0.5543111790837415],
        [5.081221447274532e-06, 0.017547763053272165, 0.7422027775716376, -0.4571910134917369, 0.3340446930199166],
        [-2.616743919341764e-08, 0.00020934843129808607, 0.020386785890459298, -0.02006239152998345, 0.19683902757373653],
        [-3.101568430040571e-08, 0.00033734949109025253, 0.06821623160637866, -0.15878348849333124, 0.1138774497175727]
    ]
NTU_from_plate_2_4_parallel_q = [
        [3.291974284016541e+29, 1.4046022258329841e+23, 9191295776559532.0, -94421155.58205885, 1.0],
        [-3.193039887202064e+27, -3.4584178939213135e+21, -744806062875625.5, -31129803.142464183, 1.0],
        [-1.4381734658615522e+25, -1.693967837916621e+19, 8744771440563.748, 7659034.457524104, 1.0],
        [5.334771832140096e+26, 3.310748737686997e+20, -318539018055488.6, -22783770.25004286, 1.0],
        [4.926301588203296e+24, 3.3624134280590442e+19, 35173571760931.96, -5815597.159534814, 1.0],
        [8.649595607153984e+22, 8.156774026772406e+17, -1108318826792.9722, -7822738.38075017, 1.0],
        [7.404592532060123e+21, 1.0301055990958646e+17, 198966807158.52313, -182772.06662951043, 1.0],
        [-6.23738327477897e+22, -1.0048009493765537e+18, 2927622159692.4263, 2050629.9961337105, 1.0],
        [2.6592386481247224e+19, 2908854062634457.0, 42550661625.26152, -259606.3621141768, 1.0],
        [9.134101063562578e+17, 206794469159654.2, 8356789568.550677, 46095.68024652044, 1.0],
        [-1.3457090730356344e+18, -697630890940867.9, -71678079579.3958, -1540469.3978697397, 1.0],
        [1.1648162142676827e+17, 16957170961125.854, -4925846474.02719, -82119.20066004853, 1.0],
        [69235668361192.89, 184631921507.40326, 113484951.21664792, 22269.071053282812, 1.0],
        [-214631136673.61963, 1622101574.6971405, 6486940.571456372, 5191.662978533358, 1.0],
        [13037201476178.576, 53359497555.82677, 41997862.035921745, 7440.558062890958, 1.0],
        [-1510382156.1036587, -72336552.91298369, -482917.9480942061, -472.4787451295257, 1.0],
        [601613270.3192981, 32675351.48453366, 399065.14389228437, 1484.22312195332, 1.0],
        [-2806385.722630836, -221612.42147962135, 234.94182591367698, 116.6915537324147, 1.0],
        [2708621.889882308, 453979.7463113455, 18565.770503192096, 257.75689787361085, 1.0],
        [-222744.307636411, 37667.70409365177, 4153.602682046006, 67.49251582259824, 1.0],
        [57484.8140277244, 29249.909883546512, 2777.238094492285, 46.96976971793605, 1.0],
        [6211.193365476433, 2807.633263716254, 215.9750919403814, -3.689822852474681, 1.0],
        [68.48406976517343, 136.01076483156524, 57.439094496787085, 7.674214233620933, 1.0],
        [3.138051268210192, 15.474492941908835, 21.438163552556862, 9.928142597214924, 1.0],
        [0.052309953216059274, 0.647521303684222, 1.5240912769957224, 1.5965807136033143, 1.0],
        [0.03724768734929924, 0.9146363469974856, 5.2570155120528295, 8.733003177321063, 1.0],
        [-0.004061747315746429, -0.14038236184631592, -1.29781295599634, -3.375368854837777, 1.0],
        [0.008747917519380184, 0.2707496260037431, 1.685727780371999, -2.445440187207637, 1.0],
        [0.0015720431608568725, 0.1353192513128876, 2.1708447802007065, -1.3315013189823655, 1.0],
        [1.6024647013909014e-05, 0.0030671976634641853, 0.10156204670681959, -0.08262254790017286, 1.0],
        [2.3229283659683647e-05, 0.008935849190909461, 0.5851500679118268, -1.384387683729736, 1.0]
    ]

NTU_from_plate_2_2_parallel_counterflow_offset = [7.5e-08, 1.5e-07, 3e-07, 6e-07, 1.2e-06, 2.4e-06, 4.8e-06, 9.6e-06, 1.92e-05, 3.84e-05, 7.68e-05, 0.0001536, 0.0003072, 0.0006144, 0.0012288,
        0.0024576, 0.0049152, 0.0098304, 0.0196608, 0.0393216, 0.0786432, 0.1572864, 0.3145728, 0.6291456, 1.2582912, 2.5165824, 5.0331648, 10.0663296,
        20.1326592, 40.2653184, 80.5306368
    ]
NTU_from_plate_2_2_parallel_counterflow_p = [
        [-1.3540947044203507e+30, -8.410868237241077e+23, -9.592458969288491e+16, -2069815723.7830818, 32.81155785582896],
        [-5.909190345903646e+27, -4.24286158072528e+21, 939084401227737.4, 468010968.1889994, 31.42526592915459],
        [2.2580839013543478e+29, 8.759887723905137e+22, -3.795586069473787e+16, -1376170474.1965516, 30.03897572979474],
        [1.792810256083172e+27, 7.204726957462461e+21, 4073730638465264.0, -336704176.572998, 28.652689407724193],
        [3.0136351918641795e+25, 1.7449506492771002e+20, -94436408765550.34, -429497368.9344826, 27.2664107356295],
        [2.5503663600480286e+24, 2.0547278237477323e+19, 21065007797896.387, -10291984.296030033, 25.8801455685103],
        [-2.068959886273596e+25, -2.0275177343647046e+20, 283363617196754.2, 99817061.51082629, 24.493906760153582],
        [7.316250745095077e+21, 4.981073282863969e+17, 4071178832783.351, -12246412.243451085, 23.10771675633743],
        [2.303093386026652e+20, 3.2523232692334844e+16, 716985370983.0242, 1890719.1416514283, 21.72161752401211],
        [-4.4715545423999264e+20, -1.4745520169872864e+17, -8317355851715.972, -92722811.75950827, 20.335686947153395],
        [3.048309794645512e+19, 3461917411846938.0, -362599062006.45734, -3395666.6901195566, 18.95006719428506],
        [1.0840830998464214e+16, 18649593079388.15, 6191089972.499226, 639683.9318348671, 17.565015462728365],
        [-56829165736312.85, 70564590547.99759, 306929905.61277986, 148301.85724583076, 16.18099391544868],
        [310867835909252.25, 2199488772836.398, 1846929222.0995405, 116534.16034655277, 14.79882120114736],
        [-334740697066.67944, -11566416197.512352, -46807636.94628418, -36055.36084122248, 13.41992479832999],
        [26840001873.396393, 1176747347.4470434, 8813678.971174415, 20286.661696956173, 12.046746116490432],
        [331913333.4983728, 40516050.23123385, 806397.6785176995, 5236.358697987059, 10.683356504174844],
        [44564097.249639705, 3136095.9802567307, 13197.27737274941, 472.46470130684173, 9.336331182644951],
        [17994131.336623687, 4944651.796285454, 176426.70513709838, 1635.990096945895, 8.015855071773482],
        [1175197.7515924468, -5987292.526887681, -1249674.1787671829, -47443.87985192143, 6.736866971917378],
        [32169.38393625651, 35667.678007831906, 3281.4651865653304, 10.308412025244055, 5.519759829746941],
        [726.6676328162513, 2656.6374294136144, 938.8331291251928, 91.97929785764381, 4.389835242348104],
        [19.69824258336419, 111.99161644970393, 66.99538082454526, 16.542529871221728, 3.3746537841440527],
        [0.14533082879935388, 4.621483129606414, 13.046154764778136, 10.666553990710677, 2.4990539191450107],
        [0.002263983168833709, 0.012748221620148423, 0.08650686829071288, 1.2031142177285348, 1.779035515465167],
        [-9.830589986253666e-05, -0.0029990944273936377, 0.07473329578861215, 0.6680604884342561, 1.2170809731605838],
        [-0.00023383933828264596, -0.055094751033442003, -0.9014330674014724, -3.2873931899225397, 0.8013800839068237],
        [-4.59305763926361e-06, -0.0023384643471826417, -0.075298488101913, -0.519264025248997, 0.5094004416089638],
        [8.296798367082449e-06, 0.005019553134541902, 0.14520784160424574, -0.1480560699285633, 0.3138448562017667],
        [-0.00010404498951688742, -0.042625544741268964, 0.07039958323658324, -0.19160439461435394, 0.18823178739081262],
        [1.9257874863594892e-08, 5.065044581803412e-05, 0.005970986973437208, 0.007799255403717845, 0.11036344917508734]
    ]

NTU_from_plate_2_2_parallel_counterflow_q = [
        [-5.109312737272847e+28, -2.7669837286303267e+22, -2979525666165644.0, -62269188.45954034, 1.0],
        [-2.3013695133305585e+26, -1.3555563147438998e+20, 34967643100737.21, 15317110.975382129, 1.0],
        [8.539176311283978e+27, 2.651062211838455e+21, -1274041588071886.2, -45590896.094170064, 1.0],
        [7.884125680680726e+25, 2.690570806322227e+20, 140725727389471.77, -11634889.668290695, 1.0],
        [1.3875230679709336e+24, 6.541525966824309e+18, -4448026600910.923, -15690760.761365911, 1.0],
        [1.1845109172174247e+23, 8.238643367203461e+17, 795468754178.3317, -365479.85193633026, 1.0],
        [-1.0019142582756757e+24, -8.086708001888148e+18, 11636573241044.895, 4092189.8130896357, 1.0],
        [4.274893801197289e+20, 2.337481097686945e+16, 171016757014.2874, -520955.8420503665, 1.0],
        [1.4582333948892862e+19, 1649920265439080.0, 33323373506.59859, 91837.83231499583, 1.0],
        [-3.154069768612261e+19, -8175627871376203.0, -420703621074.17615, -4557050.2207395835, 1.0],
        [1.9847893757166674e+18, 157724843212160.53, -19387590413.827114, -177816.88499161773, 1.0],
        [978720427117839.0, 1262120134355.849, 377566806.9648005, 37158.4755696676, 1.0],
        [-4513922446287.058, 8413417806.407876, 22154634.809136182, 9566.661311724654, 1.0],
        [38070523736321.266, 175066430198.3794, 126396868.72314765, 8093.653907458961, 1.0],
        [-61489371269.06048, -1171842872.8663423, -3846269.020383889, -2566.272998013682, 1.0],
        [5045193301.256966, 133092026.4251891, 834715.8381330054, 1750.712772778083, 1.0],
        [87384327.07596397, 5685052.186587454, 91282.50331715525, 527.4122879147639, 1.0],
        [10190075.423735108, 370265.87179662683, 1818.871879286359, 71.60265020472393, 1.0],
        [6596827.383166512, 848613.961529598, 24268.6940300083, 216.01923171376424, 1.0],
        [-1866586.5008091372, -1826397.6735039286, -233559.34184976644, -7035.606933227122, 1.0],
        [23329.113452411028, 8850.625264022498, 589.593396865346, 5.78276680823048, 1.0],
        [1025.6980944478596, 1030.4061217224696, 257.83080542412074, 23.20339328125292, 1.0],
        [36.169804435094974, 56.11202269478698, 25.35072952019013, 6.191692301948293, 1.0],
        [0.870223837663587, 4.952736276431698, 8.141700513572173, 5.001805420623979, 1.0],
        [0.0053042419079786796, 0.012398428202822248, 0.27504711840752394, 1.0888403188024758, 1.0],
        [-0.0006872648984212443, 0.006963112330393481, 0.17402749161779235, 0.7777215103341757, 1.0],
        [-0.006590838477539741, -0.19611938648031352, -1.640780687507329, -3.97716372393897, 1.0],
        [-0.00024349857853046174, -0.013728589278679348, -0.21715653326322837, -0.9520846696079065, 1.0],
        [0.0005124101312746552, 0.03261073652701299, 0.4456416681314436, -0.4360265030003868, 1.0],
        [-0.004815365117901423, -0.219395737344097, 0.35487804621679386, -0.9991702722289716, 1.0],
        [4.108812052585529e-06, 0.000985483311381511, 0.05478155149325794, 0.0804130994573366, 1.0]
    ]

NTU_from_H_2_unoptimal_offset = [7.5e-08, 1.5e-07, 3e-07, 6e-07, 1.2e-06, 2.4e-06, 4.8e-06, 9.6e-06, 1.92e-05, 3.84e-05, 7.68e-05, 0.0001536, 0.0003072, 0.0006144, 0.0012288,
        0.0024576, 0.0049152, 0.0098304, 0.0196608, 0.0393216, 0.0786432, 0.1572864, 0.3145728, 0.6291456, 1.2582912, 2.5165824, 5.0331648, 10.0663296,
        20.1326592, 40.2653184, 80.5306368
    ]
NTU_from_H_2_unoptimal_p = [
        [9.311090177068194e+30, 4.5399475564038554e+24, 3.1701651864920506e+17, -3255644601.8905473, 34.19784939508176],
        [-8.462125737049189e+28, -1.051236894580789e+23, -2.39780488292632e+16, -1034705971.0814927, 32.81155489224577],
        [-3.69265415130165e+26, -5.301822567178955e+20, 234882398081691.12, 234024752.99665904, 31.425260261916403],
        [1.410560599685134e+28, 1.093808467030952e+22, -9490122636110636.0, -687633577.4144348, 30.038964915173185],
        [1.1201146547787187e+26, 9.003076575091149e+20, 1018131781915969.1, -168283241.34922192, 28.652668818329857],
        [1.877203115154918e+24, 2.174212412689376e+19, -23504286677434.996, -213953804.07995003, 27.266371636249264],
        [1.5943258764466105e+23, 2.5692522022742523e+18, 5269608999836.621, -5146997.019028249, 25.88007152864389],
        [-1.2864224116384471e+24, -2.5151093731537924e+19, 71437838753037.27, 50054408.15140979, 24.49376699655128],
        [4.543942596226072e+20, 6.190381882242281e+16, 1011578832812.2455, -6097146.227414874, 23.107453874576674],
        [1.443032893602968e+19, 4079127223298191.0, 179961190587.62994, 950472.5517376459, 21.72112504269345],
        [-1.728525301221655e+19, -1.1407375350719094e+16, -1284373100357.226, -28343072.8472184, 20.334768568331974],
        [1.758725436886904e+18, 365882402028983.06, -92828439127.84097, -1530591.291875413, 18.948363656972532],
        [792036415908859.2, 2862760755487.6484, 1976373644.0058162, 409085.29888017545, 17.561875017574707],
        [-2786989433439.3467, 18416834507.402943, 95324460.57822925, 82900.5916379863, 16.175246819451317],
        [128921676834374.16, 719301630875.2167, 615567114.8357931, 110348.07378592492, 14.788396313036571],
        [-2794040572.6818414, -507746769.4624018, -4437994.568915218, -4358.585576142468, 13.401217944728895],
        [3853463713.3106866, 379360075.81886023, 5594276.687695783, 22313.0443512979, 12.013630281672944],
        [66137990.08414679, 19025214.66224554, 664515.59380578, 6201.159018478011, 10.625745240108772],
        [9822129.89007198, 3897096.80061018, 202285.06707868786, 2999.612381935198, 9.238376729914885],
        [-948514.706520815, 374713.5998567277, 41481.55233261393, 494.59805780216936, 7.8543751267742525],
        [-11980.837950093153, -20667.491465511925, -2068.4413590656304, 80.22238868415197, 6.481745527422999],
        [-838.008545833977, -6831.181943934252, -3093.5280521633795, -306.91642288399237, 5.139846224565047],
        [19.032578110906325, 391.13780322667174, 466.3339913926907, 133.0562843703077, 3.8685445515879007],
        [-0.6990369659716387, 25.392942977606992, 27.402588487800582, 0.6420111258460738, 2.733344882284591],
        [0.09338324529681567, 2.865351487389797, 5.114582382139363, 1.469511752917645, 1.8084168254866542],
        [-0.0007321936823915426, -0.11675608789113232, -1.0568175987314021, -1.748890445704759, 1.13225374983665],
        [0.0011779746288648211, 0.32293841158631104, 2.913257638698029, -2.4643506838886613, 0.6823222255523778],
        [5.3274221084394595e-06, 0.0014584108461436326, 0.00018851507626193995, -0.3401471788904687, 0.4011229006580691],
        [1.8271761689352308e-07, 0.0007480503587161893, 0.02957290171761351, -0.11994510120785795, 0.23177982864651042],
        [6.8784748559726126e-09, 1.0482965040695156e-05, 0.0002687892019338303, -0.025003862996259105, 0.13213333004089212],
        [8.763989208567784e-11, 3.591705984063844e-07, 4.921736181910166e-05, 0.0005333054403615145, 0.07446942413083177]
    ]
NTU_from_H_2_unoptimal_q = [
        [3.291954404217546e+29, 1.4045943998953848e+23, 9191247072077158.0, -94420499.23235528, 1.0],
        [-3.1929593654948206e+27, -3.458328656247019e+21, -744784566006064.1, -31128443.437835522, 1.0],
        [-1.4381207928261743e+25, -1.693830694462608e+19, 8745588741895.983, 7659170.596221632, 1.0],
        [5.334127451972166e+26, 3.309852357106662e+20, -318547388902977.1, -22780419.901297953, 1.0],
        [4.925904942982619e+24, 3.3621635012359696e+19, 35171092574049.785, -5815045.64860498, 1.0],
        [8.643102749014142e+22, 8.150919305905134e+17, -1107277799713.401, -7816238.671263142, 1.0],
        [7.404943670579725e+21, 1.0301814770628384e+17, 198996743041.04614, -182778.72900289771, 1.0],
        [-6.228991027766363e+22, -1.0027628939625004e+18, 2933582652790.756, 2052062.8183752794, 1.0],
        [2.6555314925123396e+19, 2905025729565769.5, 42490608344.3377, -259352.57588899546, 1.0],
        [9.139672887482086e+17, 206954301896627.38, 8364534478.701708, 46155.88226561543, 1.0],
        [-1.219999405456035e+18, -632482998158155.4, -64953233304.81653, -1392542.5072398013, 1.0],
        [1.1417868279414859e+17, 16091601606423.625, -4956298785.520264, -80089.7158649981, 1.0],
        [72281648309673.0, 194796621486.79834, 120708758.31747963, 23664.722681493673, 1.0],
        [-191688538350.4686, 1813128315.751175, 6801704.817635524, 5326.460112015771, 1.0],
        [12819548881301.445, 52995232684.17356, 42414015.75348217, 7571.915653206027, 1.0],
        [-902051651.0068148, -53121439.77681727, -359594.87348877365, -264.4646423506963, 1.0],
        [789302837.8910002, 43370445.43163683, 526337.3354614506, 1891.217157665758, 1.0],
        [20890921.591943897, 2670229.183391574, 73117.13860794589, 602.7662205426564, 1.0],
        [3563425.2597333514, 615785.8657853191, 25312.120646154457, 335.70276322683435, 1.0],
        [-5281.921891333585, 79753.7409802169, 5646.05302573743, 69.41887774964894, 1.0],
        [-10534.39102576053, -4471.762975806752, -281.81541885802784, 16.231070841093203, 1.0],
        [-2357.310740142227, -2582.904848108706, -744.555607310463, -57.3657004239234, 1.0],
        [113.868855200266, 255.96866595663204, 169.54873521825837, 35.836022237199735, 1.0],
        [6.394734078690184, 18.116469980060796, 10.094374048899589, 1.1083023536795746, 1.0],
        [0.7587189251723472, 3.0042361512189037, 3.205115156278497, 1.3209192849079732, 1.0],
        [-0.022869248133995868, -0.34815770989249484, -1.3774996352160318, -1.2636425293046998, 1.0],
        [0.060036246550538616, 1.1203157641107213, 3.7280232186165474, -3.462501131451635, 1.0],
        [0.00026946774515168733, 0.004270492842455563, -0.06591637316270788, -0.7704935865841344, 1.0],
        [0.0001068179798574021, 0.008394972657004668, 0.10682274388388532, -0.4776775454943614, 1.0],
        [1.506979766231119e-06, 0.00012838571550289634, -0.001854978747745747, -0.16888199753883318, 1.0],
        [4.521472806270594e-08, 1.1656542235467652e-05, 0.0007261231042891383, 0.01752787098221555, 1.0]
    ]

NTU_from_G_2_unoptimal_offset = [7.5e-08, 1.5e-07, 3e-07, 6e-07, 1.2e-06, 2.4e-06, 4.8e-06, 9.6e-06, 1.92e-05, 3.84e-05, 7.68e-05, 0.0001536, 0.0003072, 0.0006144, 0.0012288,
        0.0024576, 0.0049152, 0.0098304, 0.0196608, 0.0393216, 0.0786432, 0.1572864, 0.3145728, 0.6291456, 1.2582912, 2.5165824, 5.0331648, 10.0663296,
        20.1326592, 40.2653184
    ]
NTU_from_G_2_unoptimal_p = [
        [9.3109303298284e+30, 4.5398767600969564e+24, 3.170118237710197e+17, -3255582028.524916, 34.197848452021496],
        [-8.46153002817336e+28, -1.0511619908571051e+23, -2.3976144993098692e+16, -1034582403.5113205, 32.811553084100204],
        [-3.692290021887378e+26, -5.30074247026417e+20, 234950665825511.66, 234036525.3056526, 31.425256801589224],
        [1.4101005965316078e+28, 1.0930781123070384e+22, -9490843606104960.0, -687356679.2161583, 30.038958306458262],
        [1.1198607183619476e+26, 9.001227755812281e+20, 1017933807262523.2, -168239785.08146304, 28.652656224909716],
        [1.8733467374857413e+24, 2.16994206245589e+19, -23441677453694.316, -213471555.46532023, 27.26634769739391],
        [1.594512099343422e+23, 2.569743318791424e+18, 5271709548496.723, -5147454.137281263, 25.88002614751128],
        [-1.2818229253935495e+24, -2.5019202614319935e+19, 71832292062919.38, 50148771.068874955, 24.4936812282376],
        [4.526495011307341e+20, 6.168512896379921e+16, 1007792749560.6382, -6081385.246719949, 23.107292341412784],
        [1.4451218664677962e+19, 4087331357766796.5, 180399075525.08047, 953934.9722401868, 21.720822001301844],
        [-1.3869417526718001e+19, -9157013351748946.0, -1029788858691.2903, -22579365.615234394, 20.33420262961592],
        [1.6729941515616177e+18, 324820952603289.25, -94618051655.88188, -1432919.5602435556, 18.947312380207574],
        [881377079365588.4, 3273908659562.771, 2305553502.530553, 477247.7586482068, 17.5599347216217],
        [-2294712887420.415, 24687543307.214867, 107469238.04231983, 88609.84525340016, 16.171694155750217],
        [122305930099635.03, 705130609607.45, 635856450.1020223, 115721.47051671412, 14.78195757555553],
        [4611767764.284328, -109441848.88329092, -1372356.3121382846, 1370.2740105504195, 13.389707907074477],
        [10777953775.11337, 1102063066.2696054, 16002550.06022829, 59325.3942992436, 11.993442742792807],
        [177557128.393412, 49828237.759808525, 1637511.0220782922, 13518.269190691442, 10.591302235163992],
        [-11319417.192115635, -4781582.158127945, -231145.86260741882, -2382.4032164871915, 9.181987939464335],
        [4453125.420063021, 3506923.02340456, 273936.10624431854, 4847.528594239127, 7.7676966035380675],
        [13382.567476165745, 21430.76649959107, 2674.227284505346, 103.35356016929184, 6.3607932986774065],
        [543.272216259408, 4094.6104229588664, 2371.7677593267967, 332.1165814161398, 4.994049687616688],
        [1019.6252971158917, 7258.186980145784, 2649.616182665721, -234.08989484797726, 3.726702452635227],
        [-0.16061952890054718, 26.475103623327957, 43.95078747890926, 10.898179144782612, 2.631460758603036],
        [0.09149784249886796, 4.507115402896758, 15.671800736565093, 13.267529540880941, 1.761750359230347],
        [-0.00015770925539570476, -0.022822040552860745, -0.16411319759532497, 0.0669124755747916, 1.126534416642668],
        [0.00017204089843551025, 0.016550563961219227, -0.11929008580672096, -1.7458596555122052, 0.6944621419501356],
        [-7.685285829457997e-05, 0.0018127599213621482, 0.2174778970361136, -0.33456541736023027, 0.41625508736123845],
        [9.408372464685374e-05, 0.02102006446772354, 0.00912807079932442, -0.1551911187056057, 0.24418730333772493],
        [5.544070897390911e-08, 9.56760771830238e-05, 0.006378971832410667, -0.003131142559408388, 0.1408475496320552]
    ]
NTU_from_G_2_unoptimal_q = [
        [3.291899133189346e+29, 1.4045726158959873e+23, 9191111431994574.0, -94418671.74467972, 1.0],
        [-3.192736723751476e+27, -3.4580817365485486e+21, -744725064710364.8, -31124678.831225593, 1.0],
        [-1.4379756315457668e+25, -1.6934523380520071e+19, 8747844401754.986, 7659546.39574443, 1.0],
        [5.332355293465041e+26, 3.307386549899383e+20, -318570445606863.2, -22771206.601799294, 1.0],
        [4.924818318184813e+24, 3.361477658856301e+19, 35164284666572.586, -5813531.206756055, 1.0],
        [8.625452913515019e+22, 8.135001664691246e+17, -1104444855152.0972, -7798558.608923992, 1.0],
        [7.405897764320868e+21, 1.0303870840706498e+17, 199077908074.35632, -182796.37614918998, 1.0],
        [-6.206279982934437e+22, -9.972450845965889e+17, 2949730732036.1255, 2055922.8784550051, 1.0],
        [2.6456407174731583e+19, 2894792447086410.5, 42330038939.27757, -258671.97913873894, 1.0],
        [9.154740126119596e+17, 207382210576832.97, 8385210219.820697, 46316.26048683696, 1.0],
        [-9.792909081377169e+17, -507714298948915.3, -52072419949.23038, -1109132.0457241398, 1.0],
        [1.0837682471753584e+17, 13849549366085.156, -5047510075.563598, -74938.92605805417, 1.0],
        [80947665751539.0, 223424288658.0732, 140917314.1477617, 27549.357113325757, 1.0],
        [-132644052917.59738, 2311595719.622853, 7627291.298373281, 5680.979777532329, 1.0],
        [12251665633349.52, 52214132651.39238, 43847873.4458912, 7939.025429824954, 1.0],
        [357856427.292005, -13251164.49142286, -104886.64220397608, 163.45236732230813, 1.0],
        [2296389972.242021, 126515566.06074195, 1501329.2443050412, 4980.7209489861, 1.0],
        [57930714.72719192, 6980962.132745493, 178865.45887007687, 1295.833190332679, 1.0],
        [-4491299.168210062, -759374.3426606245, -28261.14356909038, -248.17491581939632, 1.0],
        [2506378.5603899686, 662130.496745444, 39393.242284047185, 630.7413500970226, 1.0],
        [11844.477007158022, 4964.567889890172, 475.3789207352269, 20.27033156384447, 1.0],
        [1416.4504720022767, 1813.2683066018321, 634.6090169991996, 68.9432285360459, 1.0],
        [2588.630195894044, 3034.5274746385926, 618.0900263390289, -61.3460322498087, 1.0],
        [6.112796092986507, 23.856734715976987, 20.0956829765361, 5.00250593541833, 1.0],
        [1.0163551310040695, 6.580584763171997, 12.526779989536907, 8.018967861683878, 1.0],
        [-0.00425071771516727, -0.057961324447462347, -0.14113094148720395, 0.3271266362038584, 1.0],
        [0.003503234523706573, 0.006625141755068311, -0.5342364646029686, -2.37092136161077, 1.0],
        [-0.00024692055494272906, 0.04414352357324488, 0.4614638247959701, -0.7287208247762657, 1.0],
        [0.00372150090621385, 0.08764328509210832, 0.012510039071747442, -0.5966700220138256, 1.0],
        [1.214635258673814e-05, 0.0015853341009969197, 0.0448060202483571, -0.0022540017940275857, 1.0]
    ]

NTU_from_P_J_2_offset = [7.5e-08, 2.25e-07, 6.75e-07, 2.025e-06, 6.075e-06, 1.8225000000000003e-05, 5.467500000000001e-05, 0.00016402500000000002, 0.000492075,
        0.001476225, 0.004428675, 0.013286025, 0.039858075, 0.119574225, 0.358722675, 1.0761680249999999, 3.2285040749999996, 9.685512224999998,
        29.056536674999997, 87.169610025]
NTU_from_P_J_2_p = [
        [1.379293737184234e+44, 5.439023353733237e+37, 4.2006081013865867e+30, 1.5600236133130092e+23, 1.0691012154169038e+16, 607948038.8625113, 35.584143946088],
        [3.977011844664325e+41, 4.3621401764830945e+35, 5.394318809746255e+28, 6.18977231622784e+20, 2371944977808139.5, -13729570.055684585, 33.386919579479546],
        [5.503399680509165e+39, 1.817486676934107e+34, 2.4835200809625483e+27, -4.94507688364635e+21, 1710942974186981.0, -323425148.685019, 31.189694983023596],
        [-5.688303341798886e+34, 1.9802128520243527e+30, 3.873601341667205e+25, 1.3266955228207854e+20, 32936950212583.0, -192901487.91097888, 28.99247043733991],
        [2.6577822797427475e+31, 3.677744229595766e+27, 9.720475078230118e+22, 7.630266992579775e+17, 1079206988084.657, -713421.4769980257, 26.795245857337733],
        [2.535644640454403e+31, 3.3854769442469036e+27, 7.414350002779228e+22, 2.7270860191982288e+17, 2032720913118.922, 2586699.473660061, 24.598021286804627],
        [-3.670558764178667e+26, -2.907829370394574e+23, -3.832557120797078e+19, -972481647974356.9, 8754962146.297441, -578040.0935389815, 22.40079670572366],
        [1.4567732266714603e+24, 2.0002302464098957e+21, 5.807908884342328e+17, 86872945369936.56, 11954307761.34507, 805855.0329613951, 20.203572198563034],
        [-1.8717456698528226e+20, -1.2176465524651423e+18, -778715222269321.5, 869869235169.809, 701696300.6903126, 123267.94767870879, 18.00634747585284],
        [-1.2720358195046216e+18, -2.994023342276172e+16, -115432726426873.97, -103528958523.52058, 5714026.505779477, -15843.156326800376, 15.809121639609089],
        [-693868088372775.0, -14665128532110.385, -40097140813.94941, -12207173.118260076, 874474.9845013419, 4765.386828156788, 13.611888987763974],
        [608526214919.5482, 213144699162.69724, 10494925952.93119, 161216968.40265524, 912961.118340575, 3645.556442288636, 11.414614080037193],
        [12400266.233880278, 55616915.46934441, 15009980.829033542, 1319417.6118004804, 48979.82322881266, 885.3320092533332, 9.217129450305661],
        [-2175425.2487209123, -50112748.10174761, -19806780.207386833, -1499087.9941959996, 37429.79938076639, -732.1927840030977, 7.019288428844824],
        [180.38230298958038, -6877.271284301687, -7818.717128329305, -1317.358943766587, 230.73216317292162, -43.76812039187138, 4.830718891488284],
        [-0.17395706235381111, -16.6766544106866, 17.378017727113797, 28.091561354012356, -3.3753097336005804, 9.746402232018756, 2.7734138119975618],
        [-0.08170110767689108, -1.9622386891531531, 31.973619141306784, -18.994072067345428, 18.01530939582972, -10.00277120433454, 1.2830169212938194],
        [-1.8673203116193894e-06, -0.0006729694269362634, -0.0038298059267577044, 0.17481160261931097, 0.5595504373144352, 1.1890765321959993, 0.5336606579343032],
        [-4.126347971671396e-07, -0.00010383779245126719, 0.0004941341620841793, -0.0036143614373294695, 0.03215823233734107, -0.13594349795793487, 0.2153400299994889],
        [1.1352761642923954e-12, 4.466300862534921e-09, 4.980466105756786e-07, -1.747924038757887e-05, 0.0005662659607976993, -0.0008124803373884537, 0.08545603782210896]
    ]
NTU_from_P_J_2_q = [
        [4.591167043547481e+42, 1.606114843399718e+36, 1.2027419595813414e+29, 4.5709455933251277e+21, 308809321150730.1, 17834210.46773281, 1.0],
        [1.4176611491239528e+40, 1.3607402099050933e+34, 1.5783578828462145e+27, 3.912507110657984e+19, 70413898494599.84, -144987.26733659825, 1.0],
        [2.1375766831490205e+38, 6.038431370476358e+32, 6.054805550705697e+25, -1.5264408314370513e+20, 53809598724793.99, -10274617.610954134, 1.0],
        [-2.141040713656621e+33, 8.876323674907398e+28, 1.4680506954801534e+24, 4.665180136962245e+18, 902143492964.3142, -6619437.068866051, 1.0],
        [1.432164429610735e+30, 1.6247132650490878e+26, 3.9301808580131244e+21, 2.9081928379120252e+16, 39088664097.274124, -14338.490436435039, 1.0],
        [1.3632572533340117e+30, 1.5022149283687727e+26, 3.0554280135011666e+21, 1.1447971515162446e+16, 83004234706.80063, 109620.15224286402, 1.0],
        [-2.429838319244439e+25, -1.5116925717630222e+22, -1.7919330403274191e+18, -42320389648484.82, 336427956.90190256, -24171.475216249233, 1.0],
        [1.029014370536014e+23, 1.1241714145850822e+20, 3.066377639010162e+16, 4603604903923.01, 614289753.7531514, 40490.27848023662, 1.0],
        [-1.7047361017944777e+19, -7.913845937988589e+16, -38110753754005.18, 56102625195.25173, 40336224.323338695, 7071.527033415434, 1.0],
        [-1.466460222837932e+17, -2368131962476749.0, -7883636266218.59, -6487219371.973534, 253874.3242902243, -916.4549397393939, 1.0],
        [-80727057098668.84, -1158659586259.3425, -3079314024.2938294, 658517.4797846868, 73213.81902982714, 383.2675505551966, 1.0],
        [154696420804.57706, 27182107403.606, 1083690882.8286123, 15089725.838928567, 83871.50963458995, 332.564715920644, 1.0],
        [14186109.875687288, 11604436.970824337, 2246816.302173693, 168932.58014667718, 5798.38653812484, 101.49810005349056, 1.0],
        [-10506354.591316702, -12816984.672097836, -3355998.4791482566, -200389.50800387145, 5079.542670150122, -101.92851768910828, 1.0],
        [-1306.7658976560958, -3118.914532403747, -1953.578672005278, -214.44697466919783, 37.10146877200857, -7.922586619768505, 1.0],
        [-3.4777812458808555, -2.66496086895917, 12.06040187108384, 9.382513303976543, 0.7618565915752666, 4.082394043941239, 1.0],
        [-0.5478521985532157, 4.4820443343765985, 21.325292193447055, -11.421627483324068, 12.178810914368531, -7.5579809400927775, 1.0],
        [-9.742102821144731e-05, -0.002065573589076824, 0.019790600076312437, 0.41450177639499663, 1.235749069760604, 2.312490295019506, 1.0],
        [-1.5856176703649368e-05, -0.00041496261982164757, 0.0018006879594633617, -0.012449317525219798, 0.13114216762570655, -0.6025944343758391, 1.0],
        [4.690094899391156e-10, 1.1078843951667822e-07, 3.7862270042457132e-06, -0.0001399289022755603, 0.006526270606084676, 0.00022681636474048618, 1.0]
    ]
NTU_from_P_J_4_offset = [7.5e-08, 1.5e-07, 3e-07, 6e-07, 1.2e-06, 2.4e-06, 4.8e-06, 9.6e-06, 1.92e-05, 3.84e-05, 7.68e-05, 0.0001536, 0.0003072, 0.0006144, 0.0012288,
        0.0024576, 0.0049152, 0.0098304, 0.0196608, 0.0393216, 0.0786432, 0.1572864, 0.3145728, 0.6291456, 1.2582912, 2.5165824, 5.0331648, 10.0663296,
        20.1326592, 40.2653184
    ]
NTU_from_P_J_4_p = [
        [1.1137179009139922e+22, 7276370402964671.0, 982286593.77106, 35.36100061385685],
        [3.9520974777651436e+20, 817675826063219.9, 326507731.5694956, 33.97470625485492],
        [-6.841863118463395e+19, -40856958897824.58, 81315681.91084157, 32.588411778703986],
        [-1.226994790378735e+20, -467148543840559.25, -248085221.6635263, 31.20211748691618],
        [6.48597266541994e+17, 11019359745896.43, 35584715.41588573, 29.8158232003719],
        [2.1079329317454016e+18, 35398661212068.19, 99510270.06019339, 28.429528868794645],
        [8935402325113268.0, 618377307638.4402, 8039429.253688355, 27.043234454056968],
        [5560958051752470.0, 452304392321.3483, 6957555.9298604345, 25.656940184631672],
        [126235735091357.48, 34964623767.870056, 1810157.0390347333, 24.270645740470233],
        [103033182132830.16, 36058139909.400696, 2072264.0804883514, 22.884351392471412],
        [-103982270484.21591, 838356119.2129722, 306578.86931115517, 21.498057055394064],
        [-7251975283452.386, -8287862367.975066, -1294281.061275808, 20.111763168535326],
        [-158106070969.9289, -328902231.04025, -61097.005884115875, 18.725469990458958],
        [-17294918972.910526, -76296368.6262589, -29707.458670753877, 17.339179506802296],
        [256922268.69636458, 5269647.432513786, 18204.805713360474, 15.952898438512468],
        [58898464.41169616, 1813429.6541062293, 10011.43433572321, 14.566648122016973],
        [446520.17751930194, 153233.62152383465, 3090.2164859982668, 13.180497004021207],
        [192422.14842327707, 48315.74348976142, 1547.3573889245085, 11.794657577802244],
        [22663.130317209336, 11333.59117949032, 700.5385580157777, 10.409766262101133],
        [1967.4393020460104, 2327.830281611502, 297.97297093031005, 9.027639452147119],
        [-318.3313733466705, 16.786387998216867, 82.34942017917173, 7.653122592980561],
        [11.458463050331853, 90.2649720625161, 49.98898169401306, 6.297942670584156],
        [0.1370275766836422, 13.527475822643838, 18.083157099365827, 4.9866549707448655],
        [-0.08811753949415488, 0.7580719231984764, 5.1618411905678885, 3.7607700748102135],
        [-0.026715037890536985, -7.268631456244228, -11.596324791801184, 2.6722136470814455],
        [0.0007400741868303971, 0.1167759145386527, 0.8860970545163467, 1.7671234621614083],
        [0.004830310767271772, 1.079863001801655, 6.078126424454076, 1.0804293879908267],
        [-0.0014226263147185095, -0.33627027530350345, -0.060678051246797524, 0.6204891876781652],
        [1.2996495928311868e-08, 7.487428756304456e-05, 0.01170061756683815, 0.34410237764731944],
        [-2.781758163682452e-07, -0.0008169026844673782, -0.06269490294979985, 0.18770099027480536]
    ]
NTU_from_P_J_4_q = [
        [3.838083335177554e+20, 222263807511839.7, 28532938.656622943, 1.0],
        [1.4833618909488933e+19, 26684604852189.977, 10002766.828354824, 1.0],
        [-2.475411036306694e+18, -1042375335783.4248, 2699804.4727502298, 1.0],
        [-4.833610251211771e+18, -15898701744151.383, -7844079.587894955, 1.0],
        [2.8854497720401664e+16, 416128712820.01227, 1249382.9866715993, 1.0],
        [9.364002670645787e+16, 1342489459568.0027, 3529555.587532283, 1.0],
        [453285058958963.44, 26079026061.61398, 312688.0379653563, 1.0],
        [286149498334759.3, 19473946611.856277, 279296.371346023, 1.0],
        [7422551752650.021, 1667363729.5110798, 78874.03269449141, 1.0],
        [6265733267256.983, 1757307727.953103, 92829.6964275974, 1.0],
        [1999652867.803483, 49852548.82714711, 15472.119794909637, 1.0],
        [-512013153282.79803, -455443511.83280444, -63707.00543582891, 1.0],
        [-12222218188.63782, -19143850.59303709, -2915.0995349095583, 1.0],
        [-1506979769.7819808, -4839415.404125513, -1525.5783956902367, 1.0],
        [29423132.775343254, 415645.0522902444, 1243.1836879724854, 1.0],
        [7318822.988441785, 154640.7646059886, 743.1491517287512, 1.0],
        [141201.0534254414, 16674.388169779257, 265.3197588068411, 1.0],
        [45153.80125997977, 5777.697281822111, 148.43127372342246, 1.0],
        [6966.966337389611, 1591.6903920455154, 77.05203947040957, 1.0],
        [917.2988119948286, 402.52765640680553, 38.61336284687747, 1.0],
        [-71.87668749611038, 26.972070330473336, 14.038555292975968, 1.0],
        [18.036042518702104, 27.128549126117615, 9.887065213985007, 1.0],
        [1.9767237853128081, 6.305760321878647, 4.800623897964247, 1.0],
        [0.04809865838967166, 1.0212459815948616, 2.083454689699847, 1.0],
        [-1.0298310153064745, -4.619556300508356, -3.90904614301033, 1.0],
        [0.016605629497247247, 0.1917630272238318, 0.7612198916261416, 1.0],
        [0.15032334712603257, 1.849779948438833, 5.776977591182038, 1.0],
        [-0.04700684698139208, -0.5503640789830394, -0.015203083681555351, 1.0],
        [7.574689191705429e-06, 0.0015757631170495458, 0.07698932879125621, 1.0],
        [-8.744918585890159e-05, -0.011691953640170439, -0.31212155570753924, 1.0]
    ]

NTU_from_P_basic_crossflow_mixed_12_offset = [7.5e-08, 1.5e-07, 3e-07, 6e-07, 1.2e-06, 2.4e-06, 4.8e-06, 9.6e-06, 1.92e-05, 3.84e-05, 7.68e-05, 0.0001536, 0.0003072, 0.0006144, 0.0012288,
        0.0024576, 0.0049152, 0.0098304, 0.0196608, 0.0393216, 0.0786432, 0.1572864, 0.3145728, 0.6291456, 1.2582912, 2.5165824, 5.0331648, 10.0663296,
        20.1326592, 40.2653184, 80.5306368, 161.0612736, 322.1225472, 644.2450944, 1288.4901888
    ]
NTU_from_P_basic_crossflow_mixed_12_p = [
        [1.0648657843926371e+28, 1.953857955950789e+22, 7266974622321844.0, 901310037.367627, 35.29646216590824],
        [-2.725970106134222e+28, -5.668910625458952e+22, -2.214546614947417e+16, -2075136770.0413792, 33.91016772641483],
        [3.317334215900722e+36, 1.83532794193258e+30, 1.3685590706146844e+21, 363848581451.6158, 32.52387319458047],
        [2.287548017339333e+25, 2.235626801942006e+20, 421302243477568.8, 237921355.15637708, 31.1375789140508],
        [3.7397053636559112e+25, 3.711246345042558e+20, 677728033156592.5, 293850388.0168432, 29.75128463980416],
        [1.0868350552139688e+30, 1.8455968608452927e+25, 4.7684805369296495e+19, 73543771834.092, 28.36499027336039],
        [-5.68877408709714e+22, -1.58579560884536e+18, -4299134851761.376, 23968028.669715144, 26.978695929717148],
        [1.6747242923170734e+22, 1.255438889589369e+18, 15434191675148.246, 27075146.549738526, 25.592401393795864],
        [1.0569269667427512e+22, 1.2502500074264484e+18, 18964792281222.582, -130738981.69876468, 24.2061071320861],
        [-2.5803066379744737e+18, 372261157232874.94, 28264835013.28186, -17050.367595574695, 22.819812804805323],
        [-9.755620022787496e+17, -725650029723333.5, -41926633472.52738, 3113611.357517518, 21.433518761319352],
        [-4.5273524706659283e+17, -909679128993800.9, -318106354724.0829, -28491925.038955502, 20.047224673431018],
        [9262291183292458.0, 19527324488391.098, 4613123691.341467, -663030.266969785, 18.660931382962406],
        [-1268775247799946.5, -14687587336923.14, -27389900433.704067, -12603645.373545203, 17.274641094810104],
        [3158577553978.4697, 71185736277.97443, 261619313.33159643, 249174.90223218722, 15.888360350134677],
        [152798849901.24323, 5254166915.823552, 29780970.83267744, 47672.85047654532, 14.502111276022058],
        [292692355448.6804, 11747722863.857492, 50327505.150009006, -93575.49190852011, 13.115964119634741],
        [1106317355.2031553, 127940907.99872206, 2104897.7308680704, 8042.20078053657, 11.7301387421864],
        [2197193.2553803152, 1285706.599464342, 94469.9714858117, 1982.9696716666438, 10.345295129780379],
        [628759.7709569605, 367127.78202066384, 28069.761808265386, 683.706593111148, 8.963325126986435],
        [528.078676943018, 2632.4181877354436, 1230.1904049453256, 174.67508689366082, 7.589305501960269],
        [-1494.2855542418577, -6088.743359266542, -2320.2207257673876, -183.19106697358848, 6.2356286720949745],
        [0.8507108729411813, 22.302501634010234, 45.4763530114471, 27.27372898572285, 4.928616927537855],
        [-1.2077329580227858, -36.62873095860811, -74.72114158071923, -30.567927983101946, 3.713928731013624],
        [0.8647680033690556, 31.52269638492401, 55.565702550395976, -4.780723933354412, 2.6509731108629784],
        [0.2985244757802897, 47.382849336432166, 448.40056689075203, 934.1164294026304, 1.7905411822239048],
        [-0.00021659913188603507, -0.031019118273585895, -0.06734409089895292, 1.4954198465032587, 1.1499737101680396],
        [2.1622908509544346e-05, 0.008770641444302658, 0.16604173854981807, 0.382168453132166, 0.7082821192347464],
        [4.196344927837933e-06, 0.0037795723343284817, 0.13015088360211113, 0.025875640008383147, 0.42214432523510087],
        [6.341115075750826e-09, 0.00019953280608039998, 0.01857524533658307, 0.00019535277989255078, 0.24534755148670773],
        [4.642762949093271e-06, 0.005095731601409439, -0.04639395438641826, 0.03610487934676299, 0.1398621181683815],
        [1.2306148922906546e-08, 4.004868303268181e-05, 0.003670103859870603, -0.030187701704396393, 0.07853403174401082],
        [4.3797422412183115e-12, 8.829865413669967e-08, 5.3467304702596584e-05, 0.0016073164432044349, 0.0435699623017177],
        [-7.062736110507592e-09, -7.668007088536754e-05, 0.0023318927789922703, -0.041990582834008715, 0.023936688503263276],
        [1.2508061870706862e-14, 2.573853599556939e-09, 7.24838458529996e-06, -0.00012330891215276327, 0.013044234435602044]
    ]

NTU_from_P_basic_crossflow_mixed_12_q = [
        [3.973671938704683e+26, 6.326552638003598e+20, 220710170167753.03, 26290926.9768005, 1.0],
        [-1.026756192847948e+27, -1.8529253231359474e+21, -678280420727251.2, -60801923.8992356, 1.0],
        [1.135562772430311e+35, 5.643879638731395e+28, 4.208089303288672e+19, 11187328400.684576, 1.0],
        [9.744383540727682e+23, 8.115450878596962e+18, 14270577510742.908, 7748023.36603521, 1.0],
        [1.5931786175761789e+24, 1.3561342279548963e+19, 23312891578604.902, 9932917.423178872, 1.0],
        [4.859688211478509e+28, 7.000358779926185e+23, 1.6811909822046449e+18, 2592795007.454948, 1.0],
        [-2.666944038287914e+21, -6.228033882597861e+16, -147002301520.06125, 903850.04214734, 1.0],
        [8.607232894250505e+20, 5.360902575722586e+16, 611331531040.4706, 1066077.3271792284, 1.0],
        [5.6923263643194255e+20, 5.5529930521810664e+16, 760135235703.9015, -5396770.908948376, 1.0],
        [-1.1447343385186822e+17, 19557170326968.332, 1212359780.557646, 1535.249044757823, 1.0],
        [-6.7087362640990504e+16, -37115893206839.71, -1786055637.3556702, 146483.33188585352, 1.0],
        [-3.4756109279027156e+16, -53271233725098.945, -16792652327.870298, -1420590.8612348323, 1.0],
        [717903881236155.8, 1149403566251.7737, 234365669.11825627, -35181.515793654195, 1.0],
        [-131068069929832.05, -1062923626979.946, -1723158048.0735118, -729415.4542761801, 1.0],
        [373138123021.75226, 5693132516.134963, 18041434.36138755, 15785.296488339987, 1.0],
        [19866365968.41335, 452347302.11955184, 2229754.0098727057, 3343.417286909013, 1.0],
        [39770464992.804535, 1030615797.581643, 3613628.7482191008, -7103.45523067367, 1.0],
        [200128825.45281503, 13652967.796242092, 190746.07900072972, 702.9359016386885, 1.0],
        [734053.0142461995, 188917.2993281958, 10859.229501012394, 201.49437641508658, 1.0],
        [209302.76040083222, 56139.34393204773, 3521.801145946451, 81.92398920910554, 1.0],
        [598.8866174601792, 715.9466859667624, 227.62193299094852, 26.319829515259006, 1.0],
        [-1479.3910297195062, -1619.331929198887, -432.5244387771081, -27.41252766738836, 1.0],
        [3.2574469229510354, 12.8571035438057, 15.091263524092597, 6.715596193986065, 1.0],
        [-5.223964240539322, -22.61021951185674, -26.126504739428494, -7.521201583422795, 1.0],
        [4.384367871411823, 20.758534092876616, 20.16342961309194, -1.3847296429245695, 1.0],
        [4.891865775589256, 81.08707101822866, 375.67471782459376, 521.9351243262005, 1.0],
        [-0.003354856095171886, -0.037975637276086216, 0.11222885303757815, 1.4337579152114632, 1.0],
        [0.0007839864026962514, 0.02894459533320549, 0.272639521725721, 0.611588697885286, 1.0],
        [0.0003025722525591445, 0.020683093661872085, 0.3104855870447247, 0.09936172614571803, 1.0],
        [1.3308588756670941e-05, 0.002314200802093514, 0.0756883464283556, 0.020616151735835784, 1.0],
        [0.0004083029953833101, 0.03304346443946263, -0.32908410954386225, 0.2683607066488552, 1.0],
        [2.7228204913745714e-06, 0.000755047941743242, 0.0447212276770666, -0.37916271374424443, 1.0],
        [4.914765639846191e-09, 5.27587799649255e-06, 0.001324873372051413, 0.03955253112545274, 1.0],
        [-4.6335857071244e-06, -0.0030716430602319663, 0.0950492776042504, -1.7528843558197416, 1.0],
        [1.2061450998792682e-10, 5.775284765597841e-07, 0.0005491863749069216, -0.008769384960859137, 1.0]
    ]



def _NTU_from_P_objective(NTU1, R1, P1, function, *args):
    '''Private function to hold the common objective function used by
    all backwards solvers for the P-NTU method.
    These methods are really hard on on floating points (overflows and divide
    by zeroes due to numbers really close to 1), so if the function fails,
    mpmath is imported and tried.
    '''
    P1_calc = function(R1, NTU1, *args)
    # Handled a larger range, not worth it
#    try:
#        P1_calc = function(R1, NTU1, **kwargs)
#    except :
#        try:
#            import mpmath
#        except ImportError:  # pragma: no cover
#            raise ValueError('For some reverse P-NTU numerical solutions, the \
#intermediary results are ill-conditioned and do not fit in a float; mpmath must \
#be installed for this calculation to proceed.')
#        globals()['exp'] = mpmath.exp
#        P1_calc = float(function(R1, NTU1, **kwargs))
#        globals()['exp'] = math.exp
    return P1_calc - P1


def _NTU_from_P_erf(NTU1, *args):
    '''Private function to hold the common objective function used by
    all backwards solvers for the P-NTU method.
    These methods are really hard on on floating points (overflows and divide
    by zeroes due to numbers really close to 1), so if the function fails,
    mpmath is imported and tried.
    '''
    R1, P1, function = args[0], args[1], args[2]
    return function(R1, NTU1, *args[3:]) - P1

def _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, function, guess, *args):
    '''Private function to solve the P-NTU method backwards, given the
    function to use, the upper and lower NTU bounds for consideration,
    and the desired P1 and R1 values.
    '''
    args2 = (R1, P1, function) + args
    try:
        if guess is not None:
            guess2 = guess
        elif NTU_min < 2.0 < NTU_max:
            guess2 = 2.0
        else:
            guess2 = NTU_min + 0.001*NTU_max
        if (NTU_min is not None and NTU_max is not None) and (guess2 < NTU_min or guess2 > NTU_max):
            guess2 = 0.5*(NTU_min + NTU_max)
        return secant(_NTU_from_P_erf, guess2, low=NTU_min, high=NTU_max, bisection=False, xtol=1e-13, args=args2)
    except:
        # secant failed. For some reason, the bisection in secant is going to wrong wrong value
        # floating point really sucks
        pass

    # Better for numerical stability if we don't need to evaluate these
    P1_max = _NTU_from_P_erf(NTU_max, *(R1, 0.0, function) + args)
    P1_min = _NTU_from_P_erf(NTU_min, *(R1, 0.0, function) + args)
    if P1 > P1_max:
        raise ValueError(f'No solution possible gives such a high P1; maximum P1={P1_max:f} at NTU1={NTU_max:f}') # numba: delete
        # raise ValueError("No solution") # numba: uncomment
    if P1 < P1_min:
        # raise ValueError("No solution") # numba: uncomment
        raise ValueError(f'No solution possible gives such a low P1; minimum P1={P1_min:f} at NTU1={NTU_min:f}') # numba: delete
    # Construct the function as a lambda expression as solvers don't support kwargs
    return brenth(_NTU_from_P_erf, NTU_min, NTU_max, args=args2)


def _NTU_max_for_P_solver(ps, qs, offsets, R1):
    '''Private function to calculate the upper bound on the NTU1 value in the
    P-NTU method. This value is calculated via a pade approximation obtained
    on the result of a global minimizer which calculated the maximum P1
    at a given R1 from ~1E-7 to approximately 100. This should suffice for
    engineering applications. This value is needed to bound the solver.
    '''
    offset_max = offsets[-1]
    for offset, p, q in zip(offsets, ps, qs):
        if R1 < offset or offset == offset_max:
            x = R1 - offset
            return horner(p, x)/horner(q, x)


def NTU_from_P_basic(P1, R1, subtype='crossflow'):
    r'''Returns the number of transfer units of a basic heat exchanger type
    with a specified (for side 1) thermal effectiveness `P1`, and heat capacity
    ratio `R1`. The supported cases are as follows:

    * Counterflow (ex. double-pipe) [analytical]
    * Parallel (ex. double pipe inefficient configuration) [analytical]
    * Crossflow, single pass, fluids unmixed [numerical]
    * Crossflow, single pass, fluid 1 mixed, fluid 2 unmixed [analytical]
    * Crossflow, single pass, fluid 2 mixed, fluid 1 unmixed [analytical]
    * Crossflow, single pass, both fluids mixed [numerical]

    The analytical solutions, for those cases they are available, are as
    follows:

    Counterflow:

    .. math::
        NTU_1 = - \frac{1}{R_{1} - 1} \ln{\left (\frac{P_{1} R_{1} - 1}{P_{1}
        - 1} \right )}

    Parallel:

    .. math::
        NTU_1 = \frac{1}{R_{1} + 1} \ln{\left (- \frac{1}{P_{1} \left(R_{1}
        + 1\right) - 1} \right )}

    Crossflow, single pass, fluid 1 mixed, fluid 2 unmixed:

    .. math::
        NTU_1 = - \frac{1}{R_{1}} \ln{\left (R_{1} \ln{\left (- \left(P_{1}
        - 1\right) e^{\frac{1}{R_{1}}} \right )} \right )}

    Crossflow, single pass, fluid 2 mixed, fluid 1 unmixed

    .. math::
        NTU_1 = - \ln{\left (\frac{1}{R_{1}} \ln{\left (- \left(P_{1} R_{1}
        - 1\right) e^{R_{1}} \right )} \right )}

    Parameters
    ----------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]
    subtype : float
        The type of heat exchanger; one of 'counterflow', 'parallel',
        'crossflow', 'crossflow approximate', 'crossflow, mixed 1',
        'crossflow, mixed 2', 'crossflow, mixed 1&2'.

    Returns
    -------
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 [-]

    Notes
    -----
    Although this function allows the thermal effectiveness desired to be
    specified, it does not mean such a high value can be obtained. An exception
    is raised when this occurs, although not always a helpful one.

    >>> NTU_from_P_basic(P1=.99, R1=.1, subtype='parallel')
    Traceback (most recent call last):
    ValueError: math domain error

    For the 'crossflow approximate' solution the function is monotonic, and a
    bounded solver is used within the range of NTU1 from 1E-11 to 1E5.

    For the full correct 'crossflow' solution, the initial guess for newton's
    method is obtained by the 'crossflow approximate' solution; the function
    may not converge because of inaccuracy performing the numerical integral
    involved.

    For the 'crossflow, mixed 1&2' solution, a bounded solver is first use, but
    the upper bound on P1 and the upper NTU1 limit is calculated from a pade
    approximation performed with mpmath.

    Examples
    --------
    >>> NTU_from_P_basic(P1=.975, R1=.1, subtype='counterflow')
    3.984769850376482
    '''
    NTU_min = 1E-11
    guess = None
    if subtype == 'counterflow':
        return -log((P1*R1 - 1.)/(P1 - 1.))/(R1 - 1.)
    elif subtype == 'parallel':
        return log(-1./(P1*(R1 + 1.) - 1.))/(R1 + 1.)
    elif subtype == 'crossflow, mixed 1':
        return -log(R1*log(-(P1 - 1.)*exp(1./R1)))/R1
    elif subtype == 'crossflow, mixed 2':
        return -log(log(-(P1*R1 - 1.)*exp(R1))/R1)
    elif subtype == 'crossflow, mixed 1&2':
        NTU_max = _NTU_max_for_P_solver(NTU_from_P_basic_crossflow_mixed_12_p,
                                        NTU_from_P_basic_crossflow_mixed_12_q,
                                        NTU_from_P_basic_crossflow_mixed_12_offset, R1)
    elif subtype == 'crossflow approximate':
        # These are tricky but also easy because P1 can always be 1
        NTU_max = 1E5
    elif subtype == 'crossflow':
        guess = NTU_from_P_basic(P1, R1, subtype='crossflow approximate')
        return secant(_NTU_from_P_objective, guess, args=(R1, P1, temperature_effectiveness_basic, 'crossflow'))
    else:
        raise ValueError('Subtype not recognized.')
    return _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, temperature_effectiveness_basic, guess, subtype)


def NTU_from_P_G(P1, R1, Ntp, optimal=True):
    r'''Returns the number of transfer units of a TEMA G type heat exchanger
    with a specified (for side 1) thermal effectiveness `P1`, heat capacity
    ratio `R1`, the number of tube passes `Ntp`, and for the two-pass case
    whether or not the inlets are arranged optimally. The supported cases are
    as follows:

    * One tube pass (tube fluid split into two streams individually mixed,
      shell fluid mixed)
    * Two tube passes (shell and tube exchanger with shell and tube fluids
      mixed in each pass at the cross section), counterflow arrangement
    * Two tube passes (shell and tube exchanger with shell and tube fluids
      mixed in each pass at the cross section), parallelflow arrangement

    Parameters
    ----------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 (shell side = 1, tube side = 2) [-]
    Ntp : int
        Number of tube passes, 1 or 2 [-]
    optimal : bool, optional
        Whether or not the arrangement is configured to give more of a
        countercurrent and efficient (True) case or an inefficient parallel
        case (only applies for two passes), [-]

    Returns
    -------
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 (shell side = 1, tube side
        = 2) [-]

    Notes
    -----
    For numbers of tube passes greater than 1 or 2, an exception is raised.

    Although this function allows the thermal effectiveness desired to be
    specified, it does not mean such a high value can be obtained. An exception
    is raised which shows the maximum possible effectiveness obtainable at the
    specified `R1` and configuration.

    >>> NTU_from_P_G(P1=1, R1=1/3., Ntp=2)
    Traceback (most recent call last):
    ValueError: No solution possible gives such a high P1; maximum P1=0.954545 at NTU1=10000.000000

    Of the three configurations, 1 pass and the optimal 2 pass have monotonic
    functions which allow for a bounded solver to work smoothly. In both cases
    a solution is searched for between NTU1 values of 1E-11 and 1E-4.

    For the 2 pass unoptimal solution, a bounded solver is first use, but
    the upper bound on P1 and the upper NTU1 limit is calculated from a pade
    approximation performed with mpmath.

    Examples
    --------
    >>> NTU_from_P_G(P1=.573, R1=1/3., Ntp=1)
    0.9999513707759524
    '''
    NTU_min = 1E-11
    function = temperature_effectiveness_TEMA_G
    if Ntp == 1 or (Ntp == 2 and optimal):
        NTU_max = 1E4
        # We could fit a curve to determine the NTU where the floating point
        # does not allow NTU to increase though, but that would be another
        # binary bisection process, different from the current pipeline
    elif Ntp == 2 and not optimal:
        NTU_max = _NTU_max_for_P_solver(NTU_from_G_2_unoptimal_p, NTU_from_G_2_unoptimal_q,
                                        NTU_from_G_2_unoptimal_offset, R1)
    else:
        raise ValueError('Supported numbers of tube passes are 1 or 2.')
    return _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, function, None, Ntp, optimal)


def NTU_from_P_J(P1, R1, Ntp):
    r'''Returns the number of transfer units of a TEMA J type heat exchanger
    with a specified (for side 1) thermal effectiveness `P1`, heat capacity
    ratio `R1`, and the number of tube passes `Ntp`. The supported cases are
    as follows:

    * One tube pass (shell fluid mixed)
    * Two tube passes (shell fluid mixed, tube pass mixed between passes)
    * Four tube passes (shell fluid mixed, tube pass mixed between passes)

    Parameters
    ----------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 (shell side = 1, tube side = 2) [-]
    Ntp : int
        Number of tube passes, 1, 2, or 4, [-]

    Returns
    -------
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 (shell side = 1, tube side
        = 2) [-]

    Notes
    -----
    For numbers of tube passes that are not 1, 2, or 4, an exception is raised.

    For the 1 tube pass case, a bounded solver is used to solve the equation
    numerically, with NTU1 ranging from 1E-11 to 1E3. NTU1 grows extremely
    quickly near its upper limit (NTU1 diverges to infinity at this maximum,
    but because the solver is bounded it will only increase up to 1000 before
    an exception is raised).

    >>> NTU_from_P_J(P1=.995024, R1=.01, Ntp=1)
    13.940758768266656
    >>> NTU_from_P_J(P1=.99502487562189, R1=.01, Ntp=1)  # doctest: +SKIP
    Traceback (most recent call last):
    ValueError: No solution possible gives such a high P1; maximum P1=0.995025 at NTU1=1000.000000

    For the 2 pass and 4 pass solution, a bounded solver is first use, but
    the upper bound on P1 and the upper NTU1 limit is calculated from a pade
    approximation performed with mpmath. These normally do not allow NTU1 to
    rise above 100.

    Examples
    --------
    >>> NTU_from_P_J(P1=.57, R1=1/3., Ntp=1)
    1.0003070138879664
    '''
    NTU_min = 1E-11
    function = temperature_effectiveness_TEMA_J
    if Ntp == 1:
        # Very often failes because at NTU=1000, there is no variation in P1
        # for instance at NTU=40, P1 already peaked and does not decline with
        # higher NTU
        NTU_max = 1E3
        # We could fit a curve to determine the NTU where the floating point
        # does not allow NTU to increase though, but that would be another
        # binary bisection process, different from the current pipeline
    elif Ntp == 2:
        NTU_max = _NTU_max_for_P_solver(NTU_from_P_J_2_p, NTU_from_P_J_2_q, NTU_from_P_J_2_offset, R1)
    elif Ntp == 4:
        NTU_max = _NTU_max_for_P_solver(NTU_from_P_J_4_p, NTU_from_P_J_4_q, NTU_from_P_J_4_offset, R1)
    else:
        raise ValueError('Supported numbers of tube passes are 1, 2, and 4.')
    return _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, function, None, Ntp)


def NTU_from_P_E(P1, R1, Ntp, optimal=True):
    r'''Returns the number of transfer units of a TEMA E type heat exchanger
    with a specified (for side 1) thermal effectiveness `P1`, heat capacity
    ratio `R1`, the number of tube passes `Ntp`, and for the two-pass case
    whether or not the inlets are arranged optimally. The supported cases are
    as follows:

    * 1-1 TEMA E, shell fluid mixed
    * 1-2 TEMA E, shell fluid mixed (this configuration is symmetric)
    * 1-2 TEMA E, shell fluid split into two steams individually mixed
    * 1-3 TEMA E, shell and tube fluids mixed, one parallel pass and two
      counterflow passes (efficient)
    * 1-3 TEMA E, shell and tube fluids mixed, two parallel passes and one
      counteflow pass (inefficient)
    * 1-N TEMA E, shall and tube fluids mixed, efficient counterflow
      orientation, N an even number

    Two of these cases have analytical solutions; the rest use numerical
    solvers of varying quality.

    The analytical solution to 1-1 TEMA E, shell fluid mixed (the same as pure
    counterflow):

    .. math::
        NTU_1 = - \frac{1}{R_{1} - 1} \ln{\left (\frac{P_{1} R_{1} - 1}{P_{1}
        - 1} \right )}

    1-2 TEMA E, shell fluid mixed:

    .. math::
        NTU_1 = \frac{2}{\sqrt{R_{1}^{2} + 1}} \ln{\left (\sqrt{\frac{P_{1}
        R_{1} - P_{1} \sqrt{R_{1}^{2} + 1} + P_{1} - 2}{P_{1} R_{1} + P_{1}
        \sqrt{R_{1}^{2} + 1} + P_{1} - 2}} \right )}

    Parameters
    ----------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 (shell side = 1, tube side = 2) [-]
    Ntp : int
        Number of tube passes, 1, 2, 3, 4, or an even number [-]
    optimal : bool, optional
        Whether or not the arrangement is configured to give more of a
        countercurrent and efficient (True) case or an inefficient parallel
        case, [-]

    Returns
    -------
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 (shell side = 1, tube side
        = 2) [-]

    Notes
    -----
    For odd numbers of tube passes greater than 3, an exception is raised.

    For the 2 pass, unoptimal case, a bounded solver is used with NTU1 between
    1E-11 and 100; the solution to any feasible P1 was found to lie in there.
    For the 4 or a higher even number of pass case, the upper limit on NTU1
    is 1000; this solver works pretty well, but as NTU1 reaches its limit the
    change in P1 is so small a smaller but also correct solution is often
    returned.

    For both the optimal and unoptimal 3 tube pass case, a solution is only
    returned if NTU1 is between 1E-11 and 10. These functions are extremely
    mathematically frustrating, and as NTU1 rises above 10 catastrophic
    cancellation quickly results in this expression finding a ZeroDivisionError.
    The use of arbitrary prevision helps little - quickly 1000 digits are needed,
    and then 1000000 digits, and so one. Using SymPy's rational number support
    works better but is extremely slow for these complicated solutions.
    Nevertheless, so long as a solution is between 1E-11 and 10, the solver is
    quite robust.

    Examples
    --------
    >>> NTU_from_P_E(P1=.58, R1=1/3., Ntp=2)
    1.0381979240816719

    '''
    NTU_min = 1E-11
    function = temperature_effectiveness_TEMA_E
    if Ntp == 1:
        return NTU_from_P_basic(P1, R1, subtype='counterflow')
    elif Ntp == 2 and optimal:
        # Nice analytical solution is available
        # There are actually two roots but one of them is complex
        x1 = R1*R1 + 1.
        return 2.*log(((P1*R1 - P1*x1**0.5 + P1 - 2.)/(P1*R1 + P1*x1**0.5 + P1 - 2.))**0.5)*(x1)**-.5
    elif Ntp == 2 and not optimal:
        NTU_max = 1E2
        # Can't find anywhere it needs to go above 70 to reach the maximum
    elif Ntp == 3 and optimal:
        # no pade could be found, just about the worst-conditioned problem
        # I've ever found
        # Higher starting values result in errors
        NTU_max = 10
    elif Ntp == 3 and not optimal:
        # no pade could be found, just about the worst-conditioned problem
        # I've ever found
        NTU_max = 10
    elif Ntp == 4 or Ntp %2 == 0:
        NTU_max = 1E3
    else:
        raise ValueError('For TEMA E shells with an odd number of tube passes more than 3, no solution is implemented.')
    return _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, function, None, Ntp, optimal)


def NTU_from_P_H(P1, R1, Ntp, optimal=True):
    r'''Returns the number of transfer units of a TEMA H type heat exchanger
    with a specified (for side 1) thermal effectiveness `P1`, heat capacity
    ratio `R1`, the number of tube passes `Ntp`, and for the two-pass case
    whether or not the inlets are arranged optimally. The supported cases are
    as follows:

    * One tube pass (tube fluid split into two streams individually mixed,
      shell fluid mixed)
    * Two tube passes (shell fluid mixed, tube pass mixed between passes)
    * Two tube passes (shell fluid mixed, tube pass mixed between passes, inlet
      tube side next to inlet shell-side)

    Parameters
    ----------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 (shell side = 1, tube side = 2) [-]
    Ntp : int
        Number of tube passes, 1, or 2, [-]
    optimal : bool, optional
        Whether or not the arrangement is configured to give more of a
        countercurrent and efficient (True) case or an inefficient parallel
        case, [-]

    Returns
    -------
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 (shell side = 1, tube side
        = 2) [-]

    Notes
    -----
    For numbers of tube passes greater than 1 or 2, an exception is raised.

    Only numerical solutions are available for this function. For the case of
    1 tube pass or the optimal 2 tube pass, the function is monotonic and a
    bounded solver is used with NTU1 between 1E-11 and 100; it will find the
    solution anywhere in that range.

    For the non-optimal 2 pass case, the function is not monotonic and a pade
    approximation was used to obtain a curve of NTU1s which give the maximum
    P1s which is used as the upper bound in the bounded solver. The lower
    bound is still 1E-11. These solvers are all robust.

    Examples
    --------
    >>> NTU_from_P_H(P1=0.573, R1=1/3., Ntp=1)
    0.9997628696891168
    '''
    NTU_min = 1E-11
    if Ntp == 1:
        NTU_max = 100
    elif Ntp == 2 and optimal:
        NTU_max = 100
    elif Ntp == 2 and not optimal:
        NTU_max = _NTU_max_for_P_solver(NTU_from_H_2_unoptimal_p, NTU_from_H_2_unoptimal_q,
                                        NTU_from_H_2_unoptimal_offset, R1)
    else:
        raise ValueError('Supported numbers of tube passes are 1 and 2.')
    return _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, temperature_effectiveness_TEMA_H,
                              None, Ntp, optimal)


def NTU_from_P_plate(P1, R1, Np1, Np2, counterflow=True,
                     passes_counterflow=True, reverse=False):
    r'''Returns the number of transfer units of a plate heat exchanger
    with a specified side 1 heat capacity ratio `R1`, side 1 number
    of transfer units `NTU1`, number of passes on sides 1 and 2 (respectively
    `Np1` and `Np2`).

    For all cases, the function also takes as arguments whether the exchanger
    is setup in an overall counter or parallel orientation `counterflow`, and
    whether or not individual stream passes are themselves counterflow or
    parallel.

    The 20 supported cases are as follows. (the first number of sides listed
    refers to side 1, and the second number refers to side 2):

    * 1 pass/1 pass parallelflow
    * 1 pass/1 pass counterflow
    * 1 pass/2 pass
    * 1 pass/3 pass or 3 pass/1 pass (with the two end passes in parallel)
    * 1 pass/3 pass or 3 pass/1 pass (with the two end passes in counterflow)
    * 1 pass/4 pass
    * 2 pass/2 pass, overall parallelflow, individual passes in parallel
    * 2 pass/2 pass, overall parallelflow, individual passes counterflow
    * 2 pass/2 pass, overall counterflow, individual passes parallelflow
    * 2 pass/2 pass, overall counterflow, individual passes counterflow
    * 2 pass/3 pass or 3 pass/2 pass, overall parallelflow
    * 2 pass/3 pass or 3 pass/2 pass, overall counterflow
    * 2 pass/4 pass or 4 pass/2 pass, overall parallel flow
    * 2 pass/4 pass or 4 pass/2 pass, overall counterflow flow

    For all except the simplest cases numerical solutions are used.

    1 pass/1 pass counterflow (also 2/2 fully counterflow):

    .. math::
        NTU_1 = - \frac{1}{R_{1} - 1} \ln{\left (\frac{P_{1} R_{1} - 1}{P_{1}
        - 1} \right )}

    1 pass/1 pass parallel flow (also 2/2 fully parallelflow):

    .. math::
        NTU_1 = \frac{1}{R_{1} + 1} \ln{\left (- \frac{1}{P_{1} \left(R_{1}
        + 1\right) - 1} \right )}

    Parameters
    ----------
    P1 : float
        Thermal effectiveness of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]
    R1 : float
        Heat capacity ratio of the heat exchanger in the P-NTU method,
        calculated with respect to stream 1 [-]
    Np1 : int
        Number of passes on side 1 [-]
    Np2 : int
        Number of passes on side 2 [-]
    counterflow : bool
        Whether or not the overall flow through the heat exchanger is in
        counterflow or parallel flow, [-]
    passes_counterflow : bool
        In addition to the overall flow direction, in some cases individual
        passes may be in counter or parallel flow; this controls that [-]
    reverse : bool
        Used **internally only** to allow cases like the 1-4 formula to work
        for the 4-1 flow case, without having to duplicate the code [-]

    Returns
    -------
    NTU1 : float
        Thermal number of transfer units of the heat exchanger in the P-NTU
        method, calculated with respect to stream 1 [-]

    Notes
    -----
    The defaults of counterflow=True and passes_counterflow=True will always
    result in the most efficient heat exchanger option, normally what is
    desired.

    If a number of passes which is not supported is provided, an exception is
    raised.

    For more details, see :obj:`temperature_effectiveness_plate`.

    Examples
    --------
    Three passes on side 1; one pass on side 2; two end passes in counterflow
    orientation.

    >>> NTU_from_P_plate(P1=0.5743, R1=1/3., Np1=3, Np2=1)
    0.9998336056090733
    '''
    NTU_min = 1E-11
    if Np1 == 1 and Np2 == 1 and counterflow:
        try:
            return -log((P1*R1 - 1.)/(P1 - 1.))/(R1 - 1.)
        except:
#            raise ValueError("impossible") # numba: uncomment
            raise ValueError('The maximum P1 obtainable at the specified R1 is %f at the limit of NTU1=inf.' %(1./R1)) # numba: delete

    elif Np1 == 1 and Np2 == 1 and not counterflow:
        try:
            return log(-1./(P1*(R1 + 1.) - 1.))/(R1 + 1.)
        except:
#            raise ValueError("impossible") # numba: uncomment
            raise ValueError('The maximum P1 obtainable at the specified R1 is %f at the limit of NTU1=inf.' %Pp(1E10, R1)) # numba: delete
    elif Np1 == 1 and Np2 == 2:
        NTU_max = 100.
    elif Np1 == 1 and Np2 == 3 and counterflow:
        NTU_max = 100.
    elif Np1 == 1 and Np2 == 3 and not counterflow:
        NTU_max = 100.
    elif Np1 == 1 and Np2 == 4:
        NTU_max = 100.
    elif Np1 == 2 and Np2 == 2:
        if counterflow and passes_counterflow:
            return NTU_from_P_plate(P1, R1, Np1=1, Np2=1, counterflow=True,
                                    passes_counterflow=True)
        elif counterflow and not passes_counterflow:
            NTU_max = 100.0
        elif not counterflow and passes_counterflow:
            NTU_max = _NTU_max_for_P_solver(NTU_from_plate_2_2_parallel_counterflow_p,
                                            NTU_from_plate_2_2_parallel_counterflow_q,
                                            NTU_from_plate_2_2_parallel_counterflow_offset, R1)
        elif not counterflow and not passes_counterflow:
            return NTU_from_P_plate(P1, R1, Np1=1, Np2=1, counterflow=False,
                                    passes_counterflow=False)
    elif Np1 == 2 and Np2 == 3:
        if counterflow:
            NTU_max = 100.0
        elif not counterflow:
            NTU_max = _NTU_max_for_P_solver(NTU_from_plate_2_3_parallel_p,
                                            NTU_from_plate_2_3_parallel_q,
                                            NTU_from_plate_2_3_parallel_offset, R1)
    elif Np1 == 2 and Np2 == 4:
        if counterflow:
            NTU_max = 100.0
        elif not counterflow:
            NTU_max = _NTU_max_for_P_solver(NTU_from_plate_2_4_parallel_p,
                                            NTU_from_plate_2_4_parallel_q,
                                            NTU_from_plate_2_4_parallel_offset, R1)
    elif not reverse:
        # Proved to work by example
        P2 = P1*R1
        R2 = 1./R1
        NTU2 = NTU_from_P_plate(R1=R2, P1=P2, Np1=Np2, Np2=Np1,
                                counterflow=counterflow,
                                passes_counterflow=passes_counterflow,
                                reverse=True)
        NTU1 = NTU2/R1
        return NTU1
    else:
        raise ValueError('Supported number of passes does not have a formula available')
    return _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, temperature_effectiveness_plate, None, Np1,
                              Np2, counterflow, passes_counterflow)


def P_NTU_method(m1, m2, Cp1, Cp2, UA=None, T1i=None, T1o=None,
                 T2i=None, T2o=None, subtype='crossflow', Ntp=1, optimal=True):
    r'''Wrapper for the various P-NTU method function calls,
    which can solve a heat exchanger. The heat capacities and mass flows
    of each stream and the type of the heat exchanger are always required.
    As additional inputs, one combination of the following inputs is required:

    * Three of the four inlet and outlet stream temperatures.
    * Temperatures for the side 1 outlet and side 2 inlet and UA
    * Temperatures for the side 1 outlet side 2 outlet and UA
    * Temperatures for the side 1 inlet and side 2 inlet and UA
    * Temperatures for the side 1 inlet and side 2 outlet and UA

    Computes the total heat exchanged as well as both temperatures of both
    streams.

    Parameters
    ----------
    m1 : float
        Mass flow rate of stream 1 (shell side = 1, tube side = 2), [kg/s]
    m2 : float
        Mass flow rate of stream 2 (shell side = 1, tube side = 2), [kg/s]
    Cp1 : float
        Averaged heat capacity of stream 1 (shell side), [J/kg/K]
    Cp2 : float
        Averaged heat capacity of stream 2 (tube side), [J/kg/K]
    UA : float, optional
        Combined Area-heat transfer coefficient term, [W/K]
    T1i : float, optional
        Inlet temperature of stream 1 (shell side), [K]
    T1o : float, optional
        Outlet temperature of stream 1 (shell side), [K]
    T2i : float, optional
        Inlet temperature of stream 2 (tube side), [K]
    T2o : float, optional
        Outlet temperature of stream 2 (tube-side), [K]
    subtype : str, optional
        The subtype of exchanger; one of 'E', 'G', 'H', 'J', 'counterflow',
        'parallel', 'crossflow', 'crossflow, mixed 1', 'crossflow, mixed 2',
        or 'crossflow, mixed 1&2'. For plate exchangers 'Np1/Np2' where `Np1`
        is the number of side 1 passes and `Np2` is the number of side 2 passes
    Ntp : int, optional
        For real heat exchangers (types 'E', 'G', 'H', and 'J'), the number of
        tube passes needss to be specified as well. Not all types support
        any number of tube passes.
    optimal : bool, optional
        For real heat exchangers (types 'E', 'G', 'H', and 'J'), there is often
        a more countercurrent (optimal) way to arrange the tube passes and a
        more parallel (optimal=False) way to arrange them. This controls that.

    Returns
    -------
    results : dict
        * Q : Heat exchanged in the heat exchanger, [W]
        * UA : Combined area-heat transfer coefficient term, [W/K]
        * T1i : Inlet temperature of stream 1, [K]
        * T1o : Outlet temperature of stream 1, [K]
        * T2i : Inlet temperature of stream 2, [K]
        * T2o : Outlet temperature of stream 2, [K]
        * P1 : Thermal effectiveness with respect to stream 1, [-]
        * P2 : Thermal effectiveness with respect to stream 2, [-]
        * R1 : Heat capacity ratio with respect to stream 1, [-]
        * R2 : Heat capacity ratio with respect to stream 2, [-]
        * C1 : The heat capacity rate of fluid 1, [W/K]
        * C2 : The heat capacity rate of fluid 2, [W/K]
        * NTU1 : Thermal Number of Transfer Units with respect to stream 1 [-]
        * NTU2 : Thermal Number of Transfer Units with respect to stream 2 [-]

    Notes
    -----
    The main equations used in this method are as follows. For the individual
    expressions used to calculate `P1`, see the `See Also` section.

    .. math::
        Q = P_1 C_1 \Delta T_{max} = P_2 C_2 \Delta T_{max}

    .. math::
        \Delta T_{max} = T_{h,i} - T_{c,i} = |T_{2,i} - T_{1,i}|

    .. math::
        R_1 = \frac{C_1}{C_2} = \frac{T_{2,i} - T_{2,o}}{T_{1,o} - T_{1, i}}

    .. math::
        R_2 = \frac{C_2}{C_1} = \frac{T_{1,o} - T_{1, i}}{T_{2,i} - T_{2,o}}

    .. math::
        R_1 = \frac{1}{R_2}

    .. math::
        NTU_1 = \frac{UA}{C_1}

    .. math::
        NTU_2 = \frac{UA}{C_2}

    .. math::
        NTU_1 = NTU_2 R_2

    .. math::
        NTU_2 = NTU_1 R_1

    .. math::
        P_1 = \frac{T_{1,o} - T_{1,i}}{T_{2,i} - T_{1,i}}

    .. math::
        P_2 = \frac{T_{2,i} - T_{2,o}}{T_{2,i} - T_{1,i}}

    .. math::
        P_1 = P_2 R_2

    .. math::
        P_2 = P_1 R_1

    .. math::
        C_1 = m_1 Cp_1

    .. math::
        C_2 = m_2 Cp_2

    Once `P1` has been calculated, there are six different cases for calculating
    the other stream temperatures depending on the two temperatures provided.
    They were derived with SymPy.

    Two known inlet temperatures:

    .. math::
        T_{1,o} = - P_{1} T_{1,i} + P_{1} T_{2,i} + T_{1,i}

    .. math::
        T_{2,o} = P_{1} R_{1} T_{1,i} - P_{1} R_{1} T_{2,i} + T_{2,i}

    Two known outlet temperatures:

    .. math::
        T_{1,i} = \frac{P_{1} R_{1} T_{1,o} + P_{1} T_{2,o}
        - T_{1,o}}{P_{1} R_{1} + P_{1} - 1}

    .. math::
        T_{2,i} = \frac{P_{1} R_{1} T_{1,o} + P_{1} T_{2,o}
        - T_{2,o}}{P_{1} R_{1} + P_{1} - 1}

    Inlet 1 known, outlet 2 known:

    .. math::
        T_{1,o} = \frac{1}{P_{1} R_{1} - 1} \left(P_{1} R_{1} T_{1,i}
        + P_{1} T_{1,i} - P_{1} T_{2,o} - T_{1,i}\right)

    .. math::
        T_{2,i} = \frac{P_{1} R_{1} T_{1,i} - T_{2,o}}{P_{1} R_{1} - 1}

    Outlet 1 known, inlet 2 known:

    .. math::
        T_{1,i} = \frac{P_{1} T_{2,i} - T_{1,o}}{P_{1} - 1}

    .. math::
        T_{2,o}  = \frac{1}{P_{1} - 1} \left(R_{1} \left(P_{1} T_{2,i}
        - T_{1,o}\right) - \left(P_{1} - 1\right) \left(R_{1} T_{1,o}
        - T_{2,i}\right)\right)

    Input and output of 2 known:

    .. math::
        T_{1,i} = \frac{1}{P_{1} R_{1}} \left(P_{1} R_{1} T_{2,i}
        - T_{2,i} + T_{2,o}\right)

    .. math::
        T_{1,o} = \frac{1}{P_{1} R_{1}} \left(P_{1} R_{1} T_{2,i}
        + \left(P_{1} - 1\right) \left(T_{2,i} - T_{2,o}\right)\right)

    Input and output of 1 known:

    .. math::
        T_{2,i} = \frac{1}{P_{1}} \left(P_{1} T_{1,i} - T_{1,i}
        + T_{1,o}\right)

    .. math::
        T_{2,o} = \frac{1}{P_{1}} \left(P_{1} R_{1} \left(T_{1,i}
        - T_{1,o}\right) + P_{1} T_{1,i} - T_{1,i} + T_{1,o}\right)

    See Also
    --------
    temperature_effectiveness_basic
    temperature_effectiveness_plate
    temperature_effectiveness_TEMA_E
    temperature_effectiveness_TEMA_G
    temperature_effectiveness_TEMA_H
    temperature_effectiveness_TEMA_J
    NTU_from_P_basic
    NTU_from_P_plate
    NTU_from_P_E
    NTU_from_P_G
    NTU_from_P_H
    NTU_from_P_J

    Examples
    --------
    Solve a heat exchanger with the UA specified, and known inlet temperatures:

    >>> from pprint import pprint
    >>> pprint(P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900,
    ... subtype='E', Ntp=4, T2i=15, T1i=130, UA=3041.75))
    {'C1': 9672.0,
     'C2': 2755.0,
     'NTU1': 0.314490281224,
     'NTU2': 1.104083484573,
     'P1': 0.173081161436,
     'P2': 0.60763738417,
     'Q': 192514.714242,
     'R1': 3.5107078039,
     'R2': 0.28484284532,
     'T1i': 130,
     'T1o': 110.095666434,
     'T2i': 15,
     'T2o': 84.878299180,
     'UA': 3041.75}

    Solve the same heat exchanger as if T1i, T2i, and T2o were known but UA was
    not:

    >>> pprint(P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900, subtype='E',
    ... Ntp=4, T1i=130, T2i=15, T2o=84.87829918042112))
    {'C1': 9672.0,
     'C2': 2755.0,
     'NTU1': 0.31449028122,
     'NTU2': 1.10408348457,
     'P1': 0.173081161436,
     'P2': 0.60763738417,
     'Q': 192514.714242,
     'R1': 3.5107078039,
     'R2': 0.2848428453,
     'T1i': 130,
     'T1o': 110.095666434,
     'T2i': 15,
     'T2o': 84.878299180,
     'UA': 3041.7499999}

    Solve a 2 pass/2 pass plate heat exchanger with overall parallel flow and
    its individual passes operating in parallel and known outlet temperatures.
    Note the overall parallel part is trigered with `optimal=False`, and the
    individual pass parallel is triggered by appending 'p' to the subtype. The
    subpass counterflow can be specified by appending 'c' instead to the
    subtype, but this is never necessary as it is the default.

    >>> pprint(P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300,
    ... T1o=126.7, T2o=26.7, subtype='2/2p', optimal=False))
    {'C1': 9672.0,
     'C2': 2755.0,
     'NTU1': 0.0310173697270,
     'NTU2': 0.10889292196,
     'P1': 0.0289452959747,
     'P2': 0.101618476467,
     'Q': 32200.0503078,
     'R1': 3.5107078039,
     'R2': 0.28484284532,
     'T1i': 130.029202885,
     'T1o': 126.7,
     'T2i': 15.0121414490,
     'T2o': 26.7,
     'UA': 300}

    References
    ----------
    .. [1] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [2] Thulukkanam, Kuppan. Heat Exchanger Design Handbook, Second Edition.
       CRC Press, 2013.
    .. [3] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    # Shellside: 1
    # Tubeside: 2
    C1 = m1*Cp1
    C2 = m2*Cp2
    R1 = C1/C2
    R2 = C2/C1

    if UA is not None:
        NTU1 = UA/C1
        NTU2 = UA/C2

        if subtype in ('counterflow', 'parallel', 'crossflow', 'crossflow, mixed 1', 'crossflow, mixed 2', 'crossflow, mixed 1&2'):
            P1 = temperature_effectiveness_basic(R1, NTU1, subtype=subtype)
        elif subtype == 'E':
            P1 = temperature_effectiveness_TEMA_E(R1=R1, NTU1=NTU1, Ntp=Ntp, optimal=optimal)
        elif subtype == 'G':
            P1 = temperature_effectiveness_TEMA_G(R1=R1, NTU1=NTU1, Ntp=Ntp, optimal=optimal)
        elif subtype == 'H':
            P1 = temperature_effectiveness_TEMA_H(R1=R1, NTU1=NTU1, Ntp=Ntp, optimal=optimal)
        elif subtype == 'J':
            P1 = temperature_effectiveness_TEMA_J(R1=R1, NTU1=NTU1, Ntp=Ntp)
        elif '/' in subtype:
            passes_counterflow = True
            Np1, end = subtype.split('/')
            if end[-1] in ['c','p']:
                passes_counterflow = end[-1] == 'c'
                end = end[0:-1]
            Np1, Np2 = int(Np1), int(end)
            P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=Np1, Np2=Np2, counterflow=optimal, passes_counterflow=passes_counterflow)
        else:
            raise ValueError("Supported types are 'E', 'G', 'H', 'J', 'counterflow',\
    'parallel', 'crossflow', 'crossflow, mixed 1', 'crossflow, mixed 2', \
    'crossflow, mixed 1&2', or 'Np1/Np2' for plate exchangers")

        possible_inputs = [(T1i, T2i), (T1o, T2o), (T1i, T2o), (T1o, T2i), (T1i, T1o), (T2i, T2o)]
        if not any(i for i in possible_inputs if None not in i):
            raise ValueError('One set of (T1i, T2i), (T1o, T2o), (T1i, T2o), (T1o, T2i), (T1i, T1o), or (T2i, T2o) is required along with UA.')

        # Deal with different temperature inputs, generated with SymPy
        if T1i and T2i:
            T2o = P1*R1*T1i - P1*R1*T2i + T2i
            T1o = -P1*T1i + P1*T2i + T1i
        elif T1o and T2o:
            T2i = (P1*R1*T1o + P1*T2o - T2o)/(P1*R1 + P1 - 1.)
            T1i = (P1*R1*T1o + P1*T2o - T1o)/(P1*R1 + P1 - 1.)
        elif T1o and T2i:
            T2o = (R1*(P1*T2i - T1o) - (P1 - 1.)*(R1*T1o - T2i))/(P1 - 1.)
            T1i = (P1*T2i - T1o)/(P1 - 1.)
        elif T1i and T2o:
            T1o = (P1*R1*T1i + P1*T1i - P1*T2o - T1i)/(P1*R1 - 1.)
            T2i = (P1*R1*T1i - T2o)/(P1*R1 - 1.)
        elif T2i and T2o:
            T1o = (P1*R1*T2i + (P1 - 1.)*(T2i - T2o))/(P1*R1)
            T1i = (P1*R1*T2i - T2i + T2o)/(P1*R1)
        elif T1i and T1o:
            T2o = (P1*R1*(T1i - T1o) + P1*T1i - T1i + T1o)/P1
            T2i = (P1*T1i - T1i + T1o)/P1
    else:
        # Case where we're solving for UA
        # Three temperatures are required
        # Ensures all four temperatures are set and Q is calculated
        if T1i is not None and T1o is not None:
            Q = m1*Cp1*(T1i-T1o)
            if T2i is not None and T2o is None:
                T2o = T2i + Q/(m2*Cp2)
            elif T2o is not None and T2i is None:
                T2i = T2o - Q/(m2*Cp2)
            elif T2o is not None and T2i is not None:
                Q2 = m2*Cp2*(T2o-T2i)
                if abs((Q-Q2)/Q) > 0.01:
                    raise ValueError('The specified heat capacities, mass flows,'
                                    ' and temperatures are inconsistent')
            else:
                raise ValueError('At least one temperature is required to be '
                                'specified on side 2.')

        elif T2i is not None and T2o is not None:
            Q = m2*Cp2*(T2o-T2i)
            if T1i is not None and T1o is None:
                T1o = T1i - Q/(m1*Cp1)
            elif T1o is not None and T1i is None:
                T1i = T1o + Q/(m1*Cp1)
            else:
                raise ValueError('At least one temperature is required to be '
                                'specified on side 2.')
        else:
            raise ValueError('Three temperatures are required to be specified '
                            'when solving for UA')

        P1 = Q/(C1*abs(T2i-T1i))
        if subtype in ('counterflow', 'parallel', 'crossflow', 'crossflow, mixed 1', 'crossflow, mixed 2', 'crossflow, mixed 1&2'):
            NTU1 = NTU_from_P_basic(P1=P1, R1=R1, subtype=subtype)
        elif subtype == 'E':
            NTU1 = NTU_from_P_E(P1=P1, R1=R1, Ntp=Ntp, optimal=optimal)
        elif subtype == 'G':
            NTU1 = NTU_from_P_G(P1=P1, R1=R1, Ntp=Ntp, optimal=optimal)
        elif subtype == 'H':
            NTU1 = NTU_from_P_H(P1=P1, R1=R1, Ntp=Ntp, optimal=optimal)
        elif subtype == 'J':
            NTU1 = NTU_from_P_J(P1=P1, R1=R1, Ntp=Ntp)
        elif '/' in subtype:
            passes_counterflow = True
            Np1, end = subtype.split('/')
            if end[-1] in ['c','p']:
                passes_counterflow = end[-1] == 'c'
                end = end[0:-1]
            Np1, Np2 = int(Np1), int(end)
            NTU1 = NTU_from_P_plate(P1=P1, R1=R1, Np1=Np1, Np2=Np2, counterflow=optimal, passes_counterflow=passes_counterflow)
        else:
            raise ValueError("Supported types are 'E', 'G', 'H', 'J', 'counterflow',\
    'parallel', 'crossflow', 'crossflow, mixed 1', 'crossflow, mixed 2', \
    'crossflow, mixed 1&2', or 'Np1/Np2' for plate exchangers")
        UA = NTU1*C1
        NTU2 = UA/C2

    Q = abs(T1i-T2i)*P1*C1
    # extra:
    P2 = P1*R1
#    effectiveness = max(C1, C2)/min(C1, C2)
    results = {'Q': Q, 'T1i': T1i, 'T1o': T1o, 'T2i': T2i, 'T2o': T2o,
          'C1': C1, 'C2': C2, 'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'NTU1': NTU1, 'NTU2': NTU2, 'UA': UA}
    return results


def F_LMTD_Fakheri(Thi, Tho, Tci, Tco, shells=1):
    r'''Calculates the log-mean temperature difference correction factor `Ft`
    for a shell-and-tube heat exchanger with one or an even number of tube
    passes, and a given number of shell passes, with the expression given in
    [1]_ and also shown in [2]_.

    .. math::
        F_t=\frac{S\ln W}{\ln \frac{1+W-S+SW}{1+W+S-SW}}

    .. math::
        S = \frac{\sqrt{R^2+1}}{R-1}

    .. math::
        W = \left(\frac{1-PR}{1-P}\right)^{1/N}

    .. math::
        R = \frac{T_{in}-T_{out}}{t_{out}-t_{in}}

    .. math::
        P = \frac{t_{out}-t_{in}}{T_{in}-t_{in}}

    If R = 1 and logarithms cannot be evaluated:

    .. math::
        W' = \frac{N-NP}{N-NP+P}

    .. math::
        F_t = \frac{\sqrt{2}\frac{1-W'}{W'}}{\ln\frac{\frac{W'}{1-W'}+\frac{1}
        {\sqrt{2}}}{\frac{W'}{1-W'}-\frac{1}{\sqrt{2}}}}

    Parameters
    ----------
    Thi : float
        Inlet temperature of hot fluid, [K]
    Tho : float
        Outlet temperature of hot fluid, [K]
    Tci : float
        Inlet temperature of cold fluid, [K]
    Tco : float
        Outlet temperature of cold fluid, [K]
    shells : int, optional
        Number of shell-side passes, [-]

    Returns
    -------
    Ft : float
        Log-mean temperature difference correction factor, [-]

    Notes
    -----
    This expression is symmetric - the same result is calculated if the cold
    side values are swapped with the hot side values. It also does not
    depend on the units of the temperature given.

    Examples
    --------
    >>> F_LMTD_Fakheri(Tci=15, Tco=85, Thi=130, Tho=110, shells=1)
    0.9438358829645933

    References
    ----------
    .. [1] Fakheri, Ahmad. "A General Expression for the Determination of the
       Log Mean Temperature Correction Factor for Shell and Tube Heat
       Exchangers." Journal of Heat Transfer 125, no. 3 (May 20, 2003): 527-30.
       doi:10.1115/1.1571078.
    .. [2] Hall, Stephen. Rules of Thumb for Chemical Engineers, Fifth Edition.
       Oxford; Waltham, MA: Butterworth-Heinemann, 2012.
    '''
    R = (Thi - Tho)/(Tco - Tci)
    P = (Tco - Tci)/(Thi - Tci)
    if R == 1.0:
        W2 = (shells - shells*P)/(shells - shells*P + P)
        return (2**0.5*(1. - W2)/W2)/log((W2/(1. - W2) + 2**-0.5)/(W2/(1. - W2) - 2**-0.5))
    else:
        W = ((1. - P*R)/(1. - P))**(1./shells)
        S = (R*R + 1.)**0.5/(R - 1.)
        return S*log(W)/log((1. + W - S + S*W)/(1. + W + S - S*W))

### Tubes

# TEMA tubes from http://www.engineeringpage.com/technology/thermal/tubesize.html
# NPSs in inches, which convert to outer diameter exactly.
_NPSs = [0.25, 0.25, 0.375, 0.375, 0.375, 0.5, 0.5, 0.625, 0.625, 0.625, 0.75, 0.75, 0.75, 0.75, 0.75, 0.875, 0.875, 0.875, 0.875, 1, 1, 1, 1, 1.25, 1.25, 1.25, 1.25, 2, 2]
_Dos = [0.00635, 0.00635, 0.009525, 0.009525, 0.009525, 0.0127, 0.0127, 0.015875, 0.015875, 0.015875, 0.01905, 0.01905, 0.01905, 0.01905, 0.01905, 0.022225, 0.022225, 0.022225, 0.022225, 0.0254, 0.0254, 0.0254, 0.0254, 0.03175, 0.03175, 0.03175, 0.03175, 0.0508, 0.0508]
_BWGs = [22, 24, 18, 20, 22, 18, 20, 16, 18, 20, 12, 14, 16, 18, 20, 14, 16, 18, 20, 12, 14, 16, 18, 10, 12, 14, 16, 12, 14]
_ts = [0.000711, 0.000559, 0.001245, 0.000889, 0.000711, 0.001245, 0.000889, 0.001651, 0.001245, 0.000889, 0.002769, 0.002108, 0.001651, 0.001245, 0.000889, 0.002108, 0.001651, 0.001245, 0.000889, 0.002769, 0.002108, 0.001651, 0.001245, 0.003404, 0.002769, 0.002108, 0.001651, 0.002769, 0.002108]
_Dis = [0.004928, 0.005232, 0.007035, 0.007747, 0.008103, 0.01021, 0.010922, 0.012573, 0.013385, 0.014097, 0.013512, 0.014834, 0.015748, 0.01656, 0.017272, 0.018009, 0.018923, 0.019735, 0.020447, 0.019862, 0.021184, 0.022098, 0.02291, 0.024942, 0.026212, 0.027534, 0.028448, 0.045262, 0.046584]

# Structure: Look up NPS, get BWGs. BWGs listed in increasing order --> decreasing thickness
TEMA_tubing = {0.25: (22, 24), 0.375: (18, 20, 22), 0.5: (18, 20),
0.625: (16, 18, 20), 0.75: (12, 14, 16, 18, 20),
0.875: (14, 16, 18, 20), 1.: (12, 14, 16, 18),
1.25: (10, 12, 14, 16), 2.: (12, 14)}

#
#for tup in TEMA_Full_Tubing:
#    Do, BWG = tup[0]/1000., tup[1]
#    t = BWG_SI[BWG_integers.index(BWG)]
#    Di = Do-2*t
#    print t*1000, Di*1000
#
def check_tubing_TEMA(NPS=None, BWG=None):
    '''
    >>> check_tubing_TEMA(2, 22)
    False
    >>> check_tubing_TEMA(0.375, 22)
    True
    '''
    if NPS in TEMA_tubing:
        if BWG in TEMA_tubing[NPS]:
            return True
    return False


def get_tube_TEMA(NPS=None, BWG=None, Do=None, Di=None, tmin=None):
    # Tube defined by a thickness and an outer diameter only, no pipe.
    # If Di or Do are specified, they must be exactly correct.
    if NPS and BWG:
        # Fully defined, guaranteed
        if not check_tubing_TEMA(NPS, BWG):
            raise ValueError('NPS and BWG Specified are not listed in TEMA')
        Do = 0.0254*NPS
        t = BWG_SI[BWG_integers.index(BWG)]
        Di = Do-2*t
    elif Do and BWG:
        NPS = Do/.0254
        if not check_tubing_TEMA(NPS, BWG):
            raise ValueError('NPS and BWG Specified are not listed in TEMA')
        t = BWG_SI[BWG_integers.index(BWG)]
        Di = Do-2*t
    elif BWG and Di:
        t = BWG_SI[BWG_integers.index(BWG)] # Will fail if BWG not int
        Do = t*2 + Di
        NPS = Do/.0254
        if not check_tubing_TEMA(NPS, BWG):
            raise ValueError('NPS and BWG Specified are not listed in TEMA')
    elif NPS and Di:
        Do = 0.0254*NPS
        t = (Do - Di)/2
        BWG = [BWG_integers[BWG_SI.index(t)]]
        if not check_tubing_TEMA(NPS, BWG):
            raise ValueError('NPS and BWG Specified are not listed in TEMA')
    elif Di and Do:
        NPS = Do/.0254
        t = (Do - Di)/2
        BWG = [BWG_integers[BWG_SI.index(t)]]
        if not check_tubing_TEMA(NPS, BWG):
            raise ValueError('NPS and BWG Specified are not listed in TEMA')
    # Begin Fuzzy matching
    elif NPS and tmin:
        Do = 0.0254*NPS
        ts = [BWG_SI[BWG_integers.index(BWG)] for BWG in TEMA_tubing[NPS]]
        ts.reverse() # Small to large
        if tmin > ts[-1]:
            raise ValueError('Specified minimum thickness is larger than available in TEMA')
        for t in ts: # Runs if at least 1 of the thicknesses are the right size.
            if tmin <= t:
                break
        BWG = [BWG_integers[BWG_SI.index(t)]]
        Di = Do-2*t
    elif Do and tmin:
        NPS = Do/.0254
        NPS, BWG, Do, Di, t = get_tube_TEMA(NPS=NPS, tmin=tmin)
    elif Di and tmin:
        raise ValueError('Not funny defined input for TEMA Schedule; multiple solutions')
    elif NPS:
        BWG = TEMA_tubing[NPS][0] # Pick the first listed size
        Do = 0.0254*NPS
        t = BWG_SI[BWG_integers.index(BWG)]
        Di = Do-2*t
    else:
        raise ValueError('Insufficient information provided')
    return NPS, BWG, Do, Di, t

TEMA_Ls_imperial = [96., 120., 144., 192., 240.] # inches
TEMA_Ls = [2.438, 3.048, 3.658, 4.877, 6.096]
HTRI_Ls_imperial = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60] # ft
HTRI_Ls = [1.829, 2.438, 3.048, 3.658, 4.267, 4.877, 5.486, 6.096, 6.706, 7.315, 8.534, 9.754, 10.973, 12.192, 13.411, 14.63, 15.85, 17.069, 18.288]


# Shells up to 120 inch in diameter.
# This is for plate shells, not pipe (up to 12 inches, pipe is used)
HEDH_shells_imperial = [12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 24., 26., 28., 30., 32., 34., 36., 38., 40., 42., 44., 46., 48., 50., 52., 54., 56., 58., 60., 63., 66., 69., 72., 75., 78., 81., 84., 87., 90., 93., 96., 99., 102., 105., 108., 111., 114., 117., 120.]
HEDH_shells = [0.3048, 0.3302, 0.3556, 0.381, 0.4064, 0.4318, 0.4572, 0.4826, 0.508, 0.5334, 0.5588, 0.6096, 0.6604, 0.7112, 0.762, 0.8128, 0.8636, 0.9144, 0.9652, 1.016, 1.0668, 1.1176, 1.1684, 1.2192, 1.27, 1.3208, 1.3716, 1.4224, 1.4732, 1.524, 1.6002, 1.6764, 1.7526, 1.8288, 1.905, 1.9812, 2.0574, 2.1336, 2.2098, 2.286, 2.3622, 2.4384, 2.5146, 2.5908, 2.667, 2.7432, 2.8194, 2.8956, 2.9718, 3.048]


HEDH_pitches = {0.25: (1.25, 1.5), 0.375: (1.330, 1.420),
0.5: (1.250, 1.310, 1.380), 0.625: (1.250, 1.300, 1.400),
0.75: (1.250, 1.330, 1.420, 1.500), 1.: (1.250, 1.312, 1.375),
1.25: (1.250,), 1.5: (1.250,), 2.: (1.250,)}

def DBundle_min(Do):
    r'''Very roughly, determines a good choice of shell diameter for a given
    tube outer diameter, according to figure 1, section 3.3.5 in [1]_.

    Parameters
    ----------
    Do : float
        Tube outer diameter, [m]

    Returns
    -------
    DShell : float
        Shell inner diameter, optional, [m]

    Notes
    -----
    This function should be used if a tube diameter is specified but not a
    shell size. DShell will have to be adjusted later, once the area
    requirement is known.
    This function is essentially a lookup table.

    Examples
    --------
    >>> DBundle_min(0.0254)
    1.0

    References
    ----------
    .. [1] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1983.
    '''
    data = ((0.006, 0.1), (0.01, 0.1), (.014, 0.3), (0.02, 0.5), (0.03, 1.0))
    for Do_tabulated, DBundle in data:
        if Do <= Do_tabulated:
            return DBundle
    return 1.5


def shell_clearance(DBundle=None, DShell=None):
    r'''Looks up the recommended clearance between a shell and tube bundle in
    a TEMA HX [1]. Either the bundle diameter or the shell diameter are needed
    provided.

    Parameters
    ----------
    DBundle : float, optional
        Outer diameter of tube bundle, [m]
    DShell : float, optional
        Shell inner diameter, [m]

    Returns
    -------
    c : float
        Shell-tube bundle clearance, [m]

    Notes
    -----
    Lower limits are extended up to the next limit where intermediate limits
    are not provided.

    Examples
    --------
    >>> shell_clearance(DBundle=1.245)
    0.0064

    References
    ----------
    .. [1] Standards of the Tubular Exchanger Manufacturers Association,
       Ninth edition, 2007, TEMA, New York.
    '''
    DShell_data = [(0.457, 0.0032), (1.016, 0.0048), (1.397, 0.0064),
                   (1.778, 0.0079), (2.159, 0.0095)]
    DBundle_data = [(0.457 - 0.0048, 0.0032), (1.016 - 0.0064, 0.0048),
                    (1.397 - 0.0079, 0.0064), (1.778 - 0.0095, 0.0079),
                    (2.159 - 0.011, 0.0095)]
    if DShell:
        for DShell_tabulated, c in DShell_data:
            if DShell < DShell_tabulated:
                return c
        return 0.011
    elif DBundle:
        for DBundle_tabulated, c in DBundle_data:
            if DBundle < DBundle_tabulated:
                return c
        return 0.011
    else:
        raise ValueError('Either DShell or DBundle must be specified')


_TEMA_baffles_refinery = [[0.0032, 0.0048, 0.0064, 0.0095, 0.0095],
[0.0048, 0.0064, 0.0095, 0.0095, 0.0127],
[0.0064, 0.0075, 0.0095, 0.0127, 0.0159],
[0.0064, 0.0095, 0.0127, 0.0159, 0.0159],
[0.0095, 0.0127, 0.0159, 0.0191, 0.0191]]

_TEMA_baffles_other = [[0.0016, 0.0032, 0.0048, 0.0064, 0.0095, 0.0095],
[0.0032, 0.0048, 0.0064, 0.0095, 0.0095, 0.0127],
[0.0048, 0.0064, 0.0075, 0.0095, 0.0127, 0.0159],
[0.0064, 0.0064, 0.0095, 0.0127, 0.0159, 0.0159],
[0.0064, 0.0095, 0.0127, 0.0127, 0.0191, 0.0191]]

def baffle_thickness(Dshell, L_unsupported, service='C'):
    r'''Determines the thickness of baffles and support plates in TEMA HX
    [1]_. Applies to latitudinal baffles along the diameter of the HX, but
    not longitudinal baffles parallel to the tubes.

    Parameters
    ----------
    Dshell : float
        Shell inner diameter, [m]
    L_unsupported : float
        Distance between tube supports, [m]
    service : str
        Service type, C, R or B, [-]

    Returns
    -------
    t : float
        Baffle or support plate thickness, [m]

    Notes
    -----
    No checks are implemented to ensure the given shell size is TEMA compatible.
    The baffles do not need to be strongas the pressure is almost the same on
    both of their sides. `L_unsupported` is a design choice; the more baffles
    in a given length, the higher the pressure drop.

    Examples
    --------
    >>> baffle_thickness(Dshell=.3, L_unsupported=50, service='R')
    0.0095

    References
    ----------
    .. [1] Standards of the Tubular Exchanger Manufacturers Association,
       Ninth edition, 2007, TEMA, New York.
    '''
    if Dshell < 0.381:
        j = 0
    elif 0.381 <= Dshell < 0.737:
        j = 1
    elif 0.737 <= Dshell < 0.991:
        j = 2
    elif 0.991 <= Dshell  < 1.524:
        j = 3
    else:
        j = 4

    if service == 'R':
        if L_unsupported <= 0.61:
            i = 0
        elif 0.61 < L_unsupported <= 0.914:
            i = 1
        elif 0.914 < L_unsupported <= 1.219:
            i = 2
        elif 1.219 < L_unsupported <= 1.524:
            i = 3
        else:
            i = 4
        t = _TEMA_baffles_refinery[j][i]

    elif service == 'C' or service == 'B':
        if L_unsupported <= 0.305:
            i = 0
        elif 0.305 < L_unsupported <= 0.610:
            i = 1
        elif 0.61 < L_unsupported <= 0.914:
            i = 2
        elif 0.914 < L_unsupported <= 1.219:
            i = 3
        elif 1.219 < L_unsupported <= 1.524:
            i = 4
        else:
            i = 5
        t = _TEMA_baffles_other[j][i]
    return t



def D_baffle_holes(do, L_unsupported):
    r'''Determines the diameter of holes in baffles for tubes according to
    TEMA [1]_. Applies for all geometries.

    Parameters
    ----------
    do : float
        Tube outer diameter, [m]
    L_unsupported : float
        Distance between tube supports, [m]

    Returns
    -------
    dB : float
        Baffle hole diameter, [m]

    Notes
    -----

    Examples
    --------
    >>> D_baffle_holes(do=.0508, L_unsupported=0.75)
    0.0516
    >>> D_baffle_holes(do=0.01905, L_unsupported=0.3)
    0.01985
    >>> D_baffle_holes(do=0.01905, L_unsupported=1.5)
    0.019450000000000002

    References
    ----------
    .. [1] Standards of the Tubular Exchanger Manufacturers Association,
       Ninth edition, 2007, TEMA, New York.
    '''
    if do > 0.0318 or L_unsupported <= 0.914: # 1-1/4 inches and 36 inches
        extra = 0.0008
    else:
        extra = 0.0004
    d = do + extra
    return d



_L_unsupported_Do =   [0.25,  0.375, 0.5,  0.628,  0.75,  0.875, 1.,   1.25,  1.5,  2.,    2.5,   3.]
_L_unsupported_steel = [0.66, 0.889, 1.118, 1.321, 1.524, 1.753, 1.88, 2.235, 2.54, 3.175, 3.175, 3.175]
_L_unsupported_aluminium = [0.559, 0.762, 0.965, 1.143, 1.321, 1.524, 1.626,
                            1.93, 2.21, 2.794, 2.794, 2.794]


def L_unsupported_max(Do, material='CS'):
    r'''Determines the maximum length of a heat exchanger tube can go without
    a support, according to TEMA [1]_. The limits provided apply for the
    worst-case temperature allowed for the material to be used at.

    Parameters
    ----------
    Do : float
        Outer tube diameter, [m]
    material : str
        Material type, either 'CS' or 'aluminium', [-]

    Returns
    -------
    L_unsupported : float
        Maximum length of unsupported tube, [m]

    Notes
    -----
    The 'CS' results is also listed as representing high alloy steel, low
    alloy steel, nickel-copper, nickel, and nickel-chromium-iron alloys.
    The 'aluminium' results are those of copper and copper alloys and
    also titanium alloys.

    The maximum and minimum tube outer diameter tabulated are 3 inch and 1/4
    inch respectively. The result is returned for the nearest tube diameter
    equal or smaller than the provided diameter, which helps ensures the
    returned tube length will not be optimistic. However, if the diameter is
    under 0.25 inches, the result will be optimistic!


    Examples
    --------
    >>> L_unsupported_max(Do=.0254, material='CS')
    1.88

    References
    ----------
    .. [1] Standards of the Tubular Exchanger Manufacturers Association,
       Ninth edition, 2007, TEMA, New York, p 5.4-5.
    '''
    Do = Do/inch # convert diameter to inches
    for i in range(12):
        if _L_unsupported_Do[i] == Do:
            break # perfect
        elif _L_unsupported_Do[i] > Do:
            i -= 1 # too big, go down a length
            break
    i = i if i < 11 else 11
    i = 0 if i == -1 else i
    if material == 'CS':
        return _L_unsupported_steel[i]
    elif material == 'aluminium':
        return _L_unsupported_aluminium[i]
    else:
        raise ValueError('Material argument should be one of "CS" or "aluminium"')


### Tube bundle count functions

square_C1s = square_Ns = triangular_C1s = triangular_Ns = None

def _load_coeffs_Phadkeb():
    global square_C1s, square_Ns, triangular_C1s, triangular_Ns
    hx_data_folder = os.path.join(os.path.dirname(__file__), 'data')
    triangular_Ns = np.load(os.path.join(hx_data_folder, "triangular_Ns_Phadkeb.npy"))
    triangular_C1s = np.load(os.path.join(hx_data_folder, "triangular_C1s_Phadkeb.npy"))
    square_Ns = np.load(os.path.join(hx_data_folder, "square_Ns_Phadkeb.npy"))
    square_C1s = np.load(os.path.join(hx_data_folder, "square_C1s_Phadkeb.npy"))

def Ntubes_Phadkeb(DBundle, Do, pitch, Ntp, angle=30):
    r'''Using tabulated values and correction factors for number of passes,
    the highly accurate method of [1]_ is used to obtain the tube count
    of a given tube bundle outer diameter for a given tube size and pitch.

    Parameters
    ----------
    DBundle : float
        Outer diameter of tube bundle, [m]
    Do : float
        Tube outer diameter, [m]
    pitch : float
        Pitch; distance between two orthogonal tube centers, [m]
    Ntp : int
        Number of tube passes, [-]
    angle : float, optional
        The angle the tubes are positioned; 30, 45, 60 or 90, [degrees]

    Returns
    -------
    Nt : int
        Total number of tubes that fit in the heat exchanger, [-]

    Notes
    -----
    For single-pass cases, the result is exact, and no tubes need to be removed
    for any reason. For 4, 6, 8 pass arrangements, a number of tubes must be
    removed to accommodate pass partition plates. The following assumptions
    are involved with that:

    * The pass partition plate is where a row of tubes would have been.
      Only one or two rows are assumed affected.
    * The thickness of partition plate is < 70% of the tube outer diameter.
    * The distance between the centerline of the partition plate and the
      centerline of the nearest row of tubes is equal to the pitch.

    This function will fail when there are more than 100,000 tubes.
    [1]_ tabulated values up to approximately 3,000 tubes derived with
    number theory. The sequesnces of integers were identified in the
    On-Line Encyclopedia of Integer Sequences (OEIS), and formulas listed in
    it were used to generate more coefficient to allow up to 100,000 tubes.
    The integer sequences are A003136, A038590, A001481, and A057961. The
    generation of coefficients for A038590 is very slow, but the rest are
    reasonably fast.

    The number of tubes that fit generally does not increase one-by-one, but by
    several.

    >>> Ntubes_Phadkeb(DBundle=1.007, Do=.028, pitch=.036, Ntp=2, angle=45.)
    558
    >>> Ntubes_Phadkeb(DBundle=1.008, Do=.028, pitch=.036, Ntp=2, angle=45.)
    574

    Because a pass partition needs to be installed in multiple tube pass
    shells, more tubes fit in an exchanger the fewer passes are used.

    >>> Ntubes_Phadkeb(DBundle=1.008, Do=.028, pitch=.036, Ntp=1, angle=45.)
    593

    Examples
    --------
    >>> Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.036, Ntp=2, angle=45.)
    782

    References
    ----------
    .. [1] Phadke, P. S., Determining tube counts for shell and tube
       exchangers, Chem. Eng., September, 91, 65-68 (1984).
    '''
    if square_C1s is None: # numba: delete
         _load_coeffs_Phadkeb() # numba: delete
    if DBundle <= Do*Ntp:
        return 0

    if Ntp == 6:
        e = 0.265
    elif Ntp == 8:
        e = 0.404
    else:
        e = 0.

    r = 0.5*(DBundle - Do)/pitch
    s = r*r
    Ns, Nr = floor(s), floor(r)
    # If Ns is between two numbers, take the smaller one
    # C1 is the number of tubes for a single pass arrangement.
    if angle == 30 or angle == 60:
        i = np.searchsorted(triangular_Ns, Ns, side='right')
        C1 = int(triangular_C1s[i-1])
    elif angle == 45 or angle == 90:
        i = np.searchsorted(square_Ns, Ns, side='right')
        C1 = int(square_C1s[i-1])

    Cx = 2*Nr + 1.

    # triangular and rotated triangular
    if (angle == 30 or angle == 60):
        w = 2*r/3**0.5
        Nw = floor(w)
        if Nw % 2 == 0:
            Cy = 3*Nw
        else:
            Cy = 3*Nw + 1
        if Ntp == 2:
            if angle == 30 :
                C2 = C1 - Cx
            else:
                C2 = C1 - Cy - 1
        else: # 4 passes, or 8; this value is needed
            C4 = C1 - Cx - Cy

    if (angle == 30 or angle == 60) and (Ntp == 6 or Ntp == 8):
        if angle == 30: # triangular
            v = 2*e*r/3**0.5 + 0.5
            Nv = floor(v)
            u = 3**0.5*Nv/2.
            if Nv % 2 == 0:
                z = (s-u*u)**0.5
            else:
                z = (s-u*u)**0.5 - 0.5
            Nz = floor(z)
            if Ntp == 6:
                C6 = C1 - Cy - 4*Nz - 1
            else:
                C8 = C4 - 4*Nz
        else: # rotated triangular
            v = 2.*e*r
            Nv = floor(v)
            u1 = 0.5*Nv
            z = (s - u1*u1)**0.5
            w1 = 2*z/2**0.5
#            w1 = 2**2**0.5 # WRONG
            u2 = 0.5*(Nv + 1)
            zs = (s-u2*u2)**0.5
            w2 = 2.*zs/3**0.5
            if Nv%2 == 0:
                z1 = 0.5*w1
                z2 = 0.5*(w2+1)
            else:
                z1 = 0.5*(w1+1)
                z2 = 0.5*w2
            Nz1 = floor(z1)
            Nz2 = floor(z2)
            if Ntp == 6:
                C6 = C1 - Cx - 4.*(Nz1 + Nz2)
            else: # 8
                C8 = C4 - 4.*(Nz1 + Nz2)

    if (angle == 45 or angle == 90):
        if angle == 90:
            Cy = Cx - 1.
            # eq 6 or 8 for c2 or c4
            if Ntp == 2:
                C2 = C1 - Cx
            else: # 4 passes, or 8; this value is needed
                C4 = C1 - Cx - Cy
        else: # rotated square
            w = r/2**0.5
            Nw = floor(w)
            Cx = 2.*Nw + 1
            Cy = Cx - 1
            if Ntp == 2:
                C2 = C1 - Cx
            else: # 4 passes, or 8; this value is needed
                C4 = C1 - Cx - Cy

    if (angle == 45 or angle == 90) and (Ntp == 6 or Ntp == 8):
        if angle == 90:
            v = e*r + 0.5
            Nv = floor(v)
            z = (s - Nv*Nv)**0.5
            Nz = floor(z)
            if Ntp == 6:
                C6 = C1 - Cy - 4*Nz - 1
            else:
                C8 = C4 - 4*Nz
        else:
            w = r/2**0.5
            Nw = floor(w)
            Cx = 2*Nw + 1

            v = 2**0.5*e*r
            Nv = floor(v)
            u1 = Nv/2**0.5
            z = (s-u1*u1)**0.5
            w1 = 2**0.5*z
            u2 = (Nv + 1)/2**0.5
            zs = (s-u2*u2)**0.5
            w2 = 2**0.5*zs
            # if Nv is odd, 21a and 22a. If even, 21b and 22b. Nz1, Nz2
            if Nv %2 == 0:
                z1 = 0.5*w1
                z2 = 0.5*(w2 + 1)
            else:
                z1 = 0.5*(w1 + 1)
                z2 = 0.5*w2
            Nz1 = floor(z1)
            Nz2 = floor(z2)
            if Ntp == 6:
                C6 = C1 - Cx - 4*(Nz1 + Nz2)
            else: # 8
                C8 = C4 - 4*(Nz1 + Nz2)

    if Ntp == 1:
        ans = C1
    elif Ntp == 2:
        ans = C2
    elif Ntp == 4:
        ans = C4
    elif Ntp == 6:
        ans = C6
    elif Ntp == 8:
        ans = C8
    else:
        raise ValueError('Only 1, 2, 4, 6, or 8 tube passes are supported')
    ans = int(ans)
    # In some cases, a negative number would be returned by these formulas
    if ans < 0:
        ans = 0 # pragma: no cover
    return ans

def to_solve_Ntubes_Phadkeb(DBundle, Do, pitch, Ntp, angle, Ntubes):
    ans = Ntubes_Phadkeb(DBundle=DBundle, Do=Do, pitch=pitch, Ntp=Ntp, angle=angle) - Ntubes
    return ans

def DBundle_for_Ntubes_Phadkeb(Ntubes, Do, pitch, Ntp, angle=30):
    r'''Determine the bundle diameter required to fit a specified number of
    tubes in a heat exchanger. Uses the highly accurate method of [1]_,
    which takes into account pitch, number of tube passes, angle,
    and tube diameter. The method is analytically correct when used in the
    other direction (calculating number of tubes from bundle diameter); in
    reverse, it is solved by bisection.

    Parameters
    ----------
    Ntubes : int
        Total number of tubes that fit in the heat exchanger, [-]
    Do : float
        Tube outer diameter, [m]
    pitch : float
        Pitch; distance between two orthogonal tube centers, [m]
    Ntp : int
        Number of tube passes, [-]
    angle : float, optional
        The angle the tubes are positioned; 30, 45, 60 or 90, [degrees]

    Returns
    -------
    DBundle : float
        Outer diameter of tube bundle, [m]

    Notes
    -----
    This function will fail when there are more than 100,000 tubes. There are
    a range of correct diameters for which there can be the given number of
    tubes; a number within that range is returned as found by bisection.

    Examples
    --------
    >>> DBundle_for_Ntubes_Phadkeb(Ntubes=782, Do=.028, pitch=.036, Ntp=2, angle=45.)
    1.1879392959379533

    References
    ----------
    .. [1] Phadke, P. S., Determining tube counts for shell and tube
       exchangers, Chem. Eng., September, 91, 65-68 (1984).
    '''
    if square_C1s is None: # numba: delete
        _load_coeffs_Phadkeb() # numba: delete
    if angle == 30 or angle == 60:
        Ns = triangular_Ns[-1]
    elif angle == 45 or angle == 90:
        Ns = square_Ns[-1]
    s = Ns + 1
    r = s**0.5
    DBundle_max = (Do + 2.*pitch*r)*(1. - 1E-8) # Cannot be exact or floor(s) will give an int too high
    return bisect(to_solve_Ntubes_Phadkeb, 0, DBundle_max, args=(Do, pitch, Ntp, angle, Ntubes))


def Ntubes_Perrys(DBundle, Do, Ntp, angle=30):
    r'''A rough equation presented in Perry's Handbook [1]_ for estimating
    the number of tubes in a tube bundle of differing geometries and tube
    sizes. Claimed accuracy of 24 tubes.

    Parameters
    ----------
    DBundle : float
        Outer diameter of tube bundle, [m]
    Do : int
        Tube outer diameter, [m]
    Ntp : float
        Number of tube passes, [-]
    angle : float
        The angle the tubes are positioned; 30, 45, 60 or 90, [degrees]

    Returns
    -------
    Nt : int
        Number of tubes, [-]

    Notes
    -----
    Perry's equation 11-74.
    Pitch equal to 1.25 times the tube outside diameter
    No other source for this equation is given.
    Experience suggests this is accurate to 40 tubes, but is often around 20
    tubes off.

    Examples
    --------
    >>> Ntubes_Perrys(DBundle=1.184, Do=.028, Ntp=2, angle=45)
    803

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. New York: McGraw-Hill Education, 2007.
    '''
    if angle == 30 or angle == 60:
        C = 0.75*DBundle/Do - 36.
        if Ntp == 1:
            Nt = 1298. + 74.86*C + 1.283*C**2 - .0078*C**3 - .0006*C**4
        elif Ntp == 2:
            Nt = 1266. + 73.58*C + 1.234*C**2 - .0071*C**3 - .0005*C**4
        elif Ntp == 4:
            Nt = 1196. + 70.79*C + 1.180*C**2 - .0059*C**3 - .0004*C**4
        elif Ntp == 6:
            Nt = 1166. + 70.72*C + 1.269*C**2 - .0074*C**3 - .0006*C**4
        else:
            raise ValueError('N passes not 1, 2, 4 or 6')
    elif angle == 45 or angle == 90:
        C = DBundle/Do - 36.
        if Ntp == 1:
            Nt = 593.6 + 33.52*C + .3782*C**2 - .0012*C**3 + .0001*C**4
        elif Ntp == 2:
            Nt = 578.8 + 33.36*C + .3847*C**2 - .0013*C**3 + .0001*C**4
        elif Ntp == 4:
            Nt = 562.0 + 33.04*C + .3661*C**2 - .0016*C**3 + .0002*C**4
        elif Ntp == 6:
            Nt = 550.4 + 32.49*C + .3873*C**2 - .0013*C**3 + .0001*C**4
        else:
            raise ValueError('N passes not 1, 2, 4 or 6')
    return int(Nt)


def Ntubes_VDI(DBundle=None, Ntp=None, Do=None, pitch=None, angle=30.):
    r'''A rough equation presented in the VDI Heat Atlas for estimating
    the number of tubes in a tube bundle of differing geometries and tube
    sizes. No accuracy estimation given.

    Parameters
    ----------
    DBundle : float
        Outer diameter of tube bundle, [m]
    Ntp : float
        Number of tube passes, [-]
    Do : float
        Tube outer diameter, [m]
    pitch : float
        Pitch; distance between two orthogonal tube centers, [m]
    angle : float
        The angle the tubes are positioned; 30, 45, 60 or 90, [degrees]

    Returns
    -------
    N : float
        Number of tubes, [-]

    Notes
    -----
    No coefficients for this method with Ntp=6 are available in [1]_. For
    consistency, estimated values were added to support 6 tube passes, f2 = 90..
    This equation is a rearranged form of that presented in [1]_.
    The calculated tube count is rounded down to an integer.

    Examples
    --------
    >>> Ntubes_VDI(DBundle=1.184, Ntp=2, Do=.028, pitch=.036, angle=30)
    966

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    if Ntp == 1:
        f2 = 0.
    elif Ntp == 2:
        f2 = 22.
    elif Ntp == 4:
        f2 = 70.
    elif Ntp == 8:
        f2 = 105.
    elif Ntp == 6:
        f2 = 90. # Estimated!
    else:
        raise ValueError('Only 1, 2, 4 and 8 passes are supported')
    if angle == 30 or angle == 60:
        f1 = 1.1
    elif angle == 45 or angle == 90:
        f1 = 1.3
    else:
        raise ValueError('Only 30, 60, 45 and 90 degree layouts are supported')

    DBundle, Do, pitch = DBundle*1000, Do*1000, pitch*1000 # convert to mm, equation is dimensional.
    t = pitch
    Ntubes = (-(-4*f1*t**4*f2**2*Do + 4*f1*t**4*f2**2*DBundle**2 + t**4*f2**4)**0.5
    - 2*f1*t**2*Do + 2*f1*t**2*DBundle**2 + t**2*f2**2) / (2*f1**2*t**4)
    return int(Ntubes)


def D_for_Ntubes_VDI(N, Ntp, Do, pitch, angle=30):
    r'''A rough equation presented in the VDI Heat Atlas for estimating
    the size of a tube bundle from a given number of tubes, number of tube
    passes, outer tube diameter, pitch, and arrangement.
    No accuracy estimation given.

    .. math::
        OTL = \sqrt{f_1 z t^2 + f_2 t \sqrt{z} - d_o}

    Parameters
    ----------
    N : float
        Number of tubes, [-]
    Ntp : float
        Number of tube passes, [-]
    Do : float
        Tube outer diameter, [m]
    pitch : float
        Pitch; distance between two orthogonal tube centers, [m]
    angle : float
        The angle the tubes are positioned; 30, 45, 60 or 90

    Returns
    -------
    DBundle : float
        Outer diameter of tube bundle, [m]

    Notes
    -----
    f1 = 1.1 for triangular, 1.3 for square patterns
    f2 is as follows: 1 pass, 0; 2 passes, 22; 4 passes, 70; 8 passes, 105.
    6 tube passes is not officially supported, only 1, 2, 4 and 8.
    However, an estimated constant has been added to support it.
    f2 = 90.

    Examples
    --------
    >>> D_for_Ntubes_VDI(N=970, Ntp=2., Do=0.00735, pitch=0.015, angle=30.)
    0.5003600119829544

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    if Ntp == 1:
        f2 = 0.
    elif Ntp == 2:
        f2 = 22.
    elif Ntp == 4:
        f2 = 70.
    elif Ntp == 6:
        f2 = 90.
    elif Ntp == 8:
        f2 = 105.
    else:
        raise ValueError('Only 1, 2, 4 and 8 passes are supported')
    if angle == 30 or angle == 60:
        f1 = 1.1
    elif angle == 45 or angle == 90:
        f1 = 1.3
    else:
        raise ValueError('Only 30, 60, 45 and 90 degree layouts are supported')
    Do, pitch = Do*1000, pitch*1000 # convert to mm, equation is dimensional.
    Dshell = (f1*N*pitch**2 + f2*N**0.5*pitch +Do)**0.5
    return Dshell/1000.


def Ntubes_HEDH(DBundle=None, Do=None, pitch=None, angle=30):
    r'''A rough equation presented in the HEDH for estimating
    the number of tubes in a tube bundle of differing geometries and tube
    sizes. No accuracy estimation given. Only 1 pass is supported.

    .. math::
        N = \frac{0.78(D_{bundle} - D_o)^2}{C_1(\text{pitch})^2}

    C1 = 0.866 for 30° and 60° layouts, and 1 for 45 and 90° layouts.

    Parameters
    ----------
    DBundle : float
        Outer diameter of tube bundle, [m]
    Do : float
        Tube outer diameter, [m]
    pitch : float
        Pitch; distance between two orthogonal tube centers, [m]
    angle : float
        The angle the tubes are positioned; 30, 45, 60 or 90, [degrees]

    Returns
    -------
    N : float
        Number of tubes, [-]

    Notes
    -----
    Seems reasonably accurate.

    Examples
    --------
    >>> Ntubes_HEDH(DBundle=1.184, Do=.028, pitch=.036, angle=30)
    928

    References
    ----------
    .. [1] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1983.
    '''
    if angle == 30 or angle == 60:
        C1 = 13/15.
    elif angle == 45 or angle == 90:
        C1 = 1.
    else:
        raise ValueError('Only 30, 60, 45 and 90 degree layouts are supported')
    Dctl = DBundle - Do
    N = 0.78*Dctl*Dctl/(C1*pitch*pitch)
    return int(N)


def DBundle_for_Ntubes_HEDH(N, Do, pitch, angle=30):
    r'''A rough equation presented in the HEDH for estimating the tube bundle
    diameter necessary to fit a given number of tubes.
    No accuracy estimation given. Only 1 pass is supported.

    .. math::
        D_{bundle} = (D_o + (\text{pitch})\sqrt{\frac{1}{0.78}}\cdot
        \sqrt{C_1\cdot N})


    C1 = 0.866 for 30° and 60° layouts, and 1 for 45 and 90° layouts.

    Parameters
    ----------
    N : float
        Number of tubes, [-]
    Do : float
        Tube outer diameter, [m]
    pitch : float
        Pitch; distance between two orthogonal tube centers, [m]
    angle : float
        The angle the tubes are positioned; 30, 45, 60 or 90, [degrees]

    Returns
    -------
    DBundle : float
        Outer diameter of tube bundle, [m]

    Notes
    -----
    Easily reversed from the main formulation.

    Examples
    --------
    >>> DBundle_for_Ntubes_HEDH(N=928, Do=.028, pitch=.036, angle=30)
    1.183993079564

    References
    ----------
    .. [1] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1983.
    '''
    if angle == 30 or angle == 60:
        C1 = 13/15.
    elif angle == 45 or angle == 90:
        C1 = 1.
    else:
        raise ValueError('Only 30, 60, 45 and 90 degree layouts are supported')
    return (Do + (1./.78)**0.5*pitch*(C1*N)**0.5)


def Ntubes(DBundle, Do, pitch, Ntp=1, angle=30, Method=None):
    r'''Calculates the number of tubes which can fit in a heat exchanger.
    The tube count is effected by the pitch, number of tube passes, and angle.

    The result is an exact number of tubes and is calculated by a very accurate
    method using number theory by default. This method is available only up to
    100,000 tubes.

    Parameters
    ----------
    DBundle : float
        Outer diameter of tube bundle, [m]
    Do : float
        Tube outer diameter, [m]
    pitch : float
        Pitch; distance between two orthogonal tube centers, [m]
    Ntp : int, optional
        Number of tube passes, [-]
    angle : float, optional
        The angle the tubes are positioned; 30, 45, 60 or 90, [degrees]

    Returns
    -------
    N : int
        Total number of tubes that fit in the heat exchanger, [-]

    Other Parameters
    ----------------
    Method : string, optional
        One of 'Phadkeb', 'HEDH', 'VDI' or 'Perry'

    See Also
    --------
    Ntubes_Phadkeb
    Ntubes_Perrys
    Ntubes_VDI
    Ntubes_HEDH
    size_bundle_from_tubecount

    Examples
    --------
    >>> Ntubes(DBundle=1.2, Do=0.025, pitch=0.03125)
    1285

    The approximations are pretty good too:

    >>> Ntubes(DBundle=1.2, Do=0.025, pitch=0.03125, Method='Perry')
    1297
    >>> Ntubes(DBundle=1.2, Do=0.025, pitch=0.03125, Method='VDI')
    1340
    >>> Ntubes(DBundle=1.2, Do=0.025, pitch=0.03125, Method='HEDH')
    1272
    '''
    if Method is None:
        Method = 'Phadkeb'

    if Method == 'Phadkeb':
        return Ntubes_Phadkeb(DBundle=DBundle, Ntp=Ntp, Do=Do, pitch=pitch, angle=angle)
    elif Method == 'HEDH':
        return Ntubes_HEDH(DBundle=DBundle, Do=Do, pitch=pitch, angle=angle)
    elif Method == 'VDI':
        return Ntubes_VDI(DBundle=DBundle, Ntp=Ntp, Do=Do, pitch=pitch, angle=angle)
    elif Method == 'Perry':
        return Ntubes_Perrys(DBundle=DBundle, Do=Do, Ntp=Ntp, angle=angle)
    else:
        raise ValueError('Method not recognized; allowable methods are '
                        '"Phadkeb", "HEDH", "VDI", and "Perry"')

def _tubecount_objf_Perry(D, Do, Ntp, angle, N):
    return Ntubes_Perrys(DBundle=D, Do=Do, Ntp=Ntp, angle=angle) - N

def size_bundle_from_tubecount(N, Do, pitch, Ntp=1, angle=30, Method=None):
    r'''Calculates the outer diameter of a tube bundle containing a specified
    number of tubes.
    The tube count is effected by the pitch, number of tube passes, and angle.

    The result is an exact number of tubes and is calculated by a very accurate
    method using number theory by default. This method is available only up to
    100,000 tubes.

    Parameters
    ----------
    N : int
        Total number of tubes that fit in the heat exchanger, [-]
    Do : float
        Tube outer diameter, [m]
    pitch : float
        Pitch; distance between two orthogonal tube centers, [m]
    Ntp : int, optional
        Number of tube passes, [-]
    angle : float, optional
        The angle the tubes are positioned; 30, 45, 60 or 90, [degrees]

    Returns
    -------
    DBundle : float
        Outer diameter of tube bundle, [m]

    Other Parameters
    ----------------
    Method : string, optional
        One of 'Phadkeb', 'HEDH', 'VDI' or 'Perry'

    See Also
    --------
    Ntubes
    DBundle_for_Ntubes_Phadkeb
    D_for_Ntubes_VDI
    DBundle_for_Ntubes_HEDH

    Notes
    -----
    The 'Perry' method is solved with a numerical solver and is very unreliable.

    Examples
    --------
    >>> size_bundle_from_tubecount(N=1285, Do=0.025, pitch=0.03125)
    1.1985676402390355
    '''
    if Method is None:
        Method2 = 'Phadkeb'
    else:
        Method2 = Method
    if Method2 == 'Phadkeb':
        return DBundle_for_Ntubes_Phadkeb(Ntubes=N, Ntp=Ntp, Do=Do, pitch=pitch, angle=angle)
    elif Method2 == 'VDI':
        return D_for_Ntubes_VDI(N=N, Ntp=Ntp, Do=Do, pitch=pitch, angle=angle)
    elif Method2 == 'HEDH':
        return DBundle_for_Ntubes_HEDH(N=N, Do=Do, pitch=pitch, angle=angle)
    elif Method2 == 'Perry':
        return brenth(_tubecount_objf_Perry, Do*5, 1000*Do, args=(Do, Ntp, angle, N))
    else:
        raise ValueError('Method not recognized; allowable methods are '
                        '"Phadkeb", "HEDH", "VDI", and "Perry"')




TEMA_heads = {'A': 'Removable Channel and Cover',
              'B': 'Bonnet (Integral Cover)',
              'C': 'Integral With Tubesheet Removable Cover',
              'N': 'Channel Integral With Tubesheet and Removable Cover',
              'D': 'Special High-Pressure Closures'}
TEMA_shells = {'E': 'One-Pass Shell',
               'F': 'Two-Pass Shell with Longitudinal Baffle',
               'G': 'Split Flow', 'H': 'Double Split Flow',
               'J': 'Divided Flow',
               'K': 'Kettle-Type Reboiler',
               'X': 'Cross Flow'}
TEMA_rears = {'L': 'Fixed Tube Sheet; Like "A" Stationary Head',
              'M': 'Fixed Tube Sheet; Like "B" Stationary Head',
              'N': 'Fixed Tube Sheet; Like "C" Stationary Head',
              'P': 'Outside Packed Floating Head',
              'S': 'Floating Head with Backing Device',
              'T': 'Pull-Through Floating Head',
              'U': 'U-Tube Bundle',
              'W': 'Externally Sealed Floating Tubesheet'}
TEMA_services = {'B': 'Chemical',
                 'R': 'Refinery',
                 'C': 'General'}
baffle_types = ['segmental', 'double segmental', 'triple segmental',
                'disk and doughnut', 'no tubes in window', 'orifice', 'rod']
