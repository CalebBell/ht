# -*- coding: utf-8 -*-
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
SOFTWARE.'''

from __future__ import division
from math import exp, log, floor, sqrt, factorial, tanh  # tanh= 1/coth
import math
from pprint import pprint
from bisect import bisect, bisect_left, bisect_right
from fluids.constants import inch, foot, degree_Fahrenheit, hour, Btu
from fluids.numerics import horner, newton, ridder
from fluids.numerics import bisect as sp_bisect
from fluids.numerics import iv
from fluids.piping import BWG_integers, BWG_inch, BWG_SI
import numpy as np

__all__ = ['effectiveness_from_NTU', 'NTU_from_effectiveness', 'calc_Cmin',
'calc_Cmax', 'calc_Cr',
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


def effectiveness_from_NTU(NTU, Cr, subtype='counterflow'):
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
        'crossflow, mixed Cmax', 'boiler', 'condenser', 'S&T', or 'nS&T' where 
        n is the number of shell and tube exchangers in a row.

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
    0.8444821799748551

    Counterflow, better than either crossflow or parallel flow:

    >>> effectiveness_from_NTU(NTU=5, Cr=0.7, subtype='counterflow')
    0.9206703686051108
    
    One shell and tube heat exchanger gives worse performance than counterflow,
    but they are designed to be economical and compact which a counterflow
    exchanger would not be. As the number of shells approaches infinity,
    the counterflow result is obtained exactly.
    
    >>> effectiveness_from_NTU(NTU=5, Cr=0.7, subtype='S&T')
    0.6834977044311439
    >>> effectiveness_from_NTU(NTU=5, Cr=0.7, subtype='50S&T')
    0.9205058702789254

    
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
    (1377.5, 9672.0, 0.14242142266335814)
    >>> NTU, eff, Q
    (2.160072595281307, 0.8312180361425988, 131675.32715043944)
    >>> Tco, Tho
    (110.59007415639887, 116.38592564614977)
    
    Alternatively, if only the outlet temperatures had been known:
        
    >>> Tco = 110.59007415639887
    >>> Tho = 116.38592564614977
    >>> Cc, Ch = Cmin, Cmax # In this case but not always
    >>> Q = eff*Cmin*Cc*Ch*(Tco - Tho)/(eff*Cmin*(Cc+Ch) - Ch*Cc)
    >>> Thi = Tho + Q/Ch
    >>> Tci = Tco - Q/Cc
    >>> Q, Tci, Thi
    (131675.32715043964, 14.999999999999858, 130.00000000000003)

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
        raise Exception('Heat capacity rate must be less than 1 by definition.')
        
    if subtype == 'counterflow':
        if Cr < 1:
            return (1. - exp(-NTU*(1. - Cr)))/(1. - Cr*exp(-NTU*(1. - Cr)))
        elif Cr == 1:
            return NTU/(1. + NTU)
    elif subtype == 'parallel':
            return (1. - exp(-NTU*(1. + Cr)))/(1. + Cr)
    elif 'S&T' in subtype:
        str_shells = subtype.split('S&T')[0]
        shells = int(str_shells) if str_shells else 1
        NTU = NTU/shells
        
        top = 1. + exp(-NTU*(1. + Cr**2)**.5)
        bottom = 1. - exp(-NTU*(1. + Cr**2)**.5)
        effectiveness = 2./(1. + Cr + (1. + Cr**2)**.5*top/bottom)
        if shells > 1:
            term = ((1. - effectiveness*Cr)/(1. - effectiveness))**shells
            effectiveness = (term - 1.)/(term - Cr)
        return effectiveness
    elif subtype == 'crossflow':
        def to_int(v, NTU, Cr):
            return (1. + NTU - v*v/(4.*Cr*NTU))*exp(-v*v/(4.*Cr*NTU))*v*float(iv(0, v))
        from scipy.integrate import quad
        int_term = quad(to_int, 0, 2.*NTU*Cr**0.5, args=(NTU, Cr))[0]
        return 1./Cr - exp(-Cr*NTU)/(2.*(Cr*NTU)**2)*int_term
    elif subtype == 'crossflow approximate':
        return 1. - exp(1./Cr*NTU**0.22*(exp(-Cr*NTU**0.78) - 1.))
    elif subtype == 'crossflow, mixed Cmin':
        return 1. -exp(-Cr**-1*(1. - exp(-Cr*NTU)))
    elif subtype ==  'crossflow, mixed Cmax':
        return (1./Cr)*(1. - exp(-Cr*(1. - exp(-NTU))))
    elif subtype in ['boiler', 'condenser']:
        return  1. - exp(-NTU)
    else:
        raise Exception('Input heat exchanger type not recognized')
        

def NTU_from_effectiveness(effectiveness, Cr, subtype='counterflow'):
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
        'crossflow, mixed Cmax', 'boiler', 'condenser', 'S&T', or 'nS&T' where 
        n is the number of shell and tube exchangers in a row.

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
    
    >>> NTU_from_effectiveness(.99, Cr=.7, subtype='5S&T')
    Traceback (most recent call last):
    Exception: The specified effectiveness is not physically possible for this configuration; the maximum effectiveness possible is 0.974122977755.

    Examples
    --------
    Worst case, parallel flow:
    
    >>> NTU_from_effectiveness(effectiveness=0.5881156068417585, Cr=0.7, subtype='parallel')
    5.000000000000012
    
    Crossflow, somewhat higher effectiveness:
        
    >>> NTU_from_effectiveness(effectiveness=0.8444821799748551, Cr=0.7, subtype='crossflow')
    5.000000000000859

    Counterflow, better than either crossflow or parallel flow:

    >>> NTU_from_effectiveness(effectiveness=0.9206703686051108, Cr=0.7, subtype='counterflow')
    5.0
    
    Shell and tube exchangers:
    
    >>> NTU_from_effectiveness(effectiveness=0.6834977044311439, Cr=0.7, subtype='S&T')
    5.000000000000071
    >>> NTU_from_effectiveness(effectiveness=0.9205058702789254, Cr=0.7, subtype='50S&T')
    4.999999999999996


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
    (0.6086956521739131, 1.1040839095588, 3041.751170834494, 15.208755854172471)

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
        raise Exception('Heat capacity rate must be less than 1 by definition.')

    if subtype == 'counterflow':
        # [2]_ gives the expression 1./(1-Cr)*log((1-Cr*eff)/(1-eff)), but
        # this is just the same equation rearranged differently.
        if Cr < 1:
            return 1./(Cr - 1.)*log((effectiveness - 1.)/(effectiveness*Cr - 1.))
        elif Cr == 1:
            return effectiveness/(1. - effectiveness)
    elif subtype == 'parallel':
        if effectiveness*(1. + Cr) > 1:
            raise Exception('The specified effectiveness is not physically \
possible for this configuration; the maximum effectiveness possible is %s.' % (1./(Cr + 1.)))
        return -log(1. - effectiveness*(1. + Cr))/(1. + Cr)
    elif 'S&T' in subtype:
        # [2]_ gives the expression
        # D = (1+Cr**2)**0.5
        # 1/D*log((2-eff*(1+Cr-D))/(2-eff*(1+Cr + D)))
        # This is confirmed numerically to be the same equation rearranged
        # differently
        str_shells = subtype.split('S&T')[0]
        shells = int(str_shells) if str_shells else 1
        
        F = ((effectiveness*Cr - 1.)/(effectiveness - 1.))**(1./shells)
        e1 = (F - 1.)/(F - Cr)
        E = (2./e1 - (1. + Cr))/(1. + Cr**2)**0.5
        
        if (E - 1.)/(E + 1.) <= 0:
            # Derived with SymPy
            max_effectiveness = (-((-Cr + sqrt(Cr**2 + 1) + 1)/(Cr + sqrt(Cr**2 + 1) - 1))**shells + 1)/(Cr - ((-Cr + sqrt(Cr**2 + 1) + 1)/(Cr + sqrt(Cr**2 + 1) - 1))**shells)
            raise Exception('The specified effectiveness is not physically \
possible for this configuration; the maximum effectiveness possible is %s.' % (max_effectiveness))
        
        NTU = -(1. + Cr*Cr)**-0.5*log((E - 1.)/(E + 1.))
        return shells*NTU
    elif subtype == 'crossflow':
        # Can't use a bisect solver here because at high NTU there's a derivative of 0
        # due to the integral term not changing when it's very near one
        guess = NTU_from_effectiveness(effectiveness, Cr, 'crossflow approximate')
        def to_solve(NTU, Cr, effectiveness):
            return effectiveness_from_NTU(NTU, Cr, subtype='crossflow') - effectiveness
        return newton(to_solve, guess, args=(Cr, effectiveness))
    elif subtype == 'crossflow approximate':
        # This will fail if NTU is more than 10,000 or less than 1E-7, but
        # this is extremely unlikely to occur in normal usage.
        # Maple and SymPy and Wolfram Alpha all failed to obtain an exact
        # analytical expression even with coefficients for 0.22 and 0.78 or 
        # with an explicit value for Cr. The function has been plotted,
        # and appears to be monotonic - there is only one solution.
        def to_solve(NTU, Cr, effectiveness):
            return (1. - exp(1./Cr*NTU**0.22*(exp(-Cr*NTU**0.78) - 1.))) - effectiveness
        return ridder(to_solve, a=1E-7, b=1E5, args=(Cr, effectiveness))
    
    elif subtype == 'crossflow, mixed Cmin':
        if Cr*log(1. - effectiveness) < -1:
            raise Exception('The specified effectiveness is not physically \
possible for this configuration; the maximum effectiveness possible is %s.' % (1. - exp(-1./Cr)))
        return -1./Cr*log(Cr*log(1. - effectiveness) + 1.)
    
    elif subtype ==  'crossflow, mixed Cmax':
        if 1./Cr*log(1. - effectiveness*Cr) < -1:
            raise Exception('The specified effectiveness is not physically \
possible for this configuration; the maximum effectiveness possible is %s.' % (((exp(Cr) - 1.0)*exp(-Cr)/Cr)))
        return -log(1. + 1./Cr*log(1. - effectiveness*Cr))
    
    elif subtype in ['boiler', 'condenser']:
        return -log(1. - effectiveness)
    else:
        raise Exception('Input heat exchanger type not recognized')


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
    0.713634370024604

    References
    ----------
    .. [1] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat 
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [2] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    try:
        return (1. - exp(-x*(1. + y)))/(1. + y)
    except ZeroDivisionError:
        return x


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
    0.9206703686051108

    References
    ----------
    .. [1] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat 
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    .. [2] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    try:
        term = exp(-x*(1. - y))
        return (1. - term)/(1. - y*term)
    except ZeroDivisionError:
        return x/(1. + x)


def effectiveness_NTU_method(mh, mc, Cph, Cpc, subtype='counterflow', Thi=None, 
                             Tho=None, Tci=None, Tco=None, UA=None):
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
    
    See also
    --------
    effectiveness_from_NTU
    NTU_from_effectiveness

    Examples
    --------
    Solve a heat exchanger to determine UA and effectiveness given the
    configuration, flows, subtype, the cold inlet/outlet temperatures, and the
    hot stream inlet temperature.
    
    >>> pprint(effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, 
    ... subtype='crossflow, mixed Cmax', Tci=15, Tco=85, Thi=130))
    {'Cmax': 9672.0,
     'Cmin': 2755.0,
     'Cr': 0.2848428453267163,
     'NTU': 1.1040839095588,
     'Q': 192850.0,
     'Tci': 15,
     'Tco': 85,
     'Thi': 130,
     'Tho': 110.06100082712986,
     'UA': 3041.751170834494,
     'effectiveness': 0.6086956521739131}
    
    Solve the same heat exchanger with the UA specified, and known inlet
    temperatures:
        
    >>> pprint(effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, 
    ... subtype='crossflow, mixed Cmax', Tci=15, Thi=130, UA=3041.75))
    {'Cmax': 9672.0,
     'Cmin': 2755.0,
     'Cr': 0.2848428453267163,
     'NTU': 1.1040834845735028,
     'Q': 192849.96310220254,
     'Tci': 15,
     'Tco': 84.99998660697007,
     'Thi': 130,
     'Tho': 110.06100464203861,
     'UA': 3041.75,
     'effectiveness': 0.6086955357127832}
    '''
    Cmin = calc_Cmin(mh=mh, mc=mc, Cph=Cph, Cpc=Cpc)
    Cmax = calc_Cmax(mh=mh, mc=mc, Cph=Cph, Cpc=Cpc)
    Cr = calc_Cr(mh=mh, mc=mc, Cph=Cph, Cpc=Cpc)
    Cc = mc*Cpc
    Ch = mh*Cph
    if UA is not None:
        NTU = NTU_from_UA(UA=UA, Cmin=Cmin)
        effectiveness = eff = effectiveness_from_NTU(NTU=NTU, Cr=Cr, subtype=subtype)
        
        possible_inputs = [(Tci, Thi), (Tci, Tho), (Tco, Thi), (Tco, Tho)]
        if not any([i for i in possible_inputs if None not in i]):
            raise Exception('One set of (Tci, Thi), (Tci, Tho), (Tco, Thi), or (Tco, Tho) are required along with UA.')
        
        if Thi and Tci:
            Q = eff*Cmin*(Thi - Tci)
        elif Tho and Tco :
            Q = eff*Cmin*Cc*Ch*(Tco - Tho)/(eff*Cmin*(Cc+Ch) - Ch*Cc)
        elif Thi and Tco:
            Q = Cmin*Cc*eff*(Tco-Thi)/(eff*Cmin - Cc)
        elif Tho and Tci:
            Q = Cmin*Ch*eff*(Tci-Tho)/(eff*Cmin - Ch)
        # The following is not used as it was decided to require complete temperature information
#        elif Tci and Tco:
#            Q = Cc*(Tco - Tci)
#        elif Tho and Thi:
#            Q = Ch*(Thi-Tho)
        # Compute the remaining temperatures with the fewest lines of code
        if Tci and not Tco:
            Tco = Tci + Q/(Cc)
        else:
            Tci = Tco - Q/(Cc)
        if Thi and not Tho:
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
                    raise Exception('The specified heat capacities, mass flows, and temperatures are inconsistent')
            else:
                raise Exception('At least one temperature is required to be specified on the cold side.')
                
        elif Tci is not None and Tco is not None:
            Q = mc*Cpc*(Tco-Tci)
            if Thi is not None and Tho is None:
                Tho = Thi - Q/(mh*Cph)
            elif Tho is not None and Thi is None:
                Thi = Tho + Q/(mh*Cph)
            else:
                raise Exception('At least one temperature is required to be specified on the cold side.')
        else:
            raise Exception('Three temperatures are required to be specified '
                            'when solving for UA')

        effectiveness = Q/Cmin/(Thi-Tci)
        NTU = NTU_from_effectiveness(effectiveness, Cr, subtype=subtype)
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
    OverflowError: long int too large to convert to float
    

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
        # Precalculate integer factorials up to N
        factorials = [factorial(i) for i in range(N)]
        K_powers = [K**j for j in range(0, N+1)]
        NKR1_powers = [NKR1**k for k in range(0, N+1)]
        exp_terms = [exp(i*NTU1_N) for i in range(-N+1, 1)]
        NKR1_powers_over_factorials = [NKR1_powers[k]/factorials[k] 
                                       for k in range(N)]
        
        # Precompute even more...
        NKR1_pows_div_factorials = [0]
        for k in NKR1_powers_over_factorials:
            NKR1_pows_div_factorials.append(NKR1_pows_div_factorials[-1]+k)
        NKR1_pows_div_factorials.pop(0)
        
        final_speed = [i*j for i, j in zip(K_powers, NKR1_pows_div_factorials)]
        
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
            raise Exception('Number of passes and rows not supported.')


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
        def to_int(v):
            v2 = v*v
            return (1. + NTU1 - v2*R1_NTU1_4_inv)*exp(-v2*R1_NTU1_4_inv)*v*float(iv(0, v))
        from scipy.integrate import quad
        int_term = quad(to_int, 0.0, 2.*NTU1*R1**0.5)[0]# args=(NTU1, R1)
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
        raise Exception('Subtype not recognized.')
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
        raise Exception('Supported numbers of tube passes are 1, 2, and 4.')
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
        raise Exception('Supported numbers of tube passes are 1 and 2.')
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
        raise Exception('Supported numbers of tube passes are 1 and 2.')
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
        raise Exception('For TEMA E shells with an odd number of tube passes more than 3, no solution is implemented.')
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
                                                   counterflow=False)
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
        R2 = 1./R1
        NTU2 = NTU1*R1
        P2 = temperature_effectiveness_plate(R1=R2, NTU1=NTU2, Np1=Np2, Np2=Np1,
                                             counterflow=counterflow, 
                                             passes_counterflow=passes_counterflow, 
                                             reverse=True)
        P1 = P2*R2
        return P1
    
    raise Exception('Supported number of passes does not have a formula available')

    
NTU_from_plate_2_3_parallel = {
    'offset': [7.5e-09, 1.4249999999999999e-08, 2.7074999999999996e-08, 5.144249999999999e-08, 9.774074999999998e-08, 1.8570742499999996e-07,
        3.528441074999999e-07, 6.704038042499998e-07, 1.2737672280749996e-06, 2.420157733342499e-06, 4.598299693350748e-06, 8.73676941736642e-06,
        1.6599861892996197e-05, 3.153973759669277e-05, 5.9925501433716265e-05, 0.0001138584527240609, 0.0002163310601757157, 0.0004110290143338598,
        0.0007809551272343336, 0.0014838147417452338, 0.002819248009315944, 0.005356571217700294, 0.010177485313630557, 0.019337222095898058,
        0.036740721982206306, 0.06980737176619198, 0.13263400635576475, 0.25200461207595304, 0.47880876294431074, 0.9097366495941903,
        1.7284996342289616, 3.2841493050350268, 6.23988367956655, 11.855778991176445, 22.525980083235243, 42.79936215814696, 81.31878810047922,
        154.5056973909105, 293.56082504272996, 557.7655675811869
    ],
    'p': [
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
    ],
    'q': [
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
}

NTU_from_plate_2_4_parallel = {
    'offset': [7.5e-08, 1.5e-07, 3e-07, 6e-07, 1.2e-06, 2.4e-06, 4.8e-06, 9.6e-06, 1.92e-05, 3.84e-05, 7.68e-05, 0.0001536, 0.0003072, 0.0006144, 0.0012288,
        0.0024576, 0.0049152, 0.0098304, 0.0196608, 0.0393216, 0.0786432, 0.1572864, 0.3145728, 0.6291456, 1.2582912, 2.5165824, 5.0331648, 10.0663296,
        20.1326592, 40.2653184, 80.5306368
    ],
    'p': [
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
    ],
    'q': [
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
}

NTU_from_plate_2_2_parallel_counterflow = {
    'offset': [7.5e-08, 1.5e-07, 3e-07, 6e-07, 1.2e-06, 2.4e-06, 4.8e-06, 9.6e-06, 1.92e-05, 3.84e-05, 7.68e-05, 0.0001536, 0.0003072, 0.0006144, 0.0012288,
        0.0024576, 0.0049152, 0.0098304, 0.0196608, 0.0393216, 0.0786432, 0.1572864, 0.3145728, 0.6291456, 1.2582912, 2.5165824, 5.0331648, 10.0663296,
        20.1326592, 40.2653184, 80.5306368
    ],
    'p': [
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
    ],
    'q': [
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
}

NTU_from_H_2_unoptimal = {
    'offset': [7.5e-08, 1.5e-07, 3e-07, 6e-07, 1.2e-06, 2.4e-06, 4.8e-06, 9.6e-06, 1.92e-05, 3.84e-05, 7.68e-05, 0.0001536, 0.0003072, 0.0006144, 0.0012288,
        0.0024576, 0.0049152, 0.0098304, 0.0196608, 0.0393216, 0.0786432, 0.1572864, 0.3145728, 0.6291456, 1.2582912, 2.5165824, 5.0331648, 10.0663296,
        20.1326592, 40.2653184, 80.5306368
    ],
    'p': [
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
    ],
    'q': [
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
}


NTU_from_G_2_unoptimal = {
    'offset': [7.5e-08, 1.5e-07, 3e-07, 6e-07, 1.2e-06, 2.4e-06, 4.8e-06, 9.6e-06, 1.92e-05, 3.84e-05, 7.68e-05, 0.0001536, 0.0003072, 0.0006144, 0.0012288,
        0.0024576, 0.0049152, 0.0098304, 0.0196608, 0.0393216, 0.0786432, 0.1572864, 0.3145728, 0.6291456, 1.2582912, 2.5165824, 5.0331648, 10.0663296,
        20.1326592, 40.2653184
    ],
    'p': [
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
    ],
    'q': [
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
}

NTU_from_P_J_2 = {
    'offset': [7.5e-08, 2.25e-07, 6.75e-07, 2.025e-06, 6.075e-06, 1.8225000000000003e-05, 5.467500000000001e-05, 0.00016402500000000002, 0.000492075,
        0.001476225, 0.004428675, 0.013286025, 0.039858075, 0.119574225, 0.358722675, 1.0761680249999999, 3.2285040749999996, 9.685512224999998,
        29.056536674999997, 87.169610025],
    'p': [
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
    ],
    'q': [
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
}

NTU_from_P_J_4 = {
    'offset': [7.5e-08, 1.5e-07, 3e-07, 6e-07, 1.2e-06, 2.4e-06, 4.8e-06, 9.6e-06, 1.92e-05, 3.84e-05, 7.68e-05, 0.0001536, 0.0003072, 0.0006144, 0.0012288,
        0.0024576, 0.0049152, 0.0098304, 0.0196608, 0.0393216, 0.0786432, 0.1572864, 0.3145728, 0.6291456, 1.2582912, 2.5165824, 5.0331648, 10.0663296,
        20.1326592, 40.2653184
    ],
    'p': [
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
    ],
    'q': [
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
}    
    
NTU_from_P_basic_crossflow_mixed_12 = {
    'offset': [7.5e-08, 1.5e-07, 3e-07, 6e-07, 1.2e-06, 2.4e-06, 4.8e-06, 9.6e-06, 1.92e-05, 3.84e-05, 7.68e-05, 0.0001536, 0.0003072, 0.0006144, 0.0012288,
        0.0024576, 0.0049152, 0.0098304, 0.0196608, 0.0393216, 0.0786432, 0.1572864, 0.3145728, 0.6291456, 1.2582912, 2.5165824, 5.0331648, 10.0663296,
        20.1326592, 40.2653184, 80.5306368, 161.0612736, 322.1225472, 644.2450944, 1288.4901888
    ],
    'p': [
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
    ],
    'q': [
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
    ],
}




def _NTU_from_P_objective(NTU1, R1, P1, function, **kwargs):
    '''Private function to hold the common objective function used by 
    all backwards solvers for the P-NTU method.
    These methods are really hard on on floating points (overflows and divide
    by zeroes due to numbers really close to 1), so if the function fails,
    mpmath is imported and tried.
    '''
    try:
        P1_calc = function(R1, NTU1, **kwargs)
    except :
        try:
            import mpmath
        except ImportError:  # pragma: no cover
            raise Exception('For some reverse P-NTU numerical solutions, the \
intermediary results are ill-conditioned and do not fit in a float; mpmath must \
be installed for this calculation to proceed.')
        globals()['exp'] = mpmath.exp
        P1_calc = float(function(R1, NTU1, **kwargs))
        globals()['exp'] = math.exp
    return P1_calc - P1


def _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, function, **kwargs):
    '''Private function to solve the P-NTU method backwards, given the
    function to use, the upper and lower NTU bounds for consideration,
    and the desired P1 and R1 values.
    '''
    P1_max = _NTU_from_P_objective(NTU_max, R1, 0, function, **kwargs)
    P1_min = _NTU_from_P_objective(NTU_min, R1, 0, function, **kwargs)
    if P1 > P1_max:
        raise ValueError('No solution possible gives such a high P1; maximum P1=%f at NTU1=%f' %(P1_max, NTU_max))
    if P1 < P1_min:
        raise ValueError('No solution possible gives such a low P1; minimum P1=%f at NTU1=%f' %(P1_min, NTU_min))
    # Construct the function as a lambda expression as solvers don't support kwargs
    to_solve = lambda NTU1: _NTU_from_P_objective(NTU1, R1, P1, function, **kwargs)
    return ridder(to_solve, NTU_min, NTU_max)


def _NTU_max_for_P_solver(data, R1):
    '''Private function to calculate the upper bound on the NTU1 value in the
    P-NTU method. This value is calculated via a pade approximation obtained
    on the result of a global minimizer which calculated the maximum P1
    at a given R1 from ~1E-7 to approximately 100. This should suffice for 
    engineering applications. This value is needed to bound the solver.
    '''
    offset_max = data['offset'][-1]
    for offset, p, q in zip(data['offset'], data['p'], data['q']):
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
        NTU_1 = - \frac{1}{R_{1} - 1} \log{\left (\frac{P_{1} R_{1} - 1}{P_{1} 
        - 1} \right )}
    
    Parallel:
    
    .. math::
        NTU_1 = \frac{1}{R_{1} + 1} \log{\left (- \frac{1}{P_{1} \left(R_{1} 
        + 1\right) - 1} \right )}
    
    Crossflow, single pass, fluid 1 mixed, fluid 2 unmixed:
        
    .. math::
        NTU_1 = - \frac{1}{R_{1}} \log{\left (R_{1} \log{\left (- \left(P_{1}
        - 1\right) e^{\frac{1}{R_{1}}} \right )} \right )}
    
    Crossflow, single pass, fluid 2 mixed, fluid 1 unmixed
    
    .. math::
        NTU_1 = - \log{\left (\frac{1}{R_{1}} \log{\left (- \left(P_{1} R_{1}
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
    function = temperature_effectiveness_basic
    if subtype == 'counterflow':
        return -log((P1*R1 - 1.)/(P1 - 1.))/(R1 - 1.)
    elif subtype == 'parallel':
        return log(-1./(P1*(R1 + 1.) - 1.))/(R1 + 1.)
    elif subtype == 'crossflow, mixed 1':
        return -log(R1*log(-(P1 - 1.)*exp(1./R1)))/R1
    elif subtype == 'crossflow, mixed 2':
        return -log(log(-(P1*R1 - 1.)*exp(R1))/R1)
    elif subtype == 'crossflow, mixed 1&2':
        NTU_max = _NTU_max_for_P_solver(NTU_from_P_basic_crossflow_mixed_12, R1)        
    elif subtype == 'crossflow approximate':
        # These are tricky but also easy because P1 can always be 1
        NTU_max = 1E5
    elif subtype == 'crossflow':
        guess = NTU_from_P_basic(P1, R1, subtype='crossflow approximate')
        to_solve = lambda NTU1 : _NTU_from_P_objective(NTU1, R1, P1, function, subtype='crossflow')
        return newton(to_solve, guess)
    else:
        raise Exception('Subtype not recognized.')
    return _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, function, subtype=subtype)


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
    0.9999513707769526
    '''
    NTU_min = 1E-11
    function = temperature_effectiveness_TEMA_G
    if Ntp == 1 or (Ntp == 2 and optimal):
        NTU_max = 1E4
        # We could fit a curve to determine the NTU where the floating point
        # does not allow NTU to increase though, but that would be another
        # binary bisection process, different from the current pipeline
    elif Ntp == 2 and not optimal:
        NTU_max = _NTU_max_for_P_solver(NTU_from_G_2_unoptimal, R1)
    else:
        raise Exception('Supported numbers of tube passes are 1 or 2.')
    return _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, function, Ntp=Ntp, optimal=optimal)


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
    13.940758760696617
    >>> NTU_from_P_J(P1=.99502487562188, R1=.01, Ntp=1)
    536.4817955951684
    >>> NTU_from_P_J(P1=.99502487562189, R1=.01, Ntp=1)
    Traceback (most recent call last):
    ValueError: No solution possible gives such a high P1; maximum P1=0.995025 at NTU1=1000.000000
    
    For the 2 pass and 4 pass solution, a bounded solver is first use, but
    the upper bound on P1 and the upper NTU1 limit is calculated from a pade
    approximation performed with mpmath. These normally do not allow NTU1 to 
    rise above 100.

    Examples
    --------
    >>> NTU_from_P_J(P1=.57, R1=1/3., Ntp=1)
    1.0003070138879648
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
        NTU_max = _NTU_max_for_P_solver(NTU_from_P_J_2, R1)
    elif Ntp == 4:
        NTU_max = _NTU_max_for_P_solver(NTU_from_P_J_4, R1)
    else:
        raise Exception('Supported numbers of tube passes are 1, 2, and 4.')
    return _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, function, Ntp=Ntp)


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
        NTU_1 = - \frac{1}{R_{1} - 1} \log{\left (\frac{P_{1} R_{1} - 1}{P_{1} 
        - 1} \right )}
    
    1-2 TEMA E, shell fluid mixed:
        
    .. math::
        NTU_1 = \frac{2}{\sqrt{R_{1}^{2} + 1}} \log{\left (\sqrt{\frac{P_{1} 
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
        raise Exception('For TEMA E shells with an odd number of tube passes more than 3, no solution is implemented.')
    return _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, function, Ntp=Ntp, optimal=optimal)


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
    0.9997628696881165
    '''
    NTU_min = 1E-11
    function = temperature_effectiveness_TEMA_H
    if Ntp == 1:
        NTU_max = 100
    elif Ntp == 2 and optimal:
        NTU_max = 100
    elif Ntp == 2 and not optimal:
        NTU_max = _NTU_max_for_P_solver(NTU_from_H_2_unoptimal, R1)
    else:
        raise Exception('Supported numbers of tube passes are 1 and 2.')
    return _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, function, Ntp=Ntp, optimal=optimal)


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
        NTU_1 = - \frac{1}{R_{1} - 1} \log{\left (\frac{P_{1} R_{1} - 1}{P_{1} 
        - 1} \right )}
    
    1 pass/1 pass parallel flow (also 2/2 fully parallelflow):
    
    .. math::
        NTU_1 = \frac{1}{R_{1} + 1} \log{\left (- \frac{1}{P_{1} \left(R_{1} 
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
    0.9998336056060733
    '''
    NTU_min = 1E-11
    function = temperature_effectiveness_plate
    if Np1 == 1 and Np2 == 1 and counterflow:
        try:
            return -log((P1*R1 - 1.)/(P1 - 1.))/(R1 - 1.)
        except ValueError:
            raise ValueError('The maximum P1 obtainable at the specified R1 is %f at the limit of NTU1=inf.' %(1./R1))
        
    elif Np1 == 1 and Np2 == 1 and not counterflow:
        try:
            return log(-1./(P1*(R1 + 1.) - 1.))/(R1 + 1.)
        except ValueError:
            raise ValueError('The maximum P1 obtainable at the specified R1 is %f at the limit of NTU1=inf.' %Pp(1E10, R1))
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
            NTU_max = 100
        elif not counterflow and passes_counterflow:
            NTU_max = _NTU_max_for_P_solver(NTU_from_plate_2_2_parallel_counterflow, R1)
        elif not counterflow and not passes_counterflow:
            return NTU_from_P_plate(P1, R1, Np1=1, Np2=1, counterflow=False, 
                                    passes_counterflow=False)
    elif Np1 == 2 and Np2 == 3:
        if counterflow:
            NTU_max = 100
        elif not counterflow:
            NTU_max = _NTU_max_for_P_solver(NTU_from_plate_2_3_parallel, R1)
    elif Np1 == 2 and Np2 == 4:
        if counterflow:
            NTU_max = 100
        elif not counterflow:
            NTU_max = _NTU_max_for_P_solver(NTU_from_plate_2_4_parallel, R1)
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
        raise Exception('Supported number of passes does not have a formula available')
    return _NTU_from_P_solver(P1, R1, NTU_min, NTU_max, function, Np1=Np1, 
                              Np2=Np2, counterflow=counterflow, 
                              passes_counterflow=passes_counterflow)


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
        
    See also
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
        
    >>> pprint(P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900, 
    ... subtype='E', Ntp=4, T2i=15, T1i=130, UA=3041.75))
    {'C1': 9672.0,
     'C2': 2755.0,
     'NTU1': 0.3144902812241522,
     'NTU2': 1.1040834845735028,
     'P1': 0.1730811614360235,
     'P2': 0.6076373841775751,
     'Q': 192514.71424206023,
     'R1': 3.5107078039927404,
     'R2': 0.2848428453267163,
     'T1i': 130,
     'T1o': 110.09566643485729,
     'T2i': 15,
     'T2o': 84.87829918042112,
     'UA': 3041.75}
    
    Solve the same heat exchanger as if T1i, T2i, and T2o were known but UA was
    not:
        
    >>> pprint(P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900, subtype='E', 
    ... Ntp=4, T1i=130, T2i=15, T2o=84.87829918042112))
    {'C1': 9672.0,
     'C2': 2755.0,
     'NTU1': 0.31449028122515194,
     'NTU2': 1.1040834845770124,
     'P1': 0.17308116143602348,
     'P2': 0.607637384177575,
     'Q': 192514.7142420602,
     'R1': 3.5107078039927404,
     'R2': 0.2848428453267163,
     'T1i': 130,
     'T1o': 110.09566643485729,
     'T2i': 15,
     'T2o': 84.87829918042112,
     'UA': 3041.7500000096693}

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
     'NTU1': 0.031017369727047148,
     'NTU2': 0.1088929219600726,
     'P1': 0.028945295974795074,
     'P2': 0.10161847646759273,
     'Q': 32200.050307849266,
     'R1': 3.5107078039927404,
     'R2': 0.2848428453267163,
     'T1i': 130.02920288542694,
     'T1o': 126.7,
     'T2i': 15.012141449056527,
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
        
        if subtype in ['counterflow', 'parallel', 'crossflow', 'crossflow, mixed 1', 'crossflow, mixed 2', 'crossflow, mixed 1&2']:
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
                passes_counterflow = True if end[-1] == 'c' else False
                end = end[0:-1]
            Np1, Np2 = int(Np1), int(end)
            P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=Np1, Np2=Np2, counterflow=optimal, passes_counterflow=passes_counterflow)
        else:
            raise Exception("Supported types are 'E', 'G', 'H', 'J', 'counterflow',\
    'parallel', 'crossflow', 'crossflow, mixed 1', 'crossflow, mixed 2', \
    'crossflow, mixed 1&2', or 'Np1/Np2' for plate exchangers")
        
        possible_inputs = [(T1i, T2i), (T1o, T2o), (T1i, T2o), (T1o, T2i), (T1i, T1o), (T2i, T2o)]
        if not any([i for i in possible_inputs if None not in i]):
            raise Exception('One set of (T1i, T2i), (T1o, T2o), (T1i, T2o), (T1o, T2i), (T1i, T1o), or (T2i, T2o) is required along with UA.')
        
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
                    raise Exception('The specified heat capacities, mass flows,'
                                    ' and temperatures are inconsistent')
            else:
                raise Exception('At least one temperature is required to be '
                                'specified on side 2.')
                
        elif T2i is not None and T2o is not None:
            Q = m2*Cp2*(T2o-T2i)
            if T1i is not None and T1o is None:
                T1o = T1i - Q/(m1*Cp1)
            elif T1o is not None and T1i is None:
                T1i = T1o + Q/(m1*Cp1)
            else:
                raise Exception('At least one temperature is required to be '
                                'specified on side 2.')
        else:
            raise Exception('Three temperatures are required to be specified '
                            'when solving for UA')
                
        P1 = Q/(C1*abs(T2i-T1i))
        if subtype in ['counterflow', 'parallel', 'crossflow', 'crossflow, mixed 1', 'crossflow, mixed 2', 'crossflow, mixed 1&2']:
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
                passes_counterflow = True if end[-1] == 'c' else False
                end = end[0:-1]
            Np1, Np2 = int(Np1), int(end)
            NTU1 = NTU_from_P_plate(P1=P1, R1=R1, Np1=Np1, Np2=Np2, counterflow=optimal, passes_counterflow=passes_counterflow)
        else:
            raise Exception("Supported types are 'E', 'G', 'H', 'J', 'counterflow',\
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
        return (2**0.5*(1. - W2)/W2)/log(((W2/(1. - W2) + 2**-0.5)/(W2/(1. - W2) - 2**-0.5)))
    else:
        W = ((1. - P*R)/(1. - P))**(1./shells)
        S = (R*R + 1.)**0.5/(R - 1.)
        return S*log(W)/log((1. + W - S + S*W)/(1. + W + S - S*W))

### Tubes

# TEMA tubes from http://www.engineeringpage.com/technology/thermal/tubesize.html
# NPSs in inches, which convert to outer diameter exactly.
_NPSs = [0.25, 0.25, 0.375, 0.375, 0.375, 0.5, 0.5, 0.625, 0.625, 0.625, 0.75, 0.75, 0.75, 0.75, 0.75, 0.875, 0.875, 0.875, 0.875, 1, 1, 1, 1, 1.25, 1.25, 1.25, 1.25, 2, 2]
_Dos = [ i/1000. for i in [6.35, 6.35, 9.525, 9.525, 9.525, 12.7, 12.7, 15.875, 15.875, 15.875, 19.05, 19.05, 19.05, 19.05, 19.05, 22.225, 22.225, 22.225, 22.225, 25.4, 25.4, 25.4, 25.4, 31.75, 31.75, 31.75, 31.75, 50.8, 50.8]]
_BWGs = [22, 24, 18, 20, 22, 18, 20, 16, 18, 20, 12, 14, 16, 18, 20, 14, 16, 18, 20, 12, 14, 16, 18, 10, 12, 14, 16, 12, 14]
_ts = [i/1000. for i in [0.711, 0.559, 1.245, 0.889, 0.711, 1.245, 0.889, 1.651, 1.245, 0.889, 2.769, 2.108, 1.651, 1.245, 0.889, 2.108, 1.651, 1.245, 0.889, 2.769, 2.108, 1.651, 1.245, 3.404, 2.769, 2.108, 1.651, 2.769, 2.108]]
_Dis = [i/1000. for i in [4.928, 5.232, 7.035, 7.747, 8.103, 10.21, 10.922, 12.573, 13.385, 14.097, 13.512, 14.834, 15.748, 16.56, 17.272, 18.009, 18.923, 19.735, 20.447, 19.862, 21.184, 22.098, 22.91, 24.942, 26.212, 27.534, 28.448, 45.262, 46.584]]

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
            raise Exception('NPS and BWG Specified are not listed in TEMA')
        Do = 0.0254*NPS
        t = BWG_SI[BWG_integers.index(BWG)]
        Di = Do-2*t
    elif Do and BWG:
        NPS = Do/.0254
        if not check_tubing_TEMA(NPS, BWG):
            raise Exception('NPS and BWG Specified are not listed in TEMA')
        t = BWG_SI[BWG_integers.index(BWG)]
        Di = Do-2*t
    elif BWG and Di:
        t = BWG_SI[BWG_integers.index(BWG)] # Will fail if BWG not int
        Do = t*2 + Di
        NPS = Do/.0254
        if not check_tubing_TEMA(NPS, BWG):
            raise Exception('NPS and BWG Specified are not listed in TEMA')
    elif NPS and Di:
        Do = 0.0254*NPS
        t = (Do - Di)/2
        BWG = [BWG_integers[BWG_SI.index(t)]]
        if not check_tubing_TEMA(NPS, BWG):
            raise Exception('NPS and BWG Specified are not listed in TEMA')
    elif Di and Do:
        NPS = Do/.0254
        t = (Do - Di)/2
        BWG = [BWG_integers[BWG_SI.index(t)]]
        if not check_tubing_TEMA(NPS, BWG):
            raise Exception('NPS and BWG Specified are not listed in TEMA')
    # Begin Fuzzy matching
    elif NPS and tmin:
        Do = 0.0254*NPS
        ts = [BWG_SI[BWG_integers.index(BWG)] for BWG in TEMA_tubing[NPS]]
        ts.reverse() # Small to large
        if tmin > ts[-1]:
            raise Exception('Specified minimum thickness is larger than available in TEMA')
        for t in ts: # Runs if at least 1 of the thicknesses are the right size.
            if tmin <= t:
                break
        BWG = [BWG_integers[BWG_SI.index(t)]]
        Di = Do-2*t
    elif Do and tmin:
        NPS = Do/.0254
        NPS, BWG, Do, Di, t = get_tube_TEMA(NPS=NPS, tmin=tmin)
    elif Di and tmin:
        raise Exception('Not funny defined input for TEMA Schedule; multiple solutions')
    elif NPS:
        BWG = TEMA_tubing[NPS][0] # Pick the first listed size
        Do = 0.0254*NPS
        t = BWG_SI[BWG_integers.index(BWG)]
        Di = Do-2*t
    else:
        raise Exception('Insufficient information provided')
    return NPS, BWG, Do, Di, t

TEMA_Ls_imperial = [96., 120., 144., 192., 240.] # inches
TEMA_Ls = [2.438, 3.048, 3.658, 4.877, 6.096]
HTRI_Ls_imperial = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60] # ft
HTRI_Ls = [round(i*foot, 3) for i in HTRI_Ls_imperial]


# Shells up to 120 inch in diameter.
# This is for plate shells, not pipe (up to 12 inches, pipe is used)
HEDH_shells_imperial = [12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 24., 26., 28., 30., 32., 34., 36., 38., 40., 42., 44., 46., 48., 50., 52., 54., 56., 58., 60., 63., 66., 69., 72., 75., 78., 81., 84., 87., 90., 93., 96., 99., 102., 105., 108., 111., 114., 117., 120.]
HEDH_shells = [round(i*inch, 6) for i in HEDH_shells_imperial]


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
    data = [(0.006, 0.1), (0.01, 0.1), (.014, 0.3), (0.02, 0.5), (0.03, 1.0)]
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
        raise Exception('Either DShell or DBundle must be specified')


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



_L_unsupported_Do = [0.25, 0.375, 0.5, 0.628, 0.75, 0.875, 1., 1.25, 1.5,
                          2., 2.5, 3.]
_L_unsupported_steel = [0.66, 0.889, 1.118, 1.321, 1.524, 1.753, 1.88, 2.235,
                        2.54, 3.175, 3.175, 3.175]
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
    i = bisect(_L_unsupported_Do, Do)-1
    i = i if i < 11 else 11 # bisect returns 1+ if above the index
    i = 0 if i == -1 else i
    if material == 'CS':
        return _L_unsupported_steel[i]
    elif material == 'aluminium':
        return _L_unsupported_aluminium[i]
    else:
        raise Exception('Material argument should be one of "CS" or "aluminium"')


### Tube bundle count functions


# 130 kB in memory as numpy arrays
triangular_Ns = np.array([0, 1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25, 27, 28, 31, 36, 37, 39, 43, 48, 49, 52, 57, 61, 63, 64, 67, 73, 75, 76, 79, 81, 84, 91, 93, 97, 100, 103, 108, 109,
111, 112, 117, 121, 124, 127, 129, 133, 139, 144, 147, 148, 151, 156, 157, 163, 169, 171, 172, 175, 181, 183, 189, 192, 193, 196, 199, 201, 208, 211, 217,
219, 223, 225, 228, 229, 237, 241, 243, 244, 247, 252, 256, 259, 268, 271, 273, 277, 279, 283, 289, 291, 292, 300, 301, 304, 307, 309, 313, 316, 324, 325,
327, 331, 333, 336, 337, 343, 349, 351, 361, 363, 364, 367, 372, 373, 379, 381, 387, 388, 397, 399, 400, 403, 409, 412, 417, 421, 427, 432, 433, 436, 439,
441, 444, 448, 453, 457, 463, 468, 469, 471, 475, 481, 484, 487, 489, 496, 499, 507, 508, 511, 513, 516, 523, 525, 529, 532, 541, 543, 547, 549, 553, 556,
559, 567, 571, 576, 577, 579, 588, 589, 592, 597, 601, 603, 604, 607, 613, 619, 624, 625, 628, 631, 633, 637, 643, 651, 652, 657, 661, 669, 673, 675, 676,
679, 684, 687, 688, 691, 700, 703, 709, 711, 721, 723, 724, 727, 729, 732, 733, 739, 741, 751, 756, 757, 763, 768, 769, 772, 775, 777, 784, 787, 793, 796,
804, 811, 813, 817, 819, 823, 829, 831, 832, 837, 841, 844, 847, 849, 853, 859, 867, 868, 871, 873, 876, 877, 883, 889, 892, 900, 903, 907, 912, 916, 919,
921, 925, 927, 931, 937, 939, 948, 949, 961, 964, 967, 972, 973, 975, 976, 981, 988, 991, 993, 997, 999, 1008, 1009, 1011, 1021, 1024, 1027, 1029, 1033,
1036, 1039, 1047, 1051, 1053, 1057, 1063, 1069, 1072, 1075, 1083, 1084, 1087, 1089, 1092, 1093, 1099, 1101, 1108, 1116, 1117, 1119, 1123, 1129, 1132, 1137,
1141, 1143, 1147, 1153, 1156, 1159, 1161, 1164, 1168, 1171, 1183, 1191, 1197, 1200, 1201, 1204, 1209, 1213, 1216, 1225, 1227, 1228, 1231, 1236, 1237, 1249,
1251, 1252, 1261, 1263, 1264, 1267, 1273, 1279, 1281, 1291, 1296, 1297, 1299, 1300, 1303, 1308, 1317, 1321, 1323, 1324, 1327, 1332, 1333, 1339, 1344, 1348,
1351, 1359, 1369, 1371, 1372, 1381, 1387, 1389, 1393, 1396, 1399, 1404, 1407, 1413, 1417, 1423, 1425, 1429, 1443, 1444, 1447, 1452, 1453, 1456, 1459, 1461,
1467, 1468, 1471, 1477, 1483, 1488, 1489, 1492, 1497, 1501, 1516, 1519, 1521, 1524, 1525, 1531, 1533, 1539, 1543, 1548, 1549, 1552, 1561, 1567, 1569, 1573,
1575, 1579, 1587, 1588, 1591, 1596, 1597, 1600, 1603, 1609, 1612, 1621, 1623, 1627, 1629, 1636, 1641, 1647, 1648, 1651, 1657, 1659, 1663, 1668, 1669, 1675,
1677, 1681, 1684, 1687, 1693, 1699, 1701, 1708, 1713, 1723, 1728, 1729, 1731, 1732, 1737, 1741, 1744, 1747, 1753, 1756, 1759, 1764, 1767, 1776, 1777, 1783,
1789, 1791, 1792, 1801, 1803, 1807, 1809, 1812, 1813, 1821, 1825, 1828, 1831, 1839, 1843, 1849, 1852, 1857, 1861, 1867, 1872, 1873, 1875, 1876, 1879, 1884,
1891, 1893, 1897, 1899, 1900, 1911, 1924, 1929, 1933, 1936, 1939, 1948, 1951, 1953, 1956, 1957, 1963, 1971, 1975, 1981, 1983, 1984, 1987, 1993, 1996, 1999,
2007, 2011, 2017, 2019, 2023, 2025, 2028, 2029, 2032, 2037, 2041, 2044, 2052, 2053, 2061, 2064, 2071, 2073, 2077, 2083, 2089, 2092, 2100, 2107, 2109, 2113,
2116, 2119, 2127, 2128, 2131, 2133, 2137, 2143, 2149, 2161, 2163, 2164, 2169, 2172, 2179, 2181, 2187, 2188, 2191, 2196, 2197, 2199, 2203, 2209, 2212, 2217,
2221, 2223, 2224, 2236, 2239, 2251, 2253, 2257, 2263, 2268, 2269, 2271, 2275, 2281, 2284, 2287, 2289, 2293, 2299, 2304, 2307, 2308, 2311, 2316, 2317, 2325,
2331, 2341, 2347, 2352, 2353, 2356, 2359, 2361, 2368, 2371, 2377, 2379, 2383, 2388, 2389, 2401, 2404, 2412, 2413, 2416, 2425, 2428, 2433, 2437, 2439, 2443,
2449, 2451, 2452, 2457, 2467, 2469, 2473, 2476, 2479, 2487, 2493, 2496, 2500, 2503, 2509, 2511, 2512, 2521, 2523, 2524, 2527, 2532, 2539, 2541, 2547, 2548,
2551, 2557, 2559, 2569, 2572, 2575, 2577, 2587, 2593, 2601, 2604, 2608, 2611, 2613, 2617, 2619, 2623, 2628, 2631, 2641, 2644, 2647, 2649, 2653, 2659, 2667,
2671, 2676, 2677, 2683, 2689, 2692, 2700, 2701, 2704, 2707, 2709, 2713, 2716, 2719, 2721, 2725, 2731, 2736, 2743, 2748, 2749, 2752, 2757, 2763, 2764, 2767,
2775, 2779, 2781, 2791, 2793, 2797, 2800, 2803, 2809, 2811, 2812, 2817, 2821, 2833, 2836, 2844, 2847, 2851, 2857, 2863, 2869, 2881, 2883, 2884, 2887, 2892,
2896, 2899, 2901, 2908, 2916, 2917, 2919, 2923, 2925, 2928, 2932, 2943, 2947, 2953, 2956, 2964, 2971, 2973, 2977, 2979, 2983, 2989, 2991, 2997, 3001, 3004,
3007, 3019, 3024, 3025, 3027, 3028, 3031, 3033, 3037, 3049, 3052, 3061, 3063, 3067, 3072, 3073, 3076, 3079, 3081, 3087, 3088, 3097, 3099, 3100, 3108, 3109,
3117, 3121, 3133, 3136, 3139, 3141, 3148, 3153, 3159, 3163, 3169, 3171, 3172, 3175, 3181, 3184, 3187, 3189, 3193, 3199, 3207, 3211, 3216, 3217, 3225, 3229,
3241, 3244, 3249, 3252, 3253, 3259, 3261, 3267, 3268, 3271, 3276, 3279, 3283, 3292, 3297, 3301, 3303, 3307, 3313, 3316, 3319, 3324, 3325, 3328, 3331, 3343,
3348, 3351, 3357, 3361, 3364, 3367, 3369, 3373, 3376, 3379, 3387, 3388, 3391, 3396, 3397, 3409, 3411, 3412, 3423, 3429, 3433, 3436, 3439, 3441, 3457, 3459,
3463, 3468, 3469, 3472, 3475, 3477, 3481, 3483, 3484, 3492, 3493, 3499, 3504, 3508, 3511, 3513, 3517, 3523, 3529, 3532, 3541, 3547, 3549, 3556, 3559, 3568,
3571, 3573, 3577, 3583, 3589, 3591, 3600, 3601, 3603, 3607, 3612, 3613, 3627, 3628, 3631, 3637, 3639, 3643, 3648, 3661, 3664, 3667, 3673, 3675, 3676, 3679,
3681, 3684, 3691, 3693, 3697, 3700, 3703, 3708, 3709, 3711, 3721, 3724, 3727, 3733, 3739, 3747, 3748, 3751, 3753, 3756, 3757, 3769, 3775, 3781, 3783, 3787,
3789, 3792, 3793, 3796, 3801, 3811, 3819, 3823, 3829, 3837, 3843, 3844, 3847, 3853, 3856, 3868, 3871, 3873, 3877, 3888, 3889, 3891, 3892, 3897, 3900, 3904,
3907, 3909, 3913, 3919, 3924, 3925, 3931, 3937, 3943, 3951, 3952, 3963, 3964, 3967, 3969, 3972, 3981, 3988, 3991, 3996, 3997, 3999, 4003, 4009, 4017, 4021,
4027, 4032, 4033, 4036, 4039, 4044, 4051, 4053, 4057, 4069, 4075, 4077, 4084, 4087, 4093, 4096, 4099, 4107, 4108, 4111, 4113, 4116, 4123, 4129, 4132, 4143,
4144, 4153, 4156, 4159, 4161, 4167, 4171, 4177, 4179, 4188, 4197, 4201, 4204, 4207, 4212, 4219, 4221, 4225, 4228, 4231, 4237, 4239, 4243, 4249, 4251, 4252,
4261, 4269, 4273, 4275, 4276, 4287, 4288, 4291, 4297, 4300, 4303, 4309, 4327, 4329, 4332, 4333, 4336, 4339, 4341, 4348, 4351, 4356, 4357, 4359, 4363, 4368,
4372, 4375, 4377, 4381, 4383, 4396, 4401, 4404, 4413, 4417, 4423, 4429, 4431, 4432, 4441, 4447, 4449, 4453, 4459, 4464, 4467, 4468, 4476, 4477, 4483, 4489,
4491, 4492, 4501, 4503, 4507, 4513, 4516, 4519, 4525, 4528, 4537, 4548, 4549, 4557, 4561, 4563, 4564, 4567, 4572, 4575, 4579, 4588, 4591, 4593, 4597, 4599,
4603, 4612, 4617, 4621, 4624, 4627, 4629, 4636, 4639, 4644, 4647, 4651, 4656, 4657, 4663, 4672, 4681, 4683, 4684, 4687, 4693, 4699, 4701, 4707, 4711, 4719,
4723, 4725, 4729, 4732, 4737, 4753, 4759, 4761, 4764, 4771, 4773, 4783, 4788, 4789, 4791, 4800, 4801, 4804, 4809, 4813, 4816, 4819, 4825, 4827, 4831, 4836,
4837, 4849, 4852, 4861, 4863, 4864, 4867, 4869, 4881, 4887, 4891, 4900, 4903, 4908, 4909, 4912, 4921, 4923, 4924, 4927, 4933, 4941, 4944, 4948, 4951, 4953,
4957, 4963, 4969, 4971, 4975, 4977, 4987, 4989, 4993, 4996, 4999, 5004, 5007, 5008, 5011, 5023, 5025, 5031, 5041, 5043, 5044, 5047, 5052, 5053, 5056, 5059,
5061, 5068, 5077, 5079, 5089, 5092, 5097, 5101, 5103, 5107, 5113, 5116, 5119, 5124, 5131, 5139, 5143, 5149, 5161, 5164, 5167, 5169, 5173, 5179, 5184, 5187,
5188, 5193, 5196, 5197, 5200, 5203, 5209, 5211, 5212, 5223, 5227, 5232, 5233, 5239, 5241, 5257, 5259, 5263, 5268, 5275, 5277, 5281, 5284, 5292, 5293, 5296,
5299, 5301, 5308, 5317, 5323, 5328, 5329, 5331, 5332, 5341, 5347, 5349, 5356, 5367, 5373, 5376, 5377, 5383, 5392, 5403, 5404, 5407, 5409, 5413, 5419, 5421,
5425, 5427, 5431, 5436, 5437, 5439, 5443, 5449, 5461, 5463, 5473, 5475, 5476, 5479, 5484, 5488, 5491, 5493, 5503, 5509, 5517, 5521, 5524, 5527, 5529, 5547,
5548, 5551, 5556, 5557, 5563, 5569, 5571, 5572, 5575, 5581, 5583, 5584, 5587, 5596, 5601, 5611, 5616, 5619, 5623, 5625, 5628, 5629, 5637, 5641, 5647, 5652,
5653, 5659, 5668, 5673, 5677, 5679, 5683, 5689, 5691, 5692, 5697, 5700, 5701, 5707, 5716, 5719, 5725, 5733, 5737, 5743, 5749, 5761, 5767, 5772, 5776, 5779,
5787, 5788, 5791, 5799, 5803, 5808, 5809, 5812, 5817, 5821, 5824, 5827, 5833, 5836, 5839, 5844, 5851, 5853, 5857, 5859, 5868, 5869, 5871, 5872, 5881, 5884,
5887, 5889, 5908, 5913, 5917, 5923, 5925, 5929, 5932, 5941, 5943, 5947, 5949, 5952, 5953, 5956, 5961, 5968, 5971, 5977, 5979, 5983, 5988, 5997, 6004, 6007,
6013, 6019, 6021, 6025, 6031, 6033, 6037, 6043, 6051, 6057, 6064, 6067, 6069, 6073, 6075, 6076, 6079, 6084, 6087, 6091, 6096, 6097, 6100, 6111, 6121, 6123,
6124, 6132, 6133, 6139, 6151, 6156, 6159, 6163, 6169, 6172, 6175, 6181, 6183, 6192, 6196, 6199, 6208, 6211, 6213, 6217, 6219, 6223, 6229, 6231, 6241, 6244,
6247, 6249, 6253, 6267, 6268, 6271, 6276, 6277, 6283, 6289, 6292, 6300, 6301, 6316, 6321, 6327, 6331, 6337, 6339, 6343, 6348, 6349, 6352, 6357, 6361, 6364,
6367, 6373, 6379, 6381, 6384, 6388, 6393, 6397, 6399, 6400, 6403, 6411, 6412, 6421, 6427, 6429, 6433, 6436, 6447, 6448, 6451, 6469, 6475, 6481, 6483, 6484,
6487, 6489, 6492, 6493, 6499, 6507, 6508, 6516, 6517, 6529, 6537, 6541, 6543, 6544, 6547, 6553, 6559, 6561, 6564, 6571, 6573, 6577, 6588, 6591, 6592, 6597,
6604, 6607, 6609, 6619, 6627, 6628, 6631, 6636, 6637, 6643, 6649, 6651, 6652, 6661, 6663, 6669, 6672, 6673, 6676, 6679, 6691, 6697, 6700, 6703, 6708, 6709,
6717, 6724, 6727, 6733, 6736, 6748, 6751, 6753, 6759, 6763, 6769, 6771, 6772, 6775, 6781, 6789, 6793, 6796, 6799, 6804, 6807, 6811, 6813, 6823, 6825, 6829,
6832, 6841, 6843, 6852, 6859, 6861, 6867, 6871, 6877, 6879, 6883, 6889, 6892, 6897, 6901, 6907, 6912, 6913, 6916, 6921, 6924, 6925, 6928, 6933, 6937, 6948,
6949, 6951, 6961, 6964, 6967, 6973, 6975, 6976, 6979, 6988, 6991, 6993, 6997, 7009, 7012, 7023, 7024, 7027, 7033, 7036, 7039, 7041, 7056, 7057, 7059, 7063,
7068, 7069, 7075, 7077, 7081, 7083, 7087, 7099, 7104, 7108, 7111, 7113, 7129, 7131, 7132, 7137, 7141, 7147, 7149, 7156, 7159, 7164, 7167, 7168, 7177, 7189,
7201, 7203, 7204, 7207, 7212, 7213, 7219, 7225, 7228, 7231, 7236, 7237, 7239, 7243, 7248, 7252, 7267, 7273, 7275, 7284, 7297, 7299, 7300, 7303, 7309, 7311,
7312, 7317, 7321, 7324, 7329, 7333, 7347, 7351, 7353, 7356, 7357, 7363, 7369, 7371, 7372, 7381, 7393, 7396, 7399, 7401, 7407, 7408, 7411, 7417, 7419, 7423,
7428, 7437, 7441, 7444, 7459, 7461, 7468, 7471, 7477, 7479, 7483, 7488, 7489, 7492, 7500, 7501, 7504, 7507, 7509, 7516, 7519, 7525, 7527, 7533, 7536, 7537,
7543, 7549, 7561, 7563, 7564, 7569, 7572, 7573, 7581, 7588, 7591, 7596, 7600, 7603, 7609, 7617, 7621, 7623, 7639, 7641, 7644, 7651, 7653, 7657, 7663, 7669,
7671, 7675, 7677, 7681, 7687, 7693, 7696, 7699, 7707, 7716, 7717, 7723, 7725, 7731, 7732, 7741, 7744, 7747, 7753, 7756, 7759, 7761, 7771, 7779, 7783, 7789,
7792, 7803, 7804, 7807, 7812, 7813, 7819, 7824, 7825, 7828, 7833, 7839, 7851, 7852, 7857, 7861, 7867, 7869, 7873, 7879, 7884, 7891, 7893, 7900, 7903, 7921,
7923, 7924, 7927, 7932, 7933, 7936, 7941, 7947, 7948, 7951, 7957, 7959, 7963, 7969, 7972, 7977, 7984, 7987, 7993, 7996, 7999, 8001, 8011, 8013, 8017, 8028,
8029, 8031, 8044, 8047, 8049, 8053, 8059, 8067, 8068, 8071, 8076, 8089, 8092, 8100, 8101, 8103, 8107, 8112, 8113, 8116, 8121, 8125, 8127, 8128, 8137, 8139,
8148, 8157, 8161, 8163, 8164, 8167, 8175, 8176, 8179, 8191, 8193, 8197, 8203, 8208, 8209, 8212, 8221, 8227, 8229, 8233, 8244, 8247, 8251, 8256, 8263, 8269,
8271, 8275, 8281, 8284, 8287, 8289, 8292, 8293, 8299, 8301, 8308, 8311, 8317, 8325, 8329, 8332, 8337, 8341, 8343, 8353, 8356, 8359, 8368, 8373, 8377, 8379,
8389, 8391, 8400, 8401, 8407, 8409, 8419, 8425, 8427, 8428, 8431, 8433, 8436, 8443, 8451, 8452, 8461, 8463, 8464, 8467, 8473, 8476, 8479, 8491, 8499, 8508,
8509, 8512, 8521, 8524, 8527, 8532, 8539, 8541, 8548, 8553, 8557, 8563, 8571, 8572, 8575, 8581, 8587, 8589, 8593, 8596, 8599, 8607, 8611, 8617, 8623, 8629,
8641, 8643, 8644, 8647, 8649, 8652, 8656, 8659, 8661, 8676, 8677, 8683, 8688, 8689, 8697, 8703, 8707, 8713, 8716, 8719, 8724, 8725, 8731, 8737, 8743, 8748,
8749, 8751, 8752, 8757, 8761, 8764, 8769, 8773, 8775, 8779, 8784, 8788, 8796, 8797, 8803, 8812, 8821, 8827, 8829, 8833, 8836, 8839, 8841, 8848, 8859, 8863,
8868, 8869, 8884, 8887, 8892, 8893, 8896, 8911, 8913, 8917, 8919, 8923, 8929, 8931, 8937, 8941, 8944, 8949, 8953, 8956, 8959, 8967, 8971, 8973, 8983, 8991,
9001, 9003, 9004, 9007, 9012, 9013, 9021, 9025, 9028, 9037, 9043, 9049, 9052, 9057, 9067, 9072, 9073, 9075, 9076, 9079, 9081, 9084, 9091, 9093, 9099, 9100,
9103, 9109, 9111, 9121, 9124, 9127, 9133, 9136, 9139, 9147, 9148, 9151, 9156, 9157, 9172, 9175, 9181, 9183, 9187, 9189, 9196, 9199, 9201, 9211, 9216, 9217,
9219, 9228, 9232, 9237, 9241, 9243, 9244, 9247, 9253, 9261, 9264, 9268, 9271, 9277, 9283, 9289, 9291, 9297, 9300, 9313, 9319, 9324, 9325, 9327, 9331, 9337,
9343, 9349, 9351, 9363, 9364, 9373, 9388, 9391, 9397, 9399, 9403, 9408, 9409, 9412, 9417, 9421, 9423, 9424, 9433, 9436, 9439, 9444, 9451, 9457, 9459, 9463,
9472, 9475, 9477, 9481, 9484, 9489, 9507, 9508, 9511, 9513, 9516, 9517, 9525, 9529, 9532, 9543, 9547, 9552, 9556, 9559, 9561, 9567, 9577, 9579, 9583, 9589,
9597, 9601, 9604, 9607, 9613, 9616, 9619, 9621, 9631, 9633, 9643, 9648, 9649, 9651, 9652, 9661, 9664, 9667, 9675, 9679, 9687, 9697, 9700, 9703, 9709, 9712,
9721, 9723, 9732, 9733, 9739, 9747, 9748, 9751, 9756, 9759, 9763, 9769, 9772, 9777, 9781, 9783, 9787, 9793, 9796, 9801, 9804, 9808, 9811, 9813, 9817, 9828,
9829, 9837, 9841, 9847, 9849, 9859, 9868, 9871, 9876, 9883, 9891, 9892, 9901, 9903, 9904, 9907, 9909, 9916, 9919, 9921, 9925, 9931, 9937, 9939, 9943, 9948,
9949, 9957, 9961, 9967, 9972, 9973, 9975, 9984, 9991, 9993, 9997, 10000, 10003, 10009, 10012, 10027, 10029, 10033, 10036, 10039, 10044, 10048, 10051, 10053,
10069, 10071, 10075, 10083, 10084, 10092, 10093, 10096, 10099, 10101, 10107, 10108, 10111, 10117, 10119, 10128, 10129, 10137, 10141, 10147, 10156, 10159,
10161, 10164, 10171, 10173, 10177, 10188, 10191, 10192, 10201, 10204, 10213, 10225, 10227, 10228, 10231, 10233, 10236, 10243, 10249, 10261, 10267, 10269,
10273, 10276, 10279, 10287, 10288, 10297, 10299, 10300, 10303, 10308, 10309, 10317, 10321, 10323, 10333, 10339, 10348, 10357, 10363, 10369, 10371, 10372,
10377, 10381, 10389, 10393, 10399, 10404, 10407, 10416, 10423, 10425, 10429, 10431, 10432, 10443, 10444, 10447, 10449, 10452, 10453, 10459, 10468, 10471,
10476, 10477, 10479, 10492, 10497, 10501, 10507, 10512, 10513, 10519, 10524, 10525, 10531, 10533, 10539, 10543, 10551, 10564, 10567, 10569, 10573, 10576,
10587, 10588, 10596, 10597, 10609, 10612, 10621, 10623, 10627, 10633, 10636, 10639, 10641, 10647, 10651, 10657, 10663, 10668, 10675, 10677, 10684, 10687,
10693, 10699, 10704, 10708, 10711, 10713, 10717, 10719, 10723, 10729, 10731, 10732, 10749, 10753, 10756, 10767, 10768, 10771, 10773, 10777, 10789, 10800,
10801, 10803, 10804, 10809, 10816, 10819, 10821, 10825, 10828, 10831, 10836, 10837, 10839, 10843, 10849, 10852, 10861, 10864, 10867, 10876, 10881, 10884,
10891, 10893, 10900, 10903, 10909, 10911, 10917, 10921, 10924, 10927, 10929, 10933, 10939, 10944, 10957, 10963, 10969, 10972, 10975, 10981, 10983, 10987,
10992, 10993, 10996, 11001, 11008, 11011, 11019, 11023, 11025, 11028, 11037, 11041, 11043, 11047, 11052, 11053, 11056, 11059, 11068, 11071, 11073, 11079,
11083, 11089, 11091, 11100, 11109, 11113, 11116, 11119, 11124, 11127, 11131, 11133, 11137, 11149, 11161, 11163, 11164, 11167, 11172, 11173, 11179, 11181,
11188, 11191, 11197, 11199, 11200, 11212, 11217, 11221, 11227, 11236, 11239, 11241, 11244, 11248, 11251, 11253, 11257, 11259, 11263, 11268, 11271, 11284,
11287, 11299, 11307, 11311, 11317, 11323, 11325, 11329, 11332, 11343, 11344, 11347, 11349, 11353, 11359, 11361, 11367, 11376, 11377, 11379, 11383, 11388,
11389, 11401, 11403, 11404, 11419, 11425, 11428, 11433, 11437, 11443, 11449, 11452, 11457, 11461, 11467, 11469, 11476, 11479, 11487, 11491, 11497, 11503,
11511, 11524, 11527, 11529, 11532, 11533, 11536, 11541, 11548, 11551, 11557, 11559, 11563, 11568, 11575, 11581, 11584, 11587, 11593, 11596, 11599, 11604,
11613, 11617, 11619, 11631, 11632, 11641, 11647, 11653, 11664, 11667, 11668, 11673, 11676, 11677, 11683, 11689, 11691, 11692, 11700, 11701, 11712, 11719,
11721, 11725, 11727, 11728, 11731, 11737, 11739, 11743, 11749, 11757, 11761, 11767, 11772, 11773, 11775, 11779, 11788, 11791, 11793, 11809, 11811, 11812,
11821, 11824, 11827, 11829, 11833, 11839, 11851, 11853, 11856, 11863, 11875, 11881, 11884, 11887, 11889, 11892, 11893, 11899, 11901, 11907, 11908, 11911,
11916, 11923, 11929, 11932, 11941, 11943, 11947, 11953, 11956, 11959, 11964, 11971, 11973, 11988, 11989, 11991, 11997, 12004, 12007, 12009, 12016, 12025,
12027, 12028, 12037, 12043, 12049, 12051, 12061, 12063, 12073, 12076, 12081, 12096, 12097, 12099, 12100, 12103, 12108, 12109, 12112, 12117, 12124, 12127,
12132, 12139, 12148, 12153, 12157, 12159, 12163, 12169, 12171, 12175, 12181, 12187, 12196, 12207, 12208, 12211, 12217, 12225, 12229, 12231, 12241, 12244,
12247, 12252, 12253, 12261, 12268, 12271, 12277, 12279, 12288, 12289, 12292, 12297, 12301, 12304, 12307, 12313, 12316, 12319, 12321, 12324, 12333, 12337,
12339, 12343, 12348, 12352, 12369, 12373, 12379, 12387, 12388, 12391, 12396, 12400, 12403, 12409, 12421, 12427, 12429, 12432, 12433, 12436, 12439, 12451,
12457, 12459, 12463, 12468, 12469, 12475, 12477, 12481, 12483, 12484, 12487, 12493, 12501, 12511, 12513, 12517, 12523, 12531, 12532, 12537, 12541, 12544,
12547, 12553, 12556, 12559, 12564, 12571, 12577, 12583, 12589, 12591, 12592, 12601, 12603, 12607, 12612, 12613, 12619, 12621, 12636, 12637, 12649, 12652,
12657, 12663, 12675, 12676, 12679, 12684, 12688, 12691, 12693, 12697, 12700, 12703, 12711, 12717, 12721, 12724, 12729, 12736, 12739, 12747, 12748, 12753,
12756, 12757, 12763, 12769, 12772, 12775, 12781, 12783, 12787, 12796, 12799, 12807, 12817, 12819, 12823, 12825, 12828, 12829, 12841, 12844, 12853, 12861,
12864, 12868, 12871, 12873, 12877, 12883, 12889, 12891, 12900, 12901, 12907, 12909, 12913, 12916, 12919, 12927, 12931, 12943, 12961, 12964, 12967, 12973,
12976, 12979, 12981, 12987, 12996, 12999, 13003, 13008, 13009, 13012, 13017, 13023, 13027, 13033, 13036, 13044, 13051, 13053, 13063, 13068, 13069, 13071,
13072, 13075, 13077, 13081, 13084, 13089, 13093, 13099, 13104, 13111, 13116, 13117, 13125, 13129, 13131, 13132, 13143, 13147, 13149, 13153, 13159, 13168,
13171, 13177, 13183, 13188, 13189, 13201, 13203, 13204, 13212, 13213, 13219, 13225, 13228, 13237, 13239, 13249, 13251, 13252, 13264, 13267, 13269, 13273,
13276, 13279, 13287, 13291, 13293, 13296, 13297, 13300, 13309, 13312, 13323, 13324, 13327, 13333, 13339, 13341, 13347, 13351, 13357, 13359, 13372, 13377,
13381, 13392, 13399, 13401, 13404, 13411, 13417, 13423, 13428, 13429, 13431, 13441, 13444, 13449, 13456, 13459, 13467, 13468, 13471, 13473, 13476, 13477,
13483, 13492, 13503, 13504, 13507, 13509, 13513, 13516, 13521, 13525, 13531, 13537, 13539, 13548, 13552, 13557, 13564, 13567, 13573, 13575, 13579, 13584,
13588, 13591, 13597, 13603, 13609, 13611, 13627, 13633, 13636, 13644, 13647, 13648, 13657, 13663, 13669, 13671, 13675, 13681, 13683, 13687, 13689, 13692,
13693, 13699, 13701, 13711, 13716, 13723, 13725, 13729, 13732, 13737, 13741, 13744, 13756, 13759, 13764, 13773, 13779, 13789, 13791, 13797, 13801, 13807,
13809, 13813, 13819, 13825, 13828, 13831, 13836, 13843, 13851, 13852, 13863, 13867, 13872, 13873, 13876, 13879, 13881, 13887, 13888, 13897, 13900, 13903,
13908, 13909, 13917, 13921, 13924, 13927, 13932, 13933, 13936, 13941, 13951, 13953, 13963, 13968, 13969, 13971, 13972, 13975, 13989, 13993, 13996, 13999,
14011, 14016, 14023, 14029, 14032, 14041, 14043, 14044, 14049, 14052, 14061, 14068, 14071, 14077, 14079, 14083, 14089, 14092, 14097, 14103, 14107, 14116,
14119, 14121, 14128, 14131, 14133, 14137, 14143, 14149, 14157, 14161, 14164, 14167, 14169, 14173, 14175, 14187, 14188, 14196, 14197, 14203, 14209, 14211,
14221, 14224, 14233, 14236, 14251, 14259, 14269, 14272, 14275, 14277, 14281, 14283, 14284, 14287, 14292, 14293, 14299, 14308, 14313, 14317, 14319, 14323,
14332, 14341, 14347, 14349, 14353, 14356, 14364, 14367, 14371, 14373, 14383, 14389, 14400, 14401, 14403, 14404, 14407, 14412, 14419, 14425, 14427, 14428,
14431, 14437, 14439, 14448, 14449, 14452, 14457, 14461, 14475, 14479, 14481, 14491, 14493, 14497, 14503, 14508, 14511, 14512, 14521, 14524, 14527, 14533,
14539, 14547, 14548, 14551, 14556, 14557, 14563, 14572, 14581, 14583, 14589, 14592, 14593, 14599, 14601, 14607, 14611, 14623, 14629, 14641, 14643, 14644,
14647, 14653, 14656, 14661, 14668, 14673, 14677, 14683, 14689, 14692, 14700, 14701, 14704, 14709, 14713, 14716, 14724, 14725, 14727, 14731, 14736, 14737,
14749, 14763, 14764, 14767, 14769, 14772, 14779, 14781, 14788, 14791, 14797, 14799, 14800, 14812, 14821, 14823, 14827, 14832, 14833, 14836, 14844, 14851,
14853, 14859, 14869, 14871, 14884, 14887, 14889, 14896, 14907, 14908, 14911, 14913, 14917, 14923, 14925, 14929, 14931, 14932, 14941, 14947, 14953, 14956,
14959, 14961, 14967, 14979, 14983, 14988, 14989, 14992, 14997, 15001, 15004, 15007, 15012, 15013, 15021, 15024, 15025, 15028, 15031, 15033, 15043, 15061,
15067, 15069, 15073, 15075, 15076, 15091, 15093, 15097, 15100, 15121, 15123, 15124, 15127, 15129, 15132, 15133, 15139, 15141, 15148, 15151, 15156, 15159,
15168, 15172, 15175, 15177, 15183, 15184, 15187, 15193, 15199, 15204, 15217, 15223, 15229, 15231, 15237, 15241, 15244, 15247, 15253, 15259, 15267, 15271,
15276, 15277, 15289, 15291, 15292, 15303, 15307, 15309, 15313, 15316, 15319, 15321, 15325, 15331, 15337, 15339, 15343, 15348, 15349, 15357, 15361, 15367,
15372, 15373, 15376, 15379, 15388, 15391, 15393, 15403, 15409, 15412, 15417, 15421, 15424, 15427, 15429, 15439, 15447, 15451, 15463, 15469, 15472, 15475,
15483, 15484, 15492, 15493, 15501, 15507, 15508, 15511, 15519, 15523, 15537, 15541, 15547, 15552, 15553, 15556, 15559, 15561, 15564, 15568, 15577, 15579,
15583, 15588, 15591, 15600, 15601, 15607, 15609, 15613, 15616, 15619, 15625, 15627, 15628, 15633, 15636, 15637, 15643, 15649, 15652, 15661, 15667, 15669,
15673, 15676, 15679, 15681, 15696, 15699, 15700, 15717, 15721, 15723, 15724, 15727, 15733, 15739, 15748, 15751, 15757, 15769, 15771, 15772, 15775, 15777,
15781, 15787, 15789, 15799, 15804, 15808, 15811, 15817, 15823, 15825, 15831, 15841, 15843, 15852, 15856, 15859, 15868, 15876, 15877, 15879, 15883, 15888,
15889, 15897, 15901, 15903, 15907, 15913, 15919, 15924, 15925, 15937, 15951, 15952, 15964, 15967, 15969, 15973, 15979, 15984, 15987, 15988, 15991, 15993,
15996, 16003, 16009, 16012, 16021, 16023, 16033, 16036, 16039, 16041, 16047, 16051, 16057, 16063, 16068, 16069, 16075, 16081, 16084, 16087, 16093, 16101,
16108, 16111, 16119, 16128, 16129, 16131, 16132, 16141, 16144, 16147, 16149, 16156, 16171, 16176, 16177, 16183, 16189, 16204, 16207, 16209, 16212, 16213,
16219, 16221, 16227, 16228, 16231, 16237, 16239, 16243, 16249, 16257, 16263, 16267, 16273, 16275, 16276, 16279, 16281, 16293, 16297, 16300, 16308, 16311,
16317, 16321, 16329, 16333, 16336, 16339, 16347, 16348, 16363, 16369, 16372, 16381, 16383, 16384, 16387, 16389, 16393, 16396, 16399, 16411, 16417, 16419,
16425, 16428, 16429, 16432, 16437, 16444, 16447, 16452, 16453, 16459, 16464, 16471, 16473, 16477, 16479, 16492, 16509, 16513, 16516, 16519, 16525, 16527,
16528, 16531, 16549, 16551, 16561, 16563, 16567, 16572, 16573, 16576, 16581, 16587, 16597, 16603, 16612, 16624, 16627, 16633, 16636, 16639, 16641, 16644,
16651, 16653, 16657, 16663, 16668, 16669, 16671, 16681, 16684, 16689, 16693, 16699, 16707, 16708, 16713, 16716, 16717, 16723, 16725, 16729, 16741, 16743,
16747, 16749, 16752, 16759, 16761, 16771, 16777, 16783, 16788, 16789, 16803, 16804, 16807, 16816, 16819, 16825, 16828, 16831, 16833, 16843, 16848, 16857,
16861, 16869, 16875, 16876, 16879, 16884, 16887, 16891, 16897, 16900, 16903, 16909, 16911, 16912, 16921, 16923, 16924, 16927, 16939, 16941, 16948, 16956,
16957, 16959, 16963, 16972, 16975, 16977, 16981, 16987, 16993, 16996, 17004, 17008, 17011, 17019, 17029, 17031, 17037, 17041, 17044, 17047, 17049, 17053,
17059, 17067, 17071, 17073, 17076, 17077, 17091, 17092, 17100, 17101, 17103, 17104, 17107, 17113, 17121, 17131, 17137, 17143, 17148, 17152, 17157, 17161,
17164, 17167, 17173, 17175, 17188, 17191, 17199, 17200, 17203, 17209, 17211, 17212, 17229, 17233, 17236, 17239, 17247, 17251, 17257, 17263, 17269, 17275,
17283, 17293, 17299, 17301, 17308, 17311, 17316, 17317, 17328, 17329, 17332, 17337, 17341, 17344, 17353, 17356, 17359, 17361, 17364, 17373, 17377, 17383,
17389, 17392, 17397, 17401, 17404, 17407, 17409, 17419, 17424, 17427, 17428, 17431, 17436, 17443, 17449, 17451, 17452, 17461, 17463, 17467, 17472, 17481,
17488, 17491, 17497, 17499, 17500, 17508, 17509, 17517, 17521, 17524, 17532, 17539, 17551, 17553, 17557, 17559, 17563, 17569, 17571, 17575, 17577, 17581,
17584, 17587, 17593, 17599, 17604, 17607, 17613, 17616, 17617, 17623, 17629, 17643, 17647, 17652, 17653, 17659, 17661, 17667, 17668, 17683, 17689, 17692,
17701, 17707, 17713, 17716, 17724, 17725, 17728, 17737, 17739, 17749, 17751, 17761, 17764, 17767, 17769, 17773, 17775, 17787, 17788, 17791, 17796, 17797,
17803, 17812, 17823, 17827, 17829, 17836, 17839, 17841, 17847, 17851, 17856, 17857, 17859, 17863, 17868, 17872, 17881, 17883, 17887, 17899, 17904, 17908,
17911, 17913, 17923, 17929, 17931, 17932, 17937, 17949, 17953, 17956, 17959, 17964, 17968, 17971, 17977, 17983, 17989, 17991, 18004, 18012, 18013, 18019,
18021, 18025, 18028, 18031, 18039, 18043, 18049, 18052, 18057, 18061, 18063, 18064, 18075, 18076, 18091, 18093, 18097, 18099, 18100, 18103, 18109, 18111,
18112, 18121, 18127, 18129, 18133, 18148, 18151, 18153, 18157, 18169, 18171, 18175, 18181, 18187, 18192, 18196, 18199, 18201, 18207, 18211, 18217, 18219,
18223, 18225, 18228, 18229, 18237, 18244, 18252, 18253, 18256, 18259, 18261, 18268, 18271, 18273, 18277, 18288, 18289, 18291, 18300, 18301, 18307, 18313,
18316, 18319, 18325, 18333, 18352, 18361, 18363, 18364, 18367, 18369, 18372, 18373, 18379, 18388, 18396, 18397, 18399, 18412, 18417, 18421, 18427, 18433,
18439, 18448, 18451, 18453, 18457, 18463, 18468, 18475, 18477, 18481, 18484, 18487, 18489, 18493, 18496, 18499, 18507, 18508, 18516, 18517, 18523, 18525,
18529, 18541, 18543, 18544, 18549, 18553, 18556, 18559, 18571, 18576, 18577, 18583, 18588, 18597, 18604, 18613, 18619, 18624, 18628, 18631, 18633, 18637,
18639, 18643, 18651, 18652, 18657, 18661, 18669, 18679, 18687, 18688, 18691, 18693, 18697, 18721, 18723, 18724, 18727, 18732, 18736, 18739, 18741, 18747,
18748, 18757, 18759, 18769, 18772, 18775, 18781, 18787, 18793, 18796, 18801, 18804, 18811, 18813, 18817, 18823, 18828, 18829, 18831, 18844, 18849, 18859,
18867, 18876, 18877, 18889, 18892, 18900, 18903, 18907, 18913, 18916, 18919, 18925, 18928, 18943, 18948, 18949, 18961, 18963, 18967, 18973, 18979, 18981,
18991, 18993, 18997, 19003, 19009, 19011, 19012, 19017, 19029, 19033, 19036, 19039, 19044, 19047, 19051, 19056, 19069, 19071, 19075, 19081, 19083, 19084,
19087, 19092, 19093, 19101, 19117, 19119, 19123, 19132, 19137, 19141, 19143, 19152, 19156, 19164, 19171, 19177, 19179, 19183, 19189, 19191, 19197, 19200,
19201, 19204, 19207, 19209, 19213, 19216, 19219, 19225, 19231, 19233, 19236, 19237, 19243, 19249, 19252, 19263, 19264, 19267, 19273, 19276, 19279, 19281,
19287, 19299, 19300, 19303, 19308, 19309, 19321, 19324, 19333, 19341, 19344, 19348, 19351, 19353, 19357, 19363, 19369, 19375, 19381, 19387, 19396, 19399,
19407, 19408, 19417, 19423, 19425, 19429, 19441, 19443, 19444, 19447, 19449, 19452, 19453, 19456, 19461, 19467, 19468, 19471, 19476, 19477, 19479, 19483,
19489, 19497, 19501, 19507, 19513, 19521, 19524, 19531, 19537, 19543, 19548, 19551, 19561, 19564, 19573, 19579, 19587, 19597, 19600, 19603, 19609, 19611,
19612, 19621, 19623, 19627, 19629, 19632, 19636, 19641, 19648, 19651, 19659, 19663, 19675, 19677, 19681, 19683, 19684, 19687, 19692, 19696, 19699, 19708,
19713, 19717, 19719, 19723, 19729, 19731, 19732, 19741, 19747, 19753, 19759, 19764, 19773, 19776, 19777, 19783, 19791, 19792, 19801, 19804, 19812, 19813,
19819, 19821, 19825, 19827, 19828, 19831, 19843, 19852, 19857, 19861, 19867, 19876, 19879, 19881, 19884, 19891, 19893, 19900, 19903, 19908, 19909, 19911,
19927, 19929, 19933, 19939, 19947, 19948, 19953, 19956, 19957, 19963, 19969, 19972, 19983, 19984, 19989, 19993, 19996, 19999, 20007, 20011, 20016, 20017,
20019, 20023, 20028, 20029, 20032, 20037, 20041, 20044, 20047, 20059, 20071, 20073, 20083, 20089, 20091, 20092, 20100, 20101, 20107, 20109, 20113, 20124,
20127, 20137, 20143, 20149, 20151, 20161, 20164, 20167, 20172, 20173, 20176, 20181, 20188, 20191, 20197, 20199, 20208, 20209, 20212, 20221, 20224, 20233,
20236, 20239, 20244, 20253, 20259, 20269, 20272, 20275, 20277, 20287, 20289, 20293, 20307, 20308, 20311, 20313, 20316, 20323, 20325, 20341, 20343, 20347,
20353, 20356, 20359, 20367, 20368, 20371, 20379, 20388, 20389, 20397, 20404, 20407, 20412, 20419, 20421, 20425, 20428, 20431, 20433, 20439, 20443, 20449,
20452, 20461, 20464, 20467, 20469, 20475, 20476, 20479, 20487, 20491, 20496, 20497, 20509, 20521, 20523, 20524, 20527, 20529, 20533, 20551, 20556, 20557,
20563, 20569, 20572, 20575, 20577, 20583, 20593, 20596, 20599, 20601, 20611, 20613, 20629, 20631, 20637, 20641, 20644, 20649, 20653, 20656, 20659, 20667,
20668, 20671, 20676, 20683, 20691, 20692, 20701, 20703, 20707, 20716, 20719, 20721, 20725, 20731, 20736, 20739, 20743, 20748, 20749, 20752, 20761, 20763,
20767, 20772, 20773, 20775, 20784, 20788, 20797, 20799, 20800, 20809, 20811, 20812, 20836, 20839, 20844, 20847, 20848, 20853, 20857, 20863, 20881, 20883,
20887, 20892, 20899, 20901, 20908, 20917, 20919, 20923, 20925, 20928, 20929, 20932, 20937, 20941, 20947, 20956, 20959, 20964, 20971, 20973, 20979, 20983,
20989, 20991, 21001, 21007, 21013, 21019, 21025, 21027, 21028, 21031, 21036, 21037, 21049, 21052, 21061, 21067, 21069, 21072, 21073, 21081, 21097, 21099,
21100, 21108, 21117, 21121, 21123, 21124, 21127, 21133, 21136, 21139, 21151, 21157, 21163, 21168, 21169, 21171, 21172, 21175, 21177, 21184, 21187, 21189,
21193, 21196, 21204, 21207, 21211, 21217, 21223, 21225, 21231, 21232, 21243, 21247, 21249, 21259, 21261, 21268, 21277, 21283, 21289, 21292, 21297, 21312,
21313, 21316, 21319, 21324, 21325, 21328, 21333, 21337, 21339, 21343, 21349, 21364, 21379, 21387, 21388, 21391, 21393, 21396, 21397, 21409, 21411, 21421,
21423, 21424, 21427, 21433, 21441, 21447, 21451, 21457, 21463, 21468, 21469, 21475, 21477, 21481, 21487, 21492, 21493, 21499, 21501, 21504, 21508, 21511,
21517, 21523, 21529, 21531, 21532, 21541, 21553, 21559, 21567, 21568, 21577, 21589, 21601, 21603, 21609, 21612, 21613, 21616, 21619, 21621, 21628, 21631,
21636, 21639, 21649, 21652, 21657, 21661, 21673, 21675, 21676, 21679, 21684, 21691, 21693, 21697, 21700, 21708, 21711, 21717, 21724, 21727, 21729, 21733,
21739, 21744, 21748, 21751, 21756, 21757, 21763, 21772, 21775, 21787, 21793, 21796, 21799, 21801, 21817, 21819, 21823, 21825, 21841, 21844, 21847, 21852,
21853, 21859, 21871, 21883, 21891, 21892, 21897, 21900, 21901, 21904, 21907, 21909, 21916, 21925, 21927, 21931, 21933, 21936, 21937, 21943, 21951, 21952,
21961, 21963, 21964, 21972, 21973, 21979, 21987, 21991, 21997, 21999, 22003, 22009, 22012, 22021, 22027, 22036, 22039, 22041, 22051, 22053, 22059, 22063,
22068, 22071, 22075, 22084, 22087, 22089, 22093, 22096, 22107, 22108, 22111, 22113, 22116, 22123, 22129, 22141, 22143, 22147, 22153, 22159, 22171, 22177,
22179, 22183, 22188, 22189, 22192, 22197, 22201, 22203, 22204, 22213, 22221, 22224, 22225, 22228, 22233, 22237, 22249, 22251, 22252, 22257, 22267, 22269,
22273, 22276, 22279, 22284, 22288, 22291, 22300, 22303, 22309, 22311, 22323, 22324, 22332, 22336, 22348, 22351, 22357, 22369, 22377, 22381, 22383, 22384,
22387, 22393, 22399, 22404, 22411, 22413, 22431, 22437, 22441, 22444, 22447, 22449, 22453, 22459, 22464, 22467, 22476, 22477, 22483, 22489, 22492, 22500,
22501, 22503, 22512, 22516, 22519, 22521, 22527, 22531, 22537, 22543, 22548, 22549, 22557, 22564, 22567, 22573, 22575, 22579, 22581, 22588, 22599, 22603,
22608, 22611, 22612, 22621, 22629, 22633, 22636, 22639, 22647, 22651, 22657, 22669, 22672, 22675, 22681, 22683, 22687, 22689, 22692, 22699, 22707, 22708,
22711, 22716, 22717, 22719, 22723, 22732, 22741, 22743, 22747, 22753, 22756, 22764, 22768, 22771, 22773, 22777, 22783, 22788, 22789, 22800, 22801, 22804,
22807, 22809, 22813, 22819, 22827, 22828, 22831, 22849, 22851, 22861, 22863, 22864, 22867, 22869, 22876, 22897, 22900, 22903, 22909, 22917, 22921, 22923,
22932, 22948, 22953, 22959, 22963, 22969, 22971, 22972, 22975, 22981, 22987, 22989, 22993, 22996, 22999, 23007, 23011, 23013, 23017, 23025, 23029, 23031,
23041, 23043, 23044, 23047, 23053, 23059, 23061, 23068, 23071, 23079, 23088, 23097, 23101, 23104, 23107, 23116, 23119, 23121, 23125, 23131, 23143, 23148,
23149, 23151, 23152, 23164, 23167, 23169, 23173, 23175, 23179, 23191, 23193, 23196, 23197, 23203, 23209, 23212, 23223, 23227, 23232, 23233, 23236, 23241,
23248, 23251, 23257, 23259, 23263, 23268, 23269, 23275, 23277, 23281, 23283, 23284, 23293, 23296, 23308, 23311, 23313, 23317, 23332, 23337, 23344, 23347,
23349, 23353, 23356, 23367, 23371, 23376, 23377, 23383, 23389, 23401, 23404, 23409, 23412, 23413, 23421, 23425, 23428, 23431, 23436, 23439, 23457, 23467,
23472, 23473, 23475, 23476, 23484, 23488, 23491, 23497, 23499, 23503, 23509, 23517, 23521, 23524, 23527, 23536, 23539, 23548, 23553, 23556, 23557, 23563,
23569, 23571, 23581, 23583, 23587, 23593, 23599, 23601, 23607, 23611, 23619, 23623, 23629, 23632, 23637, 23652, 23653, 23668, 23671, 23673, 23677, 23679,
23689, 23692, 23700, 23707, 23709, 23716, 23719, 23725, 23728, 23731, 23737, 23743, 23761, 23763, 23764, 23767, 23769, 23772, 23773, 23779, 23781, 23788,
23791, 23796, 23799, 23803, 23808, 23812, 23823, 23824, 23827, 23833, 23839, 23841, 23844, 23853, 23857, 23863, 23869, 23871, 23872, 23877, 23884, 23887,
23889, 23893, 23899, 23907, 23908, 23911, 23916, 23917, 23929, 23931, 23932, 23952, 23959, 23961, 23971, 23977, 23979, 23988, 23997, 24001, 24003, 24007,
24016, 24019, 24025, 24028, 24031, 24033, 24037, 24039, 24043, 24049, 24051, 24052, 24061, 24073, 24076, 24079, 24084, 24087, 24091, 24093, 24097, 24100,
24103, 24109, 24121, 24124, 24132, 24133, 24141, 24147, 24148, 24151, 24159, 24163, 24169, 24172, 24175, 24177, 24181, 24187, 24193, 24199, 24201, 24204,
24213, 24217, 24223, 24228, 24229, 24241, 24247, 24253, 24256, 24267, 24268, 24271, 24276, 24283, 24292, 24300, 24301, 24303, 24304, 24307, 24309, 24316,
24321, 24325, 24336, 24337, 24339, 24348, 24349, 24363, 24364, 24367, 24373, 24375, 24379, 24381, 24384, 24388, 24391, 24397, 24400, 24411, 24417, 24421,
24427, 24439, 24444, 24451, 24457, 24469, 24471, 24481, 24483, 24484, 24489, 24492, 24493, 24496, 24499, 24501, 24511, 24517, 24525, 24528, 24529, 24532,
24537, 24547, 24553, 24556, 24571, 24573, 24577, 24579, 24583, 24589, 24591, 24601, 24604, 24609, 24613, 24619, 24624, 24627, 24631, 24636, 24643, 24649,
24652, 24661, 24663, 24676, 24681, 24687, 24688, 24691, 24697, 24699, 24700, 24703, 24709, 24724, 24727, 24732, 24733, 24741, 24753, 24757, 24763, 24768,
24775, 24781, 24784, 24787, 24789, 24793, 24796, 24799, 24807, 24811, 24813, 24823, 24825, 24829, 24832, 24841, 24843, 24844, 24847, 24852, 24859, 24861,
24867, 24868, 24876, 24877, 24879, 24889, 24892, 24897, 24901, 24903, 24907, 24913, 24916, 24919, 24924, 24925, 24933, 24943, 24949, 24951, 24961, 24964,
24967, 24975, 24976, 24979, 24987, 24988, 24991, 24996, 24997, 25011, 25012, 25023, 25029, 25033, 25039, 25057, 25059, 25068, 25072, 25077, 25081, 25084,
25087, 25099, 25104, 25108, 25111, 25117, 25119, 25123, 25129, 25131, 25132, 25137, 25141, 25147, 25153, 25156, 25159, 25167, 25168, 25171, 25173, 25183,
25189, 25200, 25203, 25204, 25207, 25213, 25219, 25221, 25225, 25227, 25237, 25243, 25249, 25257, 25261, 25264, 25273, 25275, 25281, 25284, 25291, 25293,
25299, 25303, 25308, 25309, 25321, 25324, 25327, 25329, 25339, 25348, 25353, 25356, 25357, 25363, 25372, 25383, 25389, 25392, 25393, 25396, 25401, 25408,
25411, 25417, 25419, 25423, 25428, 25437, 25441, 25444, 25447, 25453, 25456, 25459, 25468, 25471, 25473, 25477, 25492, 25497, 25501, 25513, 25516, 25519,
25524, 25525, 25527, 25531, 25536, 25537, 25552, 25561, 25563, 25567, 25572, 25579, 25581, 25588, 25591, 25596, 25600, 25603, 25609, 25612, 25617, 25621,
25623, 25627, 25633, 25639, 25644, 25648, 25657, 25659, 25669, 25671, 25675, 25681, 25684, 25689, 25693, 25699, 25708, 25711, 25713, 25716, 25717, 25725,
25732, 25741, 25743, 25744, 25747, 25753, 25759, 25761, 25767, 25771, 25779, 25788, 25792, 25797, 25801, 25804, 25819, 25821, 25825, 25831, 25833, 25837,
25843, 25849, 25851, 25867, 25869, 25873, 25876, 25879, 25887, 25900, 25903, 25909, 25921, 25923, 25924, 25929, 25932, 25933, 25936, 25939, 25941, 25947,
25948, 25951, 25956, 25963, 25968, 25969, 25972, 25975, 25977, 25981, 25983, 25987, 25996, 25999, 26011, 26017, 26028, 26029, 26031, 26032, 26041, 26047,
26049, 26053, 26064, 26067, 26068, 26071, 26083, 26089, 26091, 26101, 26107, 26109, 26113, 26116, 26119, 26121, 26131, 26139, 26143, 26148, 26149, 26157,
26161, 26164, 26172, 26173, 26175, 26176, 26188, 26193, 26203, 26209, 26211, 26212, 26221, 26227, 26229, 26233, 26236, 26239, 26244, 26247, 26251, 26253,
26256, 26257, 26263, 26269, 26271, 26275, 26283, 26284, 26287, 26292, 26293, 26299, 26307, 26308, 26317, 26319, 26325, 26337, 26347, 26352, 26353, 26359,
26364, 26368, 26371, 26377, 26383, 26388, 26391, 26407, 26409, 26413, 26416, 26425, 26428, 26431, 26436, 26437, 26443, 26449, 26463, 26467, 26476, 26479,
26481, 26487, 26497, 26499, 26508, 26509, 26512, 26517, 26523, 26524, 26533, 26539, 26544, 26548, 26551, 26557, 26569, 26572, 26575, 26577, 26581, 26589,
26596, 26599, 26604, 26607, 26608, 26617, 26623, 26629, 26641, 26644, 26647, 26652, 26661, 26676, 26677, 26679, 26683, 26688, 26689, 26692, 26701, 26704,
26713, 26716, 26725, 26731, 26733, 26737, 26739, 26751, 26757, 26761, 26764, 26769, 26779, 26787, 26788, 26791, 26793, 26797, 26800, 26803, 26811, 26812,
26821, 26823, 26827, 26832, 26833, 26836, 26839, 26847, 26859, 26863, 26868, 26869, 26875, 26877, 26881, 26893, 26896, 26899, 26901, 26908, 26913, 26919,
26923, 26929, 26932, 26944, 26947, 26949, 26953, 26959, 26971, 26973, 26983, 26992, 27001, 27003, 27004, 27009, 27012, 27021, 27031, 27036, 27037, 27039,
27043, 27052, 27061, 27063, 27067, 27073, 27075, 27076, 27079, 27084, 27088, 27091, 27097, 27100, 27103, 27109, 27111, 27121, 27124, 27127, 27129, 27133,
27139, 27147, 27151, 27156, 27157, 27171, 27172, 27175, 27184, 27187, 27196, 27201, 27211, 27216, 27219, 27223, 27225, 27228, 27229, 27237, 27241, 27243,
27244, 27252, 27253, 27259, 27271, 27273, 27277, 27279, 27283, 27292, 27297, 27300, 27309, 27316, 27325, 27327, 27328, 27331, 27333, 27337, 27343, 27349,
27361, 27363, 27364, 27367, 27372, 27373, 27381, 27391, 27397, 27399, 27403, 27408, 27409, 27417, 27427, 27433, 27436, 27441, 27444, 27451, 27453, 27457,
27468, 27469, 27471, 27475, 27481, 27484, 27487, 27493, 27508, 27516, 27517, 27525, 27529, 27532, 27541, 27543, 27547, 27549, 27556, 27559, 27561], dtype=np.int32)

triangular_C1s = np.array([1, 7, 13, 19, 31, 37, 43, 55, 61, 73, 85, 91, 97, 109, 121, 127, 139, 151, 163, 169, 187, 199, 211, 223, 235, 241, 253, 265, 271, 283, 295, 301, 313, 337, 349,
361, 367, 379, 385, 397, 409, 421, 433, 439, 451, 463, 475, 499, 511, 517, 535, 547, 559, 571, 583, 595, 613, 625, 637, 649, 661, 673, 685, 691, 703, 721,
733, 745, 757, 769, 793, 805, 817, 823, 835, 847, 859, 871, 877, 889, 913, 925, 931, 955, 967, 979, 1003, 1015, 1027, 1039, 1045, 1057, 1069, 1075, 1099,
1111, 1123, 1135, 1147, 1159, 1165, 1177, 1189, 1201, 1213, 1225, 1237, 1261, 1273, 1285, 1303, 1309, 1333, 1345, 1357, 1369, 1381, 1393, 1405, 1417, 1429,
1453, 1459, 1483, 1495, 1507, 1519, 1531, 1555, 1561, 1573, 1585, 1597, 1615, 1627, 1639, 1651, 1663, 1675, 1687, 1711, 1723, 1735, 1759, 1765, 1777, 1789,
1801, 1813, 1831, 1843, 1867, 1879, 1891, 1903, 1915, 1921, 1945, 1957, 1969, 1981, 1993, 2017, 2029, 2053, 2065, 2077, 2083, 2095, 2107, 2125, 2149, 2161,
2173, 2185, 2197, 2209, 2221, 2233, 2245, 2257, 2263, 2275, 2287, 2299, 2335, 2347, 2371, 2383, 2395, 2407, 2419, 2431, 2437, 2455, 2479, 2491, 2503, 2515,
2527, 2539, 2563, 2575, 2587, 2611, 2623, 2635, 2647, 2653, 2665, 2677, 2689, 2713, 2725, 2737, 2749, 2773, 2779, 2791, 2803, 2815, 2839, 2857, 2869, 2893,
2905, 2917, 2929, 2941, 2965, 2989, 3001, 3013, 3025, 3037, 3049, 3055, 3067, 3079, 3091, 3103, 3115, 3121, 3145, 3169, 3181, 3193, 3205, 3217, 3241, 3253,
3259, 3283, 3295, 3307, 3319, 3331, 3343, 3355, 3367, 3403, 3415, 3427, 3439, 3463, 3481, 3493, 3505, 3511, 3535, 3547, 3559, 3571, 3595, 3607, 3619, 3631,
3643, 3655, 3667, 3679, 3691, 3697, 3721, 3745, 3757, 3781, 3793, 3805, 3817, 3829, 3853, 3865, 3877, 3889, 3901, 3919, 3931, 3943, 3949, 3973, 3985, 4009,
4021, 4033, 4045, 4057, 4069, 4081, 4093, 4105, 4117, 4141, 4153, 4177, 4189, 4195, 4219, 4231, 4243, 4255, 4267, 4303, 4315, 4339, 4345, 4357, 4381, 4405,
4417, 4429, 4447, 4459, 4471, 4483, 4495, 4507, 4519, 4531, 4543, 4567, 4579, 4591, 4615, 4639, 4651, 4675, 4687, 4693, 4705, 4717, 4729, 4741, 4753, 4765,
4777, 4795, 4807, 4819, 4831, 4855, 4879, 4891, 4903, 4927, 4939, 4957, 4969, 4993, 5005, 5029, 5041, 5065, 5077, 5089, 5101, 5125, 5137, 5161, 5173, 5185,
5197, 5221, 5239, 5251, 5257, 5269, 5293, 5305, 5317, 5329, 5341, 5353, 5377, 5389, 5401, 5413, 5425, 5437, 5461, 5473, 5509, 5527, 5539, 5551, 5563, 5587,
5599, 5611, 5623, 5635, 5647, 5671, 5683, 5695, 5707, 5719, 5731, 5737, 5749, 5773, 5797, 5809, 5815, 5839, 5851, 5875, 5887, 5899, 5911, 5923, 5935, 5947,
5959, 5971, 5995, 6007, 6031, 6043, 6055, 6067, 6079, 6103, 6109, 6121, 6145, 6157, 6169, 6181, 6205, 6217, 6229, 6235, 6283, 6295, 6307, 6319, 6331, 6343,
6355, 6367, 6379, 6391, 6409, 6433, 6445, 6457, 6469, 6481, 6493, 6505, 6517, 6529, 6553, 6565, 6577, 6613, 6625, 6637, 6649, 6661, 6673, 6697, 6715, 6727,
6739, 6751, 6763, 6775, 6787, 6793, 6817, 6829, 6841, 6865, 6877, 6901, 6913, 6925, 6961, 6985, 6997, 7009, 7015, 7039, 7051, 7063, 7087, 7099, 7123, 7147,
7159, 7171, 7195, 7207, 7219, 7231, 7243, 7255, 7267, 7279, 7291, 7303, 7315, 7327, 7333, 7351, 7363, 7375, 7399, 7423, 7447, 7459, 7471, 7483, 7495, 7519,
7531, 7555, 7567, 7579, 7591, 7603, 7639, 7663, 7675, 7681, 7705, 7717, 7741, 7753, 7765, 7777, 7789, 7813, 7825, 7849, 7861, 7873, 7885, 7897, 7909, 7915,
7927, 7951, 7963, 7987, 7999, 8011, 8017, 8041, 8053, 8065, 8089, 8101, 8125, 8137, 8149, 8161, 8185, 8209, 8221, 8233, 8245, 8269, 8281, 8293, 8305, 8329,
8341, 8353, 8359, 8371, 8383, 8395, 8407, 8431, 8443, 8467, 8479, 8491, 8509, 8533, 8557, 8581, 8593, 8605, 8617, 8629, 8653, 8665, 8677, 8689, 8719, 8731,
8743, 8767, 8779, 8791, 8803, 8815, 8827, 8839, 8863, 8887, 8911, 8923, 8947, 8959, 8971, 8983, 8995, 9019, 9031, 9043, 9055, 9061, 9073, 9097, 9109, 9121,
9133, 9139, 9151, 9187, 9199, 9211, 9223, 9235, 9271, 9283, 9295, 9307, 9331, 9343, 9355, 9367, 9391, 9403, 9409, 9433, 9445, 9469, 9493, 9505, 9517, 9541,
9553, 9565, 9589, 9601, 9613, 9625, 9649, 9661, 9685, 9697, 9709, 9721, 9733, 9745, 9757, 9763, 9787, 9805, 9817, 9841, 9853, 9877, 9889, 9901, 9913, 9925,
9937, 9961, 9973, 9985, 9997, 10009, 10021, 10033, 10045, 10057, 10081, 10093, 10105, 10141, 10153, 10165, 10177, 10183, 10195, 10219, 10231, 10279, 10291,
10303, 10315, 10339, 10351, 10363, 10387, 10411, 10435, 10453, 10477, 10489, 10501, 10513, 10537, 10549, 10561, 10567, 10579, 10603, 10627, 10639, 10651,
10663, 10675, 10699, 10711, 10723, 10747, 10759, 10771, 10795, 10807, 10831, 10867, 10879, 10891, 10903, 10915, 10939, 10951, 10963, 10969, 10981, 10993,
11017, 11029, 11041, 11053, 11077, 11089, 11101, 11113, 11119, 11143, 11155, 11167, 11191, 11215, 11227, 11251, 11263, 11275, 11299, 11311, 11323, 11335,
11359, 11377, 11401, 11413, 11425, 11437, 11449, 11461, 11473, 11497, 11521, 11533, 11545, 11557, 11569, 11581, 11605, 11629, 11641, 11677, 11689, 11701,
11713, 11725, 11749, 11761, 11779, 11791, 11803, 11815, 11827, 11833, 11857, 11869, 11893, 11905, 11941, 11953, 11977, 11989, 12001, 12013, 12025, 12037,
12049, 12061, 12085, 12097, 12109, 12121, 12133, 12145, 12157, 12169, 12175, 12223, 12235, 12247, 12259, 12283, 12295, 12307, 12319, 12331, 12355, 12379,
12391, 12403, 12427, 12439, 12451, 12463, 12487, 12511, 12523, 12535, 12547, 12553, 12565, 12589, 12601, 12625, 12631, 12643, 12667, 12679, 12703, 12715,
12727, 12739, 12751, 12763, 12775, 12799, 12811, 12823, 12835, 12847, 12883, 12907, 12919, 12931, 12943, 12955, 12991, 13003, 13027, 13051, 13057, 13081,
13093, 13105, 13129, 13141, 13165, 13177, 13189, 13201, 13213, 13225, 13237, 13261, 13273, 13297, 13309, 13327, 13339, 13363, 13375, 13387, 13399, 13411,
13423, 13435, 13447, 13459, 13471, 13483, 13501, 13537, 13549, 13561, 13573, 13585, 13597, 13609, 13621, 13633, 13645, 13657, 13669, 13693, 13717, 13741,
13753, 13765, 13777, 13801, 13825, 13849, 13873, 13885, 13909, 13921, 13945, 13963, 13975, 13987, 13999, 14011, 14047, 14059, 14071, 14077, 14089, 14101,
14125, 14137, 14149, 14161, 14173, 14185, 14233, 14245, 14257, 14269, 14281, 14305, 14317, 14329, 14353, 14365, 14377, 14389, 14407, 14419, 14431, 14443,
14467, 14479, 14503, 14527, 14539, 14563, 14587, 14599, 14611, 14623, 14647, 14659, 14683, 14695, 14707, 14731, 14743, 14767, 14779, 14791, 14803, 14827,
14839, 14845, 14857, 14875, 14899, 14911, 14923, 14947, 14995, 15007, 15019, 15031, 15055, 15067, 15079, 15091, 15115, 15127, 15151, 15163, 15187, 15199,
15211, 15223, 15235, 15259, 15271, 15283, 15307, 15325, 15349, 15361, 15385, 15397, 15409, 15433, 15457, 15469, 15481, 15493, 15505, 15517, 15529, 15541,
15553, 15577, 15589, 15601, 15625, 15649, 15661, 15685, 15703, 15727, 15739, 15751, 15763, 15775, 15799, 15805, 15817, 15829, 15841, 15865, 15877, 15889,
15901, 15925, 15937, 15961, 15973, 15985, 15997, 16021, 16033, 16057, 16081, 16093, 16105, 16117, 16129, 16153, 16201, 16213, 16225, 16237, 16249, 16261,
16273, 16291, 16303, 16315, 16339, 16363, 16375, 16387, 16399, 16411, 16423, 16435, 16459, 16471, 16483, 16519, 16531, 16549, 16573, 16585, 16597, 16609,
16633, 16657, 16669, 16681, 16693, 16717, 16729, 16741, 16753, 16765, 16771, 16795, 16807, 16831, 16843, 16855, 16867, 16879, 16891, 16903, 16915, 16927,
16951, 16975, 16987, 17011, 17047, 17071, 17083, 17095, 17119, 17131, 17143, 17155, 17167, 17203, 17215, 17251, 17263, 17269, 17281, 17305, 17329, 17341,
17365, 17377, 17389, 17395, 17407, 17419, 17443, 17455, 17479, 17503, 17515, 17527, 17539, 17563, 17587, 17611, 17623, 17635, 17647, 17659, 17683, 17695,
17707, 17719, 17743, 17761, 17773, 17785, 17797, 17809, 17857, 17869, 17881, 17905, 17917, 17929, 17941, 17953, 17965, 17989, 18001, 18025, 18037, 18049,
18061, 18085, 18097, 18109, 18121, 18133, 18145, 18157, 18169, 18181, 18193, 18205, 18217, 18241, 18247, 18253, 18277, 18313, 18325, 18349, 18361, 18373,
18397, 18421, 18433, 18445, 18469, 18493, 18505, 18517, 18529, 18541, 18553, 18565, 18577, 18601, 18625, 18637, 18661, 18685, 18709, 18721, 18733, 18745,
18769, 18781, 18787, 18835, 18847, 18859, 18871, 18883, 18895, 18907, 18919, 18931, 18943, 18955, 18967, 18979, 18991, 19027, 19039, 19063, 19075, 19099,
19111, 19123, 19135, 19147, 19159, 19177, 19201, 19213, 19237, 19261, 19273, 19297, 19309, 19321, 19339, 19351, 19375, 19411, 19423, 19435, 19459, 19471,
19483, 19495, 19519, 19543, 19555, 19567, 19591, 19603, 19615, 19627, 19639, 19663, 19687, 19699, 19711, 19723, 19735, 19771, 19783, 19795, 19819, 19831,
19855, 19867, 19885, 19897, 19909, 19933, 19945, 19957, 19969, 19993, 20005, 20017, 20029, 20041, 20065, 20083, 20107, 20155, 20167, 20179, 20191, 20203,
20215, 20239, 20251, 20263, 20275, 20287, 20311, 20323, 20335, 20359, 20371, 20383, 20395, 20401, 20425, 20449, 20461, 20473, 20485, 20497, 20509, 20521,
20545, 20569, 20593, 20605, 20617, 20629, 20653, 20665, 20677, 20689, 20701, 20725, 20737, 20785, 20797, 20833, 20845, 20857, 20869, 20893, 20917, 20941,
20959, 20971, 20983, 20995, 21007, 21019, 21043, 21049, 21073, 21085, 21109, 21121, 21145, 21157, 21181, 21193, 21205, 21217, 21229, 21241, 21253, 21277,
21289, 21301, 21325, 21337, 21349, 21361, 21373, 21397, 21421, 21433, 21457, 21469, 21481, 21499, 21511, 21535, 21559, 21583, 21595, 21607, 21619, 21631,
21643, 21655, 21679, 21703, 21715, 21739, 21751, 21763, 21787, 21799, 21823, 21847, 21859, 21871, 21895, 21907, 21919, 21931, 21943, 21955, 21967, 21979,
21991, 22003, 22009, 22045, 22057, 22075, 22087, 22099, 22111, 22159, 22171, 22195, 22207, 22231, 22243, 22267, 22279, 22303, 22315, 22327, 22339, 22351,
22375, 22387, 22411, 22435, 22447, 22459, 22471, 22483, 22495, 22507, 22531, 22543, 22555, 22591, 22603, 22627, 22645, 22669, 22681, 22693, 22729, 22741,
22753, 22765, 22777, 22789, 22813, 22837, 22849, 22861, 22873, 22885, 22921, 22945, 22969, 22981, 22993, 23005, 23011, 23035, 23047, 23071, 23083, 23107,
23119, 23131, 23143, 23155, 23179, 23191, 23203, 23215, 23227, 23233, 23257, 23269, 23293, 23305, 23317, 23329, 23353, 23365, 23389, 23413, 23425, 23437,
23461, 23473, 23485, 23497, 23521, 23545, 23557, 23581, 23605, 23617, 23629, 23641, 23689, 23701, 23713, 23737, 23749, 23761, 23773, 23785, 23809, 23815,
23827, 23839, 23863, 23875, 23887, 23911, 23923, 23935, 23959, 23971, 23983, 23995, 24001, 24013, 24037, 24061, 24073, 24121, 24145, 24157, 24169, 24181,
24193, 24217, 24229, 24241, 24253, 24265, 24277, 24301, 24313, 24325, 24349, 24361, 24373, 24379, 24415, 24427, 24439, 24463, 24487, 24499, 24511, 24523,
24547, 24571, 24583, 24595, 24607, 24631, 24643, 24655, 24679, 24691, 24703, 24739, 24751, 24763, 24787, 24799, 24823, 24835, 24847, 24859, 24883, 24895,
24919, 24931, 24943, 24955, 24967, 24973, 24985, 24997, 25021, 25033, 25039, 25063, 25111, 25123, 25135, 25147, 25159, 25171, 25195, 25207, 25219, 25243,
25255, 25267, 25279, 25303, 25315, 25327, 25351, 25363, 25375, 25399, 25411, 25435, 25447, 25459, 25471, 25483, 25507, 25519, 25531, 25543, 25561, 25573,
25597, 25621, 25645, 25657, 25669, 25693, 25717, 25729, 25753, 25777, 25789, 25801, 25825, 25837, 25849, 25861, 25873, 25897, 25921, 25945, 25957, 25969,
25981, 25993, 26005, 26017, 26029, 26077, 26101, 26131, 26143, 26155, 26167, 26179, 26191, 26197, 26221, 26245, 26257, 26269, 26293, 26305, 26317, 26353,
26389, 26413, 26425, 26437, 26449, 26461, 26473, 26497, 26509, 26521, 26533, 26545, 26557, 26569, 26593, 26605, 26629, 26641, 26665, 26677, 26701, 26725,
26737, 26761, 26785, 26797, 26809, 26827, 26863, 26875, 26887, 26899, 26911, 26923, 26935, 26959, 26971, 26995, 27019, 27031, 27043, 27055, 27067, 27091,
27103, 27115, 27139, 27151, 27163, 27175, 27181, 27205, 27229, 27241, 27253, 27265, 27289, 27313, 27337, 27349, 27361, 27373, 27397, 27409, 27421, 27433,
27457, 27463, 27475, 27487, 27523, 27547, 27559, 27571, 27583, 27595, 27619, 27631, 27643, 27655, 27667, 27679, 27715, 27739, 27751, 27799, 27823, 27835,
27847, 27859, 27871, 27883, 27895, 27931, 27955, 27967, 27991, 28003, 28015, 28027, 28039, 28051, 28063, 28075, 28081, 28105, 28117, 28141, 28153, 28177,
28201, 28213, 28237, 28249, 28261, 28267, 28279, 28303, 28327, 28351, 28375, 28387, 28399, 28423, 28447, 28471, 28483, 28507, 28519, 28543, 28555, 28579,
28591, 28603, 28615, 28639, 28651, 28663, 28687, 28693, 28717, 28741, 28753, 28765, 28777, 28789, 28801, 28813, 28825, 28837, 28861, 28885, 28897, 28921,
28933, 28945, 28957, 28993, 29005, 29017, 29041, 29065, 29077, 29089, 29101, 29113, 29161, 29173, 29185, 29209, 29221, 29233, 29245, 29257, 29269, 29293,
29305, 29317, 29329, 29335, 29347, 29371, 29383, 29401, 29449, 29461, 29473, 29485, 29509, 29521, 29545, 29557, 29581, 29593, 29605, 29617, 29641, 29653,
29665, 29689, 29701, 29713, 29725, 29749, 29773, 29785, 29797, 29809, 29821, 29845, 29869, 29881, 29893, 29905, 29929, 29941, 29953, 29965, 29977, 29989,
30043, 30067, 30079, 30091, 30103, 30115, 30139, 30151, 30175, 30187, 30199, 30211, 30223, 30235, 30259, 30283, 30295, 30307, 30319, 30343, 30355, 30367,
30379, 30415, 30427, 30439, 30451, 30475, 30499, 30511, 30523, 30535, 30541, 30577, 30589, 30601, 30625, 30637, 30649, 30661, 30673, 30721, 30727, 30739,
30763, 30787, 30811, 30835, 30847, 30859, 30883, 30907, 30919, 30931, 30943, 30955, 30967, 30991, 31003, 31015, 31039, 31051, 31063, 31075, 31099, 31111,
31135, 31159, 31183, 31207, 31219, 31243, 31267, 31291, 31303, 31315, 31327, 31351, 31363, 31375, 31393, 31417, 31429, 31453, 31465, 31477, 31489, 31513,
31525, 31537, 31561, 31573, 31585, 31597, 31609, 31621, 31633, 31645, 31657, 31669, 31693, 31699, 31723, 31735, 31747, 31771, 31783, 31807, 31831, 31855,
31867, 31879, 31891, 31915, 31927, 31951, 31963, 31975, 31987, 32035, 32047, 32059, 32065, 32077, 32101, 32125, 32137, 32149, 32161, 32197, 32209, 32221,
32245, 32257, 32269, 32317, 32329, 32353, 32365, 32377, 32389, 32413, 32425, 32437, 32461, 32485, 32509, 32521, 32533, 32569, 32581, 32593, 32617, 32629,
32641, 32653, 32665, 32677, 32689, 32701, 32725, 32743, 32767, 32791, 32803, 32815, 32839, 32851, 32863, 32875, 32899, 32905, 32917, 32941, 32953, 32965,
32977, 33001, 33013, 33037, 33049, 33061, 33073, 33097, 33109, 33121, 33133, 33145, 33193, 33205, 33217, 33229, 33253, 33265, 33277, 33289, 33301, 33313,
33325, 33337, 33349, 33361, 33373, 33397, 33403, 33427, 33451, 33463, 33475, 33487, 33499, 33523, 33535, 33559, 33583, 33607, 33619, 33643, 33667, 33679,
33691, 33715, 33739, 33751, 33763, 33787, 33799, 33823, 33835, 33847, 33895, 33907, 33919, 33931, 33943, 33955, 33967, 34015, 34027, 34039, 34051, 34075,
34087, 34105, 34123, 34147, 34171, 34183, 34195, 34219, 34231, 34255, 34267, 34279, 34303, 34339, 34351, 34363, 34375, 34387, 34399, 34423, 34435, 34447,
34459, 34471, 34483, 34507, 34531, 34555, 34567, 34591, 34603, 34615, 34627, 34639, 34651, 34663, 34675, 34687, 34711, 34735, 34771, 34795, 34819, 34831,
34861, 34885, 34897, 34909, 34921, 34933, 34945, 34981, 34993, 35005, 35017, 35029, 35053, 35065, 35077, 35101, 35113, 35125, 35137, 35149, 35161, 35185,
35233, 35245, 35257, 35281, 35293, 35305, 35317, 35335, 35347, 35383, 35395, 35407, 35431, 35443, 35467, 35479, 35491, 35503, 35515, 35539, 35563, 35569,
35593, 35605, 35617, 35629, 35641, 35665, 35677, 35689, 35713, 35737, 35773, 35785, 35797, 35809, 35821, 35833, 35857, 35869, 35881, 35893, 35905, 35917,
35929, 35953, 36001, 36013, 36025, 36037, 36061, 36073, 36097, 36109, 36121, 36133, 36157, 36169, 36181, 36193, 36217, 36229, 36253, 36265, 36289, 36295,
36319, 36331, 36343, 36367, 36379, 36403, 36427, 36439, 36451, 36463, 36475, 36487, 36499, 36511, 36535, 36547, 36559, 36565, 36577, 36589, 36601, 36649,
36661, 36697, 36709, 36733, 36745, 36757, 36781, 36805, 36817, 36841, 36853, 36865, 36877, 36889, 36913, 36925, 36937, 36949, 36973, 37009, 37015, 37027,
37051, 37063, 37087, 37099, 37123, 37135, 37147, 37159, 37183, 37207, 37219, 37243, 37255, 37279, 37303, 37315, 37327, 37351, 37363, 37375, 37387, 37399,
37435, 37459, 37471, 37495, 37507, 37543, 37567, 37579, 37603, 37615, 37627, 37639, 37651, 37675, 37687, 37711, 37723, 37729, 37741, 37765, 37789, 37801,
37813, 37837, 37849, 37855, 37879, 37903, 37915, 37939, 37951, 37963, 37975, 37999, 38011, 38023, 38047, 38071, 38083, 38095, 38143, 38155, 38167, 38191,
38203, 38215, 38227, 38239, 38251, 38275, 38287, 38311, 38323, 38347, 38371, 38383, 38395, 38407, 38419, 38431, 38449, 38473, 38521, 38533, 38545, 38593,
38605, 38617, 38629, 38665, 38677, 38689, 38701, 38725, 38749, 38761, 38773, 38785, 38797, 38821, 38833, 38845, 38857, 38869, 38893, 38905, 38917, 38929,
38965, 38977, 38989, 39001, 39013, 39037, 39049, 39061, 39085, 39109, 39121, 39127, 39151, 39175, 39199, 39211, 39229, 39253, 39265, 39277, 39289, 39301,
39325, 39337, 39349, 39373, 39397, 39409, 39421, 39445, 39457, 39469, 39493, 39505, 39517, 39529, 39541, 39553, 39565, 39577, 39589, 39613, 39625, 39661,
39673, 39685, 39697, 39709, 39721, 39745, 39769, 39793, 39805, 39829, 39853, 39865, 39877, 39889, 39901, 39925, 39937, 39961, 39973, 39997, 40015, 40027,
40051, 40075, 40087, 40099, 40111, 40135, 40147, 40159, 40171, 40183, 40195, 40207, 40219, 40243, 40255, 40267, 40279, 40291, 40315, 40327, 40339, 40351,
40363, 40375, 40423, 40435, 40447, 40465, 40477, 40501, 40537, 40549, 40573, 40585, 40597, 40633, 40645, 40657, 40669, 40681, 40693, 40729, 40753, 40759,
40771, 40783, 40795, 40819, 40831, 40843, 40855, 40867, 40891, 40903, 40915, 40963, 40975, 40987, 40999, 41011, 41023, 41059, 41071, 41083, 41095, 41119,
41131, 41155, 41179, 41191, 41215, 41239, 41251, 41263, 41287, 41299, 41311, 41335, 41359, 41383, 41407, 41419, 41443, 41455, 41467, 41491, 41503, 41515,
41521, 41545, 41569, 41593, 41605, 41617, 41641, 41665, 41689, 41701, 41713, 41725, 41737, 41761, 41773, 41797, 41815, 41839, 41863, 41875, 41887, 41899,
41947, 41959, 41983, 41995, 42007, 42031, 42043, 42055, 42067, 42091, 42115, 42127, 42163, 42175, 42187, 42199, 42211, 42235, 42259, 42283, 42289, 42301,
42313, 42325, 42349, 42361, 42385, 42397, 42409, 42433, 42445, 42457, 42469, 42481, 42493, 42517, 42529, 42541, 42553, 42565, 42613, 42625, 42649, 42661,
42685, 42697, 42709, 42733, 42745, 42757, 42781, 42805, 42817, 42853, 42877, 42889, 42901, 42913, 42925, 42937, 42949, 42961, 42985, 42997, 43021, 43033,
43045, 43063, 43075, 43087, 43099, 43111, 43135, 43159, 43171, 43189, 43213, 43237, 43249, 43261, 43285, 43309, 43321, 43333, 43357, 43369, 43405, 43417,
43429, 43441, 43465, 43477, 43501, 43525, 43549, 43561, 43573, 43585, 43597, 43621, 43645, 43669, 43681, 43693, 43705, 43729, 43753, 43765, 43777, 43789,
43801, 43813, 43825, 43849, 43855, 43927, 43939, 43951, 43963, 43987, 44011, 44035, 44047, 44071, 44083, 44095, 44107, 44131, 44143, 44167, 44179, 44191,
44215, 44239, 44251, 44275, 44299, 44311, 44335, 44347, 44371, 44383, 44395, 44407, 44431, 44443, 44455, 44479, 44491, 44515, 44527, 44539, 44545, 44557,
44581, 44593, 44605, 44617, 44641, 44665, 44677, 44701, 44719, 44743, 44755, 44791, 44803, 44815, 44839, 44851, 44899, 44911, 44923, 44935, 44959, 44971,
44983, 44995, 45019, 45031, 45043, 45055, 45067, 45091, 45103, 45115, 45139, 45151, 45163, 45175, 45187, 45199, 45223, 45235, 45247, 45271, 45295, 45307,
45319, 45355, 45367, 45379, 45403, 45415, 45439, 45451, 45475, 45499, 45511, 45529, 45541, 45553, 45577, 45601, 45613, 45637, 45649, 45661, 45673, 45685,
45697, 45709, 45721, 45745, 45757, 45769, 45781, 45805, 45817, 45829, 45877, 45889, 45901, 45925, 45943, 45955, 45979, 46003, 46027, 46075, 46087, 46099,
46111, 46123, 46147, 46159, 46171, 46183, 46195, 46207, 46219, 46243, 46255, 46279, 46291, 46303, 46315, 46321, 46345, 46369, 46381, 46393, 46417, 46441,
46453, 46465, 46489, 46501, 46513, 46525, 46537, 46549, 46561, 46597, 46609, 46621, 46633, 46645, 46669, 46693, 46717, 46741, 46753, 46765, 46777, 46825,
46837, 46861, 46885, 46897, 46909, 46933, 46957, 46993, 47017, 47041, 47053, 47065, 47077, 47089, 47101, 47125, 47143, 47167, 47179, 47191, 47203, 47215,
47227, 47239, 47263, 47275, 47287, 47299, 47323, 47347, 47359, 47365, 47389, 47401, 47425, 47437, 47449, 47473, 47485, 47497, 47509, 47521, 47545, 47569,
47581, 47605, 47617, 47641, 47653, 47689, 47713, 47725, 47737, 47761, 47773, 47785, 47797, 47809, 47821, 47845, 47857, 47881, 47893, 47905, 47917, 47941,
47953, 47959, 47971, 48019, 48031, 48043, 48067, 48079, 48091, 48103, 48115, 48139, 48151, 48187, 48211, 48223, 48247, 48259, 48271, 48295, 48307, 48319,
48331, 48343, 48355, 48379, 48391, 48403, 48415, 48451, 48487, 48511, 48523, 48571, 48583, 48595, 48607, 48619, 48631, 48643, 48655, 48679, 48691, 48715,
48727, 48739, 48751, 48763, 48769, 48793, 48811, 48859, 48883, 48895, 48907, 48919, 48943, 48955, 48979, 48991, 49015, 49039, 49051, 49075, 49087, 49099,
49123, 49135, 49147, 49159, 49171, 49183, 49195, 49207, 49243, 49255, 49279, 49291, 49315, 49327, 49339, 49363, 49387, 49411, 49423, 49435, 49459, 49471,
49483, 49495, 49519, 49543, 49555, 49591, 49603, 49615, 49627, 49639, 49657, 49681, 49693, 49741, 49753, 49765, 49777, 49789, 49801, 49813, 49825, 49849,
49897, 49909, 49933, 49945, 49969, 49981, 49993, 50005, 50017, 50041, 50065, 50077, 50089, 50113, 50137, 50161, 50173, 50185, 50197, 50221, 50233, 50245,
50257, 50293, 50299, 50311, 50323, 50335, 50359, 50371, 50395, 50419, 50431, 50443, 50467, 50491, 50503, 50515, 50521, 50545, 50557, 50569, 50593, 50605,
50629, 50641, 50653, 50665, 50689, 50701, 50725, 50749, 50761, 50785, 50797, 50809, 50821, 50833, 50857, 50869, 50881, 50905, 50929, 50941, 50965, 50977,
51001, 51013, 51025, 51049, 51085, 51097, 51121, 51145, 51169, 51181, 51193, 51205, 51229, 51241, 51253, 51277, 51301, 51325, 51337, 51349, 51361, 51379,
51391, 51415, 51427, 51439, 51451, 51463, 51475, 51511, 51523, 51547, 51571, 51583, 51595, 51619, 51643, 51655, 51667, 51703, 51727, 51739, 51751, 51763,
51775, 51781, 51793, 51841, 51853, 51865, 51889, 51925, 51949, 51973, 51997, 52009, 52021, 52033, 52045, 52057, 52081, 52105, 52129, 52141, 52165, 52177,
52201, 52213, 52219, 52231, 52243, 52267, 52279, 52291, 52303, 52315, 52339, 52351, 52363, 52375, 52387, 52411, 52423, 52435, 52459, 52471, 52483, 52495,
52507, 52531, 52543, 52591, 52603, 52627, 52651, 52663, 52687, 52699, 52723, 52735, 52783, 52807, 52819, 52831, 52843, 52855, 52867, 52879, 52903, 52915,
52927, 52939, 52951, 52975, 52999, 53011, 53035, 53059, 53071, 53077, 53089, 53113, 53137, 53149, 53161, 53173, 53197, 53221, 53245, 53257, 53281, 53293,
53311, 53335, 53347, 53359, 53371, 53395, 53407, 53431, 53443, 53455, 53467, 53479, 53527, 53575, 53587, 53599, 53611, 53623, 53635, 53659, 53671, 53695,
53707, 53719, 53731, 53743, 53755, 53767, 53779, 53791, 53839, 53851, 53863, 53875, 53887, 53911, 53923, 53935, 53953, 53965, 53989, 54025, 54037, 54049,
54097, 54109, 54133, 54145, 54157, 54169, 54193, 54205, 54229, 54241, 54265, 54277, 54301, 54313, 54325, 54337, 54349, 54361, 54385, 54397, 54409, 54433,
54445, 54469, 54481, 54493, 54505, 54517, 54529, 54541, 54553, 54565, 54601, 54613, 54661, 54673, 54685, 54697, 54709, 54721, 54745, 54769, 54781, 54793,
54799, 54823, 54847, 54853, 54877, 54901, 54913, 54949, 54973, 54997, 55009, 55033, 55045, 55057, 55069, 55081, 55105, 55129, 55141, 55153, 55165, 55189,
55201, 55225, 55249, 55261, 55273, 55285, 55309, 55333, 55357, 55369, 55393, 55405, 55429, 55441, 55453, 55465, 55477, 55489, 55501, 55513, 55525, 55549,
55561, 55573, 55585, 55597, 55633, 55645, 55669, 55681, 55693, 55705, 55717, 55729, 55753, 55765, 55783, 55831, 55843, 55855, 55879, 55903, 55927, 55939,
55951, 55975, 55987, 55999, 56023, 56035, 56059, 56071, 56083, 56107, 56119, 56131, 56155, 56191, 56203, 56215, 56227, 56239, 56251, 56263, 56287, 56323,
56335, 56347, 56371, 56377, 56401, 56413, 56425, 56473, 56485, 56509, 56533, 56545, 56557, 56569, 56581, 56593, 56605, 56617, 56629, 56653, 56665, 56677,
56683, 56695, 56707, 56719, 56731, 56755, 56767, 56779, 56827, 56839, 56851, 56863, 56887, 56899, 56911, 56923, 56935, 56947, 56959, 56995, 57019, 57031,
57043, 57055, 57067, 57079, 57103, 57127, 57151, 57175, 57199, 57211, 57223, 57235, 57259, 57271, 57295, 57343, 57355, 57379, 57403, 57415, 57427, 57439,
57451, 57499, 57511, 57523, 57535, 57547, 57559, 57577, 57589, 57613, 57637, 57649, 57661, 57685, 57697, 57721, 57733, 57745, 57757, 57769, 57805, 57817,
57841, 57853, 57877, 57901, 57913, 57925, 57937, 57949, 57967, 57991, 58003, 58015, 58039, 58063, 58087, 58099, 58123, 58159, 58171, 58195, 58219, 58231,
58243, 58267, 58279, 58291, 58315, 58327, 58339, 58363, 58375, 58387, 58411, 58423, 58435, 58447, 58459, 58471, 58489, 58513, 58537, 58549, 58561, 58585,
58609, 58633, 58657, 58669, 58693, 58705, 58717, 58729, 58753, 58765, 58789, 58813, 58849, 58861, 58873, 58885, 58897, 58921, 58933, 58957, 58969, 58981,
59005, 59017, 59029, 59053, 59077, 59101, 59113, 59125, 59149, 59161, 59173, 59185, 59221, 59245, 59257, 59269, 59281, 59293, 59305, 59329, 59341, 59353,
59365, 59377, 59401, 59407, 59431, 59443, 59479, 59491, 59503, 59515, 59527, 59551, 59563, 59581, 59605, 59629, 59641, 59653, 59665, 59677, 59689, 59713,
59737, 59785, 59797, 59809, 59821, 59869, 59881, 59917, 59929, 59941, 59953, 59977, 59989, 60013, 60061, 60073, 60085, 60097, 60109, 60121, 60133, 60157,
60169, 60193, 60217, 60229, 60241, 60253, 60277, 60289, 60301, 60325, 60343, 60367, 60379, 60427, 60439, 60463, 60475, 60499, 60511, 60535, 60559, 60571,
60583, 60595, 60607, 60619, 60631, 60655, 60679, 60703, 60715, 60727, 60739, 60751, 60763, 60775, 60787, 60799, 60823, 60847, 60871, 60895, 60907, 60931,
60943, 60955, 60991, 61003, 61015, 61027, 61051, 61063, 61087, 61099, 61111, 61123, 61147, 61159, 61165, 61177, 61189, 61213, 61237, 61285, 61309, 61327,
61339, 61363, 61375, 61399, 61411, 61423, 61435, 61447, 61471, 61483, 61507, 61519, 61543, 61555, 61567, 61579, 61603, 61615, 61627, 61639, 61651, 61675,
61699, 61711, 61723, 61747, 61759, 61783, 61795, 61807, 61819, 61831, 61843, 61855, 61879, 61891, 61915, 61939, 61951, 61963, 61975, 61987, 61999, 62035,
62047, 62059, 62071, 62095, 62119, 62143, 62155, 62203, 62215, 62227, 62275, 62281, 62305, 62317, 62341, 62353, 62365, 62377, 62413, 62425, 62437, 62449,
62461, 62485, 62497, 62521, 62545, 62557, 62569, 62593, 62605, 62629, 62653, 62665, 62689, 62701, 62713, 62737, 62749, 62773, 62797, 62809, 62827, 62875,
62899, 62911, 62923, 62935, 62983, 62995, 63007, 63019, 63031, 63043, 63055, 63067, 63079, 63091, 63103, 63115, 63139, 63175, 63199, 63211, 63217, 63241,
63253, 63265, 63277, 63289, 63301, 63325, 63337, 63361, 63373, 63385, 63409, 63421, 63433, 63445, 63457, 63481, 63493, 63505, 63517, 63529, 63553, 63577,
63589, 63601, 63613, 63625, 63649, 63661, 63709, 63721, 63733, 63757, 63781, 63793, 63817, 63841, 63865, 63877, 63889, 63901, 63925, 63937, 63961, 63973,
63985, 63997, 64021, 64033, 64057, 64069, 64081, 64105, 64129, 64141, 64195, 64207, 64231, 64243, 64255, 64279, 64303, 64315, 64327, 64339, 64351, 64363,
64387, 64399, 64411, 64435, 64447, 64471, 64483, 64501, 64513, 64525, 64537, 64573, 64597, 64621, 64645, 64657, 64681, 64729, 64741, 64765, 64777, 64789,
64801, 64825, 64837, 64849, 64861, 64873, 64885, 64897, 64921, 64945, 64957, 64969, 64981, 65005, 65017, 65029, 65053, 65065, 65077, 65101, 65125, 65143,
65155, 65167, 65179, 65191, 65203, 65239, 65251, 65263, 65287, 65311, 65323, 65347, 65359, 65383, 65395, 65443, 65467, 65479, 65491, 65503, 65527, 65539,
65551, 65563, 65575, 65587, 65611, 65635, 65647, 65659, 65671, 65695, 65743, 65755, 65767, 65779, 65791, 65803, 65815, 65839, 65863, 65875, 65899, 65911,
65923, 65935, 65947, 65971, 65983, 65995, 66007, 66019, 66031, 66043, 66055, 66067, 66079, 66085, 66121, 66133, 66145, 66157, 66175, 66187, 66211, 66247,
66259, 66271, 66283, 66295, 66331, 66343, 66355, 66403, 66415, 66427, 66439, 66451, 66475, 66499, 66511, 66535, 66559, 66607, 66619, 66631, 66643, 66667,
66679, 66703, 66715, 66727, 66751, 66763, 66775, 66787, 66811, 66847, 66859, 66871, 66883, 66895, 66907, 66919, 66931, 66955, 66967, 66979, 66991, 67003,
67015, 67063, 67075, 67087, 67093, 67117, 67141, 67165, 67177, 67189, 67201, 67225, 67249, 67261, 67285, 67309, 67321, 67333, 67345, 67369, 67405, 67417,
67441, 67453, 67465, 67477, 67489, 67513, 67537, 67549, 67561, 67585, 67597, 67609, 67633, 67657, 67669, 67681, 67693, 67705, 67741, 67753, 67765, 67777,
67789, 67813, 67837, 67861, 67879, 67903, 67927, 67951, 67963, 67987, 67999, 68011, 68035, 68047, 68083, 68089, 68125, 68137, 68161, 68173, 68185, 68209,
68221, 68233, 68257, 68269, 68293, 68317, 68329, 68353, 68365, 68389, 68413, 68425, 68449, 68461, 68485, 68509, 68521, 68533, 68545, 68593, 68605, 68617,
68629, 68641, 68677, 68701, 68713, 68737, 68761, 68797, 68821, 68833, 68845, 68869, 68893, 68917, 68929, 68953, 68965, 68977, 69013, 69025, 69037, 69061,
69073, 69097, 69103, 69127, 69139, 69151, 69163, 69187, 69211, 69223, 69235, 69259, 69271, 69295, 69319, 69331, 69355, 69367, 69391, 69403, 69415, 69427,
69439, 69463, 69475, 69487, 69511, 69535, 69547, 69559, 69583, 69595, 69607, 69613, 69661, 69673, 69685, 69709, 69721, 69733, 69745, 69757, 69769, 69781,
69805, 69817, 69841, 69853, 69865, 69877, 69901, 69913, 69925, 69949, 69973, 69985, 69997, 70021, 70033, 70057, 70069, 70081, 70099, 70111, 70123, 70147,
70171, 70195, 70219, 70231, 70255, 70267, 70291, 70303, 70315, 70327, 70351, 70375, 70387, 70399, 70411, 70423, 70447, 70459, 70471, 70483, 70495, 70507,
70519, 70531, 70567, 70579, 70603, 70627, 70651, 70663, 70675, 70687, 70711, 70723, 70735, 70759, 70771, 70783, 70831, 70843, 70855, 70867, 70891, 70903,
70915, 70963, 70987, 71011, 71023, 71047, 71059, 71071, 71089, 71101, 71113, 71125, 71137, 71161, 71185, 71209, 71221, 71233, 71245, 71257, 71269, 71293,
71305, 71317, 71329, 71353, 71365, 71371, 71419, 71431, 71443, 71455, 71467, 71491, 71503, 71515, 71539, 71551, 71575, 71587, 71599, 71623, 71695, 71707,
71719, 71731, 71755, 71767, 71779, 71803, 71815, 71827, 71839, 71851, 71875, 71887, 71899, 71911, 71935, 71947, 71959, 71983, 71995, 72019, 72031, 72043,
72055, 72067, 72091, 72097, 72109, 72121, 72145, 72157, 72181, 72205, 72229, 72241, 72253, 72301, 72325, 72349, 72373, 72385, 72397, 72409, 72433, 72445,
72469, 72481, 72493, 72505, 72517, 72529, 72541, 72565, 72589, 72601, 72613, 72637, 72649, 72661, 72673, 72685, 72697, 72709, 72745, 72757, 72769, 72793,
72805, 72817, 72865, 72877, 72901, 72913, 72925, 72937, 72949, 72961, 72973, 72997, 73009, 73033, 73045, 73057, 73069, 73081, 73087, 73135, 73141, 73153,
73177, 73213, 73249, 73273, 73297, 73309, 73321, 73345, 73369, 73393, 73405, 73417, 73429, 73453, 73477, 73501, 73513, 73525, 73549, 73561, 73573, 73585,
73597, 73645, 73669, 73681, 73705, 73729, 73741, 73753, 73765, 73777, 73789, 73801, 73813, 73837, 73849, 73873, 73897, 73921, 73933, 73945, 73957, 73981,
73993, 74005, 74017, 74041, 74053, 74077, 74089, 74101, 74137, 74149, 74161, 74179, 74191, 74239, 74251, 74275, 74287, 74311, 74323, 74335, 74347, 74371,
74395, 74419, 74431, 74443, 74455, 74479, 74503, 74515, 74527, 74539, 74551, 74575, 74587, 74611, 74635, 74647, 74671, 74683, 74695, 74719, 74731, 74755,
74767, 74779, 74815, 74827, 74839, 74851, 74875, 74887, 74911, 74923, 74947, 74953, 74965, 74989, 75001, 75049, 75061, 75085, 75109, 75133, 75145, 75157,
75169, 75181, 75193, 75205, 75211, 75235, 75247, 75295, 75307, 75319, 75343, 75355, 75379, 75391, 75403, 75415, 75427, 75439, 75463, 75475, 75487, 75499,
75523, 75535, 75547, 75595, 75607, 75619, 75631, 75655, 75667, 75691, 75739, 75751, 75763, 75775, 75787, 75799, 75811, 75835, 75859, 75907, 75919, 75931,
75943, 75955, 75979, 76003, 76015, 76051, 76063, 76075, 76099, 76111, 76135, 76147, 76171, 76183, 76195, 76219, 76231, 76243, 76249, 76273, 76297, 76309,
76321, 76345, 76393, 76417, 76429, 76441, 76453, 76465, 76489, 76501, 76513, 76537, 76549, 76561, 76573, 76585, 76597, 76609, 76633, 76657, 76669, 76681,
76705, 76717, 76729, 76747, 76759, 76771, 76795, 76807, 76831, 76843, 76855, 76879, 76891, 76915, 76939, 76951, 76963, 76999, 77023, 77035, 77059, 77071,
77095, 77107, 77119, 77143, 77167, 77191, 77203, 77215, 77239, 77251, 77275, 77287, 77299, 77317, 77329, 77341, 77353, 77377, 77401, 77425, 77437, 77461,
77485, 77521, 77533, 77545, 77557, 77569, 77581, 77593, 77605, 77629, 77653, 77677, 77701, 77725, 77749, 77761, 77785, 77797, 77821, 77845, 77881, 77893,
77917, 77929, 77941, 77953, 77965, 77977, 77989, 78001, 78013, 78025, 78049, 78085, 78097, 78109, 78121, 78133, 78157, 78181, 78205, 78217, 78265, 78277,
78289, 78301, 78313, 78337, 78367, 78379, 78391, 78415, 78439, 78451, 78463, 78487, 78499, 78511, 78523, 78535, 78547, 78559, 78571, 78577, 78589, 78637,
78661, 78685, 78709, 78733, 78757, 78769, 78781, 78805, 78817, 78829, 78841, 78865, 78877, 78889, 78901, 78913, 78949, 78961, 78985, 78997, 79021, 79033,
79081, 79093, 79105, 79141, 79153, 79177, 79201, 79213, 79225, 79249, 79273, 79285, 79297, 79309, 79321, 79345, 79357, 79381, 79393, 79405, 79417, 79435,
79459, 79483, 79495, 79507, 79519, 79567, 79579, 79591, 79603, 79615, 79627, 79651, 79663, 79675, 79687, 79699, 79747, 79771, 79795, 79807, 79819, 79831,
79843, 79867, 79879, 79915, 79927, 79951, 79963, 79987, 79999, 80011, 80035, 80047, 80059, 80083, 80095, 80107, 80131, 80155, 80167, 80179, 80191, 80203,
80215, 80239, 80263, 80275, 80287, 80311, 80323, 80335, 80347, 80359, 80371, 80395, 80407, 80431, 80449, 80461, 80485, 80521, 80527, 80539, 80587, 80611,
80623, 80635, 80659, 80671, 80683, 80707, 80731, 80743, 80755, 80767, 80791, 80815, 80827, 80839, 80851, 80863, 80887, 80899, 80911, 80923, 80947, 80971,
80995, 81007, 81019, 81031, 81055, 81103, 81127, 81139, 81151, 81163, 81175, 81187, 81211, 81247, 81271, 81283, 81307, 81331, 81343, 81355, 81367, 81391,
81403, 81427, 81439, 81463, 81475, 81487, 81499, 81571, 81583, 81607, 81619, 81625, 81637, 81661, 81685, 81709, 81733, 81745, 81757, 81769, 81793, 81805,
81817, 81829, 81853, 81865, 81877, 81889, 81913, 81937, 81961, 81973, 81985, 82009, 82021, 82033, 82045, 82057, 82081, 82105, 82117, 82129, 82141, 82153,
82177, 82189, 82213, 82225, 82249, 82261, 82297, 82309, 82333, 82345, 82351, 82375, 82399, 82411, 82423, 82435, 82459, 82471, 82483, 82519, 82531, 82555,
82567, 82591, 82603, 82627, 82639, 82651, 82663, 82675, 82699, 82711, 82729, 82741, 82753, 82765, 82789, 82813, 82837, 82861, 82873, 82897, 82909, 82921,
82933, 82945, 82969, 82981, 83029, 83053, 83065, 83089, 83113, 83125, 83137, 83149, 83185, 83197, 83221, 83233, 83245, 83269, 83317, 83329, 83341, 83389,
83413, 83437, 83449, 83461, 83485, 83497, 83509, 83521, 83533, 83545, 83557, 83569, 83581, 83593, 83617, 83641, 83653, 83665, 83677, 83701, 83713, 83749,
83773, 83785, 83809, 83827, 83851, 83863, 83887, 83911, 83923, 83935, 83947, 83959, 83983, 83995, 84007, 84019, 84031, 84043, 84055, 84067, 84091, 84115,
84127, 84139, 84151, 84163, 84175, 84199, 84211, 84223, 84229, 84253, 84277, 84301, 84313, 84325, 84349, 84361, 84385, 84409, 84421, 84457, 84469, 84493,
84517, 84529, 84541, 84565, 84577, 84589, 84613, 84637, 84661, 84673, 84685, 84709, 84733, 84745, 84757, 84769, 84781, 84793, 84817, 84841, 84865, 84889,
84901, 84907, 84919, 84943, 84967, 84979, 84991, 85003, 85027, 85051, 85075, 85099, 85111, 85123, 85135, 85147, 85171, 85183, 85219, 85231, 85255, 85279,
85291, 85315, 85339, 85351, 85375, 85387, 85399, 85411, 85423, 85447, 85459, 85471, 85543, 85555, 85567, 85591, 85615, 85627, 85639, 85651, 85675, 85699,
85711, 85723, 85735, 85759, 85771, 85783, 85831, 85855, 85867, 85891, 85903, 85915, 85927, 85939, 85951, 85975, 85999, 86017, 86029, 86053, 86065, 86089,
86113, 86125, 86137, 86143, 86167, 86179, 86203, 86227, 86239, 86287, 86299, 86323, 86347, 86359, 86371, 86395, 86407, 86419, 86431, 86443, 86455, 86467,
86491, 86503, 86515, 86527, 86539, 86575, 86587, 86611, 86623, 86647, 86671, 86683, 86695, 86707, 86719, 86743, 86767, 86779, 86791, 86803, 86815, 86827,
86851, 86863, 86911, 86947, 86959, 86971, 86983, 86995, 87019, 87031, 87055, 87067, 87091, 87103, 87121, 87133, 87157, 87169, 87205, 87217, 87229, 87241,
87253, 87277, 87289, 87337, 87361, 87373, 87385, 87433, 87445, 87457, 87469, 87481, 87493, 87505, 87517, 87541, 87553, 87565, 87589, 87601, 87613, 87625,
87637, 87661, 87673, 87685, 87697, 87709, 87721, 87757, 87781, 87805, 87817, 87829, 87853, 87877, 87889, 87901, 87913, 87937, 87949, 87973, 87985, 87997,
88009, 88033, 88045, 88069, 88081, 88087, 88111, 88123, 88159, 88183, 88207, 88219, 88231, 88255, 88273, 88285, 88333, 88345, 88369, 88381, 88393, 88405,
88417, 88429, 88441, 88465, 88477, 88525, 88537, 88561, 88573, 88597, 88609, 88621, 88645, 88657, 88681, 88717, 88741, 88753, 88765, 88777, 88789, 88801,
88813, 88837, 88861, 88873, 88885, 88897, 88921, 88933, 88945, 88969, 88993, 89005, 89017, 89029, 89053, 89077, 89089, 89101, 89125, 89137, 89185, 89209,
89233, 89257, 89269, 89293, 89317, 89341, 89353, 89365, 89377, 89389, 89413, 89431, 89443, 89491, 89503, 89527, 89551, 89575, 89587, 89599, 89611, 89623,
89647, 89671, 89683, 89707, 89731, 89743, 89755, 89767, 89791, 89815, 89827, 89839, 89851, 89863, 89875, 89899, 89911, 89923, 89935, 89947, 89959, 89983,
89995, 90019, 90031, 90055, 90067, 90079, 90133, 90145, 90157, 90181, 90193, 90205, 90217, 90229, 90241, 90253, 90265, 90277, 90313, 90337, 90361, 90373,
90385, 90409, 90421, 90433, 90457, 90469, 90481, 90493, 90517, 90529, 90553, 90571, 90583, 90595, 90619, 90631, 90643, 90655, 90679, 90691, 90715, 90739,
90775, 90799, 90811, 90823, 90871, 90883, 90895, 90907, 90919, 90943, 90967, 90979, 90991, 91015, 91027, 91039, 91051, 91063, 91075, 91123, 91147, 91159,
91183, 91219, 91243, 91255, 91267, 91291, 91315, 91327, 91339, 91351, 91363, 91375, 91387, 91399, 91423, 91435, 91483, 91507, 91519, 91543, 91555, 91567,
91579, 91591, 91615, 91627, 91639, 91651, 91675, 91687, 91693, 91729, 91753, 91765, 91777, 91789, 91813, 91825, 91837, 91861, 91909, 91921, 91933, 91945,
91957, 91969, 91981, 92005, 92017, 92029, 92077, 92083, 92107, 92131, 92143, 92155, 92167, 92191, 92215, 92227, 92251, 92275, 92323, 92335, 92347, 92359,
92383, 92407, 92419, 92431, 92455, 92479, 92491, 92503, 92527, 92551, 92563, 92599, 92611, 92623, 92647, 92659, 92683, 92695, 92707, 92719, 92731, 92755,
92767, 92779, 92791, 92803, 92827, 92839, 92845, 92857, 92869, 92893, 92905, 92917, 92941, 92977, 92989, 93001, 93013, 93037, 93049, 93061, 93109, 93133,
93157, 93181, 93193, 93205, 93217, 93241, 93253, 93277, 93289, 93301, 93313, 93337, 93361, 93373, 93385, 93397, 93409, 93457, 93469, 93493, 93517, 93529,
93553, 93577, 93601, 93613, 93625, 93637, 93649, 93673, 93685, 93709, 93733, 93757, 93781, 93793, 93817, 93829, 93841, 93853, 93865, 93889, 93901, 93925,
93937, 93961, 93979, 93991, 94003, 94027, 94039, 94051, 94063, 94075, 94087, 94105, 94129, 94141, 94165, 94189, 94201, 94213, 94237, 94249, 94273, 94285,
94297, 94321, 94345, 94357, 94393, 94405, 94417, 94429, 94441, 94453, 94465, 94501, 94525, 94537, 94549, 94561, 94609, 94621, 94633, 94657, 94681, 94705,
94717, 94729, 94741, 94753, 94765, 94777, 94801, 94813, 94837, 94849, 94873, 94885, 94897, 94921, 94933, 94957, 94969, 94981, 94993, 95005, 95017, 95029,
95041, 95053, 95077, 95089, 95113, 95137, 95161, 95185, 95191, 95215, 95227, 95239, 95251, 95275, 95287, 95311, 95335, 95347, 95359, 95371, 95395, 95419,
95431, 95455, 95479, 95491, 95503, 95527, 95539, 95551, 95563, 95575, 95611, 95635, 95659, 95671, 95683, 95707, 95731, 95743, 95767, 95779, 95791, 95815,
95839, 95863, 95875, 95887, 95899, 95911, 95935, 95947, 95959, 96007, 96019, 96031, 96079, 96091, 96103, 96115, 96121, 96157, 96169, 96181, 96205, 96229,
96265, 96277, 96301, 96313, 96337, 96349, 96367, 96415, 96427, 96439, 96463, 96475, 96499, 96523, 96535, 96571, 96583, 96607, 96631, 96655, 96667, 96679,
96691, 96703, 96715, 96739, 96787, 96799, 96811, 96823, 96847, 96859, 96871, 96883, 96895, 96907, 96919, 96931, 96979, 96991, 97003, 97027, 97039, 97063,
97075, 97087, 97111, 97123, 97147, 97171, 97195, 97219, 97231, 97267, 97279, 97291, 97303, 97315, 97339, 97363, 97375, 97387, 97399, 97423, 97447, 97459,
97471, 97495, 97507, 97519, 97531, 97543, 97549, 97573, 97609, 97645, 97657, 97669, 97717, 97741, 97753, 97765, 97777, 97801, 97813, 97825, 97849, 97861,
97873, 97897, 97945, 97957, 97981, 97993, 98005, 98017, 98029, 98041, 98065, 98077, 98089, 98101, 98113, 98137, 98149, 98161, 98179, 98203, 98227, 98251,
98263, 98275, 98323, 98335, 98347, 98359, 98383, 98407, 98419, 98431, 98443, 98467, 98491, 98503, 98527, 98551, 98575, 98587, 98599, 98611, 98623, 98647,
98671, 98683, 98695, 98707, 98731, 98755, 98761, 98773, 98797, 98821, 98833, 98845, 98881, 98893, 98905, 98917, 98929, 98941, 98953, 98977, 98989, 99001,
99013, 99037, 99049, 99061, 99073, 99085, 99109, 99133, 99145, 99157, 99181, 99205, 99217, 99241, 99253, 99265, 99277, 99301, 99313, 99385, 99397, 99409,
99433, 99445, 99457, 99505, 99517, 99541, 99565, 99577, 99589, 99613, 99625, 99637, 99661, 99685, 99697, 99721, 99733, 99745, 99757, 99781, 99793, 99805,
99829, 99841, 99853, 99865, 99877, 99889, 99925, 99937, 99943, 99991, 100003], dtype=np.int32)   

square_Ns = np.array([0, 1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 26, 29, 32, 34, 36, 37, 40, 41, 45, 49, 50, 52, 53, 58, 61, 64, 65, 68, 72, 73, 74, 80, 81, 82, 85, 89, 90,
97, 98, 100, 101, 104, 106, 109, 113, 116, 117, 121, 122, 125, 128, 130, 136, 137, 144, 145, 146, 148, 149, 153, 157, 160, 162, 164, 169, 170, 173, 178,
180, 181, 185, 193, 194, 196, 197, 200, 202, 205, 208, 212, 218, 221, 225, 226, 229, 232, 233, 234, 241, 242, 244, 245, 250, 256, 257, 260, 261, 265, 269,
272, 274, 277, 281, 288, 289, 290, 292, 293, 296, 298, 305, 306, 313, 314, 317, 320, 324, 325, 328, 333, 337, 338, 340, 346, 349, 353, 356, 360, 361, 362,
365, 369, 370, 373, 377, 386, 388, 389, 392, 394, 397, 400, 401, 404, 405, 409, 410, 416, 421, 424, 425, 433, 436, 441, 442, 445, 449, 450, 452, 457, 458,
461, 464, 466, 468, 477, 481, 482, 484, 485, 488, 490, 493, 500, 505, 509, 512, 514, 520, 521, 522, 529, 530, 533, 538, 541, 544, 545, 548, 549, 554, 557,
562, 565, 569, 576, 577, 578, 580, 584, 585, 586, 592, 593, 596, 601, 605, 610, 612, 613, 617, 625, 626, 628, 629, 634, 637, 640, 641, 648, 650, 653, 656,
657, 661, 666, 673, 674, 676, 677, 680, 685, 689, 692, 697, 698, 701, 706, 709, 712, 720, 722, 724, 725, 729, 730, 733, 738, 740, 745, 746, 754, 757, 761,
765, 769, 772, 773, 776, 778, 784, 785, 788, 793, 794, 797, 800, 801, 802, 808, 809, 810, 818, 820, 821, 829, 832, 833, 841, 842, 845, 848, 850, 853, 857,
865, 866, 872, 873, 877, 881, 882, 884, 890, 898, 900, 901, 904, 905, 909, 914, 916, 922, 925, 928, 929, 932, 936, 937, 941, 949, 953, 954, 961, 962, 964,
965, 968, 970, 976, 977, 980, 981, 985, 986, 997, 1000, 1009, 1010, 1013, 1017, 1018, 1021, 1024, 1025, 1028, 1033, 1037, 1040, 1042, 1044, 1049, 1053,
1058, 1060, 1061, 1066, 1069, 1073, 1076, 1082, 1088, 1089, 1090, 1093, 1096, 1097, 1098, 1105, 1108, 1109, 1114, 1117, 1124, 1125, 1129, 1130, 1138, 1145,
1152, 1153, 1154, 1156, 1157, 1160, 1165, 1168, 1170, 1172, 1181, 1184, 1186, 1189, 1192, 1193, 1201, 1202, 1205, 1210, 1213, 1217, 1220, 1224, 1225, 1226,
1229, 1233, 1234, 1237, 1241, 1249, 1250, 1252, 1256, 1258, 1261, 1268, 1274, 1277, 1280, 1282, 1285, 1289, 1296, 1297, 1300, 1301, 1305, 1306, 1312, 1313,
1314, 1321, 1322, 1325, 1332, 1341, 1345, 1346, 1348, 1352, 1354, 1360, 1361, 1369, 1370, 1373, 1377, 1378, 1381, 1384, 1385, 1394, 1396, 1402, 1405, 1409,
1412, 1413, 1417, 1418, 1421, 1424, 1429, 1433, 1440, 1444, 1445, 1448, 1450, 1453, 1458, 1460, 1465, 1466, 1469, 1476, 1480, 1481, 1489, 1490, 1492, 1493,
1508, 1513, 1514, 1517, 1521, 1522, 1525, 1530, 1537, 1538, 1544, 1546, 1549, 1552, 1553, 1556, 1557, 1565, 1568, 1570, 1573, 1576, 1585, 1586, 1588, 1594,
1597, 1600, 1601, 1602, 1604, 1609, 1613, 1616, 1618, 1620, 1621, 1625, 1629, 1636, 1637, 1640, 1642, 1649, 1657, 1658, 1664, 1665, 1666, 1669, 1681, 1682,
1684, 1685, 1690, 1693, 1696, 1697, 1700, 1706, 1709, 1714, 1717, 1721, 1730, 1732, 1733, 1737, 1741, 1744, 1745, 1746, 1753, 1754, 1762, 1764, 1765, 1768,
1769, 1773, 1777, 1780, 1781, 1789, 1796, 1800, 1801, 1802, 1805, 1808, 1810, 1813, 1818, 1825, 1828, 1832, 1844, 1845, 1849, 1850, 1853, 1856, 1858, 1861,
1864, 1865, 1872, 1873, 1874, 1877, 1882, 1885, 1889, 1898, 1901, 1906, 1908, 1913, 1921, 1922, 1924, 1928, 1930, 1933, 1936, 1937, 1940, 1945, 1949, 1952,
1954, 1960, 1961, 1962, 1970, 1972, 1973, 1985, 1989, 1993, 1994, 1997, 2000, 2005, 2009, 2017, 2018, 2020, 2025, 2026, 2029, 2034, 2036, 2041, 2042, 2045,
2048, 2050, 2053, 2056, 2057, 2061, 2066, 2069, 2074, 2080, 2081, 2084, 2088, 2089, 2097, 2098, 2105, 2106, 2113, 2116, 2117, 2120, 2122, 2125, 2129, 2132,
2137, 2138, 2141, 2146, 2152, 2153, 2161, 2164, 2165, 2169, 2173, 2176, 2178, 2180, 2186, 2192, 2194, 2196, 2197, 2205, 2209, 2210, 2213, 2216, 2218, 2221,
2225, 2228, 2234, 2237, 2245, 2248, 2249, 2250, 2257, 2258, 2260, 2269, 2273, 2276, 2281, 2285, 2290, 2293, 2297, 2304, 2305, 2306, 2308, 2309, 2312, 2313,
2314, 2320, 2329, 2330, 2333, 2336, 2340, 2341, 2344, 2349, 2353, 2357, 2362, 2368, 2372, 2377, 2378, 2381, 2384, 2385, 2386, 2389, 2393, 2401, 2402, 2404,
2405, 2410, 2417, 2420, 2421, 2425, 2426, 2434, 2437, 2440, 2441, 2448, 2450, 2452, 2458, 2465, 2466, 2468, 2473, 2474, 2477, 2482, 2493, 2498, 2500, 2501,
2504, 2509, 2512, 2516, 2521, 2522, 2525, 2529, 2533, 2536, 2545, 2548, 2549, 2554, 2557, 2560, 2561, 2564, 2570, 2578, 2581, 2592, 2593, 2594, 2597, 2600,
2601, 2602, 2605, 2609, 2610, 2612, 2617, 2621, 2624, 2626, 2628, 2633, 2637, 2642, 2644, 2645, 2650, 2657, 2664, 2665, 2669, 2677, 2682, 2689, 2690, 2692,
2693, 2696, 2701, 2704, 2705, 2708, 2713, 2720, 2722, 2725, 2729, 2738, 2740, 2741, 2745, 2746, 2749, 2753, 2754, 2756, 2762, 2768, 2770, 2777, 2785, 2788,
2789, 2792, 2797, 2801, 2804, 2809, 2810, 2813, 2817, 2818, 2824, 2825, 2826, 2833, 2834, 2836, 2837, 2842, 2845, 2848, 2853, 2857, 2858, 2861, 2866, 2873,
2880, 2885, 2888, 2890, 2896, 2897, 2900, 2906, 2909, 2916, 2917, 2920, 2925, 2929, 2930, 2932, 2938, 2941, 2952, 2953, 2957, 2960, 2962, 2965, 2969, 2977,
2978, 2980, 2984, 2986, 2989, 2993, 2997, 3001, 3005, 3016, 3025, 3026, 3028, 3029, 3033, 3034, 3037, 3041, 3042, 3044, 3049, 3050, 3060, 3061, 3065, 3074,
3076, 3077, 3085, 3088, 3089, 3092, 3098, 3104, 3106, 3109, 3112, 3114, 3121, 3125, 3130, 3133, 3136, 3137, 3140, 3141, 3145, 3146, 3152, 3161, 3169, 3170,
3172, 3176, 3177, 3181, 3185, 3188, 3194, 3200, 3202, 3204, 3205, 3208, 3209, 3217, 3218, 3221, 3226, 3229, 3232, 3233, 3236, 3240, 3242, 3249, 3250, 3253,
3257, 3258, 3265, 3272, 3274, 3277, 3280, 3281, 3284, 3285, 3293, 3298, 3301, 3305, 3313, 3314, 3316, 3321, 3328, 3329, 3330, 3332, 3338, 3341, 3349, 3357,
3361, 3362, 3364, 3365, 3368, 3370, 3373, 3380, 3385, 3386, 3389, 3392, 3393, 3394, 3400, 3412, 3413, 3418, 3425, 3428, 3433, 3434, 3442, 3445, 3449, 3457,
3460, 3461, 3464, 3466, 3469, 3474, 3481, 3482, 3485, 3488, 3490, 3492, 3497, 3501, 3505, 3506, 3508, 3509, 3517, 3524, 3528, 3529, 3530, 3533, 3536, 3538,
3541, 3545, 3546, 3554, 3557, 3560, 3562, 3573, 3577, 3578, 3581, 3589, 3592, 3593, 3600, 3601, 3602, 3604, 3609, 3610, 3613, 3616, 3617, 3620, 3625, 3626,
3636, 3637, 3645, 3649, 3650, 3653, 3656, 3664, 3665, 3673, 3677, 3681, 3688, 3690, 3697, 3698, 3700, 3701, 3706, 3709, 3712, 3716, 3721, 3722, 3725, 3728,
3730, 3733, 3737, 3744, 3746, 3748, 3754, 3757, 3761, 3764, 3769, 3770, 3778, 3785, 3789, 3793, 3796, 3797, 3802, 3805, 3809, 3812, 3816, 3821, 3825, 3826,
3833, 3842, 3844, 3845, 3848, 3853, 3856, 3860, 3865, 3866, 3869, 3872, 3874, 3877, 3880, 3881, 3889, 3890, 3893, 3897, 3898, 3904, 3908, 3917, 3920, 3922,
3924, 3925, 3929, 3940, 3944, 3946, 3961, 3965, 3969, 3970, 3973, 3977, 3978, 3985, 3986, 3988, 3989, 3994, 4000, 4001, 4005, 4010, 4013, 4018, 4021, 4033,
4034, 4036, 4040, 4041, 4045, 4049, 4050, 4052, 4057, 4058, 4068, 4069, 4072, 4073, 4082, 4084, 4090, 4093, 4096, 4097, 4100, 4105, 4106, 4112, 4113, 4114,
4121, 4122, 4129, 4132, 4133, 4138, 4141, 4145, 4148, 4149, 4153, 4157, 4160, 4162, 4165, 4168, 4176, 4177, 4178, 4181, 4194, 4196, 4201, 4205, 4210, 4212,
4217, 4225, 4226, 4229, 4232, 4234, 4240, 4241, 4244, 4250, 4253, 4258, 4261, 4264, 4265, 4273, 4274, 4276, 4282, 4285, 4289, 4292, 4293, 4297, 4304, 4306,
4321, 4322, 4325, 4328, 4329, 4330, 4337, 4338, 4346, 4349, 4352, 4356, 4357, 4360, 4361, 4365, 4369, 4372, 4373, 4381, 4384, 4385, 4388, 4392, 4394, 4397,
4405, 4409, 4410, 4418, 4420, 4421, 4426, 4432, 4436, 4437, 4441, 4442, 4450, 4453, 4456, 4457, 4468, 4469, 4474, 4477, 4481, 4489, 4490, 4493, 4496, 4498,
4500, 4505, 4513, 4514, 4516, 4517, 4520, 4525, 4537, 4538, 4545, 4546, 4549, 4552, 4553, 4561, 4562, 4570, 4573, 4580, 4581, 4586, 4589, 4594, 4597, 4608,
4610, 4612, 4616, 4618, 4621, 4624, 4625, 4626, 4628, 4633, 4637, 4640, 4645, 4649, 4657, 4658, 4660, 4666, 4672, 4673, 4680, 4682, 4685, 4688, 4689, 4693,
4698, 4705, 4706, 4709, 4714, 4717, 4721, 4724, 4729, 4733, 4736, 4744, 4745, 4753, 4754, 4756, 4761, 4762, 4765, 4768, 4770, 4772, 4777, 4778, 4786, 4789,
4793, 4797, 4801, 4802, 4804, 4805, 4808, 4810, 4813, 4817, 4820, 4825, 4834, 4840, 4842, 4849, 4850, 4852, 4861, 4868, 4869, 4874, 4877, 4880, 4882, 4885,
4889, 4896, 4900, 4901, 4904, 4905, 4909, 4913, 4916, 4925, 4930, 4932, 4933, 4936, 4937, 4941, 4946, 4948, 4949, 4954, 4957, 4961, 4964, 4969, 4973, 4981,
4985, 4986, 4993, 4996, 5000, 5002, 5008, 5009, 5013, 5017, 5018, 5021, 5024, 5032, 5041, 5042, 5044, 5045, 5050, 5057, 5058, 5065, 5066, 5069, 5072, 5077,
5081, 5085, 5090, 5096, 5098, 5101, 5105, 5108, 5113, 5114, 5120, 5121, 5122, 5125, 5128, 5140, 5141, 5153, 5156, 5161, 5162, 5165, 5184, 5185, 5186, 5188,
5189, 5193, 5194, 5197, 5200, 5202, 5204, 5209, 5210, 5213, 5218, 5220, 5224, 5233, 5234, 5237, 5242, 5245, 5248, 5249, 5252, 5256, 5261, 5265, 5266, 5273,
5274, 5281, 5284, 5288, 5290, 5297, 5300, 5305, 5309, 5314, 5317, 5321, 5328, 5329, 5330, 5333, 5337, 5338, 5341, 5345, 5353, 5354, 5364, 5365, 5378, 5380,
5381, 5384, 5386, 5389, 5392, 5393, 5402, 5408, 5409, 5410, 5413, 5416, 5417, 5426, 5429, 5437, 5440, 5441, 5444, 5445, 5449, 5450, 5458, 5465, 5473, 5476,
5477, 5480, 5482, 5485, 5490, 5492, 5498, 5501, 5506, 5508, 5512, 5513, 5517, 5521, 5524, 5525, 5536, 5537, 5540, 5545, 5553, 5554, 5557, 5569, 5570, 5573,
5576, 5578, 5581, 5584, 5585, 5594, 5597, 5602, 5608, 5617, 5618, 5620, 5625, 5626, 5629, 5634, 5636, 5641, 5645, 5648, 5650, 5652, 5653, 5657, 5661, 5666,
5668, 5669, 5672, 5674, 5684, 5689, 5690, 5693, 5696, 5701, 5706, 5713, 5714, 5716, 5717, 5722, 5725, 5729, 5732, 5733, 5737, 5741, 5746, 5749, 5760, 5765,
5769, 5770, 5776, 5777, 5780, 5785, 5792, 5794, 5800, 5801, 5809, 5812, 5813, 5818, 5821, 5825, 5832, 5834, 5837, 5840, 5849, 5850, 5857, 5858, 5860, 5861,
5864, 5869, 5876, 5877, 5881, 5882, 5897, 5904, 5905, 5906, 5913, 5914, 5917, 5920, 5924, 5929, 5930, 5933, 5938, 5941, 5945, 5949, 5953, 5954, 5956, 5960,
5965, 5968, 5972, 5978, 5981, 5986, 5989, 5993, 5994, 6001, 6002, 6005, 6010, 6025, 6029, 6032, 6037, 6050, 6052, 6053, 6056, 6057, 6058, 6065, 6066, 6068,
6073, 6074, 6082, 6084, 6085, 6088, 6089, 6093, 6098, 6100, 6101, 6109, 6113, 6120, 6121, 6122, 6125, 6130, 6133, 6137, 6145, 6148, 6152, 6154, 6161, 6165,
6170, 6173, 6176, 6178, 6184, 6185, 6196, 6197, 6201, 6205, 6208, 6212, 6217, 6218, 6221, 6224, 6228, 6229, 6241, 6242, 6245, 6250, 6253, 6257, 6260, 6266,
6269, 6272, 6273, 6274, 6277, 6280, 6282, 6290, 6292, 6301, 6304, 6305, 6309, 6317, 6322, 6329, 6337, 6338, 6340, 6341, 6344, 6352, 6353, 6354, 6361, 6362,
6370, 6373, 6376, 6381, 6385, 6388, 6389, 6397, 6400, 6401, 6404, 6408, 6409, 6410, 6413, 6416, 6418, 6421, 6425, 6434, 6436, 6437, 6442, 6445, 6449, 6452,
6458, 6464, 6466, 6469, 6472, 6473, 6480, 6481, 6484, 6485, 6497, 6498, 6500, 6505, 6506, 6514, 6516, 6521, 6525, 6529, 6530, 6544, 6548, 6553, 6554, 6560,
6561, 6562, 6565, 6568, 6569, 6570, 6577, 6581, 6586, 6596, 6597, 6602, 6605, 6610, 6613, 6617, 6625, 6626, 6628, 6632, 6637, 6641, 6642, 6649, 6653, 6656,
6658, 6660, 6661, 6664, 6673, 6676, 6682, 6689, 6697, 6698, 6701, 6705, 6709, 6713, 6714, 6722, 6724, 6725, 6728, 6730, 6733, 6736, 6737, 6740, 6746, 6749,
6757, 6760, 6761, 6770, 6772, 6773, 6778, 6781, 6784, 6786, 6788, 6793, 6800, 6805, 6813, 6817, 6824, 6826, 6829, 6833, 6836, 6841, 6845, 6849, 6850, 6856,
6857, 6865, 6866, 6868, 6869, 6877, 6884, 6885, 6889, 6890, 6893, 6898, 6905, 6914, 6917, 6920, 6921, 6922, 6925, 6928, 6929, 6932, 6938, 6948, 6949, 6953,
6957, 6961, 6962, 6964, 6970, 6976, 6977, 6980, 6984, 6989, 6994, 6997, 7001, 7002, 7010, 7012, 7013, 7016, 7018, 7025, 7033, 7034, 7045, 7048, 7056, 7057,
7058, 7060, 7065, 7066, 7069, 7072, 7076, 7081, 7082, 7085, 7090, 7092, 7093, 7105, 7108, 7109, 7114, 7120, 7121, 7124, 7129, 7137, 7141, 7145, 7146, 7154,
7156, 7157, 7162, 7165, 7173, 7177, 7178, 7184, 7186, 7193, 7200, 7202, 7204, 7208, 7209, 7213, 7218, 7220, 7225, 7226, 7229, 7232, 7234, 7237, 7240, 7241,
7250, 7252, 7253, 7261, 7265, 7272, 7274, 7281, 7289, 7290, 7297, 7298, 7300, 7301, 7306, 7309, 7312, 7321, 7325, 7328, 7330, 7333, 7345, 7346, 7349, 7354,
7361, 7362, 7369, 7373, 7376, 7380, 7381, 7389, 7393, 7394, 7396, 7397, 7400, 7402, 7405, 7412, 7417, 7418, 7421, 7424, 7432, 7433, 7442, 7444, 7445, 7450,
7453, 7456, 7457, 7460, 7461, 7465, 7466, 7474, 7477, 7481, 7488, 7489, 7492, 7496, 7497, 7501, 7508, 7514, 7517, 7522, 7528, 7529, 7537, 7538, 7540, 7541,
7549, 7556, 7561, 7565, 7569, 7570, 7573, 7577, 7578, 7585, 7586, 7589, 7592, 7594, 7604, 7605, 7610, 7618, 7621, 7624, 7625, 7632, 7633, 7642, 7649, 7650,
7652, 7666, 7669, 7673, 7677, 7681, 7684, 7685, 7688, 7690, 7693, 7696, 7706, 7709, 7712, 7713, 7717, 7720, 7730, 7732, 7738, 7741, 7744, 7745, 7748, 7753,
7754, 7757, 7760, 7762, 7765, 7769, 7778, 7780, 7785, 7786, 7789, 7793, 7794, 7796, 7801, 7808, 7813, 7816, 7817, 7825, 7829, 7834, 7837, 7840, 7841, 7844,
7848, 7850, 7853, 7857, 7858, 7865, 7873, 7877, 7880, 7888, 7892, 7893, 7897, 7901, 7913, 7921, 7922, 7925, 7929, 7930, 7933, 7937, 7938, 7940, 7946, 7949,
7954, 7956, 7957, 7969, 7970, 7972, 7976, 7978, 7985, 7988, 7993, 8000, 8002, 8005, 8009, 8010, 8017, 8020, 8021, 8026, 8033, 8036, 8042, 8045, 8053, 8065,
8066, 8068, 8069, 8072, 8077, 8080, 8081, 8082, 8089, 8090, 8093, 8098, 8100, 8101, 8104, 8105, 8109, 8114, 8116, 8117, 8125, 8136, 8138, 8144, 8145, 8146,
8149, 8161, 8164, 8168, 8177, 8180, 8181, 8185, 8186, 8192, 8194, 8200, 8209, 8210, 8212, 8221, 8224, 8226, 8228, 8233, 8237, 8242, 8244, 8245, 8249, 8258,
8264, 8266, 8269, 8273, 8276, 8281, 8282, 8285, 8290, 8293, 8296, 8297, 8298, 8306, 8314, 8317, 8320, 8321, 8324, 8325, 8329, 8330, 8333, 8336, 8345, 8352,
8353, 8354, 8356, 8357, 8361, 8362, 8369, 8377, 8381, 8388, 8389, 8392, 8402, 8405, 8410, 8420, 8424, 8425, 8429, 8433, 8434, 8450, 8452, 8458, 8461, 8464,
8465, 8468, 8469, 8473, 8477, 8480, 8482, 8485, 8488, 8489, 8497, 8500, 8501, 8506, 8513, 8516, 8521, 8522, 8528, 8530, 8537, 8541, 8545, 8546, 8548, 8552,
8564, 8570, 8573, 8577, 8578, 8581, 8584, 8585, 8586, 8593, 8594, 8597, 8605, 8608, 8609, 8612, 8621, 8629, 8633, 8641, 8642, 8644, 8649, 8650, 8653, 8656,
8658, 8660, 8665, 8669, 8674, 8676, 8677, 8681, 8685, 8689, 8692, 8693, 8698, 8704, 8705, 8712, 8713, 8714, 8720, 8722, 8725, 8730, 8737, 8738, 8741, 8744,
8746, 8749, 8753, 8761, 8762, 8765, 8768, 8770, 8776, 8784, 8788, 8793, 8794, 8801, 8810, 8818, 8820, 8821, 8825, 8829, 8833, 8836, 8837, 8840, 8842, 8845,
8849, 8852, 8857, 8861, 8864, 8865, 8869, 8872, 8874, 8882, 8884, 8885, 8893, 8900, 8905, 8906, 8912, 8914, 8917, 8929, 8933, 8936, 8938, 8941, 8945, 8948,
8954, 8957, 8962, 8969, 8973, 8978, 8980, 8986, 8989, 8992, 8993, 8996, 9000, 9001, 9005, 9010, 9013, 9025, 9026, 9028, 9029, 9032, 9034, 9040, 9041, 9049,
9050, 9061, 9065, 9074, 9076, 9077, 9081, 9089, 9090, 9092, 9098, 9104, 9106, 9109, 9113, 9117, 9122, 9124, 9125, 9133, 9137, 9140, 9146, 9153, 9157, 9160,
9161, 9162, 9169, 9172, 9173, 9178, 9181, 9188, 9189, 9193, 9194, 9197, 9209, 9216, 9217, 9220, 9221, 9224, 9225, 9232, 9236, 9241, 9242, 9245, 9248, 9250,
9252, 9256, 9257, 9265, 9266, 9274, 9277, 9280, 9281, 9290, 9293, 9297, 9298, 9305, 9314, 9316, 9320, 9325, 9332, 9333, 9337, 9341, 9344, 9346, 9349, 9360,
9364, 9365, 9370, 9376, 9377, 9378, 9385, 9386, 9389, 9396, 9397, 9409, 9410, 9412, 9413, 9418, 9421, 9425, 9428, 9433, 9434, 9437, 9441, 9442, 9445, 9448,
9457, 9458, 9461, 9466, 9469, 9472, 9473, 9477, 9488, 9490, 9497, 9505, 9506, 9508, 9509, 9512, 9521, 9522, 9524, 9529, 9530, 9533, 9536, 9540, 9544, 9549,
9553, 9554, 9556, 9565, 9572, 9577, 9578, 9586, 9593, 9594, 9601, 9602, 9604, 9605, 9608, 9610, 9613, 9616, 9620, 9621, 9626, 9629, 9634, 9640, 9649, 9650,
9653, 9657, 9661, 9665, 9668, 9673, 9677, 9680, 9684, 9685, 9689, 9697, 9698, 9700, 9701, 9704, 9721, 9722, 9725, 9733, 9736, 9738, 9745, 9748, 9749, 9754,
9760, 9764, 9769, 9770, 9773, 9778, 9781, 9792, 9797, 9800, 9801, 9802, 9805, 9808, 9809, 9810, 9817, 9818, 9826, 9829, 9832, 9833, 9837, 9841, 9850, 9857,
9860, 9864, 9865, 9866, 9872, 9873, 9874, 9881, 9882, 9892, 9893, 9896, 9898, 9901, 9908, 9914, 9922, 9925, 9928, 9929, 9938, 9941, 9945, 9946, 9949, 9953,
9962, 9965, 9970, 9972, 9973, 9981, 9985, 9986, 9992, 9997, 10000, 10001, 10004, 10009, 10016, 10018, 10025, 10026, 10034, 10036, 10037, 10042, 10045,
10048, 10049, 10053, 10057, 10061, 10064, 10069, 10081, 10082, 10084, 10085, 10088, 10090, 10093, 10100, 10114, 10116, 10121, 10125, 10130, 10132, 10133,
10138, 10141, 10144, 10145, 10154, 10161, 10162, 10169, 10170, 10177, 10180, 10181, 10192, 10193, 10196, 10201, 10202, 10205, 10210, 10216, 10217, 10225,
10226, 10228, 10229, 10237, 10240, 10242, 10244, 10249, 10250, 10253, 10256, 10265, 10273, 10280, 10282, 10285, 10289, 10301, 10305, 10306, 10309, 10312,
10313, 10321, 10322, 10324, 10330, 10333, 10337, 10345, 10357, 10361, 10368, 10369, 10370, 10372, 10376, 10377, 10378, 10386, 10388, 10394, 10397, 10400,
10404, 10405, 10408, 10413, 10418, 10420, 10421, 10426, 10429, 10433, 10436, 10440, 10441, 10445, 10448, 10453, 10457, 10466, 10468, 10469, 10474, 10477,
10484, 10485, 10489, 10490, 10496, 10498, 10501, 10504, 10512, 10513, 10517, 10522, 10525, 10529, 10530, 10532, 10537, 10546, 10548, 10553, 10562, 10565,
10568, 10573, 10576, 10580, 10585, 10589, 10594, 10597, 10600, 10601, 10609, 10610, 10613, 10618, 10625, 10628, 10629, 10634, 10642, 10645, 10656, 10657,
10658, 10660, 10666, 10673, 10674, 10676, 10682, 10685, 10690, 10693, 10701, 10705, 10706, 10708, 10709, 10728, 10729, 10730, 10733, 10737, 10753, 10756,
10760, 10762, 10765, 10768, 10769, 10772, 10777, 10778, 10781, 10784, 10786, 10789, 10804, 10805, 10809, 10816, 10817, 10818, 10820, 10825, 10826, 10829,
10832, 10834, 10837, 10841, 10845, 10852, 10853, 10858, 10861, 10865, 10874, 10877, 10880, 10882, 10888, 10889, 10890, 10897, 10898, 10900, 10909, 10916,
10917, 10930, 10933, 10937, 10946, 10949, 10952, 10953, 10954, 10957, 10960, 10961, 10964, 10970, 10973, 10980, 10984, 10985, 10993, 10996, 11002, 11009,
11012, 11016, 11024, 11025, 11026, 11029, 11034, 11041, 11042, 11045, 11048, 11050, 11057, 11061, 11065, 11069, 11072, 11074, 11080, 11089, 11090, 11093,
11097, 11101, 11105, 11106, 11108, 11113, 11114, 11117, 11125, 11133, 11138, 11140, 11141, 11146, 11149, 11152, 11156, 11161, 11162, 11168, 11169, 11170,
11173, 11177, 11185, 11188, 11194, 11197, 11204, 11213, 11216, 11221, 11225, 11234, 11236, 11237, 11240, 11241, 11245, 11250, 11252, 11257, 11258, 11261,
11268, 11272, 11273, 11281, 11282, 11285, 11290, 11296, 11300, 11304, 11306, 11314, 11317, 11321, 11322, 11329, 11332, 11336, 11338, 11344, 11345, 11348,
11349, 11353, 11357, 11365, 11368, 11369, 11378, 11380, 11386, 11392, 11393, 11401, 11402, 11405, 11412, 11413, 11417, 11425, 11426, 11428, 11432, 11434,
11437, 11441, 11444, 11449, 11450, 11453, 11458, 11461, 11464, 11465, 11466, 11474, 11482, 11485, 11489, 11492, 11493, 11497, 11498, 11509, 11513, 11520,
11521, 11525, 11530, 11538, 11540, 11545, 11549, 11552, 11554, 11560, 11565, 11570, 11581, 11584, 11588, 11593, 11597, 11600, 11601, 11602, 11617, 11618,
11621, 11624, 11626, 11629, 11633, 11636, 11642, 11645, 11650, 11657, 11664, 11665, 11668, 11673, 11674, 11677, 11680, 11681, 11689, 11698, 11700, 11701,
11705, 11709, 11713, 11714, 11716, 11717, 11720, 11722, 11728, 11729, 11737, 11738, 11745, 11752, 11754, 11762, 11764, 11765, 11773, 11777, 11785, 11789,
11794, 11801, 11808, 11809, 11810, 11812, 11813, 11817, 11821, 11826, 11828, 11833, 11834, 11840, 11848, 11849, 11858, 11860, 11861, 11866, 11876, 11881,
11882, 11885, 11889, 11890, 11897, 11898, 11905, 11906, 11908, 11909, 11912, 11917, 11920, 11925, 11930, 11933, 11936, 11941, 11944, 11945, 11953, 11956,
11962, 11965, 11969, 11972, 11978, 11981, 11986, 11988, 12002, 12004, 12005, 12010, 12013, 12017, 12020, 12025, 12037, 12041, 12049, 12050, 12053, 12058,
12064, 12069, 12073, 12074, 12077, 12085, 12097, 12100, 12101, 12104, 12105, 12106, 12109, 12112, 12113, 12114, 12116, 12125, 12130, 12132, 12136, 12137,
12146, 12148, 12149, 12157, 12161, 12164, 12168, 12170, 12176, 12178, 12181, 12185, 12186, 12193, 12196, 12197, 12200, 12202, 12205, 12209, 12218, 12221,
12226, 12233, 12240, 12241, 12242, 12244, 12249, 12250, 12253, 12260, 12266, 12269, 12274, 12277, 12281, 12289, 12290, 12296, 12301, 12304, 12308, 12317,
12321, 12322, 12325, 12329, 12330, 12337, 12340, 12346, 12349, 12352, 12356, 12357, 12365, 12368, 12370, 12373, 12377, 12385, 12389, 12392, 12393, 12394,
12401, 12402, 12409, 12410, 12413, 12416, 12421, 12424, 12429, 12433, 12434, 12436, 12437, 12442, 12448, 12456, 12457, 12458, 12461, 12465, 12469, 12473,
12482, 12484, 12490, 12493, 12497, 12500, 12505, 12506, 12514, 12517, 12520, 12532, 12538, 12541, 12544, 12545, 12546, 12548, 12553, 12554, 12557, 12560,
12564, 12569, 12577, 12580, 12584, 12589, 12593, 12601, 12602, 12605, 12608, 12610, 12613, 12618, 12625, 12629, 12634, 12637, 12641, 12644, 12645, 12653,
12658, 12665, 12674, 12676, 12680, 12681, 12682, 12688, 12689, 12697, 12701, 12704, 12706, 12708, 12713, 12717, 12721, 12722, 12724, 12725, 12740, 12745,
12746, 12752, 12753, 12757, 12762, 12769, 12770, 12773, 12776, 12778, 12781, 12785, 12789, 12794, 12800, 12802, 12805, 12808, 12809, 12816, 12818, 12820,
12821, 12826, 12829, 12832, 12833, 12836, 12841, 12842, 12850, 12853, 12861, 12868, 12869, 12872, 12874, 12884, 12889, 12890, 12893, 12897, 12898, 12904,
12905, 12913, 12916, 12917, 12928, 12932, 12937, 12938, 12941, 12944, 12946, 12953, 12960, 12961, 12962, 12965, 12968, 12970, 12973, 12985, 12994, 12996,
12997, 13000, 13001, 13005, 13009, 13010, 13012, 13021, 13025, 13028, 13032, 13033, 13037, 13042, 13045, 13049, 13050, 13058, 13060, 13061, 13073, 13077,
13085, 13088, 13093, 13096, 13105, 13106, 13108, 13109, 13117, 13120, 13121, 13122, 13124, 13130, 13136, 13138, 13140, 13141, 13154, 13162, 13165, 13169,
13172, 13177, 13181, 13185, 13189, 13192, 13194, 13204, 13210, 13213, 13217, 13220, 13221, 13225, 13226, 13229, 13234, 13241, 13249, 13250, 13252, 13253,
13256, 13261, 13264, 13273, 13274, 13282, 13284, 13285, 13289, 13297, 13298, 13306, 13309, 13312, 13313, 13316, 13320, 13322, 13325, 13328, 13329, 13337,
13345, 13346, 13352, 13357, 13364, 13369, 13378, 13381, 13385, 13394, 13396, 13397, 13401, 13402, 13410, 13417, 13418, 13421, 13426, 13428, 13429, 13437,
13441, 13444, 13445, 13448, 13450, 13456, 13457, 13460, 13465, 13466, 13469, 13472, 13474, 13477, 13480, 13481, 13492, 13498, 13505, 13513, 13514, 13520,
13522, 13525, 13537, 13540, 13544, 13546, 13549, 13553, 13556, 13562, 13565, 13568, 13572, 13573, 13576, 13577, 13586, 13597, 13600, 13610, 13613, 13617,
13621, 13625, 13626, 13633, 13634, 13637, 13645, 13648, 13649, 13652, 13653, 13658, 13666, 13669, 13672, 13673, 13681, 13682, 13689, 13690, 13693, 13697,
13698, 13700, 13705, 13709, 13712, 13714, 13721, 13725, 13729, 13730, 13732, 13736, 13738, 13745, 13753, 13754, 13757, 13765, 13768, 13769, 13770, 13778,
13780, 13781, 13786, 13789, 13793, 13796, 13801, 13810, 13817, 13828, 13829, 13833, 13834, 13837, 13840, 13841, 13842, 13844, 13850, 13856, 13858, 13864,
13873, 13876, 13877, 13885, 13896, 13897, 13898, 13901, 13906, 13913, 13914, 13921, 13922, 13924, 13925, 13928, 13933, 13940, 13941, 13945, 13949, 13952,
13954, 13957, 13960, 13968, 13969, 13973, 13977, 13978, 13985, 13988, 13994, 13997, 14002, 14004, 14005, 14009, 14013, 14020, 14024, 14026, 14029, 14032,
14033, 14036, 14045, 14050, 14057, 14065, 14066, 14068, 14081, 14085, 14089, 14090, 14093, 14096, 14112, 14114, 14116, 14120, 14125, 14130, 14132, 14138,
14144, 14149, 14152, 14153, 14157, 14161, 14162, 14164, 14165, 14170, 14173, 14177, 14180, 14184, 14185, 14186, 14197, 14209, 14210, 14213, 14216, 14218,
14221, 14225, 14228, 14240, 14242, 14248, 14249, 14257, 14258, 14261, 14265, 14274, 14281, 14282, 14285, 14290, 14292, 14293, 14297, 14305, 14308, 14309,
14312, 14314, 14321, 14324, 14330, 14341, 14346, 14354, 14356, 14357, 14365, 14368, 14369, 14372, 14373, 14381, 14386, 14389, 14393, 14400, 14401, 14404,
14408, 14409, 14416, 14417, 14418, 14425, 14426, 14436, 14437, 14440, 14449, 14450, 14452, 14453, 14458, 14461, 14464, 14468, 14473, 14474, 14480, 14481,
14482, 14485, 14489, 14500, 14501, 14504, 14506, 14517, 14521, 14522, 14530, 14533, 14537, 14544, 14545, 14548, 14549, 14557, 14561, 14562, 14569, 14578,
14580, 14585, 14589, 14593, 14594, 14596, 14600, 14602, 14612, 14618, 14621, 14624, 14625, 14629, 14633, 14641, 14642, 14645, 14650, 14653, 14656, 14657,
14660, 14661, 14666, 14669, 14677, 14681, 14689, 14690, 14692, 14698, 14701, 14705, 14708, 14713, 14717, 14722, 14724, 14733, 14737, 14738, 14741, 14746,
14752, 14753, 14760, 14761, 14762, 14765, 14778, 14785, 14786, 14788, 14792, 14794, 14797, 14800, 14801, 14804, 14810, 14813, 14821, 14824, 14825, 14834,
14836, 14837, 14841, 14842, 14845, 14848, 14864, 14866, 14869, 14884, 14885, 14888, 14890, 14893, 14897, 14900, 14906, 14909, 14912, 14913, 14914, 14920,
14922, 14929, 14930, 14932, 14933, 14945, 14948, 14954, 14957, 14962, 14965, 14969, 14976, 14977, 14978, 14984, 14985, 14989, 14992, 14994, 15002, 15005,
15013, 15016, 15017, 15021, 15025, 15028, 15034, 15041, 15044, 15049, 15053, 15056, 15058, 15061, 15073, 15074, 15076, 15077, 15080, 15082, 15098, 15101,
15109, 15112, 15121, 15122, 15125, 15129, 15130, 15133, 15137, 15138, 15140, 15145, 15146, 15149, 15154, 15156, 15161, 15165, 15170, 15172, 15173, 15178,
15184, 15185, 15188, 15193, 15205, 15208, 15210, 15217, 15220, 15229, 15233, 15236, 15237, 15241, 15242, 15245, 15248, 15250, 15264, 15266, 15269, 15273,
15277, 15284, 15289, 15293, 15298, 15300, 15304, 15305, 15313, 15317, 15325, 15329, 15332, 15337, 15338, 15341, 15346, 15349, 15353, 15354, 15361, 15362,
15368, 15370, 15373, 15376, 15377, 15380, 15381, 15385, 15386, 15392, 15397, 15401, 15412, 15413, 15418, 15424, 15425, 15426, 15434, 15440, 15445, 15453,
15457, 15460, 15461, 15464, 15473, 15476, 15481, 15482, 15488, 15489, 15490, 15493, 15496, 15497, 15506, 15508, 15509, 15514, 15520, 15524, 15529, 15530,
15533, 15538, 15541, 15545, 15556, 15560, 15569, 15570, 15572, 15577, 15578, 15581, 15586, 15588, 15592, 15597, 15601, 15602, 15605, 15613, 15616, 15625,
15626, 15629, 15632, 15633, 15634, 15641, 15649, 15650, 15658, 15661, 15665, 15668, 15669, 15674, 15677, 15680, 15682, 15685, 15688, 15689, 15696, 15700,
15705, 15706, 15714, 15716, 15725, 15730, 15733, 15737, 15746, 15749, 15754, 15760, 15761, 15769, 15773, 15776, 15777, 15784, 15786, 15793, 15794, 15797,
15802, 15805, 15809, 15817, 15821, 15826, 15842, 15844, 15845, 15850, 15857, 15858, 15860, 15866, 15874, 15876, 15877, 15880, 15881, 15885, 15889, 15892,
15898, 15901, 15905, 15908, 15912, 15913, 15914, 15921, 15925, 15929, 15937, 15938, 15940, 15944, 15949, 15952, 15956, 15957, 15970, 15973, 15976, 15977,
15986, 15993, 15997, 16000, 16001, 16004, 16010, 16018, 16020, 16021, 16025, 16029, 16033, 16034, 16040, 16042, 16045, 16052, 16057, 16061, 16066, 16069,
16072, 16073, 16081, 16084, 16085, 16090, 16097, 16101, 16105, 16106, 16109, 16129, 16130, 16132, 16133, 16136, 16138, 16141, 16144, 16145, 16153, 16154,
16160, 16162, 16164, 16165, 16178, 16180, 16186, 16189, 16193, 16196, 16200, 16201, 16202, 16208, 16209, 16210, 16217, 16218, 16228, 16229, 16232, 16234,
16237, 16241, 16245, 16249, 16250, 16253, 16265, 16272, 16273, 16276, 16277, 16285, 16288, 16290, 16292, 16298, 16301, 16317, 16322, 16325, 16328, 16333,
16336, 16337, 16349, 16354, 16360, 16361, 16362, 16369, 16370, 16372, 16381, 16384, 16385, 16388, 16393, 16400, 16405, 16409, 16417, 16418, 16420, 16421,
16424, 16425, 16433, 16441, 16442, 16448, 16452, 16453, 16456, 16465, 16466, 16474, 16477, 16481, 16484, 16488, 16490, 16493, 16498, 16501, 16505, 16513,
16516, 16525, 16528, 16529, 16532, 16538, 16546, 16552, 16553, 16561, 16562, 16564, 16565, 16570, 16573, 16577, 16580, 16586, 16589, 16592, 16594, 16596,
16601, 16605, 16609, 16612, 16613, 16628, 16633, 16634, 16640, 16641, 16642, 16645, 16648, 16649, 16650, 16657, 16658, 16660, 16661, 16666, 16672, 16673,
16677, 16690, 16693, 16704, 16705, 16706, 16708, 16712, 16714, 16717, 16722, 16724, 16729, 16733, 16738, 16741, 16745, 16749, 16754, 16757, 16762, 16769,
16776, 16778, 16781, 16784, 16785, 16801, 16804, 16805, 16810, 16820, 16825, 16829, 16837, 16840, 16848, 16850, 16857, 16858, 16861, 16865, 16866, 16868,
16889, 16893, 16897, 16900, 16901, 16904, 16909, 16913, 16916, 16921, 16922, 16925, 16928, 16930, 16936, 16937, 16938, 16945, 16946, 16949, 16954, 16960,
16964, 16965, 16970, 16976, 16978, 16981, 16993, 16994, 17000, 17001, 17002, 17009, 17012, 17021, 17026, 17029, 17032, 17033, 17041, 17042, 17044, 17053,
17056, 17057, 17060, 17065, 17069, 17074, 17077, 17082, 17090, 17092, 17093, 17096, 17101, 17104, 17109, 17113, 17117, 17125, 17128, 17137, 17140, 17141,
17146, 17153, 17154, 17156, 17161, 17162, 17165, 17168, 17170, 17172, 17173, 17177, 17186, 17188, 17189, 17194, 17197, 17209, 17210, 17216, 17217, 17218,
17221, 17224, 17225, 17242, 17245, 17257, 17258, 17261, 17266, 17282, 17284, 17285, 17288, 17289, 17293, 17297, 17298, 17300, 17305, 17306, 17312, 17316,
17317, 17320, 17321, 17330, 17333, 17338, 17341, 17345, 17348, 17352, 17354, 17357, 17362, 17370, 17377, 17378, 17384, 17386, 17389, 17393, 17396, 17397,
17401, 17405, 17408, 17410, 17417, 17424, 17425, 17426, 17428, 17429, 17433, 17440, 17444, 17449, 17450, 17460, 17473, 17474, 17476, 17477, 17482, 17485,
17488, 17489, 17492, 17497, 17498, 17505, 17506, 17509, 17522, 17524, 17525, 17530, 17533, 17536, 17540, 17541, 17545, 17552, 17557, 17561, 17568, 17569,
17573, 17576, 17581, 17585, 17586, 17588, 17593, 17597, 17602, 17609, 17620, 17629, 17636, 17640, 17642, 17645, 17649, 17650, 17657, 17658, 17665, 17666,
17669, 17672, 17674, 17680, 17681, 17684, 17689, 17690, 17693, 17698, 17704, 17705, 17713, 17714, 17722, 17725, 17728, 17729, 17730, 17737, 17738, 17741,
17744, 17748, 17749, 17753, 17757, 17761, 17764, 17768, 17770, 17777, 17785, 17786, 17789, 17797, 17800, 17810, 17812, 17824, 17828, 17833, 17834, 17837,
17849, 17858, 17861, 17865, 17866, 17872, 17873, 17876, 17881, 17882, 17885, 17890, 17893, 17896, 17901, 17905, 17908, 17909, 17914, 17921, 17924, 17929,
17937, 17938, 17945, 17946, 17953, 17956, 17957, 17960, 17965, 17972, 17973, 17977, 17978, 17981, 17984, 17986, 17989, 17992, 18000, 18002, 18005, 18010,
18013, 18020, 18026, 18029, 18037, 18041, 18045, 18049, 18050, 18052, 18056, 18058, 18061, 18064, 18065, 18068, 18077, 18080, 18081, 18082, 18085, 18089,
18097, 18098, 18100, 18121, 18122, 18125, 18130, 18133, 18148, 18149, 18152, 18153, 18154, 18162, 18169, 18173, 18178, 18180, 18181, 18184, 18185, 18196,
18208, 18212, 18217, 18218, 18225, 18226, 18229, 18233, 18234, 18241, 18244, 18245, 18248, 18250, 18253, 18257, 18261, 18265, 18266, 18269, 18274, 18277,
18280, 18281, 18289, 18292, 18301, 18306, 18313, 18314, 18317, 18320, 18322, 18324, 18325, 18329, 18338, 18341, 18344, 18346, 18353, 18356, 18362, 18365,
18369, 18376, 18378, 18385, 18386, 18388, 18394, 18397, 18401, 18405, 18409, 18413, 18418, 18421, 18432, 18433, 18434, 18440, 18442, 18448, 18450, 18457,
18461, 18464, 18472, 18473, 18477, 18481, 18482, 18484, 18485, 18490, 18493, 18496, 18497, 18500, 18504, 18505, 18512, 18513, 18514, 18517, 18521, 18530,
18532, 18541, 18545, 18548, 18549, 18553, 18554, 18560, 18562, 18577, 18580, 18581, 18586, 18589, 18593, 18594, 18596, 18605, 18610, 18617, 18621, 18625,
18628, 18629, 18632, 18637, 18640, 18649, 18650, 18661, 18664, 18665, 18666, 18674, 18682, 18685, 18688, 18692, 18698, 18701, 18709, 18713, 18720, 18721,
18728, 18729, 18730, 18737, 18740, 18749, 18752, 18754, 18756, 18757, 18761, 18769, 18770, 18772, 18773, 18778, 18785, 18792, 18793, 18794, 18797, 18801,
18805, 18818, 18820, 18824, 18826, 18833, 18836, 18842, 18845, 18850, 18853, 18856, 18857, 18866, 18868, 18869, 18873, 18874, 18882, 18884, 18889, 18890,
18896, 18901, 18913, 18914, 18916, 18917, 18922, 18925, 18932, 18937, 18938, 18944, 18945, 18946, 18954, 18965, 18973, 18976, 18980, 18985, 18989, 18994,
18997, 19001, 19009, 19010, 19012, 19013, 19016, 19017, 19018, 19024, 19025, 19037, 19042, 19044, 19045, 19048, 19053, 19058, 19060, 19061, 19066, 19069,
19072, 19073, 19080, 19081, 19088, 19093, 19097, 19098, 19105, 19106, 19108, 19109, 19112, 19121, 19125, 19130, 19133, 19141, 19144, 19154, 19156, 19157,
19161, 19165, 19169, 19172, 19181, 19186, 19188, 19193, 19202, 19204, 19208, 19210, 19213, 19216, 19220, 19225, 19226, 19232, 19233, 19237, 19240, 19242,
19249, 19252, 19253, 19258, 19265, 19268, 19269, 19273, 19277, 19280, 19289, 19298, 19300, 19301, 19306, 19309, 19314, 19321, 19322, 19325, 19330, 19333,
19336, 19337, 19345, 19346, 19354, 19357, 19360, 19368, 19370, 19373, 19377, 19378, 19381, 19385, 19394, 19396, 19400, 19402, 19405, 19408, 19409, 19417,
19421, 19429, 19433, 19441, 19442, 19444, 19445, 19449, 19450, 19453, 19457, 19465, 19466, 19469, 19472, 19476, 19477, 19485, 19489, 19490, 19493, 19496,
19498, 19501, 19508, 19517, 19520, 19521, 19528, 19538, 19540, 19541, 19546, 19549, 19553, 19556, 19557, 19562, 19573, 19577, 19584, 19585, 19594, 19597,
19600, 19601, 19602, 19604, 19609, 19610, 19616, 19618, 19620, 19625, 19633, 19634, 19636, 19637, 19645, 19649, 19652, 19658, 19661, 19664, 19666, 19669,
19674, 19681, 19682, 19697, 19700, 19709, 19714, 19717, 19720, 19721, 19728, 19729, 19730, 19732, 19744, 19746, 19748, 19753, 19762, 19764, 19769, 19773,
19777, 19784, 19786, 19792, 19793, 19796, 19801, 19802, 19805, 19813, 19816, 19825, 19828, 19841, 19844, 19845, 19850, 19853, 19856, 19858, 19861, 19865,
19876, 19881, 19882, 19885, 19889, 19890, 19892, 19897, 19898, 19906, 19913, 19917, 19924, 19925, 19930, 19937, 19940, 19944, 19945, 19946, 19949, 19961,
19962, 19970, 19972, 19973, 19981, 19984, 19989, 19993, 19994, 19997, 20000, 20002, 20005, 20008, 20017, 20018, 20021, 20025, 20029, 20032, 20036, 20041,
20050, 20052, 20065, 20068, 20072, 20074, 20077, 20084, 20089, 20090, 20096, 20098, 20101, 20105, 20106, 20113, 20114, 20117, 20122, 20128, 20129, 20133,
20137, 20138, 20149, 20161, 20162, 20164, 20165, 20168, 20170, 20173, 20176, 20177, 20180, 20186, 20189, 20200, 20201, 20205, 20213, 20221, 20225, 20228,
20232, 20233, 20241, 20242, 20245, 20249, 20250, 20260, 20261, 20264, 20266, 20269, 20276, 20281, 20282, 20285, 20288, 20290, 20297, 20308, 20313, 20322,
20324, 20329, 20333, 20338, 20340, 20341, 20345, 20353, 20354, 20357, 20360, 20362, 20365, 20369, 20381, 20384, 20386, 20389, 20392, 20393, 20402, 20404,
20410, 20413, 20417, 20420, 20421, 20432, 20434, 20441, 20449, 20450, 20452, 20453, 20456, 20457, 20458, 20465, 20474, 20477, 20480, 20484, 20485, 20488,
20498, 20500, 20506, 20509, 20512, 20513, 20521, 20525, 20529, 20530, 20533, 20546, 20549, 20557, 20560, 20561, 20564, 20565, 20570, 20578, 20593, 20602,
20605, 20609, 20610, 20612, 20617, 20618, 20621, 20624, 20626, 20629, 20637, 20641, 20642, 20644, 20645, 20648, 20660, 20665, 20666, 20673, 20674, 20681,
20689, 20690, 20693, 20705, 20714, 20717, 20722, 20725, 20736, 20737, 20738, 20740, 20744, 20745, 20749, 20752, 20753, 20754, 20756, 20761, 20765, 20772,
20773, 20776, 20781, 20785, 20788, 20789, 20794, 20800, 20808, 20809, 20810, 20813, 20816, 20817, 20825, 20826, 20836, 20840, 20842, 20849, 20852, 20857,
20858, 20866, 20869, 20872, 20873, 20880, 20882, 20885, 20890, 20893, 20896, 20897, 20905, 20906, 20914, 20917, 20921, 20929, 20932, 20933, 20936, 20938,
20948, 20954, 20961, 20968, 20969, 20970, 20978, 20980, 20981, 20992, 20996, 20997, 21001, 21002, 21005, 21008, 21013, 21017, 21024, 21025, 21026, 21029,
21034, 21037, 21041, 21044, 21050, 21053, 21058, 21060, 21061, 21064, 21069, 21073, 21074, 21085, 21089, 21092, 21096, 21097, 21101, 21106, 21121, 21124,
21125, 21130, 21136, 21141, 21145, 21146, 21149, 21152, 21157, 21160, 21169, 21170, 21177, 21178, 21188, 21193, 21194, 21200, 21202, 21205, 21213, 21217,
21218, 21220, 21221, 21226, 21233, 21236, 21250, 21253, 21256, 21257, 21258, 21265, 21268, 21269, 21277, 21281, 21284, 21289, 21290, 21305, 21312, 21313,
21314, 21316, 21317, 21320, 21325, 21332, 21341, 21346, 21348, 21349, 21352, 21361, 21364, 21365, 21370, 21377, 21380, 21386, 21389, 21393, 21397, 21401,
21402, 21410, 21412, 21416, 21418, 21425, 21429, 21433, 21437, 21445, 21449, 21456, 21458, 21460, 21465, 21466, 21473, 21474, 21481, 21485, 21493, 21501,
21506, 21509, 21512, 21517, 21520, 21521, 21524, 21529, 21530, 21533, 21536, 21537, 21538, 21541, 21544, 21554, 21556, 21557, 21562, 21568, 21569, 21572,
21577, 21578, 21589, 21601, 21605, 21608, 21609, 21610, 21613, 21617, 21618, 21625, 21632, 21634, 21636, 21640, 21645, 21649, 21650, 21652, 21658, 21661,
21664, 21668, 21673, 21674, 21677, 21682, 21685, 21689, 21690, 21697, 21701, 21704, 21706, 21709, 21713, 21716, 21722, 21730, 21737, 21745, 21748, 21753,
21754, 21757, 21760, 21764, 21773, 21776, 21778, 21780, 21785, 21789, 21794, 21796, 21800, 21805, 21809, 21817, 21818, 21821, 21825, 21832, 21834, 21841,
21845, 21853, 21860, 21865, 21866, 21874, 21881, 21892, 21893, 21898, 21901, 21904, 21905, 21906, 21908, 21913, 21914, 21920, 21922, 21925, 21928, 21929,
21933, 21937, 21940, 21941, 21946, 21953, 21960, 21961, 21968, 21969, 21970, 21977, 21985, 21986, 21992, 21997, 22001, 22004, 22009, 22013, 22018, 22021,
22024, 22025, 22032, 22037, 22045, 22048, 22049, 22050, 22052, 22058, 22061, 22068, 22069, 22073, 22082, 22084, 22090, 22093, 22096, 22100, 22105, 22109,
22114, 22117, 22122, 22129, 22130, 22133, 22138, 22144, 22148, 22153, 22157, 22160, 22178, 22180, 22181, 22185, 22186, 22189, 22193, 22194, 22201, 22202,
22205, 22210, 22212, 22213, 22216, 22217, 22226, 22228, 22229, 22234, 22237, 22250, 22257, 22261, 22265, 22266, 22273, 22276, 22277, 22280, 22282, 22285,
22292, 22293, 22298, 22301, 22304, 22312, 22313, 22321, 22322, 22324, 22336, 22338, 22340, 22345, 22346, 22349, 22354, 22369, 22370, 22373, 22376, 22381,
22385, 22388, 22393, 22394, 22397, 22405, 22408, 22409, 22417, 22426, 22432, 22433, 22437, 22441, 22442, 22445, 22450, 22453, 22457, 22465, 22468, 22469,
22472, 22474, 22480, 22481, 22482, 22490, 22500, 22501, 22504, 22509, 22514, 22516, 22522, 22525, 22529, 22536, 22541, 22544, 22546, 22549, 22562, 22564,
22565, 22570, 22573, 22580, 22581, 22585, 22589, 22592, 22600, 22601, 22608, 22612, 22613, 22621, 22625, 22628, 22633, 22634, 22637, 22642, 22644, 22658,
22664, 22669, 22672, 22676, 22681, 22685, 22688, 22689, 22690, 22696, 22697, 22698, 22706, 22709, 22714, 22717, 22721, 22725, 22730, 22736, 22738, 22741,
22745, 22753, 22756, 22760, 22761, 22765, 22769, 22772, 22777, 22784, 22786, 22789, 22797, 22801, 22802, 22804, 22805, 22810, 22817, 22824, 22826, 22829,
22834, 22837, 22849, 22850, 22852, 22853, 22856, 22861, 22864, 22865, 22868, 22873, 22874, 22877, 22882, 22888, 22898, 22900, 22901, 22905, 22906, 22916,
22921, 22922, 22928, 22930, 22932, 22937, 22941, 22945, 22948, 22949, 22961, 22964, 22970, 22973, 22978, 22984, 22985, 22986, 22993, 22994, 22996, 22997,
23013, 23017, 23018, 23021, 23026, 23029, 23040, 23041, 23042, 23049, 23050, 23053, 23057, 23060, 23076, 23080, 23081, 23090, 23098, 23101, 23104, 23105,
23108, 23113, 23117, 23120, 23125, 23129, 23130, 23137, 23140, 23141, 23153, 23162, 23165, 23168, 23173, 23176, 23185, 23186, 23189, 23194, 23197, 23200,
23201, 23202, 23204, 23209, 23225, 23229, 23234, 23236, 23242, 23245, 23248, 23252, 23257, 23258, 23266, 23269, 23272, 23273, 23284, 23285, 23290, 23293,
23297, 23300, 23314, 23321, 23328, 23329, 23330, 23333, 23336, 23337, 23341, 23346, 23348, 23353, 23354, 23357, 23360, 23362, 23365, 23369, 23373, 23377,
23378, 23393, 23396, 23400, 23402, 23409, 23410, 23413, 23417, 23418, 23425, 23426, 23428, 23432, 23434, 23440, 23444, 23445, 23456, 23458, 23461, 23465,
23473, 23474, 23476, 23477, 23481, 23490, 23497, 23504, 23508, 23509, 23524, 23525, 23528, 23530, 23533, 23537, 23545, 23546, 23549, 23553, 23554, 23557,
23561, 23569, 23570, 23578, 23581, 23585, 23588, 23589, 23593, 23602, 23605, 23609, 23616, 23618, 23620, 23624, 23626, 23629, 23633, 23634, 23642, 23645,
23652, 23656, 23657, 23665, 23666, 23668, 23669, 23677, 23680, 23689, 23696, 23697, 23698, 23701, 23716, 23717, 23720, 23722, 23725, 23729, 23732, 23733,
23741, 23752, 23753, 23761, 23762, 23764, 23765, 23770, 23773, 23778, 23780, 23789, 23794, 23796, 23797, 23801, 23805, 23809, 23810, 23812, 23813, 23816,
23818, 23824, 23825, 23833, 23834, 23837, 23840, 23850, 23857, 23860, 23866, 23869, 23872, 23873, 23882, 23885, 23888, 23890, 23893, 23906, 23909, 23912,
23913, 23917, 23924, 23929, 23930, 23938, 23941, 23944, 23945, 23953, 23956, 23957, 23962, 23965, 23972, 23976, 23977, 23981, 23985, 23993, 24001, 24004,
24005, 24008, 24010, 24020, 24021, 24025, 24026, 24029, 24034, 24037, 24040, 24041, 24049, 24050, 24061, 24065, 24074, 24077, 24082, 24085, 24089, 24093,
24097, 24098, 24100, 24106, 24109, 24113, 24116, 24121, 24125, 24128, 24133, 24137, 24138, 24146, 24148, 24154, 24157, 24161, 24169, 24170, 24181, 24193,
24194, 24197, 24200, 24201, 24202, 24208, 24210, 24212, 24217, 24218, 24221, 24224, 24226, 24228, 24229, 24232, 24237, 24245, 24250, 24260, 24264, 24272,
24274, 24281, 24292, 24293, 24296, 24298, 24305, 24309, 24313, 24314, 24317, 24322, 24328, 24329, 24336, 24337, 24340, 24341, 24345, 24349, 24352, 24356,
24361, 24362, 24370, 24372, 24373, 24385, 24386, 24389, 24392, 24394, 24400, 24401, 24404, 24410, 24413, 24417, 24418, 24421, 24425, 24433, 24436, 24442,
24445, 24452, 24457, 24461, 24466, 24469, 24473, 24480, 24481, 24482, 24484, 24488, 24498, 24500, 24505, 24506, 24509, 24517, 24520, 24525, 24532, 24533,
24538, 24545, 24548, 24554, 24557, 24561, 24562, 24565, 24578, 24580, 24592, 24593, 24601, 24602, 24608, 24616, 24625, 24634, 24641, 24642, 24644, 24649,
24650, 24653, 24658, 24660, 24665, 24669, 24674, 24677, 24680, 24685, 24692, 24697, 24698, 24701, 24704, 24705, 24709, 24712, 24713, 24714, 24730, 24733,
24736, 24737, 24740, 24741, 24745, 24746, 24749, 24754, 24770, 24777, 24778, 24781, 24784, 24785, 24786, 24788, 24793, 24797, 24802, 24804, 24805, 24809,
24818, 24820, 24821, 24826, 24832, 24841, 24842, 24845, 24848, 24853, 24858, 24865, 24866, 24868, 24869, 24872, 24874, 24877, 24884, 24889, 24896, 24901,
24905, 24912, 24914, 24916, 24917, 24922, 24925, 24929, 24930, 24938, 24941, 24946, 24949, 24953, 24961, 24964, 24965, 24968, 24973, 24977, 24980, 24986,
24989, 24993, 24994, 25000, 25009, 25010, 25012, 25013, 25028, 25033, 25034, 25037, 25040, 25045, 25049, 25057, 25064, 25065, 25073, 25076, 25082, 25085,
25088, 25090, 25092, 25096, 25097, 25101, 25105, 25106, 25108, 25114, 25117, 25120, 25121, 25128, 25129, 25133, 25138, 25153, 25154, 25160, 25168, 25169,
25173, 25177, 25178, 25181, 25186, 25189, 25202, 25204, 25205, 25209, 25210, 25216, 25220, 25225, 25226, 25229, 25236, 25237, 25250, 25253, 25258, 25261,
25268, 25274, 25281, 25282, 25285, 25288, 25290, 25297, 25301, 25306, 25309, 25313, 25316, 25317, 25321, 25325, 25330, 25337, 25345, 25348, 25349, 25352,
25353, 25357, 25360, 25362, 25364, 25373, 25376, 25378, 25381, 25385, 25394, 25397, 25402, 25405, 25408, 25409, 25412, 25416, 25425, 25426, 25433, 25434,
25442, 25444, 25448, 25450, 25453, 25457, 25469, 25477, 25480, 25490, 25492, 25493, 25497, 25504, 25505, 25506, 25514, 25524, 25525, 25529, 25533, 25537,
25538, 25540, 25541, 25546, 25549, 25552, 25556, 25561, 25562, 25565, 25570, 25577, 25578, 25588, 25589, 25600, 25601, 25604, 25605, 25609, 25610, 25616,
25618, 25621, 25625, 25632, 25633, 25636, 25640, 25642, 25649, 25652, 25657, 25658, 25664, 25666, 25672, 25673, 25677, 25681, 25682, 25684, 25693, 25700,
25705, 25706, 25713, 25717, 25721, 25722, 25733, 25736, 25738, 25741, 25744, 25748, 25749, 25765, 25768, 25769, 25777, 25778, 25780, 25786, 25789, 25793,
25794, 25796, 25801, 25805, 25808, 25810, 25825, 25826, 25832, 25834, 25841, 25849, 25856, 25857, 25864, 25873, 25874, 25876, 25877, 25882, 25888, 25889,
25892, 25906, 25909, 25913, 25920, 25921, 25922, 25924, 25925, 25930, 25933, 25936, 25937, 25940, 25945, 25946, 25957, 25961, 25965, 25969, 25970, 25981,
25985, 25988, 25992, 25994, 25997, 26000, 26002, 26010, 26017, 26018, 26020, 26021, 26024, 26029, 26041, 26042, 26045, 26050, 26053, 26056, 26064, 26065,
26066, 26073, 26074, 26077, 26084, 26090, 26093, 26098, 26100, 26113, 26116, 26117, 26120, 26122, 26129, 26141, 26146, 26153, 26154, 26161, 26165, 26170,
26176, 26177, 26181, 26185, 26186, 26189, 26192, 26209, 26210, 26212, 26216, 26218, 26221, 26225, 26233, 26234, 26237, 26240, 26242, 26244, 26245, 26248,
26249, 26253, 26260, 26261, 26269, 26272, 26276, 26280, 26281, 26282, 26293, 26297, 26305, 26308, 26309, 26317, 26321, 26324, 26325, 26329, 26330, 26333,
26338, 26344, 26353, 26354, 26357, 26361, 26362, 26365, 26370, 26377, 26378, 26384, 26388, 26393, 26401, 26405, 26408, 26413, 26417, 26420, 26426, 26434,
26437, 26440, 26441, 26442, 26449, 26450, 26452, 26458, 26468, 26469, 26482, 26485, 26489, 26497, 26498, 26500, 26501, 26504, 26506, 26509, 26512, 26513,
26522, 26525, 26528, 26533, 26545, 26546, 26548, 26557, 26561, 26564, 26568, 26569, 26570, 26573, 26577, 26578, 26585, 26594, 26596, 26597, 26605, 26612,
26613, 26618, 26624, 26626, 26632, 26633, 26640, 26641, 26644, 26645, 26650, 26656, 26658, 26665, 26669, 26674, 26681, 26685, 26689, 26690, 26692, 26693,
26701, 26704, 26705, 26713, 26714, 26717, 26721, 26725, 26728, 26729, 26737, 26738, 26741, 26756, 26762, 26765, 26770, 26773, 26777, 26788, 26792, 26793,
26794, 26801, 26802, 26804, 26813, 26820, 26821, 26825, 26833, 26834, 26836, 26842, 26849, 26852, 26856, 26858, 26861, 26869, 26874, 26881, 26882, 26888,
26890, 26893, 26896, 26897, 26900, 26901, 26905, 26912, 26914, 26920, 26921, 26930, 26932, 26937, 26938, 26941, 26944, 26945, 26948, 26953, 26954, 26960,
26962, 26965, 26969, 26973, 26977, 26981, 26984, 26989, 26993, 26996, 27009, 27010, 27017, 27026, 27028, 27040, 27044, 27045, 27050, 27053, 27061, 27065,
27073, 27074, 27077, 27080, 27085, 27088, 27092, 27098, 27101, 27106, 27109, 27112, 27121, 27124, 27130, 27136, 27144, 27145, 27146, 27149, 27152, 27154,
27157, 27161, 27169, 27172, 27173, 27185, 27194, 27197, 27200, 27205, 27217, 27220, 27225, 27226, 27229, 27233, 27234, 27241, 27242, 27245, 27250, 27252,
27253, 27257, 27261, 27266, 27268, 27274, 27277, 27281, 27289, 27290, 27293, 27296, 27297, 27298, 27304, 27306, 27316, 27325, 27329, 27332, 27333, 27337,
27338, 27344, 27346, 27353, 27361, 27362, 27364, 27365, 27369, 27378, 27380, 27385, 27386, 27389, 27394, 27396, 27397, 27400, 27409, 27410, 27418, 27421,
27424, 27425, 27428, 27437, 27441, 27442, 27449, 27450, 27457, 27458, 27460, 27464, 27469, 27472, 27476, 27481, 27490, 27505, 27506, 27508, 27509, 27514,
27521, 27529, 27530, 27536, 27538, 27540, 27541, 27549, 27556, 27557, 27560, 27562, 27565, 27572, 27578, 27581, 27585, 27586, 27592, 27593, 27602, 27605,
27613, 27617, 27620, 27625, 27634, 27637, 27653, 27656, 27658, 27666, 27668, 27673, 27674, 27677, 27680, 27682, 27684, 27685, 27688, 27689, 27693, 27697,
27700, 27701, 27709, 27712, 27716, 27725, 27728, 27733, 27737, 27746, 27749, 27752, 27754, 27757, 27765, 27770, 27773, 27781, 27785, 27792, 27793, 27794,
27796, 27801, 27802, 27809, 27812, 27817, 27826, 27828, 27829, 27833, 27842, 27844, 27845, 27848, 27850, 27856, 27857, 27865, 27866, 27869, 27877, 27880,
27881, 27882, 27889, 27890, 27893, 27898, 27901, 27904, 27905, 27908, 27914, 27917, 27920, 27925, 27936, 27938, 27941, 27946, 27953, 27954, 27956, 27961,
27970, 27976, 27977, 27981, 27985, 27988, 27989, 27994, 27997, 28001, 28004, 28008, 28009, 28010, 28013, 28018, 28026, 28033, 28037, 28040, 28048, 28052,
28057, 28058, 28064, 28066, 28069, 28072, 28081, 28085, 28089, 28090, 28093, 28097, 28100, 28109, 28114, 28121, 28125, 28130, 28132, 28136, 28145, 28157,
28162, 28169, 28170, 28178, 28180, 28181, 28186, 28192, 28193, 28197, 28201, 28205, 28213, 28224, 28225, 28228, 28229, 28232, 28233, 28240, 28249, 28250,
28260, 28264, 28265, 28269, 28273, 28276, 28277, 28285, 28288, 28289, 28297, 28298, 28304, 28305, 28306, 28309, 28314, 28322, 28324, 28328, 28330, 28333,
28340, 28345, 28346, 28349, 28354, 28360, 28368, 28370, 28372, 28373, 28381, 28393, 28394, 28397, 28409, 28417, 28418, 28420, 28421, 28426, 28429, 28432,
28433, 28436, 28442, 28445, 28449, 28450, 28453, 28456, 28465, 28477, 28480, 28484, 28493, 28496, 28498, 28505, 28513, 28514, 28516, 28517, 28521, 28522,
28530, 28537, 28541, 28548, 28549, 28561, 28562, 28564, 28565, 28570, 28573, 28577, 28580, 28584, 28585, 28586, 28593, 28594, 28597, 28601, 28610, 28616,
28618, 28621, 28624, 28625, 28628, 28629, 28642, 28645, 28648, 28649, 28657, 28660, 28661, 28665, 28669, 28673, 28682, 28685, 28692, 28697, 28705, 28708,
28712, 28714, 28717, 28729, 28730, 28736, 28738, 28741, 28744, 28745, 28746, 28753, 28757, 28762, 28769, 28772, 28778, 28781, 28786, 28789, 28793, 28800,
28802, 28808, 28813, 28816, 28817, 28818, 28825, 28832, 28834, 28836, 28837, 28845, 28849, 28850, 28852, 28872, 28873, 28874, 28880, 28881, 28885, 28898,
28900, 28901, 28904, 28906, 28909, 28913, 28916, 28921, 28922, 28925, 28928, 28933, 28936, 28946, 28948, 28949, 28953, 28960, 28961, 28962, 28964, 28970,
28978, 28981, 28989, 29000, 29002, 29005, 29008, 29009, 29012, 29017, 29021, 29033, 29034, 29041, 29042, 29044, 29045, 29053, 29057, 29060, 29061, 29065,
29066, 29069, 29074, 29077, 29081, 29088, 29090, 29096, 29097, 29098, 29101, 29105, 29114, 29122, 29124, 29125, 29129, 29137, 29138, 29153, 29156, 29160,
29161, 29170, 29173, 29178, 29185, 29186, 29188, 29189, 29192, 29200, 29201, 29204, 29209, 29221, 29224, 29236, 29237, 29241, 29242, 29245, 29248, 29250,
29257, 29258, 29261, 29266, 29269, 29273, 29277, 29282, 29284, 29285, 29290, 29297, 29300, 29305, 29306, 29312, 29313, 29314, 29320, 29321, 29322, 29332,
29333, 29338, 29341, 29345, 29353, 29354, 29362, 29377, 29378, 29380, 29384, 29385, 29389, 29396, 29401, 29402, 29405, 29410, 29416, 29426, 29429, 29434,
29437, 29444, 29448, 29449, 29453, 29461, 29466, 29473, 29474, 29476, 29482, 29485, 29489, 29492, 29493, 29497, 29501, 29504, 29506, 29520, 29521, 29522,
29524, 29525, 29529, 29530, 29537, 29549, 29556, 29565, 29569, 29570, 29572, 29573, 29576, 29581, 29584, 29585, 29588, 29593, 29594, 29597, 29600, 29602,
29608, 29609, 29620, 29626, 29629, 29633, 29637, 29641, 29642, 29645, 29648, 29650, 29653, 29665, 29668, 29669, 29672, 29674, 29682, 29684, 29690, 29696,
29705, 29709, 29717, 29725, 29728, 29732, 29738, 29741, 29745, 29753, 29761, 29765, 29768, 29770, 29776, 29780, 29786, 29789, 29794, 29800, 29801, 29809,
29812, 29817, 29818, 29824, 29825, 29826, 29828, 29833, 29837, 29840, 29844, 29857, 29858, 29860, 29861, 29864, 29866, 29873, 29881, 29889, 29890, 29896,
29905, 29908, 29914, 29917, 29921, 29924, 29929, 29930, 29933, 29938, 29945, 29952, 29954, 29956, 29957, 29961, 29965, 29968, 29970, 29978, 29984, 29988,
29989, 29993, 30004, 30005, 30010, 30013, 30017, 30025, 30026, 30029, 30032, 30034, 30037, 30042, 30050, 30053, 30056, 30068, 30069, 30073, 30082, 30088,
30089, 30097, 30098, 30106, 30109, 30112, 30113, 30116, 30122, 30125, 30133, 30137, 30141, 30145, 30146, 30148, 30152, 30154, 30157, 30160, 30161, 30164,
30169, 30181, 30185, 30193, 30196, 30197, 30202, 30209, 30213, 30218, 30224, 30233, 30241, 30242, 30244, 30249, 30250, 30253, 30258, 30260, 30265, 30266,
30269, 30274, 30276, 30277, 30280, 30285, 30290, 30292, 30293, 30298, 30301, 30308, 30312, 30313, 30322, 30325, 30329, 30330, 30340, 30341, 30344, 30346,
30356, 30357, 30361, 30365, 30368, 30370, 30376, 30377, 30386, 30389, 30397, 30410, 30413, 30416, 30420, 30421, 30425, 30433, 30434, 30440, 30445, 30449,
30458, 30465, 30466, 30469, 30472, 30474, 30482, 30484, 30490, 30493, 30496, 30497, 30500, 30501, 30505, 30509, 30517, 30528, 30529, 30532, 30537, 30538,
30545, 30546, 30553, 30554, 30557, 30565, 30568, 30577, 30578, 30581, 30586, 30589, 30593, 30596, 30600, 30605, 30608, 30610, 30617, 30625, 30626, 30629,
30634, 30637, 30641, 30649, 30650, 30658, 30661, 30664, 30665, 30673, 30674, 30676, 30677, 30682, 30685, 30689, 30692, 30697, 30698, 30706, 30708, 30713,
30717, 30722, 30724, 30725, 30733, 30736, 30740, 30746, 30749, 30752, 30754, 30757, 30760, 30762, 30769, 30770, 30772, 30773, 30781, 30784, 30794, 30802,
30805, 30809, 30817, 30821, 30824, 30825, 30826, 30829, 30836, 30841, 30848, 30850, 30852, 30853, 30865, 30868, 30869, 30880, 30881, 30890, 30893, 30897,
30901, 30906, 30914, 30920, 30922, 30925, 30928, 30929, 30937, 30941, 30946, 30949, 30952, 30953, 30962, 30964, 30976, 30977, 30978, 30980, 30985, 30986,
30992, 30994, 31001, 31005, 31012, 31013, 31016, 31018, 31025, 31028, 31033, 31037, 31040, 31041, 31048, 31049, 31057, 31058, 31060, 31061, 31066, 31069,
31076, 31081, 31082, 31085, 31090, 31097, 31105, 31109, 31112, 31113, 31117, 31120, 31121, 31138, 31140, 31144, 31145, 31149, 31153, 31154, 31156, 31162,
31172, 31176, 31177, 31181, 31184, 31189, 31193, 31194, 31201, 31202, 31204, 31205, 31210, 31213, 31221, 31225, 31226, 31232, 31237, 31249, 31250, 31252,
31253, 31258, 31264, 31265, 31266, 31268, 31277, 31282, 31285, 31298, 31300, 31301, 31313, 31316, 31321, 31322, 31329, 31330, 31333, 31336, 31337, 31338,
31345, 31348, 31354, 31357, 31360, 31364, 31365, 31370, 31373, 31376, 31378, 31385, 31392, 31393, 31397, 31400, 31409, 31410, 31412, 31417, 31421, 31428,
31429, 31432, 31433, 31450, 31460, 31466, 31469, 31473, 31474, 31477, 31481, 31489, 31492, 31498, 31501, 31505, 31508, 31509, 31513, 31517, 31520, 31522,
31525, 31529, 31538, 31541, 31545, 31546, 31552, 31554, 31561, 31568, 31572, 31573, 31581, 31585, 31586, 31588, 31594, 31601, 31604, 31609, 31610, 31613,
31618, 31634, 31637, 31642, 31645, 31649, 31652, 31653, 31657, 31681, 31684, 31685, 31688, 31690, 31693, 31697, 31700, 31705, 31709, 31714, 31716, 31720,
31721, 31729, 31732, 31733, 31741, 31748, 31752, 31753, 31754, 31760, 31761, 31762, 31765, 31769, 31770, 31778, 31781, 31784, 31793, 31796, 31797, 31802,
31805, 31810, 31813, 31816, 31817, 31824], dtype=np.int32)

square_C1s = np.array([1, 5, 9, 13, 21, 25, 29, 37, 45, 49, 57, 61, 69, 81, 89, 97, 101, 109, 113, 121, 129, 137, 145, 149, 161, 169, 177, 185, 193, 197, 213, 221, 225, 233, 241,
249, 253, 261, 277, 285, 293, 301, 305, 317, 325, 333, 341, 349, 357, 365, 373, 377, 385, 401, 405, 421, 429, 437, 441, 457, 465, 473, 481, 489, 497, 505,
509, 517, 529, 545, 553, 561, 569, 577, 593, 601, 609, 613, 621, 633, 641, 657, 665, 673, 681, 697, 709, 717, 725, 733, 741, 749, 757, 761, 769, 777, 793,
797, 805, 821, 829, 845, 853, 861, 869, 877, 885, 889, 901, 917, 925, 933, 941, 949, 965, 973, 981, 989, 997, 1005, 1009, 1033, 1041, 1049, 1057, 1069,
1085, 1093, 1101, 1109, 1117, 1125, 1129, 1137, 1153, 1161, 1177, 1185, 1201, 1209, 1217, 1225, 1229, 1237, 1245, 1257, 1265, 1273, 1281, 1289, 1305, 1313,
1321, 1329, 1353, 1361, 1369, 1373, 1389, 1405, 1413, 1425, 1433, 1441, 1449, 1457, 1465, 1473, 1481, 1489, 1505, 1513, 1517, 1533, 1541, 1549, 1565, 1581,
1597, 1605, 1609, 1617, 1633, 1641, 1649, 1653, 1669, 1685, 1693, 1701, 1709, 1725, 1733, 1741, 1749, 1757, 1765, 1781, 1789, 1793, 1801, 1813, 1829, 1837,
1853, 1861, 1869, 1877, 1885, 1893, 1901, 1917, 1925, 1933, 1941, 1961, 1969, 1977, 1993, 2001, 2009, 2017, 2025, 2029, 2053, 2061, 2069, 2077, 2085, 2093,
2101, 2109, 2121, 2129, 2145, 2161, 2177, 2185, 2201, 2209, 2217, 2225, 2233, 2241, 2249, 2253, 2261, 2285, 2289, 2305, 2313, 2321, 2337, 2353, 2361, 2377,
2385, 2393, 2409, 2417, 2425, 2433, 2441, 2449, 2453, 2469, 2477, 2493, 2501, 2509, 2521, 2529, 2537, 2545, 2553, 2561, 2569, 2585, 2593, 2601, 2609, 2617,
2629, 2637, 2661, 2669, 2693, 2701, 2709, 2725, 2733, 2741, 2749, 2757, 2765, 2769, 2785, 2801, 2809, 2821, 2837, 2845, 2861, 2869, 2877, 2885, 2893, 2917,
2925, 2933, 2941, 2949, 2957, 2965, 2981, 2989, 2997, 3001, 3017, 3025, 3041, 3045, 3061, 3069, 3077, 3085, 3093, 3109, 3125, 3133, 3149, 3157, 3173, 3181,
3189, 3197, 3205, 3209, 3233, 3241, 3249, 3265, 3281, 3289, 3297, 3305, 3313, 3317, 3333, 3341, 3357, 3365, 3381, 3389, 3397, 3405, 3409, 3425, 3433, 3441,
3449, 3457, 3489, 3497, 3505, 3513, 3521, 3529, 3545, 3553, 3569, 3577, 3593, 3597, 3605, 3613, 3625, 3641, 3657, 3673, 3681, 3697, 3705, 3713, 3721, 3729,
3745, 3753, 3761, 3769, 3777, 3793, 3801, 3809, 3817, 3833, 3841, 3853, 3861, 3869, 3877, 3885, 3893, 3909, 3917, 3937, 3945, 3953, 3969, 3985, 3993, 4001,
4009, 4017, 4025, 4041, 4049, 4053, 4061, 4085, 4093, 4109, 4117, 4125, 4141, 4149, 4157, 4165, 4189, 4197, 4205, 4221, 4229, 4237, 4249, 4257, 4273, 4281,
4293, 4309, 4317, 4325, 4341, 4349, 4357, 4373, 4389, 4397, 4405, 4421, 4429, 4437, 4445, 4461, 4469, 4477, 4485, 4493, 4501, 4509, 4513, 4537, 4545, 4569,
4577, 4581, 4597, 4613, 4621, 4637, 4645, 4661, 4669, 4677, 4693, 4701, 4709, 4725, 4741, 4749, 4765, 4777, 4785, 4809, 4825, 4841, 4849, 4857, 4865, 4873,
4881, 4889, 4897, 4905, 4921, 4925, 4941, 4949, 4957, 4973, 4989, 4997, 5005, 5013, 5025, 5033, 5041, 5049, 5057, 5065, 5073, 5081, 5089, 5097, 5129, 5137,
5145, 5153, 5169, 5177, 5193, 5201, 5209, 5217, 5233, 5241, 5249, 5261, 5273, 5281, 5297, 5321, 5329, 5337, 5345, 5369, 5377, 5385, 5393, 5409, 5417, 5433,
5441, 5449, 5457, 5465, 5473, 5489, 5497, 5505, 5513, 5521, 5525, 5541, 5557, 5573, 5581, 5589, 5605, 5621, 5629, 5637, 5649, 5657, 5673, 5681, 5689, 5705,
5713, 5721, 5745, 5753, 5761, 5769, 5785, 5789, 5813, 5829, 5837, 5845, 5853, 5861, 5877, 5885, 5893, 5901, 5909, 5917, 5949, 5957, 5973, 5981, 5989, 5997,
6005, 6021, 6025, 6041, 6049, 6065, 6073, 6077, 6093, 6109, 6125, 6133, 6141, 6149, 6157, 6173, 6181, 6197, 6213, 6221, 6237, 6253, 6261, 6269, 6277, 6293,
6309, 6317, 6325, 6333, 6349, 6361, 6369, 6377, 6385, 6393, 6409, 6417, 6433, 6437, 6461, 6469, 6477, 6485, 6493, 6501, 6509, 6525, 6541, 6549, 6557, 6565,
6573, 6581, 6589, 6605, 6613, 6621, 6625, 6641, 6657, 6665, 6697, 6705, 6721, 6729, 6737, 6745, 6761, 6769, 6777, 6785, 6793, 6809, 6817, 6833, 6841, 6845,
6861, 6869, 6877, 6885, 6893, 6909, 6917, 6921, 6953, 6961, 6969, 6977, 6985, 7009, 7017, 7025, 7033, 7049, 7057, 7073, 7089, 7105, 7113, 7129, 7137, 7145,
7153, 7161, 7177, 7193, 7201, 7209, 7213, 7229, 7237, 7245, 7253, 7265, 7273, 7289, 7305, 7321, 7337, 7345, 7353, 7369, 7377, 7385, 7393, 7409, 7417, 7425,
7433, 7441, 7449, 7465, 7473, 7481, 7497, 7505, 7513, 7521, 7525, 7533, 7541, 7573, 7589, 7597, 7605, 7613, 7637, 7645, 7653, 7661, 7677, 7685, 7693, 7705,
7713, 7721, 7753, 7761, 7769, 7777, 7785, 7793, 7809, 7817, 7825, 7845, 7861, 7869, 7885, 7893, 7909, 7917, 7933, 7957, 7965, 7981, 7989, 8005, 8013, 8021,
8029, 8037, 8045, 8061, 8069, 8085, 8093, 8109, 8113, 8121, 8129, 8137, 8161, 8173, 8181, 8197, 8205, 8221, 8229, 8237, 8245, 8253, 8269, 8277, 8285, 8293,
8301, 8309, 8317, 8341, 8349, 8357, 8389, 8405, 8413, 8421, 8429, 8445, 8453, 8461, 8469, 8485, 8497, 8513, 8521, 8529, 8545, 8553, 8577, 8585, 8597, 8613,
8621, 8637, 8645, 8653, 8661, 8669, 8685, 8693, 8701, 8717, 8725, 8741, 8757, 8765, 8773, 8781, 8789, 8797, 8809, 8825, 8841, 8849, 8857, 8865, 8889, 8897,
8905, 8921, 8929, 8937, 8945, 8961, 8969, 8977, 8985, 8993, 9001, 9009, 9033, 9041, 9057, 9061, 9085, 9093, 9101, 9125, 9133, 9141, 9145, 9153, 9169, 9193,
9209, 9225, 9233, 9249, 9265, 9273, 9281, 9289, 9305, 9313, 9329, 9337, 9353, 9361, 9377, 9385, 9393, 9401, 9417, 9425, 9433, 9449, 9465, 9477, 9493, 9501,
9517, 9525, 9541, 9549, 9557, 9569, 9577, 9585, 9609, 9625, 9633, 9649, 9665, 9673, 9689, 9705, 9713, 9721, 9729, 9737, 9745, 9753, 9761, 9769, 9777, 9785,
9809, 9825, 9841, 9845, 9853, 9869, 9877, 9909, 9917, 9925, 9941, 9949, 9965, 9981, 9989, 9997, 10005, 10021, 10029, 10037, 10049, 10057, 10065, 10081,
10089, 10097, 10105, 10113, 10121, 10129, 10137, 10145, 10161, 10169, 10177, 10185, 10189, 10221, 10229, 10237, 10245, 10261, 10269, 10277, 10293, 10309,
10325, 10333, 10349, 10365, 10381, 10389, 10405, 10413, 10421, 10429, 10437, 10445, 10453, 10469, 10477, 10485, 10501, 10517, 10525, 10533, 10545, 10557,
10573, 10581, 10597, 10605, 10629, 10645, 10653, 10661, 10669, 10685, 10693, 10717, 10725, 10733, 10741, 10765, 10773, 10781, 10797, 10805, 10837, 10845,
10853, 10869, 10877, 10885, 10893, 10901, 10909, 10913, 10921, 10953, 10961, 10977, 10985, 11001, 11009, 11025, 11033, 11041, 11049, 11057, 11065, 11069,
11077, 11093, 11101, 11117, 11133, 11141, 11157, 11165, 11173, 11181, 11197, 11213, 11221, 11229, 11237, 11245, 11261, 11269, 11277, 11289, 11305, 11313,
11329, 11337, 11345, 11353, 11361, 11369, 11385, 11417, 11425, 11433, 11441, 11449, 11465, 11489, 11505, 11513, 11521, 11537, 11545, 11553, 11561, 11569,
11585, 11593, 11597, 11621, 11629, 11645, 11653, 11661, 11669, 11681, 11689, 11713, 11721, 11737, 11745, 11761, 11769, 11777, 11785, 11793, 11817, 11825,
11833, 11841, 11873, 11881, 11897, 11905, 11913, 11929, 11937, 11945, 11961, 11977, 11985, 11993, 12001, 12025, 12033, 12041, 12057, 12061, 12077, 12093,
12101, 12109, 12125, 12141, 12149, 12165, 12169, 12185, 12193, 12209, 12217, 12225, 12241, 12257, 12265, 12273, 12281, 12289, 12297, 12305, 12321, 12329,
12353, 12361, 12377, 12393, 12401, 12417, 12449, 12453, 12469, 12485, 12501, 12517, 12533, 12541, 12549, 12557, 12565, 12581, 12589, 12605, 12621, 12629,
12637, 12645, 12661, 12669, 12677, 12693, 12701, 12717, 12725, 12737, 12745, 12753, 12761, 12769, 12785, 12793, 12801, 12817, 12825, 12841, 12849, 12853,
12869, 12893, 12909, 12917, 12925, 12933, 12941, 12957, 12965, 12973, 12981, 12989, 12997, 13013, 13029, 13045, 13053, 13061, 13069, 13085, 13093, 13109,
13117, 13125, 13133, 13141, 13157, 13165, 13173, 13181, 13205, 13221, 13229, 13237, 13273, 13281, 13289, 13293, 13309, 13325, 13333, 13341, 13373, 13381,
13389, 13397, 13413, 13429, 13437, 13445, 13453, 13461, 13477, 13485, 13501, 13509, 13517, 13525, 13533, 13549, 13557, 13581, 13589, 13605, 13621, 13629,
13637, 13653, 13661, 13669, 13673, 13681, 13697, 13705, 13721, 13737, 13745, 13753, 13769, 13777, 13793, 13801, 13809, 13825, 13833, 13849, 13857, 13865,
13869, 13901, 13909, 13917, 13925, 13933, 13949, 13957, 13965, 13989, 14005, 14013, 14021, 14029, 14045, 14053, 14061, 14069, 14073, 14089, 14097, 14105,
14121, 14137, 14169, 14177, 14193, 14201, 14209, 14225, 14249, 14265, 14273, 14289, 14297, 14305, 14313, 14329, 14337, 14345, 14361, 14377, 14393, 14401,
14409, 14425, 14433, 14441, 14445, 14461, 14469, 14477, 14485, 14493, 14505, 14537, 14545, 14561, 14577, 14585, 14601, 14617, 14625, 14633, 14649, 14665,
14673, 14681, 14689, 14705, 14713, 14729, 14737, 14745, 14753, 14761, 14777, 14793, 14809, 14817, 14833, 14841, 14849, 14857, 14865, 14873, 14881, 14913,
14921, 14929, 14945, 14949, 14957, 14973, 14981, 14997, 15005, 15021, 15029, 15037, 15045, 15053, 15069, 15077, 15081, 15089, 15097, 15105, 15137, 15145,
15153, 15169, 15193, 15201, 15209, 15217, 15233, 15257, 15265, 15273, 15281, 15289, 15297, 15305, 15321, 15329, 15345, 15353, 15361, 15373, 15397, 15405,
15421, 15429, 15445, 15453, 15477, 15509, 15517, 15525, 15533, 15541, 15549, 15557, 15565, 15573, 15581, 15589, 15597, 15613, 15621, 15629, 15645, 15661,
15669, 15677, 15685, 15705, 15721, 15729, 15737, 15745, 15761, 15777, 15785, 15793, 15809, 15813, 15821, 15837, 15853, 15877, 15893, 15901, 15917, 15933,
15949, 15957, 15965, 15973, 15989, 16005, 16013, 16021, 16029, 16045, 16053, 16061, 16069, 16077, 16085, 16101, 16133, 16141, 16157, 16173, 16181, 16189,
16205, 16221, 16237, 16241, 16273, 16281, 16289, 16297, 16305, 16313, 16321, 16345, 16357, 16365, 16373, 16389, 16405, 16413, 16429, 16437, 16445, 16453,
16461, 16469, 16485, 16493, 16509, 16525, 16533, 16541, 16557, 16565, 16573, 16581, 16589, 16597, 16605, 16613, 16621, 16645, 16661, 16669, 16677, 16693,
16709, 16717, 16729, 16761, 16769, 16777, 16793, 16801, 16817, 16833, 16841, 16849, 16881, 16889, 16905, 16913, 16921, 16929, 16945, 16953, 16961, 16977,
16989, 16997, 17013, 17021, 17029, 17037, 17045, 17061, 17069, 17085, 17093, 17101, 17109, 17117, 17141, 17149, 17165, 17181, 17193, 17201, 17217, 17225,
17241, 17257, 17265, 17273, 17281, 17289, 17297, 17313, 17329, 17337, 17345, 17353, 17401, 17409, 17417, 17433, 17449, 17457, 17465, 17473, 17481, 17497,
17505, 17521, 17529, 17537, 17545, 17561, 17569, 17585, 17593, 17601, 17617, 17629, 17645, 17665, 17681, 17697, 17705, 17713, 17721, 17737, 17745, 17769,
17777, 17785, 17793, 17809, 17817, 17833, 17841, 17849, 17857, 17865, 17873, 17889, 17897, 17905, 17913, 17921, 17937, 17945, 17953, 17961, 17969, 17993,
18009, 18017, 18025, 18033, 18041, 18065, 18073, 18081, 18097, 18105, 18121, 18125, 18141, 18165, 18197, 18205, 18213, 18237, 18245, 18261, 18269, 18277,
18285, 18293, 18317, 18321, 18329, 18345, 18361, 18369, 18393, 18401, 18417, 18433, 18441, 18449, 18457, 18473, 18481, 18489, 18505, 18513, 18521, 18537,
18545, 18553, 18561, 18577, 18593, 18601, 18605, 18621, 18637, 18645, 18661, 18693, 18701, 18709, 18725, 18733, 18749, 18765, 18773, 18781, 18789, 18797,
18813, 18829, 18845, 18853, 18869, 18877, 18893, 18909, 18933, 18941, 18957, 18965, 18977, 18993, 19001, 19009, 19017, 19033, 19049, 19057, 19073, 19081,
19089, 19097, 19109, 19125, 19133, 19141, 19149, 19157, 19181, 19189, 19205, 19213, 19229, 19237, 19245, 19261, 19277, 19285, 19293, 19309, 19325, 19333,
19349, 19365, 19381, 19397, 19405, 19413, 19421, 19429, 19445, 19453, 19461, 19477, 19509, 19517, 19525, 19533, 19541, 19549, 19557, 19565, 19573, 19577,
19585, 19601, 19625, 19649, 19657, 19673, 19689, 19697, 19701, 19717, 19725, 19733, 19749, 19757, 19789, 19797, 19805, 19813, 19845, 19853, 19861, 19877,
19885, 19893, 19901, 19917, 19933, 19949, 19957, 19965, 19973, 19981, 19989, 20005, 20013, 20021, 20029, 20045, 20053, 20061, 20069, 20081, 20097, 20105,
20113, 20145, 20161, 20169, 20177, 20185, 20193, 20217, 20225, 20233, 20249, 20257, 20273, 20281, 20289, 20297, 20305, 20321, 20329, 20337, 20345, 20353,
20361, 20369, 20385, 20401, 20405, 20437, 20453, 20461, 20469, 20477, 20485, 20509, 20517, 20533, 20541, 20549, 20557, 20573, 20589, 20593, 20609, 20641,
20649, 20657, 20673, 20681, 20689, 20705, 20721, 20729, 20737, 20753, 20769, 20785, 20801, 20833, 20841, 20849, 20857, 20865, 20881, 20889, 20905, 20913,
20921, 20929, 20945, 20953, 20961, 20969, 20977, 20993, 21001, 21017, 21033, 21041, 21057, 21065, 21073, 21081, 21089, 21101, 21125, 21137, 21153, 21161,
21169, 21177, 21193, 21201, 21217, 21233, 21257, 21265, 21281, 21289, 21305, 21313, 21321, 21329, 21345, 21353, 21361, 21385, 21401, 21409, 21425, 21433,
21441, 21449, 21457, 21465, 21473, 21497, 21505, 21529, 21537, 21545, 21561, 21569, 21585, 21593, 21601, 21609, 21625, 21629, 21661, 21677, 21685, 21701,
21709, 21717, 21733, 21741, 21749, 21773, 21781, 21805, 21813, 21821, 21829, 21837, 21853, 21861, 21869, 21873, 21881, 21913, 21921, 21929, 21945, 21953,
21969, 21985, 21993, 22001, 22009, 22025, 22033, 22041, 22049, 22057, 22081, 22097, 22105, 22121, 22129, 22133, 22141, 22149, 22165, 22181, 22189, 22197,
22213, 22229, 22245, 22253, 22285, 22301, 22309, 22325, 22341, 22349, 22357, 22365, 22381, 22389, 22405, 22413, 22429, 22445, 22461, 22469, 22477, 22485,
22501, 22509, 22525, 22533, 22541, 22557, 22565, 22573, 22581, 22593, 22609, 22617, 22633, 22641, 22649, 22657, 22665, 22701, 22709, 22717, 22725, 22733,
22741, 22757, 22773, 22805, 22813, 22821, 22837, 22853, 22861, 22869, 22877, 22893, 22901, 22909, 22925, 22949, 22957, 22973, 22981, 22989, 22997, 23021,
23029, 23045, 23053, 23085, 23093, 23101, 23109, 23125, 23133, 23141, 23157, 23165, 23181, 23189, 23197, 23205, 23213, 23217, 23233, 23257, 23265, 23281,
23297, 23305, 23313, 23329, 23337, 23345, 23353, 23365, 23373, 23389, 23413, 23429, 23437, 23445, 23461, 23469, 23485, 23493, 23509, 23517, 23525, 23533,
23541, 23549, 23557, 23565, 23581, 23589, 23613, 23621, 23629, 23637, 23645, 23653, 23661, 23693, 23701, 23709, 23717, 23725, 23757, 23769, 23785, 23793,
23801, 23809, 23841, 23849, 23857, 23873, 23881, 23889, 23913, 23929, 23945, 23953, 23961, 23993, 24001, 24017, 24025, 24033, 24057, 24065, 24073, 24081,
24089, 24097, 24105, 24121, 24153, 24157, 24173, 24181, 24197, 24205, 24221, 24229, 24237, 24245, 24261, 24277, 24285, 24301, 24309, 24313, 24329, 24345,
24353, 24361, 24369, 24385, 24393, 24409, 24425, 24433, 24449, 24465, 24481, 24489, 24497, 24505, 24513, 24529, 24537, 24553, 24561, 24569, 24593, 24601,
24609, 24625, 24633, 24641, 24657, 24665, 24689, 24697, 24705, 24713, 24729, 24737, 24745, 24761, 24777, 24785, 24793, 24809, 24817, 24833, 24845, 24861,
24885, 24893, 24925, 24933, 24941, 24945, 24961, 24977, 24985, 25001, 25017, 25033, 25049, 25065, 25073, 25081, 25089, 25105, 25113, 25121, 25137, 25145,
25161, 25169, 25185, 25193, 25209, 25225, 25233, 25249, 25257, 25265, 25281, 25289, 25305, 25321, 25329, 25337, 25345, 25361, 25377, 25385, 25393, 25401,
25417, 25425, 25433, 25445, 25453, 25461, 25477, 25493, 25501, 25509, 25517, 25557, 25565, 25581, 25589, 25605, 25613, 25629, 25637, 25653, 25661, 25693,
25709, 25717, 25733, 25741, 25745, 25761, 25785, 25793, 25809, 25817, 25825, 25833, 25841, 25849, 25857, 25865, 25881, 25889, 25921, 25937, 25945, 25953,
25961, 25969, 25977, 25985, 25997, 26013, 26029, 26045, 26053, 26069, 26077, 26085, 26093, 26101, 26109, 26125, 26141, 26149, 26173, 26181, 26197, 26213,
26221, 26237, 26245, 26253, 26261, 26269, 26285, 26293, 26309, 26317, 26325, 26349, 26357, 26365, 26373, 26381, 26405, 26429, 26445, 26453, 26477, 26485,
26493, 26501, 26537, 26545, 26553, 26561, 26565, 26581, 26597, 26605, 26621, 26629, 26645, 26653, 26669, 26677, 26693, 26709, 26741, 26749, 26757, 26765,
26773, 26781, 26789, 26805, 26821, 26829, 26845, 26861, 26869, 26877, 26885, 26893, 26909, 26917, 26925, 26933, 26941, 26957, 26989, 26997, 27013, 27021,
27029, 27045, 27053, 27061, 27069, 27085, 27093, 27109, 27117, 27133, 27141, 27145, 27169, 27185, 27193, 27209, 27225, 27241, 27249, 27257, 27265, 27273,
27281, 27297, 27305, 27321, 27329, 27337, 27345, 27361, 27365, 27373, 27381, 27397, 27405, 27429, 27445, 27453, 27469, 27477, 27485, 27493, 27509, 27517,
27525, 27541, 27557, 27565, 27581, 27589, 27597, 27613, 27621, 27629, 27645, 27661, 27669, 27677, 27685, 27709, 27717, 27725, 27729, 27737, 27769, 27777,
27809, 27817, 27825, 27841, 27849, 27857, 27873, 27881, 27889, 27905, 27913, 27921, 27937, 27945, 27969, 28001, 28017, 28025, 28033, 28049, 28057, 28065,
28073, 28089, 28097, 28113, 28121, 28129, 28153, 28161, 28169, 28177, 28181, 28197, 28205, 28221, 28229, 28237, 28253, 28269, 28277, 28293, 28325, 28333,
28345, 28353, 28369, 28377, 28385, 28393, 28409, 28417, 28425, 28449, 28481, 28497, 28513, 28521, 28537, 28545, 28561, 28577, 28585, 28593, 28601, 28617,
28625, 28641, 28649, 28657, 28665, 28697, 28705, 28713, 28729, 28745, 28753, 28761, 28777, 28785, 28793, 28809, 28817, 28825, 28841, 28849, 28857, 28865,
28881, 28889, 28905, 28913, 28917, 28933, 28949, 28957, 28965, 28989, 28997, 29005, 29013, 29021, 29029, 29041, 29073, 29081, 29097, 29105, 29137, 29153,
29161, 29169, 29185, 29193, 29209, 29217, 29225, 29233, 29249, 29257, 29273, 29289, 29313, 29321, 29337, 29345, 29353, 29361, 29369, 29377, 29393, 29401,
29417, 29433, 29441, 29449, 29457, 29473, 29481, 29497, 29505, 29513, 29525, 29541, 29557, 29565, 29581, 29589, 29637, 29645, 29653, 29669, 29677, 29685,
29693, 29709, 29717, 29725, 29733, 29741, 29749, 29765, 29773, 29781, 29789, 29797, 29829, 29837, 29853, 29861, 29869, 29885, 29901, 29909, 29913, 29921,
29937, 29953, 29961, 29969, 29985, 29993, 30001, 30017, 30033, 30041, 30057, 30065, 30081, 30089, 30097, 30113, 30129, 30137, 30145, 30149, 30181, 30189,
30197, 30205, 30213, 30245, 30253, 30261, 30269, 30277, 30293, 30301, 30325, 30333, 30349, 30357, 30373, 30381, 30397, 30405, 30413, 30421, 30453, 30461,
30469, 30485, 30509, 30525, 30533, 30541, 30549, 30573, 30581, 30589, 30597, 30613, 30621, 30629, 30637, 30653, 30661, 30669, 30685, 30701, 30709, 30717,
30725, 30741, 30753, 30757, 30781, 30813, 30821, 30837, 30853, 30861, 30869, 30885, 30893, 30901, 30909, 30917, 30933, 30957, 30965, 30997, 31005, 31021,
31029, 31037, 31045, 31053, 31069, 31077, 31085, 31101, 31109, 31117, 31125, 31133, 31141, 31149, 31173, 31189, 31197, 31205, 31213, 31245, 31253, 31261,
31277, 31293, 31309, 31325, 31333, 31341, 31349, 31365, 31373, 31381, 31397, 31417, 31433, 31449, 31457, 31465, 31473, 31497, 31505, 31521, 31537, 31545,
31553, 31569, 31577, 31593, 31601, 31617, 31625, 31641, 31649, 31665, 31669, 31677, 31693, 31709, 31725, 31733, 31757, 31773, 31781, 31797, 31813, 31829,
31845, 31853, 31869, 31877, 31885, 31901, 31909, 31917, 31925, 31933, 31949, 31957, 31973, 31981, 31989, 31997, 32005, 32017, 32025, 32057, 32073, 32081,
32097, 32121, 32129, 32137, 32153, 32169, 32177, 32185, 32201, 32217, 32249, 32257, 32265, 32281, 32289, 32305, 32321, 32337, 32345, 32353, 32369, 32377,
32401, 32409, 32417, 32425, 32441, 32457, 32473, 32481, 32489, 32505, 32513, 32529, 32533, 32541, 32573, 32581, 32589, 32597, 32605, 32613, 32621, 32629,
32645, 32669, 32681, 32697, 32705, 32721, 32729, 32745, 32761, 32777, 32785, 32793, 32801, 32817, 32833, 32849, 32857, 32865, 32873, 32881, 32889, 32897,
32905, 32913, 32921, 32937, 32953, 32969, 32977, 32993, 33001, 33017, 33025, 33033, 33049, 33057, 33081, 33089, 33105, 33113, 33129, 33137, 33145, 33161,
33169, 33185, 33193, 33209, 33217, 33225, 33257, 33265, 33273, 33281, 33305, 33313, 33317, 33333, 33341, 33349, 33389, 33397, 33405, 33421, 33437, 33453,
33461, 33469, 33481, 33513, 33521, 33537, 33545, 33561, 33569, 33585, 33601, 33625, 33641, 33657, 33673, 33681, 33689, 33697, 33705, 33737, 33745, 33753,
33761, 33769, 33785, 33793, 33809, 33817, 33825, 33833, 33849, 33865, 33873, 33881, 33889, 33897, 33913, 33929, 33937, 33949, 33965, 33973, 33989, 34013,
34021, 34037, 34045, 34053, 34061, 34077, 34093, 34101, 34109, 34125, 34133, 34165, 34173, 34189, 34205, 34213, 34221, 34229, 34237, 34253, 34261, 34285,
34293, 34301, 34309, 34325, 34349, 34357, 34373, 34381, 34393, 34401, 34409, 34417, 34433, 34449, 34457, 34473, 34481, 34497, 34505, 34537, 34545, 34553,
34561, 34577, 34585, 34593, 34609, 34621, 34637, 34653, 34661, 34677, 34685, 34693, 34701, 34749, 34757, 34765, 34781, 34789, 34797, 34805, 34821, 34837,
34853, 34861, 34869, 34885, 34901, 34909, 34917, 34925, 34933, 34941, 34973, 34981, 34989, 35005, 35021, 35029, 35037, 35053, 35061, 35069, 35077, 35085,
35101, 35117, 35125, 35133, 35149, 35157, 35173, 35181, 35189, 35197, 35205, 35213, 35237, 35253, 35265, 35281, 35297, 35305, 35337, 35357, 35373, 35381,
35397, 35405, 35413, 35421, 35429, 35445, 35453, 35485, 35501, 35509, 35533, 35541, 35549, 35557, 35565, 35573, 35589, 35597, 35605, 35621, 35629, 35637,
35653, 35661, 35677, 35685, 35701, 35717, 35725, 35733, 35741, 35757, 35765, 35773, 35781, 35797, 35805, 35821, 35829, 35845, 35853, 35877, 35893, 35901,
35909, 35917, 35925, 35941, 35949, 35953, 35977, 35993, 36009, 36025, 36033, 36049, 36057, 36065, 36073, 36089, 36097, 36121, 36129, 36137, 36145, 36161,
36177, 36185, 36201, 36225, 36241, 36249, 36265, 36281, 36289, 36293, 36309, 36333, 36349, 36381, 36397, 36405, 36413, 36421, 36429, 36453, 36461, 36469,
36477, 36493, 36501, 36509, 36517, 36533, 36541, 36549, 36557, 36589, 36613, 36621, 36625, 36641, 36649, 36657, 36673, 36681, 36697, 36705, 36713, 36721,
36745, 36753, 36769, 36777, 36809, 36817, 36833, 36841, 36857, 36865, 36873, 36889, 36897, 36905, 36921, 36937, 36945, 36953, 36969, 37001, 37017, 37025,
37041, 37049, 37057, 37065, 37073, 37081, 37097, 37105, 37113, 37129, 37137, 37145, 37153, 37161, 37177, 37193, 37201, 37225, 37229, 37245, 37261, 37277,
37285, 37297, 37313, 37329, 37337, 37369, 37377, 37385, 37401, 37409, 37425, 37433, 37441, 37457, 37473, 37497, 37513, 37521, 37529, 37537, 37545, 37561,
37569, 37577, 37585, 37601, 37609, 37625, 37641, 37649, 37665, 37673, 37689, 37697, 37705, 37721, 37737, 37753, 37769, 37817, 37825, 37833, 37841, 37865,
37881, 37889, 37905, 37913, 37921, 37929, 37945, 37961, 37969, 37981, 37989, 38005, 38021, 38029, 38037, 38045, 38053, 38061, 38077, 38109, 38125, 38133,
38149, 38165, 38173, 38181, 38189, 38197, 38205, 38213, 38225, 38241, 38249, 38257, 38273, 38289, 38297, 38313, 38321, 38329, 38353, 38361, 38377, 38393,
38409, 38417, 38425, 38441, 38457, 38465, 38473, 38481, 38489, 38505, 38513, 38529, 38537, 38545, 38553, 38561, 38569, 38577, 38593, 38609, 38617, 38625,
38641, 38657, 38669, 38685, 38733, 38741, 38757, 38781, 38797, 38805, 38821, 38829, 38837, 38845, 38861, 38869, 38885, 38893, 38901, 38917, 38933, 38941,
38949, 38957, 38965, 38981, 38989, 39021, 39029, 39037, 39045, 39053, 39061, 39069, 39077, 39085, 39093, 39101, 39109, 39117, 39125, 39133, 39149, 39165,
39181, 39189, 39193, 39201, 39217, 39225, 39233, 39257, 39289, 39313, 39321, 39329, 39345, 39361, 39369, 39377, 39381, 39413, 39429, 39437, 39445, 39453,
39469, 39485, 39493, 39501, 39509, 39541, 39549, 39557, 39565, 39573, 39581, 39597, 39605, 39637, 39645, 39653, 39685, 39701, 39709, 39717, 39725, 39741,
39757, 39765, 39773, 39805, 39813, 39821, 39837, 39845, 39861, 39877, 39885, 39893, 39909, 39917, 39925, 39933, 39941, 39949, 39957, 39965, 39973, 39997,
40013, 40029, 40037, 40045, 40061, 40069, 40077, 40089, 40105, 40121, 40129, 40137, 40145, 40161, 40169, 40177, 40189, 40205, 40237, 40245, 40253, 40261,
40293, 40309, 40317, 40325, 40333, 40341, 40357, 40365, 40373, 40381, 40405, 40413, 40421, 40429, 40445, 40453, 40469, 40477, 40485, 40501, 40509, 40517,
40525, 40533, 40565, 40581, 40589, 40597, 40605, 40621, 40637, 40645, 40653, 40661, 40669, 40677, 40685, 40701, 40709, 40725, 40733, 40749, 40757, 40773,
40789, 40793, 40809, 40841, 40849, 40873, 40881, 40897, 40905, 40921, 40945, 40953, 40961, 40969, 40977, 40985, 41001, 41009, 41033, 41041, 41057, 41073,
41089, 41097, 41113, 41121, 41129, 41137, 41153, 41161, 41177, 41185, 41201, 41217, 41225, 41229, 41245, 41277, 41285, 41293, 41309, 41325, 41333, 41341,
41357, 41373, 41389, 41397, 41405, 41421, 41429, 41445, 41453, 41461, 41477, 41493, 41501, 41517, 41533, 41545, 41561, 41569, 41585, 41593, 41601, 41633,
41641, 41657, 41665, 41681, 41689, 41705, 41713, 41729, 41737, 41753, 41769, 41777, 41793, 41801, 41809, 41817, 41825, 41833, 41849, 41857, 41905, 41913,
41921, 41929, 41961, 41969, 41977, 41985, 42001, 42017, 42025, 42033, 42049, 42065, 42081, 42089, 42097, 42105, 42121, 42129, 42137, 42145, 42153, 42161,
42177, 42185, 42193, 42201, 42217, 42229, 42253, 42265, 42273, 42289, 42305, 42313, 42321, 42329, 42337, 42345, 42361, 42393, 42401, 42417, 42449, 42457,
42473, 42497, 42505, 42529, 42537, 42553, 42561, 42577, 42593, 42601, 42609, 42617, 42633, 42641, 42657, 42665, 42673, 42681, 42689, 42697, 42721, 42737,
42745, 42761, 42777, 42809, 42817, 42825, 42841, 42857, 42873, 42881, 42889, 42897, 42913, 42921, 42929, 42937, 42945, 42953, 42961, 42969, 42981, 43005,
43013, 43021, 43029, 43053, 43069, 43077, 43085, 43093, 43101, 43125, 43133, 43149, 43157, 43173, 43181, 43197, 43213, 43221, 43229, 43245, 43253, 43261,
43277, 43281, 43313, 43321, 43337, 43345, 43361, 43369, 43385, 43401, 43417, 43425, 43433, 43449, 43457, 43473, 43489, 43497, 43505, 43513, 43537, 43545,
43569, 43577, 43585, 43593, 43601, 43617, 43625, 43641, 43649, 43657, 43673, 43681, 43689, 43697, 43705, 43709, 43733, 43741, 43749, 43781, 43789, 43805,
43837, 43845, 43853, 43869, 43885, 43893, 43909, 43925, 43933, 43949, 43965, 43981, 43989, 43997, 44005, 44013, 44029, 44037, 44045, 44061, 44069, 44077,
44085, 44093, 44101, 44109, 44133, 44157, 44165, 44197, 44213, 44221, 44229, 44245, 44261, 44277, 44293, 44301, 44305, 44313, 44321, 44337, 44369, 44385,
44393, 44401, 44417, 44425, 44441, 44449, 44457, 44469, 44485, 44493, 44509, 44541, 44549, 44557, 44573, 44581, 44597, 44613, 44621, 44637, 44653, 44669,
44677, 44685, 44693, 44717, 44725, 44741, 44749, 44765, 44773, 44789, 44797, 44813, 44829, 44845, 44853, 44869, 44885, 44901, 44909, 44917, 44941, 44957,
44965, 44981, 44989, 45005, 45013, 45021, 45037, 45045, 45053, 45061, 45077, 45085, 45133, 45141, 45149, 45157, 45165, 45181, 45189, 45197, 45213, 45225,
45233, 45249, 45257, 45265, 45281, 45297, 45305, 45329, 45337, 45345, 45353, 45361, 45369, 45405, 45413, 45429, 45437, 45445, 45453, 45461, 45477, 45485,
45501, 45509, 45525, 45541, 45549, 45581, 45597, 45605, 45613, 45621, 45637, 45653, 45669, 45677, 45685, 45693, 45709, 45717, 45725, 45733, 45741, 45749,
45765, 45781, 45789, 45805, 45813, 45821, 45829, 45845, 45869, 45877, 45893, 45901, 45909, 45917, 45949, 45957, 45965, 45969, 45977, 46009, 46033, 46041,
46049, 46057, 46073, 46081, 46089, 46097, 46113, 46129, 46145, 46177, 46185, 46193, 46209, 46241, 46249, 46257, 46265, 46281, 46289, 46297, 46305, 46313,
46321, 46337, 46345, 46353, 46369, 46385, 46393, 46409, 46417, 46433, 46441, 46449, 46453, 46469, 46477, 46501, 46509, 46517, 46533, 46541, 46549, 46565,
46589, 46597, 46605, 46621, 46637, 46653, 46669, 46677, 46685, 46693, 46701, 46713, 46745, 46753, 46769, 46785, 46793, 46817, 46833, 46849, 46857, 46865,
46873, 46889, 46897, 46905, 46921, 46929, 46945, 46961, 46977, 46985, 46993, 47001, 47033, 47041, 47049, 47065, 47073, 47081, 47097, 47113, 47121, 47129,
47145, 47161, 47169, 47177, 47185, 47193, 47217, 47241, 47249, 47273, 47281, 47297, 47305, 47313, 47321, 47329, 47337, 47345, 47353, 47361, 47393, 47401,
47409, 47417, 47433, 47441, 47449, 47457, 47473, 47485, 47517, 47533, 47541, 47553, 47569, 47601, 47609, 47617, 47625, 47633, 47641, 47657, 47689, 47697,
47705, 47713, 47729, 47745, 47753, 47761, 47777, 47785, 47809, 47817, 47833, 47849, 47857, 47873, 47881, 47889, 47897, 47913, 47921, 47953, 47961, 47977,
47985, 47993, 48001, 48009, 48017, 48033, 48041, 48065, 48073, 48089, 48097, 48121, 48145, 48153, 48161, 48169, 48177, 48185, 48193, 48201, 48217, 48225,
48233, 48241, 48257, 48289, 48297, 48301, 48309, 48325, 48333, 48365, 48373, 48389, 48405, 48413, 48421, 48429, 48445, 48453, 48477, 48485, 48493, 48509,
48525, 48541, 48573, 48589, 48597, 48605, 48613, 48629, 48645, 48653, 48657, 48665, 48681, 48689, 48705, 48713, 48721, 48729, 48745, 48753, 48769, 48777,
48793, 48809, 48817, 48833, 48841, 48857, 48865, 48881, 48889, 48905, 48921, 48937, 48945, 48953, 48961, 48969, 48977, 48985, 48993, 49009, 49025, 49041,
49049, 49077, 49093, 49101, 49109, 49117, 49125, 49133, 49141, 49165, 49173, 49181, 49213, 49221, 49229, 49245, 49261, 49269, 49277, 49293, 49309, 49325,
49333, 49357, 49373, 49381, 49389, 49397, 49445, 49461, 49469, 49477, 49485, 49493, 49501, 49517, 49525, 49541, 49549, 49565, 49573, 49581, 49589, 49605,
49621, 49629, 49637, 49669, 49677, 49685, 49701, 49717, 49729, 49745, 49761, 49785, 49801, 49809, 49841, 49849, 49857, 49861, 49869, 49885, 49893, 49909,
49917, 49933, 49941, 49949, 49965, 49981, 49997, 50005, 50021, 50037, 50061, 50077, 50085, 50101, 50117, 50125, 50141, 50149, 50157, 50165, 50181, 50189,
50197, 50213, 50221, 50229, 50245, 50261, 50269, 50277, 50293, 50301, 50317, 50333, 50357, 50373, 50381, 50389, 50405, 50421, 50437, 50445, 50453, 50461,
50477, 50485, 50493, 50501, 50517, 50525, 50541, 50557, 50565, 50573, 50589, 50597, 50613, 50617, 50633, 50649, 50681, 50689, 50697, 50705, 50713, 50729,
50745, 50761, 50777, 50785, 50793, 50825, 50833, 50849, 50857, 50865, 50873, 50881, 50893, 50909, 50917, 50925, 50933, 50949, 50957, 50973, 50981, 50989,
50997, 51005, 51021, 51037, 51045, 51053, 51093, 51101, 51117, 51125, 51133, 51149, 51165, 51181, 51189, 51205, 51213, 51229, 51237, 51245, 51253, 51277,
51293, 51301, 51309, 51317, 51325, 51357, 51373, 51381, 51389, 51397, 51413, 51421, 51429, 51433, 51465, 51481, 51505, 51529, 51561, 51577, 51585, 51593,
51609, 51617, 51625, 51649, 51657, 51673, 51681, 51689, 51697, 51705, 51713, 51745, 51753, 51761, 51769, 51777, 51793, 51801, 51833, 51841, 51857, 51873,
51889, 51897, 51905, 51929, 51937, 51945, 51953, 51961, 51969, 51977, 51985, 51993, 52005, 52021, 52037, 52053, 52061, 52069, 52085, 52093, 52109, 52125,
52133, 52141, 52157, 52173, 52189, 52197, 52213, 52221, 52229, 52237, 52253, 52257, 52273, 52289, 52297, 52305, 52329, 52337, 52345, 52361, 52369, 52385,
52393, 52401, 52417, 52433, 52441, 52449, 52481, 52489, 52497, 52505, 52521, 52537, 52545, 52561, 52569, 52585, 52593, 52601, 52633, 52641, 52649, 52665,
52689, 52705, 52713, 52721, 52737, 52745, 52761, 52777, 52785, 52801, 52825, 52849, 52873, 52881, 52897, 52913, 52921, 52945, 52953, 52961, 52977, 52993,
53001, 53009, 53017, 53025, 53041, 53077, 53085, 53093, 53109, 53125, 53133, 53141, 53149, 53173, 53177, 53193, 53209, 53217, 53225, 53241, 53257, 53273,
53281, 53297, 53305, 53337, 53353, 53361, 53377, 53385, 53393, 53409, 53441, 53449, 53457, 53473, 53481, 53489, 53497, 53505, 53513, 53521, 53529, 53537,
53545, 53553, 53569, 53585, 53601, 53617, 53641, 53649, 53657, 53673, 53689, 53697, 53705, 53713, 53721, 53729, 53737, 53753, 53761, 53793, 53801, 53809,
53825, 53841, 53849, 53865, 53873, 53881, 53885, 53893, 53909, 53925, 53957, 53965, 53981, 53997, 54013, 54021, 54029, 54037, 54053, 54061, 54077, 54085,
54093, 54101, 54117, 54125, 54173, 54189, 54205, 54213, 54221, 54237, 54253, 54261, 54277, 54293, 54301, 54317, 54325, 54333, 54337, 54361, 54377, 54393,
54401, 54417, 54425, 54441, 54449, 54465, 54473, 54481, 54489, 54505, 54513, 54521, 54529, 54545, 54553, 54569, 54577, 54585, 54601, 54609, 54617, 54625,
54633, 54641, 54649, 54657, 54665, 54681, 54689, 54693, 54741, 54749, 54757, 54773, 54789, 54805, 54813, 54821, 54845, 54861, 54877, 54885, 54901, 54909,
54917, 54949, 54957, 54965, 54973, 54981, 54997, 55013, 55021, 55029, 55037, 55053, 55077, 55093, 55109, 55117, 55133, 55141, 55157, 55165, 55181, 55197,
55205, 55213, 55221, 55237, 55245, 55261, 55269, 55277, 55293, 55301, 55317, 55325, 55341, 55365, 55373, 55381, 55389, 55405, 55421, 55445, 55453, 55461,
55477, 55485, 55493, 55497, 55505, 55537, 55545, 55553, 55557, 55589, 55605, 55613, 55621, 55637, 55645, 55661, 55669, 55693, 55701, 55709, 55725, 55733,
55741, 55757, 55765, 55781, 55789, 55805, 55813, 55821, 55829, 55837, 55853, 55869, 55885, 55893, 55901, 55925, 55949, 55981, 55997, 56005, 56013, 56029,
56045, 56053, 56069, 56077, 56093, 56109, 56117, 56125, 56141, 56157, 56165, 56173, 56189, 56205, 56221, 56229, 56245, 56261, 56269, 56277, 56301, 56309,
56317, 56325, 56333, 56341, 56373, 56381, 56397, 56401, 56409, 56425, 56441, 56449, 56457, 56465, 56481, 56489, 56497, 56505, 56513, 56529, 56545, 56553,
56585, 56601, 56609, 56641, 56649, 56657, 56673, 56681, 56697, 56705, 56717, 56725, 56741, 56749, 56757, 56765, 56781, 56789, 56797, 56813, 56821, 56829,
56845, 56853, 56861, 56869, 56893, 56901, 56933, 56973, 56989, 56997, 57013, 57021, 57029, 57037, 57053, 57061, 57069, 57085, 57101, 57117, 57125, 57133,
57149, 57157, 57165, 57181, 57189, 57197, 57209, 57225, 57233, 57241, 57249, 57281, 57289, 57321, 57329, 57361, 57369, 57377, 57385, 57417, 57425, 57433,
57441, 57449, 57465, 57481, 57489, 57505, 57513, 57521, 57529, 57537, 57553, 57569, 57577, 57585, 57609, 57617, 57633, 57641, 57649, 57657, 57665, 57681,
57689, 57705, 57721, 57729, 57737, 57753, 57769, 57777, 57793, 57801, 57809, 57825, 57841, 57849, 57857, 57881, 57885, 57893, 57909, 57925, 57933, 57941,
57965, 57973, 57981, 57989, 57997, 58013, 58021, 58029, 58037, 58045, 58061, 58069, 58077, 58089, 58105, 58137, 58145, 58161, 58177, 58185, 58193, 58201,
58209, 58241, 58257, 58265, 58281, 58289, 58297, 58305, 58313, 58329, 58337, 58353, 58369, 58385, 58393, 58409, 58417, 58425, 58433, 58457, 58473, 58481,
58489, 58521, 58529, 58545, 58561, 58569, 58585, 58601, 58625, 58633, 58641, 58657, 58673, 58681, 58689, 58721, 58729, 58737, 58745, 58753, 58769, 58777,
58793, 58809, 58817, 58825, 58841, 58857, 58873, 58881, 58889, 58897, 58905, 58913, 58929, 58941, 58957, 58965, 58973, 58989, 59037, 59045, 59053, 59061,
59069, 59077, 59093, 59105, 59121, 59137, 59145, 59161, 59177, 59185, 59201, 59249, 59265, 59273, 59289, 59297, 59313, 59321, 59329, 59337, 59345, 59353,
59369, 59385, 59393, 59409, 59417, 59425, 59433, 59441, 59449, 59473, 59481, 59497, 59513, 59521, 59537, 59545, 59553, 59569, 59577, 59585, 59617, 59633,
59649, 59657, 59665, 59673, 59681, 59697, 59705, 59713, 59721, 59729, 59745, 59761, 59785, 59793, 59801, 59805, 59837, 59845, 59861, 59877, 59893, 59901,
59909, 59917, 59925, 59933, 59949, 59957, 59965, 59981, 60005, 60013, 60029, 60045, 60061, 60077, 60085, 60093, 60125, 60141, 60149, 60157, 60165, 60181,
60189, 60197, 60205, 60221, 60237, 60245, 60253, 60269, 60285, 60301, 60309, 60317, 60321, 60353, 60361, 60369, 60377, 60401, 60409, 60417, 60425, 60433,
60465, 60473, 60481, 60489, 60505, 60513, 60529, 60537, 60545, 60553, 60569, 60585, 60593, 60601, 60625, 60633, 60641, 60649, 60665, 60669, 60677, 60701,
60717, 60725, 60733, 60749, 60781, 60797, 60805, 60821, 60829, 60837, 60869, 60877, 60885, 60893, 60901, 60917, 60925, 60941, 60965, 60981, 60997, 61005,
61021, 61029, 61037, 61045, 61053, 61061, 61069, 61077, 61093, 61101, 61125, 61133, 61141, 61173, 61181, 61189, 61197, 61205, 61213, 61229, 61237, 61253,
61269, 61277, 61285, 61293, 61301, 61317, 61333, 61341, 61349, 61357, 61373, 61381, 61397, 61413, 61421, 61429, 61445, 61453, 61461, 61469, 61477, 61493,
61509, 61517, 61529, 61545, 61549, 61573, 61581, 61613, 61621, 61637, 61653, 61685, 61701, 61709, 61717, 61733, 61749, 61757, 61773, 61781, 61789, 61797,
61805, 61837, 61845, 61853, 61869, 61877, 61901, 61909, 61917, 61925, 61957, 61989, 61997, 62013, 62029, 62037, 62045, 62053, 62061, 62069, 62085, 62093,
62109, 62125, 62133, 62141, 62157, 62165, 62173, 62181, 62189, 62197, 62229, 62237, 62245, 62293, 62301, 62309, 62317, 62325, 62349, 62357, 62373, 62381,
62389, 62421, 62429, 62433, 62441, 62473, 62481, 62513, 62521, 62537, 62545, 62561, 62569, 62577, 62593, 62617, 62633, 62641, 62657, 62665, 62681, 62689,
62697, 62705, 62713, 62729, 62737, 62745, 62777, 62785, 62793, 62801, 62817, 62825, 62845, 62861, 62877, 62893, 62909, 62917, 62925, 62949, 62957, 62965,
62973, 62981, 63005, 63013, 63029, 63045, 63061, 63069, 63085, 63093, 63101, 63117, 63125, 63141, 63149, 63165, 63173, 63181, 63197, 63205, 63213, 63229,
63237, 63245, 63261, 63269, 63277, 63285, 63301, 63305, 63337, 63345, 63361, 63369, 63385, 63393, 63409, 63417, 63433, 63457, 63465, 63481, 63513, 63529,
63553, 63569, 63577, 63585, 63601, 63617, 63633, 63641, 63657, 63673, 63681, 63697, 63705, 63713, 63729, 63745, 63753, 63769, 63777, 63793, 63801, 63809,
63825, 63833, 63841, 63857, 63865, 63873, 63889, 63897, 63929, 63937, 63945, 63953, 63969, 63977, 63993, 64001, 64017, 64025, 64033, 64041, 64049, 64057,
64069, 64077, 64109, 64125, 64141, 64157, 64165, 64173, 64189, 64197, 64209, 64233, 64241, 64257, 64265, 64273, 64289, 64305, 64321, 64329, 64337, 64345,
64377, 64393, 64409, 64441, 64449, 64457, 64465, 64481, 64489, 64513, 64521, 64537, 64545, 64553, 64561, 64577, 64593, 64609, 64625, 64641, 64657, 64665,
64673, 64681, 64713, 64729, 64745, 64753, 64769, 64793, 64809, 64817, 64825, 64833, 64841, 64849, 64857, 64873, 64889, 64905, 64921, 64937, 64945, 64953,
64961, 64969, 64985, 65001, 65009, 65041, 65049, 65057, 65073, 65097, 65101, 65117, 65125, 65157, 65165, 65181, 65189, 65197, 65205, 65213, 65221, 65237,
65253, 65261, 65269, 65277, 65285, 65301, 65309, 65317, 65333, 65357, 65369, 65377, 65393, 65409, 65417, 65425, 65449, 65465, 65473, 65489, 65505, 65513,
65529, 65537, 65545, 65553, 65569, 65577, 65585, 65601, 65617, 65633, 65649, 65665, 65673, 65681, 65713, 65721, 65729, 65745, 65753, 65761, 65769, 65777,
65785, 65793, 65801, 65809, 65825, 65833, 65849, 65865, 65881, 65897, 65905, 65913, 65929, 65937, 65945, 65953, 65969, 65985, 65993, 66001, 66009, 66045,
66053, 66069, 66085, 66101, 66117, 66125, 66149, 66165, 66173, 66189, 66197, 66205, 66213, 66229, 66245, 66261, 66269, 66277, 66285, 66309, 66317, 66333,
66341, 66349, 66397, 66413, 66421, 66429, 66445, 66461, 66469, 66477, 66485, 66493, 66501, 66533, 66549, 66557, 66565, 66573, 66581, 66605, 66613, 66629,
66637, 66645, 66649, 66665, 66673, 66681, 66697, 66705, 66745, 66761, 66769, 66785, 66793, 66809, 66825, 66833, 66841, 66857, 66873, 66889, 66905, 66921,
66929, 66937, 66945, 66957, 66965, 66997, 67021, 67029, 67037, 67053, 67061, 67077, 67093, 67109, 67117, 67133, 67149, 67157, 67173, 67197, 67213, 67221,
67229, 67237, 67253, 67269, 67285, 67293, 67301, 67325, 67333, 67341, 67373, 67389, 67405, 67413, 67421, 67453, 67469, 67477, 67493, 67501, 67509, 67525,
67533, 67541, 67549, 67565, 67573, 67581, 67597, 67605, 67613, 67621, 67637, 67653, 67661, 67669, 67677, 67693, 67701, 67717, 67733, 67741, 67749, 67757,
67765, 67773, 67781, 67789, 67797, 67805, 67837, 67853, 67857, 67873, 67881, 67889, 67897, 67929, 67941, 67957, 67965, 67981, 68013, 68021, 68045, 68053,
68069, 68077, 68085, 68093, 68101, 68109, 68125, 68141, 68157, 68165, 68181, 68197, 68205, 68213, 68221, 68237, 68245, 68261, 68269, 68301, 68309, 68325,
68333, 68341, 68357, 68365, 68381, 68389, 68397, 68405, 68413, 68421, 68437, 68445, 68461, 68469, 68493, 68509, 68525, 68533, 68541, 68549, 68573, 68581,
68589, 68597, 68629, 68653, 68669, 68685, 68709, 68717, 68725, 68741, 68749, 68757, 68765, 68777, 68809, 68817, 68825, 68841, 68849, 68865, 68881, 68905,
68913, 68921, 68929, 68937, 68953, 68969, 68977, 68993, 69009, 69017, 69025, 69033, 69065, 69073, 69089, 69097, 69105, 69113, 69121, 69129, 69145, 69153,
69169, 69177, 69185, 69209, 69217, 69225, 69241, 69257, 69273, 69285, 69301, 69317, 69333, 69341, 69357, 69365, 69381, 69389, 69397, 69405, 69413, 69461,
69477, 69485, 69493, 69509, 69517, 69525, 69541, 69549, 69557, 69565, 69573, 69581, 69589, 69605, 69621, 69637, 69653, 69685, 69693, 69701, 69709, 69717,
69729, 69745, 69761, 69777, 69785, 69801, 69809, 69825, 69833, 69841, 69849, 69857, 69873, 69905, 69913, 69929, 69961, 69969, 69977, 69985, 69993, 70009,
70025, 70041, 70049, 70057, 70065, 70081, 70097, 70105, 70121, 70153, 70161, 70169, 70177, 70193, 70209, 70241, 70249, 70257, 70265, 70273, 70289, 70305,
70313, 70321, 70337, 70353, 70361, 70369, 70377, 70393, 70401, 70409, 70425, 70433, 70441, 70449, 70457, 70465, 70473, 70481, 70505, 70513, 70529, 70545,
70561, 70569, 70581, 70597, 70613, 70621, 70629, 70661, 70681, 70689, 70705, 70721, 70729, 70745, 70753, 70801, 70817, 70825, 70833, 70841, 70849, 70857,
70873, 70881, 70897, 70929, 70937, 70953, 70969, 70985, 70993, 71001, 71025, 71041, 71049, 71057, 71065, 71073, 71105, 71113, 71129, 71137, 71145, 71153,
71169, 71177, 71185, 71193, 71209, 71217, 71233, 71265, 71273, 71281, 71297, 71305, 71313, 71329, 71337, 71345, 71361, 71369, 71377, 71401, 71417, 71425,
71433, 71441, 71457, 71473, 71481, 71497, 71505, 71537, 71545, 71553, 71561, 71569, 71577, 71593, 71609, 71613, 71629, 71637, 71653, 71669, 71677, 71685,
71701, 71717, 71725, 71741, 71757, 71781, 71797, 71805, 71813, 71821, 71829, 71861, 71869, 71885, 71893, 71901, 71917, 71925, 71929, 71953, 71961, 71977,
71993, 72009, 72017, 72033, 72041, 72057, 72065, 72073, 72081, 72113, 72121, 72137, 72145, 72153, 72169, 72177, 72185, 72209, 72225, 72233, 72241, 72249,
72257, 72289, 72297, 72305, 72321, 72329, 72345, 72353, 72361, 72369, 72385, 72401, 72425, 72433, 72441, 72457, 72465, 72481, 72489, 72505, 72513, 72529,
72533, 72549, 72565, 72581, 72589, 72613, 72653, 72669, 72685, 72701, 72733, 72749, 72773, 72789, 72821, 72829, 72837, 72845, 72861, 72869, 72877, 72885,
72893, 72917, 72925, 72933, 72941, 72949, 72973, 72989, 72997, 73013, 73021, 73037, 73045, 73053, 73069, 73085, 73093, 73101, 73109, 73133, 73141, 73157,
73189, 73197, 73205, 73229, 73237, 73245, 73249, 73265, 73281, 73289, 73297, 73305, 73321, 73329, 73345, 73353, 73361, 73369, 73385, 73393, 73409, 73417,
73425, 73441, 73449, 73465, 73473, 73497, 73505, 73517, 73533, 73549, 73557, 73565, 73589, 73621, 73629, 73645, 73653, 73669, 73677, 73693, 73701, 73717,
73733, 73749, 73757, 73765, 73773, 73789, 73797, 73813, 73821, 73837, 73845, 73853, 73861, 73885, 73901, 73933, 73949, 73957, 73989, 74005, 74013, 74021,
74029, 74037, 74045, 74061, 74077, 74085, 74093, 74125, 74133, 74141, 74149, 74157, 74173, 74181, 74189, 74197, 74213, 74221, 74229, 74237, 74245, 74261,
74269, 74285, 74293, 74301, 74317, 74333, 74341, 74357, 74365, 74373, 74389, 74397, 74405, 74413, 74437, 74453, 74457, 74473, 74489, 74505, 74553, 74569,
74585, 74593, 74601, 74609, 74617, 74625, 74637, 74653, 74669, 74685, 74693, 74701, 74733, 74741, 74749, 74757, 74773, 74781, 74789, 74805, 74821, 74829,
74837, 74853, 74861, 74869, 74893, 74901, 74917, 74925, 74941, 74965, 74973, 74989, 74997, 75005, 75013, 75021, 75029, 75061, 75069, 75085, 75093, 75101,
75109, 75117, 75125, 75133, 75141, 75149, 75165, 75173, 75189, 75205, 75221, 75237, 75253, 75261, 75269, 75285, 75301, 75309, 75317, 75325, 75357, 75365,
75373, 75389, 75405, 75413, 75421, 75437, 75453, 75465, 75481, 75489, 75505, 75513, 75529, 75545, 75553, 75601, 75609, 75625, 75633, 75641, 75649, 75665,
75697, 75705, 75713, 75721, 75745, 75761, 75769, 75777, 75785, 75793, 75825, 75841, 75849, 75857, 75865, 75873, 75881, 75897, 75913, 75929, 75937, 75953,
75961, 75977, 75985, 75993, 76005, 76013, 76021, 76037, 76053, 76061, 76077, 76085, 76101, 76109, 76117, 76125, 76133, 76149, 76157, 76189, 76221, 76237,
76245, 76261, 76277, 76285, 76293, 76309, 76317, 76325, 76341, 76357, 76373, 76381, 76389, 76397, 76405, 76413, 76425, 76433, 76449, 76465, 76481, 76497,
76505, 76513, 76529, 76545, 76561, 76569, 76577, 76593, 76609, 76625, 76633, 76641, 76665, 76681, 76689, 76705, 76713, 76721, 76737, 76745, 76769, 76785,
76801, 76809, 76825, 76833, 76849, 76865, 76881, 76889, 76897, 76913, 76921, 76929, 76937, 76945, 76953, 76969, 77017, 77025, 77033, 77041, 77057, 77081,
77089, 77097, 77105, 77121, 77129, 77137, 77153, 77161, 77169, 77201, 77209, 77225, 77241, 77249, 77265, 77273, 77281, 77297, 77329, 77345, 77361, 77373,
77389, 77401, 77449, 77465, 77473, 77489, 77505, 77513, 77537, 77545, 77561, 77577, 77585, 77593, 77609, 77625, 77633, 77649, 77657, 77665, 77681, 77689,
77705, 77713, 77721, 77737, 77753, 77761, 77777, 77785, 77793, 77801, 77817, 77825, 77841, 77849, 77857, 77873, 77881, 77889, 77897, 77913, 77921, 77937,
77953, 77961, 77969, 78001, 78009, 78017, 78025, 78033, 78041, 78057, 78065, 78081, 78089, 78105, 78113, 78121, 78137, 78145, 78153, 78161, 78169, 78177,
78185, 78201, 78233, 78241, 78249, 78257, 78265, 78281, 78305, 78321, 78337, 78353, 78361, 78369, 78385, 78393, 78409, 78413, 78429, 78437, 78469, 78477,
78493, 78501, 78509, 78517, 78525, 78549, 78565, 78597, 78621, 78629, 78637, 78645, 78653, 78661, 78677, 78693, 78709, 78717, 78733, 78749, 78757, 78765,
78773, 78805, 78809, 78841, 78857, 78865, 78873, 78881, 78897, 78905, 78913, 78929, 78937, 78953, 78961, 78969, 78985, 79001, 79009, 79017, 79025, 79057,
79065, 79073, 79081, 79097, 79105, 79129, 79137, 79145, 79153, 79161, 79169, 79177, 79193, 79201, 79233, 79257, 79265, 79273, 79281, 79289, 79321, 79329,
79345, 79353, 79361, 79369, 79381, 79389, 79421, 79437, 79453, 79469, 79477, 79485, 79493, 79509, 79517, 79533, 79541, 79565, 79597, 79613, 79645, 79653,
79661, 79669, 79677, 79685, 79701, 79709, 79725, 79733, 79749, 79757, 79773, 79789, 79797, 79813, 79829, 79845, 79853, 79861, 79869, 79877, 79901, 79909,
79925, 79933, 79941, 79949, 79957, 79981, 79989, 79997, 80005, 80021, 80037, 80053, 80061, 80093, 80101, 80109, 80125, 80141, 80149, 80157, 80181, 80189,
80197, 80205, 80217, 80233, 80241, 80257, 80273, 80281, 80289, 80297, 80305, 80321, 80337, 80345, 80353, 80361, 80369, 80381, 80389, 80405, 80421, 80429,
80461, 80469, 80477, 80485, 80525, 80533, 80541, 80573, 80589, 80597, 80613, 80621, 80629, 80637, 80645, 80661, 80669, 80677, 80685, 80701, 80709, 80717,
80725, 80749, 80781, 80789, 80797, 80805, 80829, 80837, 80845, 80853, 80869, 80877, 80885, 80901, 80909, 80925, 80933, 80949, 80965, 80973, 80989, 80997,
81029, 81037, 81045, 81053, 81061, 81093, 81101, 81133, 81157, 81173, 81181, 81189, 81197, 81205, 81213, 81237, 81253, 81261, 81277, 81285, 81301, 81309,
81317, 81325, 81333, 81341, 81357, 81365, 81373, 81377, 81393, 81401, 81449, 81465, 81473, 81481, 81497, 81513, 81529, 81537, 81553, 81569, 81585, 81593,
81609, 81617, 81633, 81649, 81653, 81669, 81677, 81709, 81717, 81741, 81749, 81757, 81773, 81781, 81789, 81797, 81805, 81821, 81837, 81861, 81869, 81877,
81885, 81917, 81925, 81933, 81941, 81957, 81965, 81981, 81997, 82005, 82029, 82037, 82045, 82061, 82077, 82093, 82125, 82133, 82149, 82157, 82165, 82173,
82189, 82205, 82213, 82221, 82229, 82245, 82253, 82261, 82269, 82277, 82293, 82301, 82317, 82325, 82341, 82365, 82381, 82397, 82405, 82421, 82429, 82433,
82465, 82481, 82489, 82497, 82529, 82537, 82553, 82561, 82569, 82585, 82601, 82617, 82625, 82633, 82649, 82657, 82665, 82673, 82681, 82689, 82713, 82729,
82745, 82761, 82777, 82793, 82801, 82809, 82817, 82833, 82841, 82857, 82873, 82889, 82897, 82913, 82921, 82929, 82945, 82961, 82969, 82985, 82993, 83009,
83025, 83033, 83041, 83057, 83073, 83089, 83097, 83109, 83125, 83133, 83149, 83165, 83173, 83189, 83197, 83205, 83213, 83245, 83253, 83261, 83277, 83285,
83293, 83301, 83317, 83341, 83349, 83373, 83389, 83405, 83413, 83421, 83429, 83445, 83453, 83457, 83473, 83481, 83489, 83505, 83537, 83545, 83561, 83569,
83601, 83609, 83617, 83625, 83633, 83641, 83649, 83657, 83673, 83681, 83689, 83713, 83761, 83769, 83777, 83793, 83801, 83809, 83817, 83833, 83849, 83881,
83889, 83897, 83905, 83913, 83929, 83937, 83945, 83953, 83961, 83985, 84001, 84009, 84017, 84033, 84049, 84057, 84065, 84097, 84113, 84129, 84137, 84153,
84169, 84185, 84193, 84201, 84209, 84217, 84225, 84241, 84249, 84297, 84305, 84313, 84321, 84329, 84337, 84345, 84353, 84369, 84377, 84393, 84401, 84409,
84417, 84425, 84441, 84449, 84461, 84477, 84501, 84509, 84525, 84537, 84545, 84561, 84569, 84585, 84593, 84609, 84617, 84633, 84641, 84673, 84681, 84689,
84697, 84713, 84745, 84761, 84777, 84785, 84801, 84809, 84817, 84833, 84841, 84857, 84865, 84897, 84905, 84913, 84929, 84953, 84961, 84977, 85001, 85017,
85025, 85041, 85049, 85057, 85065, 85081, 85097, 85105, 85121, 85137, 85153, 85161, 85169, 85177, 85193, 85201, 85217, 85225, 85241, 85273, 85281, 85297,
85305, 85313, 85329, 85345, 85361, 85369, 85385, 85401, 85409, 85417, 85441, 85457, 85473, 85489, 85501, 85509, 85525, 85541, 85557, 85565, 85581, 85597,
85629, 85637, 85645, 85661, 85677, 85685, 85701, 85717, 85725, 85733, 85749, 85765, 85773, 85781, 85789, 85797, 85805, 85821, 85829, 85853, 85861, 85869,
85877, 85885, 85893, 85901, 85909, 85925, 85933, 85941, 85949, 85981, 85989, 86001, 86025, 86041, 86049, 86065, 86073, 86081, 86089, 86113, 86121, 86137,
86145, 86161, 86169, 86193, 86201, 86209, 86217, 86225, 86233, 86257, 86265, 86273, 86289, 86297, 86313, 86329, 86337, 86345, 86361, 86377, 86393, 86401,
86409, 86417, 86449, 86457, 86473, 86481, 86489, 86505, 86513, 86521, 86525, 86541, 86573, 86581, 86613, 86629, 86637, 86645, 86661, 86677, 86685, 86701,
86717, 86733, 86749, 86757, 86773, 86837, 86853, 86869, 86877, 86885, 86893, 86909, 86917, 86925, 86941, 86957, 86973, 86981, 86989, 87005, 87013, 87021,
87037, 87045, 87069, 87077, 87085, 87093, 87117, 87141, 87149, 87157, 87165, 87173, 87181, 87189, 87197, 87213, 87229, 87245, 87253, 87269, 87285, 87293,
87301, 87317, 87325, 87333, 87341, 87349, 87365, 87373, 87381, 87389, 87405, 87421, 87429, 87437, 87453, 87457, 87481, 87489, 87505, 87521, 87529, 87537,
87553, 87585, 87593, 87601, 87605, 87621, 87629, 87661, 87669, 87677, 87693, 87701, 87717, 87725, 87741, 87765, 87773, 87789, 87797, 87813, 87821, 87829,
87845, 87853, 87869, 87885, 87901, 87909, 87941, 87949, 87965, 87973, 87981, 87989, 87997, 88005, 88021, 88037, 88053, 88061, 88069, 88093, 88101, 88117,
88125, 88133, 88141, 88149, 88157, 88165, 88173, 88181, 88189, 88221, 88229, 88253, 88269, 88277, 88301, 88309, 88317, 88333, 88357, 88389, 88405, 88413,
88445, 88461, 88469, 88485, 88501, 88517, 88533, 88541, 88557, 88565, 88573, 88589, 88597, 88613, 88629, 88633, 88657, 88665, 88673, 88681, 88689, 88705,
88737, 88769, 88785, 88793, 88809, 88817, 88825, 88833, 88841, 88857, 88873, 88881, 88889, 88897, 88913, 88945, 88953, 88961, 88969, 88981, 88997, 89005,
89021, 89037, 89069, 89085, 89093, 89101, 89109, 89125, 89133, 89149, 89165, 89181, 89197, 89205, 89213, 89229, 89237, 89253, 89269, 89285, 89301, 89317,
89325, 89333, 89341, 89349, 89357, 89373, 89389, 89413, 89429, 89437, 89453, 89461, 89477, 89485, 89493, 89509, 89517, 89533, 89541, 89557, 89565, 89573,
89581, 89597, 89613, 89621, 89629, 89645, 89653, 89673, 89681, 89697, 89729, 89745, 89753, 89777, 89793, 89801, 89817, 89825, 89833, 89857, 89865, 89881,
89897, 89905, 89921, 89929, 89937, 89969, 89985, 89993, 90001, 90033, 90041, 90049, 90057, 90073, 90081, 90097, 90105, 90121, 90129, 90145, 90153, 90161,
90177, 90185, 90201, 90209, 90217, 90225, 90273, 90281, 90289, 90305, 90313, 90329, 90337, 90345, 90361, 90377, 90393, 90401, 90409, 90425, 90441, 90449,
90457, 90469, 90477, 90493, 90501, 90509, 90517, 90525, 90549, 90565, 90581, 90589, 90597, 90613, 90629, 90653, 90661, 90669, 90685, 90693, 90701, 90709,
90741, 90749, 90785, 90793, 90801, 90817, 90825, 90841, 90849, 90857, 90865, 90913, 90921, 90929, 90937, 90953, 90961, 90969, 90977, 90993, 91001, 91009,
91025, 91041, 91049, 91065, 91073, 91105, 91121, 91137, 91145, 91153, 91161, 91169, 91177, 91185, 91193, 91209, 91225, 91241, 91273, 91289, 91297, 91313,
91321, 91337, 91345, 91361, 91369, 91377, 91393, 91401, 91417, 91425, 91441, 91449, 91457, 91473, 91481, 91489, 91497, 91529, 91537, 91545, 91561, 91569,
91585, 91593, 91601, 91617, 91625, 91633, 91665, 91673, 91681, 91705, 91721, 91745, 91753, 91761, 91769, 91777, 91793, 91801, 91825, 91829, 91837, 91853,
91861, 91893, 91909, 91917, 91933, 91941, 91949, 91965, 91973, 91977, 91985, 92001, 92033, 92041, 92065, 92081, 92089, 92097, 92105, 92113, 92129, 92145,
92153, 92161, 92169, 92177, 92209, 92225, 92241, 92257, 92273, 92289, 92305, 92337, 92345, 92361, 92369, 92377, 92385, 92401, 92417, 92449, 92457, 92465,
92473, 92481, 92489, 92505, 92513, 92521, 92529, 92545, 92553, 92561, 92569, 92577, 92585, 92601, 92617, 92633, 92649, 92665, 92673, 92681, 92689, 92705,
92721, 92737, 92745, 92769, 92785, 92801, 92809, 92825, 92833, 92849, 92857, 92873, 92881, 92889, 92897, 92905, 92909, 92941, 92957, 92973, 92981, 92997,
93021, 93029, 93037, 93053, 93069, 93077, 93085, 93093, 93109, 93117, 93125, 93133, 93149, 93173, 93189, 93221, 93229, 93237, 93245, 93261, 93277, 93293,
93309, 93317, 93349, 93357, 93365, 93413, 93421, 93429, 93437, 93445, 93461, 93469, 93477, 93493, 93505, 93537, 93545, 93561, 93577, 93585, 93593, 93617,
93633, 93649, 93665, 93673, 93689, 93697, 93721, 93729, 93737, 93745, 93753, 93769, 93777, 93793, 93801, 93817, 93833, 93841, 93857, 93865, 93873, 93881,
93897, 93913, 93929, 93937, 93945, 93953, 93961, 93969, 93981, 94013, 94029, 94037, 94069, 94077, 94093, 94101, 94117, 94125, 94157, 94165, 94181, 94197,
94205, 94213, 94221, 94237, 94253, 94285, 94301, 94309, 94325, 94349, 94357, 94365, 94373, 94381, 94389, 94397, 94421, 94437, 94461, 94469, 94485, 94517,
94541, 94549, 94557, 94565, 94581, 94589, 94597, 94605, 94613, 94621, 94629, 94661, 94669, 94677, 94693, 94709, 94717, 94725, 94733, 94741, 94757, 94789,
94797, 94805, 94813, 94821, 94837, 94853, 94861, 94869, 94877, 94893, 94901, 94917, 94925, 94933, 94941, 94949, 94957, 94965, 94981, 94989, 95001, 95033,
95049, 95065, 95073, 95081, 95093, 95125, 95141, 95157, 95189, 95197, 95205, 95213, 95229, 95237, 95245, 95253, 95261, 95285, 95301, 95317, 95349, 95357,
95365, 95373, 95381, 95389, 95405, 95421, 95437, 95453, 95461, 95477, 95485, 95493, 95509, 95525, 95541, 95549, 95573, 95589, 95613, 95629, 95637, 95653,
95669, 95677, 95693, 95709, 95717, 95725, 95741, 95749, 95757, 95765, 95781, 95789, 95797, 95805, 95837, 95845, 95861, 95869, 95877, 95885, 95893, 95909,
95925, 95933, 95965, 95973, 95981, 95989, 95997, 96013, 96021, 96029, 96037, 96053, 96069, 96093, 96101, 96109, 96133, 96149, 96157, 96173, 96189, 96209,
96217, 96233, 96257, 96265, 96281, 96289, 96313, 96321, 96329, 96337, 96353, 96369, 96377, 96385, 96393, 96401, 96417, 96425, 96433, 96441, 96449, 96465,
96473, 96481, 96489, 96497, 96505, 96529, 96545, 96561, 96593, 96601, 96617, 96621, 96629, 96637, 96653, 96661, 96677, 96709, 96717, 96725, 96733, 96749,
96765, 96773, 96805, 96813, 96821, 96837, 96845, 96869, 96877, 96885, 96901, 96909, 96917, 96941, 96949, 96957, 96973, 96981, 96989, 97005, 97013, 97029,
97037, 97045, 97061, 97077, 97109, 97125, 97133, 97157, 97165, 97181, 97189, 97197, 97205, 97213, 97229, 97245, 97261, 97269, 97273, 97281, 97289, 97305,
97321, 97329, 97345, 97353, 97369, 97401, 97409, 97417, 97425, 97441, 97489, 97497, 97505, 97521, 97537, 97545, 97553, 97569, 97585, 97601, 97617, 97633,
97641, 97649, 97665, 97673, 97681, 97697, 97713, 97721, 97737, 97753, 97761, 97769, 97793, 97809, 97817, 97825, 97841, 97857, 97873, 97881, 97889, 97905,
97913, 97921, 97929, 97937, 97945, 97953, 97961, 97969, 97977, 97985, 98001, 98009, 98025, 98033, 98049, 98057, 98065, 98089, 98105, 98113, 98121, 98129,
98157, 98173, 98181, 98189, 98197, 98245, 98253, 98261, 98269, 98277, 98293, 98301, 98325, 98341, 98357, 98365, 98373, 98381, 98385, 98417, 98425, 98433,
98441, 98449, 98465, 98481, 98497, 98505, 98513, 98521, 98553, 98569, 98585, 98601, 98617, 98633, 98641, 98649, 98657, 98681, 98689, 98705, 98713, 98729,
98745, 98753, 98769, 98777, 98785, 98833, 98849, 98857, 98865, 98881, 98889, 98897, 98905, 98913, 98921, 98929, 98953, 98969, 98977, 98985, 98993, 99001,
99017, 99025, 99073, 99089, 99105, 99113, 99129, 99137, 99153, 99161, 99177, 99185, 99193, 99201, 99209, 99225, 99241, 99257, 99265, 99273, 99281, 99297,
99329, 99345, 99353, 99361, 99377, 99393, 99409, 99417, 99433, 99441, 99449, 99465, 99477, 99493, 99509, 99525, 99541, 99557, 99581, 99613, 99629, 99645,
99653, 99685, 99693, 99701, 99709, 99725, 99733, 99741, 99745, 99761, 99769, 99785, 99793, 99801, 99817, 99825, 99841, 99849, 99865, 99881, 99889, 99897,
99905, 99913, 99929, 99945, 99961, 99977, 99985, 100001], dtype=np.int32)


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
        raise Exception('Only 1, 2, 4, 6, or 8 tube passes are supported')
    ans = int(ans)
    # In some cases, a negative number would be returned by these formulas
    if ans < 0:
        ans = 0 # pragma: no cover
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
    if angle == 30 or angle == 60:
        Ns = triangular_Ns[-1]
    elif angle == 45 or angle == 90:
        Ns = square_Ns[-1]
    s = Ns + 1
    r = s**0.5
    DBundle_max = (Do + 2.*pitch*r)*(1. - 1E-8) # Cannot be exact or floor(s) will give an int too high
    def to_solve(DBundle):
        ans = Ntubes_Phadkeb(DBundle=DBundle, Do=Do, pitch=pitch, Ntp=Ntp, angle=angle) - Ntubes
        return ans
    return sp_bisect(to_solve, 0, DBundle_max)


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
            raise Exception('N passes not 1, 2, 4 or 6')
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
            raise Exception('N passes not 1, 2, 4 or 6')
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
        raise Exception('Only 1, 2, 4 and 8 passes are supported')
    if angle == 30 or angle == 60:
        f1 = 1.1
    elif angle == 45 or angle == 90:
        f1 = 1.3
    else:
        raise Exception('Only 30, 60, 45 and 90 degree layouts are supported')

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
        raise Exception('Only 1, 2, 4 and 8 passes are supported')
    if angle == 30 or angle == 60:
        f1 = 1.1
    elif angle == 45 or angle == 90:
        f1 = 1.3
    else:
        raise Exception('Only 30, 60, 45 and 90 degree layouts are supported')
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
        raise Exception('Only 30, 60, 45 and 90 degree layouts are supported')
    Dctl = DBundle - Do
    N = 0.78*Dctl**2/C1/pitch**2
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
    1.1839930795640605

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
        raise Exception('Only 30, 60, 45 and 90 degree layouts are supported')
    return (Do + (1./.78)**0.5*pitch*(C1*N)**0.5)


def Ntubes(DBundle, Do, pitch, Ntp=1, angle=30, Method=None, 
           AvailableMethods=False):
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
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to calculate the tube count

    Other Parameters
    ----------------
    Method : string, optional
        One of 'Phadkeb', 'HEDH', 'VDI' or 'Perry'
    AvailableMethods : bool, optional
        If True, function will consider which methods which can be used to
        calculate the tube count with the given inputs

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
    def list_methods():
        methods = ['Phadkeb']
        if Ntp == 1:
            methods.append('HEDH')
        if Ntp in [1, 2, 4, 8]:
            methods.append('VDI')
        if Ntp in [1, 2, 4, 6]:
             # Also restricted to 1.25 pitch ratio but not hard coded
            methods.append('Perry')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
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
        raise Exception('Method not recognized; allowable methods are '
                        '"Phadkeb", "HEDH", "VDI", and "Perry"')


def size_bundle_from_tubecount(N, Do, pitch, Ntp=1, angle=30, Method=None,
                               AvailableMethods=False):
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
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to calculate the tube count

    Other Parameters
    ----------------
    Method : string, optional
        One of 'Phadkeb', 'HEDH', 'VDI' or 'Perry'
    AvailableMethods : bool, optional
        If True, function will consider which methods which can be used to
        calculate the tube count with the given inputs

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
    def list_methods():
        methods = ['Phadkeb']
        if Ntp == 1:
            methods.append('HEDH')
        if Ntp in [1, 2, 4, 8]:
            methods.append('VDI')
        if Ntp in [1, 2, 4, 6]:
             # Also restricted to 1.25 pitch ratio but not hard coded
            methods.append('Perry')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = 'Phadkeb'
        
    if Method == 'Phadkeb':
        return DBundle_for_Ntubes_Phadkeb(Ntubes=N, Ntp=Ntp, Do=Do, pitch=pitch, angle=angle)
    elif Method == 'VDI':
        return D_for_Ntubes_VDI(N=N, Ntp=Ntp, Do=Do, pitch=pitch, angle=angle)
    elif Method == 'HEDH':
        return DBundle_for_Ntubes_HEDH(N=N, Do=Do, pitch=pitch, angle=angle)
    elif Method == 'Perry':
        to_solve = lambda D : Ntubes_Perrys(DBundle=D, Do=Do, Ntp=Ntp, angle=angle) - N
        return ridder(to_solve, Do*5, 1000*Do)
    else:
        raise Exception('Method not recognized; allowable methods are '
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
