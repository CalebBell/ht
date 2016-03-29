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
from math import exp, log, floor
from math import tanh # 1/coth
from scipy.interpolate import interp1d
from fluids.piping import BWG_integers, BWG_inch, BWG_SI
TEMA_R_to_metric = 0.17611018

__all__ = ['effectiveness_from_NTU', 'calc_Cmin', 'calc_Cmax', 'calc_Cr',
'NTU_from_UA', 'UA_from_NTU', 'check_tubing_TEMA', 'get_tube_TEMA',
'DBundle_min', 'shell_clearance', 'baffle_thickness', 'D_baffle_holes',
'L_unsupported_max', 'Ntubes_Perrys', 'Ntubes_VDI', 'Ntubes_Phadkeb',
'Ntubes_HEDH', 'Ntubes', 'D_for_Ntubes_VDI']


# TODO: Implement selection algorithms for heat exchangers from
# Systematic Procedure for Selection of Heat Exchangers
# 10.1243/PIME_PROC_1983_197_006_02


def effectiveness_from_NTU(NTU, Cr, Ntp=1, shells=1, counterflow=True,
                           subtype='double-pipe', Cmin_mixed=False, Cmax_mixed=False):
    r'''Returns the effectiveness of a heat exchanger with a given NTU and Cr,
    number of tube passes, number of shells, and if it is counterflow or not,
    and if it is the subtype double-pipe, and if the shell and/or tube fluids
    are mixed or not.

    For Parallel Double pipe exchangers:

    .. math::
        \epsilon = \frac{1 - \exp[-NTU(1+C_r)]}{1+C_r}

    For Counterflow Double pipe exchangers:

    .. math::
        \epsilon = \frac{1 - \exp[-NTU(1-C_r)]}{1-C_r\exp[-NTU(1-C_r)]},\; C_r < 1

        \epsilon = \frac{NTU}{1+NTU},\; C_r = 1

    For Shell-and-tube exchangers with one shell pass, 2n tube passes:

    .. math::
        \epsilon_1 = 2\left\{1 + C_r + \sqrt{1+C_r^2}\times\frac{1+\exp
        [-(NTU)_1\sqrt{1+C_r^2}]}{1-\exp[-(NTU)_1\sqrt{1+C_r^2}]}\right\}^{-1}

    For Shell-and-tube exchangers with one shell pass, 2n tube passes:

    .. math::
        \epsilon = \left[\left(\frac{1-\epsilon_1 C_r}{1-\epsilon_1}\right)^2
        -1\right]\left[\left(\frac{1-\epsilon_1 C_r}{1-\epsilon_1}\right)^n
        - C_r\right]^{-1}

    For Cross-flow (single-pass) exchangers with both fluids unmixed:

    .. math::
        \epsilon = 1 - \exp\left[\left(\frac{1}{C_r}\right)
        (NTU)^{0.22}\left\{\exp\left[C_r(NTU)^{0.78}\right]-1\right\}\right]

    For Cross-flow (single-pass) exchangers with Cmax mixed:

    .. math::
        \epsilon = \left(\frac{1}{C_r}\right)(1 - \exp\left\{-C_r[1-\exp(-NTU)]\right\})

    For Cross-flow (single-pass) exchangers with Cmin mixed:

    .. math::
        \epsilon = 1 - \exp(-C_r^{-1}\{1 - \exp[-C_r(NTU)]\})

    For all other cases, and especially for boilers and condensers:

    .. math::
        \epsilon = 1 - \exp(-NTU)

    Parameters
    ----------
    NTU : float
        Thermal Number of Transfer Units [-]
    Cr : float
        The heat capacity rate ratio, of the smaller fluid to the larger
        fluid, [-]
    Ntp : float, optional
        Number of tube passes, [-]
    shells: float, optional
        Number of shells in series
    counterflow : bool, optional
        Whether the exchanger is counterflow or co-current
    subtype : str, optional
        The subtype of exchanger; one of 'shell and tube' or 'double-pipe'
    Cmin_mixed : bool, optional
        Whether or not the minimum heat capacity rate fluid is mized in the
        heat exchanger
    Cmax_mixed : bool, optional
        Whether or not the minimum heat capacity rate fluid is mized in the
        heat exchanger

    Returns
    -------
    effectiveness : float
        The thermal effectiveness of the heat exchanger, [-]

    Notes
    -----
    Many more correlations exist.

    Examples
    --------

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    if subtype == 'double-pipe':
        if not counterflow:
            effectiveness = (1-exp(-NTU*(1+Cr)))/(1+Cr)
        else:
            if Cr < 1:
                effectiveness = (1-exp(-NTU*(1-Cr)))/(1-Cr*exp(-NTU*(1-Cr)))
            elif Cr == 1:
                effectiveness = NTU/(1+NTU)
    elif Ntp > 1 and subtype == 'shell and tube':
        if not counterflow:
            raise Exception('Formulas for S&T are only for counterflow, not parallel')
        if Ntp % 2:
            raise Exception('For shell and tube exchangers with 2n tube passes, odd tube numbers not allowed')
        top = 1+exp(-NTU*(1+Cr**2)**.5)
        bottom = 1-exp(-NTU*(1+Cr**2)**.5)
        effectiveness = 2.0/(1+Cr+(1+Cr**2)**.5*top/bottom)
        if shells > 1:
            term = ((1-effectiveness*Cr)/(1-effectiveness))**shells
            effectiveness = (term-1)/(term-Cr)
    elif Ntp == 1 and subtype == 'shell and tube':
        if not Cmin_mixed and not Cmax_mixed:
            effectiveness = 1-exp(1./Cr*NTU**.22*(exp(-Cr*NTU**.78)-1))
        elif Cmax_mixed:
            effectiveness = (1./Cr)*(1-exp(-Cr**-1*(1-exp(-NTU))))
        elif Cmin_mixed:
            effectiveness = 1-exp(-Cr**-1*(1-exp(-Cr*NTU)))
    else:
        effectiveness = 1-exp(-NTU)
    return effectiveness







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
        The heat capacity rate of the smaller fluid, [-]

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
    Cmin = min(Ch, Cc)
    return Cmin
#print [calc_Cmin(mh=22., mc=5.5, Cph=2200, Cpc=4400.)]


def calc_Cmax(mh, mc, Cph, Cpc):
    r'''Returns the heat capacity rate for the maximum stream
    having flows `mh` and `mc`, with averaged heat capacities `Cph` and `Cpc`.

    .. math::
        C_c = m_cC_{p,c}

        C_h = m_h C_{p,h}

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
        The heat capacity rate of the larger fluid, [-]

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
    Cmax = max(Ch, Cc)
    return Cmax

#print [calc_Cmax(mh=22., mc=5.5, Cph=2200, Cpc=4400.)]

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
        fluid, [-]

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
    Cr = Cmin/Cmax
    return Cr

#print [calc_Cr(mh=22., mc=5.5, Cph=2200, Cpc=4400.)]


def NTU_from_UA(UA, Cmin):
    r'''Returns the Number of Transfer Units for a heat exchanger having
    UA, and with Cmin heat capacity rate.

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
    NTU = UA/Cmin
    return NTU

#print [NTU_from_UA(4400., 22.)]


def UA_from_NTU(NTU, Cmin):
    r'''Returns the combined Area-heat transfer term for a heat exchanger
    having a specified NTU, and with Cmin heat capacity rate.

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
        Combined Area-heat transfer coefficient term, [W/K]

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
    UA = NTU*Cmin
    return UA

#print [UA_from_NTU(200., 22.)]

#print NTU_effectiveness(3.5, .2345, n=3.0, Type='Counterflow')
#print NTU_NTU(.67, 9, .238, Type='Counterflow', n=3.0)



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

#TEMA_Full_Tubing = [(6.35,22), (6.35,24), (6.35,26), (6.35,27), (9.53,18), (9.53,20), (9.53,22), (9.53,24), (12.7,16), (12.7,16), (12.7,20), (12.7,22), (15.88,12), (15.88,13), (15.88,14), (15.88,15), (15.88,16), (15.88,17), (15.88,18), (15.88,19), (15.88,20), (19.05,10), (19.05,11), (19.05,12), (19.05,13), (19.05,14), (19.05,15), (19.05,16), (19.05,17), (19.05,18), (19.05,20), (22.23,10), (22.23,11), (22.23,12), (22.23,13), (22.23,14), (22.23,15), (22.23,16), (22.23,17), (22.23,18), (22.23,20), (25.4,8), (25.4,10), (25.4,11), (25.4,12), (25.4,13), (25.4,14), (25.4,15), (25.4,16), (25.4,18), (25.4,20), (31.75,7), (31.75,8), (31.75,10), (31.75,11), (31.75,12), (31.75,13), (31.75,14), (31.75,16), (31.75,18), (31.75,20), (38.1,10), (38.1,12), (38.1,14), (38.1,16), (50.8,11), (50.8,12), (50.8,13), (50.8,14), (63.5,10), (63.5,12), (63.5,14), (76.2,10), (76.2,12), (76.2,14)]
#BWG_integers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
#BWG_inch = [0.34, 0.3, 0.284, 0.259, 0.238, 0.22, 0.203, 0.18, 0.165, 0.148, 0.134, 0.12, 0.109, 0.095, 0.083, 0.072, 0.065, 0.058, 0.049, 0.042, 0.035, 0.032, 0.028, 0.025, 0.022, 0.02, 0.018, 0.016, 0.014, 0.013, 0.012, 0.01, 0.009, 0.008, 0.007, 0.005, 0.004]
#BWG_SI = [round(i*.0254,6) for i in BWG_inch]
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
HTRI_Ls = [round(i*0.3048, 3) for i in HTRI_Ls_imperial]


# Shells up to 120 inch in diameter.
# This is for plate shells, not pipe (up to 12 inches, pipe is used)
HEDH_shells_imperial = [12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 24., 26., 28., 30., 32., 34., 36., 38., 40., 42., 44., 46., 48., 50., 52., 54., 56., 58., 60., 63., 66., 69., 72., 75., 78., 81., 84., 87., 90., 93., 96., 99., 102., 105., 108., 111., 114., 117., 120.]
HEDH_shells = [round(i*0.0254, 6) for i in HEDH_shells_imperial]


HEDH_pitches = {0.25: (1.25, 1.5), 0.375: (1.330, 1.420),
0.5: (1.250, 1.310, 1.380), 0.625: (1.250, 1.300, 1.400),
0.75: (1.250, 1.330, 1.420, 1.500), 1.: (1.250, 1.312, 1.375),
1.25: (1.250), 1.5: (1.250), 2.: (1.250)}

def DBundle_min(Do):
    r'''Very roughly, determines a good choice of shell diameter for a given
    tube outer diameter, according to figure 1, section 3.3.5 in [1]_.

    Inputs
    ------
    Do : float
        Tube outer diameter, [m]

    Returns
    -------
    DShell : float
        Shell inner diameter, optional, [m]

    Notes
    -----
    This function should be used if a tube diameter is specified but not a
    shell size. DShell will have to be adjusted later, once the ara requirement
    is known.

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
    if Do <= 0.006:
        DBundle = 0.1
    elif Do <= 0.01:
        DBundle = 0.1
    elif Do <= 0.014:
        DBundle = 0.3
    elif Do <= 0.02:
        DBundle = 0.5
    elif Do <= 0.03:
        DBundle = 1.
    else:
        DBundle = 1.5
    return DBundle


def shell_clearance(DBundle=None, DShell=None):
    r'''Determines the clearance between a shell and tube bundle in a TEMA HX
    [1].

    Inputs
    ------
        DShell : float
            Shell inner diameter, optional, [m]
        DBundle : float
            Outer diameter of tube bundle, optional, [m]

    Returns
    -------
        c : float
            Shell-tube bundle clearance, [m]

    Notes
    -----
    Lower limits are extended up to the next limit where intermediate limits
    are not provided. Only one of shell diameter or bundle are required.

    Examples
    --------
    >>> shell_clearance(DBundle=1.245)
    0.0064

    References
    ----------
    .. [1] Standards of the Tubular Exchanger Manufacturers Association,
       Ningth ednition, 2007, TEMA, New York.
    '''
    if DShell:
        if DShell< 0.457:
            c = 0.0032
        elif DShell < 1.016:
            c = 0.0048
        elif DShell < 1.397:
            c = 0.0064
        elif DShell < 1.778:
            c = 0.0079
        elif DShell < 2.159:
            c = 0.0095
        else:
            c = 0.011

    elif DBundle:
        if DBundle < 0.457 - 0.0048:
            c = 0.0032
        elif DBundle < 1.016 - 0.0064:
            c = 0.0048
        elif DBundle < 1.397 - 0.0079:
            c = 0.0064
        elif DBundle < 1.778 - 0.0095:
            c = 0.0079
        elif DBundle <2.159 - 0.011:
            c = 0.0095
        else:
            c = 0.011
    else:
        raise Exception('DShell or DBundle must be specified')
    return c


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

def baffle_thickness(Dshell=None, L_unsupported=None, service='C'):
    r'''Determines the thickness of baffles and support plates in TEMA HX
    [1]_. Does not apply to longitudinal baffles.

    Parameters
    ----------
    Dshell : float
        Shell inner diameter, [m]
    L_unsupported: float
        Distance between tube supports, [m]
    service: str
        Service type, C, R or B, [-]

    Returns
    -------
    t : float
        Baffle or support plate thickness, [m]

    Notes
    -----
    No checks are provided to ensure sizes are TEMA compatible.
    As pressure concerns are not relevant, these are simple.
    Mandatory sizes. Uses specified limits in mm.

    Examples
    --------
    >>> baffle_thickness(Dshell=.3, L_unsupported=50, service='R')
    0.0095

    References
    ----------
    .. [1] Standards of the Tubular Exchanger Manufacturers Association,
       Ningth ednition, 2007, TEMA, New York.
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



def D_baffle_holes(do=None, L_unsupported=None):
    r'''Determines the diameter of holes in baffles for tubes according to
    TEMA [1]_. Applies for all geometries.

    Parameters
    ----------
    do : float
        Tube outer diameter, [m]
    L_unsupported: float
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
       Ningth ednition, 2007, TEMA, New York.
    '''
    if do > 0.0318 or L_unsupported <= 0.914: # 1-1/4 inches and 36 inches
        extra = 0.0008
    else:
        extra = 0.0004
    d = do + extra
    return d


_L_unsupported_steel = [0.66, 0.889, 1.118, 1.321, 1.524, 1.753, 1.88, 2.235, 2.54, 3.175, 3.175, 3.175]
_L_unsupported_aluminium = [0.559, 0.762, 0.965, 1.143, 1.321, 1.524, 1.626, 1.93, 2.21, 2.794, 2.794, 2.794]
_L_unsupported_lengths = [0.25, 0.375, 0.5, 0.628, 0.75, 0.875, 1., 1.25, 1.5, 2., 2.5, 3.]

def L_unsupported_max(NPS=None, material='CS'):
    r'''Determines the maximum length of unsupported tube acording to
    TEMA [1]_. Temperature limits are ignored.

    Inputs
    ------
    NPS : float
        Nominal pipe size, [in]
    material: str
        Material type, CS or other for the other list

    Returns
    -------
    L_unsupported : float
        Maximum length of unsupported tube, [m]

    Notes
    -----
    Interpolation of available sizes is probably possible.

    Examples
    --------
    >>> L_unsupported_max(NPS=1.5, material='CS')
    2.54

    References
    ----------
    .. [1] Standards of the Tubular Exchanger Manufacturers Association,
       Ningth ednition, 2007, TEMA, New York.
    '''
    if NPS in _L_unsupported_lengths:
        i = _L_unsupported_lengths.index(NPS)
    else:
        raise Exception('Tube size not in list length unsupported list')
    if material == 'CS':
        L = _L_unsupported_steel[i]
    else:
        L = _L_unsupported_aluminium[i]
    return L


### Tube bundle count functions

def Ntubes_Perrys(DBundle=None, do=None, Ntp=None, angle=30):
    r'''A rough equation presented in Perry's Handbook [1]_ for estimating
    the number of tubes in a tube bundle of differing geometries and tube
    sizes. Claimed accuracy of 24 tubes.

    Parameters
    ----------
    DBundle : float
        Outer diameter of tube bundle, [m]
    Ntp : float
        Number of tube passes, [-]
    do : float
        Tube outer diameter, [m]
    angle : float
        The angle the tubes are positioned; 30, 45, 60 or 90

    Returns
    -------
    Nt : float
        Number of tubes, [-]

    Notes
    -----
    Perry's equation 11-74.
    Pitch equal to 1.25 times the tube outside diameter
    No other source for this equation is given.
    Experience suggests this is accurate to 40 tubes, but is often around 20 tubes off.

    Examples
    --------
    >>> [[Ntubes_Perrys(DBundle=1.184, Ntp=i, do=.028, angle=j) for i in [1,2,4,6]] for j in [30, 45, 60, 90]]
    [[1001, 973, 914, 886], [819, 803, 784, 769], [1001, 973, 914, 886], [819, 803, 784, 769]]

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. New York: McGraw-Hill Education, 2007.
    '''
    if angle == 30 or angle == 60:
        C = 0.75*DBundle/do - 36.
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
        C = DBundle/do - 36.
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
    Nt = int(Nt)
    return Nt




def Ntubes_VDI(DBundle=None, Ntp=None, do=None, pitch=None, angle=30.):
    r'''A rough equation presented in the VDI Heat Atlas for estimating
    the number of tubes in a tube bundle of differing geometries and tube
    sizes. No accuracy estimation given.

    Parameters
    ----------
    DBundle : float
        Outer diameter of tube bundle, [m]
    Ntp : float
        Number of tube passes, [-]
    do : float
        Tube outer diameter, [m]
    pitch : float
        Pitch; distance between two orthogonal tube centers, [m]
    angle : float
        The angle the tubes are positioned; 30, 45, 60 or 90

    Returns
    -------
    Nt : float
        Number of tubes, [-]

    Notes
    -----
    6 tube passes is not officially supported, only 1, 2, 4 and 8.
    However, an estimated constant has been added to support it.
    f2 = 90.
    This equation is a rearranged for of that presented in [1]_.
    Calculated tube count is rounded down to an integer.

    Examples
    --------
    >>> [[Ntubes_VDI(DBundle=1.184, Ntp=i, do=.028, pitch=.036, angle=j) for i in [1,2,4,6,8]] for j in [30, 45, 60, 90]]
    [[983, 966, 929, 914, 903], [832, 818, 790, 778, 769], [983, 966, 929, 914, 903], [832, 818, 790, 778, 769]]

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

    DBundle, do, pitch = DBundle*1000, do*1000, pitch*1000 # convert to mm, equation is dimensional.
    t = pitch
    Ntubes = (-(-4*f1*t**4*f2**2*do + 4*f1*t**4*f2**2*DBundle**2 + t**4*f2**4)**0.5
    - 2*f1*t**2*do + 2*f1*t**2*DBundle**2 + t**2*f2**2) / (2*f1**2*t**4)
    Ntubes = int(Ntubes)
    return Ntubes

#print [[Ntubes_VDI(DBundle=1.184, Ntp=i, do=.028, pitch=.036, angle=j) for i in [1,2,4,6,8]] for j in [30, 45, 60, 90]]

#print [[Ntubes_VDI(DBundle=1.184, Ntp=i, do=.028, pitch=.036, angle=j) for i in [1,2,4,8]] for j in [30, 45, 60, 90]]
#    >>> [Ntubes_Phadkeb(DBundle=1.200-.008*2, do=.028, pitch=.036, Ntp=i, angle=45.) for i in [1,2,4,6,8]]
#    [805, 782, 760, 698, 680]
#








_triangular_Ns = [0, 1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25, 27, 28, 31, 36, 37, 39, 43, 48, 49, 52, 57, 61, 63, 64, 67, 73, 75, 76, 79, 81, 84, 91, 93, 97, 100, 103, 108, 109, 111, 112, 117, 121, 124, 127, 129, 133, 139, 144, 147, 148, 151, 156, 163, 167, 169, 171, 172, 175, 181, 183, 189, 192, 193, 196, 199, 201, 208, 211, 217, 219, 223, 225, 228, 229, 237, 241, 243, 244, 247, 252, 256, 259, 268, 271, 273, 277, 279, 283, 289, 291, 292, 300, 301, 304, 307, 309, 313, 316, 324, 325, 327, 331, 333, 336, 337, 343, 349, 351, 361, 363, 364, 367, 372, 373, 379, 381, 387, 388, 397, 399, 400, 403, 409, 412, 417, 421, 427, 432, 433, 436, 439, 441, 444, 448, 453, 457, 463, 468, 469, 471, 475, 481, 484, 487, 489, 496, 499, 507, 508, 511, 513, 516, 523, 525, 529, 532, 541, 543, 547, 549, 553, 556, 559, 567, 571, 576, 577, 579, 588, 589, 592, 597, 601, 603, 604, 607, 613, 619, 624, 625, 628, 631, 633, 637, 643, 651, 652, 657, 661, 669, 673, 675, 676, 679, 684, 687, 688, 691, 700, 703, 709, 711, 721, 723, 724, 727, 729, 732, 733, 739, 741, 751, 756, 757, 763, 768, 769, 772, 775, 777, 784, 787, 793, 796, 804, 811, 813, 817, 819, 823, 829, 831, 832, 837, 841, 844, 847, 849, 853, 859, 867, 868, 871, 873, 876, 877, 883, 889, 892, 900, 903, 907, 912, 916, 919, 921, 925, 927, 931, 937, 939, 948, 949, 961, 964, 967, 972, 973, 975, 976, 981, 988, 991, 993, 997, 999]
_triangular_C1s = [1, 7, 13, 19, 31, 37, 43, 55, 61, 73, 85, 91, 97, 109, 121, 127, 139, 151, 163, 169, 187, 199, 211, 223, 235, 241, 253, 265, 271, 283, 295, 301, 313, 337, 349, 361, 367, 379, 385, 397, 409, 421, 433, 439, 451, 463, 475, 499, 511, 517, 535, 547, 559, 571, 583, 595, 613, 625, 637, 649, 661, 673, 685, 691, 703, 721, 733, 745, 757, 769, 793, 805, 817, 823, 835, 847, 859, 871, 877, 889, 913, 925, 931, 955, 967, 979, 1003, 1015, 1027, 1039, 1045, 1057, 1069, 1075, 1099, 1111, 1123, 1135, 1147, 1159, 1165, 1177, 1189, 1201, 1213, 1225, 1237, 1261, 1273, 1285, 1303, 1309, 1333, 1345, 1357, 1369, 1381, 1393, 1405, 1417, 1429, 1453, 1459, 1483, 1495, 1507, 1519, 1531, 1555, 1561, 1573, 1585, 1597, 1615, 1627, 1639, 1651, 1663, 1675, 1687, 1711, 1723, 1735, 1759, 1765, 1777, 1789, 1801, 1813, 1831, 1843, 1867, 1879, 1891, 1903, 1915, 1921, 1945, 1957, 1969, 1981, 1993, 2017, 2029, 2053, 2065, 2077, 2083, 2095, 2107, 2125, 2149, 2161, 2173, 2185, 2197, 2209, 2221, 2233, 2245, 2257, 2263, 2275, 2287, 2299, 2335, 2347, 2371, 2383, 2395, 2407, 2419, 2431, 2437, 2455, 2479, 2491, 2503, 2515, 2527, 2539, 2563, 2575, 2587, 2611, 2623, 2635, 2647, 2653, 2665, 2677, 2689, 2713, 2725, 2737, 2749, 2773, 2779, 2791, 2803, 2815, 2839, 2857, 2869, 2893, 2905, 2917, 2929, 2941, 2965, 2989, 3001, 3013, 3025, 3037, 3049, 3055, 3067, 3079, 3091, 3103, 3115, 3121, 3145, 3169, 3181, 3193, 3205, 3217, 3241, 3253, 3259, 3283, 3295, 3307, 3319, 3331, 3343, 3355, 3367, 3403, 3415, 3427, 3439, 3463, 3481, 3493, 3505, 3511, 3535, 3547, 3559, 3571, 3595, 3607, 3619, 3631, 3643]
Phadkeb_triangular = interp1d(_triangular_Ns, _triangular_C1s, copy=False, kind='zero')
_square_Ns = [0, 1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 26, 29, 32, 34, 36, 37, 40, 41, 45, 49, 50, 52, 53, 58, 61, 64, 65, 68, 72, 73, 74, 80, 81, 82, 85, 89, 90, 97, 98, 100, 101, 104, 106, 109, 113, 116, 117, 121, 122, 125, 128, 130, 136, 137, 144, 145, 146, 148, 149, 153, 157, 160, 162, 164, 169, 170, 173, 178, 180, 181, 185, 193, 194, 196, 197, 200, 202, 205, 208, 212, 218, 221, 225, 226, 229, 232, 233, 234, 241, 242, 244, 245, 250, 256, 257, 260, 261, 265, 269, 272, 274, 277, 281, 288, 289, 290, 292, 293, 296, 298, 305, 306, 313, 314, 317, 320, 324, 325, 328, 333, 337, 338, 340, 346, 349, 353, 356, 360, 361, 362, 365, 369, 370, 373, 377, 386, 388, 389, 392, 394, 397, 400, 401, 404, 405, 409, 410, 416, 421, 424, 425, 433, 436, 441, 442, 445, 449, 450, 452, 457, 458, 461, 464, 466, 468, 477, 481, 482, 484, 485, 488, 490, 493, 500, 505, 509, 512, 514, 520, 521, 522, 529, 530, 533, 538, 541, 544, 545, 548, 549, 554, 557, 562, 565, 569, 576, 577, 578, 580, 584, 585, 586, 592, 593, 596, 601, 605, 610, 612, 613, 617, 625, 626, 628, 629, 634, 637, 640, 641, 648, 650, 653, 656, 657, 661, 666, 673, 674, 676, 677, 680, 685, 689, 692, 697, 698, 701, 706, 709, 712, 720, 722, 724, 725, 729, 730, 733, 738, 740, 745, 746, 754, 757, 761, 765, 769, 772, 773, 776, 778, 784, 785, 788, 793, 794, 797, 800, 801, 802, 808, 809, 810, 818, 820, 821, 829, 832, 833, 841, 842, 845, 848, 850, 853, 857, 865, 866, 872, 873, 877, 881, 882, 884, 890, 898, 900, 901, 904, 905, 909, 914, 916, 922, 925, 928, 929, 932, 936, 937, 941, 949, 953, 954, 961, 962, 964, 965, 968, 970, 976, 977, 980, 981, 985, 986, 997, 1000]
_square_C1s = [1, 5, 9, 13, 21, 25, 29, 37, 45, 49, 57, 61, 69, 81, 89, 97, 101, 109, 113, 121, 129, 137, 145, 149, 161, 169, 177, 185, 193, 197, 213, 221, 225, 233, 241, 249, 253, 261, 277, 285, 293, 301, 305, 317, 325, 333, 341, 349, 357, 365, 373, 377, 385, 401, 405, 421, 429, 437, 441, 457, 465, 473, 481, 489, 497, 506, 509, 517, 529, 545, 553, 561, 569, 577, 593, 601, 609, 613, 621, 633, 641, 657, 665, 673, 681, 697, 709, 717, 725, 733, 741, 749, 757, 761, 769, 777, 793, 797, 805, 821, 829, 845, 853, 861, 869, 877, 885, 889, 901, 917, 925, 933, 941, 949, 965, 973, 981, 989, 997, 1005, 1009, 1033, 1041, 1049, 1057, 1069, 1085, 1093, 1101, 1109, 1117, 1125, 1129, 1137, 1153, 1161, 1177, 1185, 1201, 1209, 1217, 1225, 1229, 1237, 1245, 1257, 1265, 1273, 1281, 1289, 1305, 1313, 1321, 1329, 1353, 1361, 1369, 1373, 1389, 1405, 1413, 1425, 1433, 1441, 1449, 1457, 1465, 1473, 1481, 1489, 1505, 1513, 1517, 1533, 1541, 1549, 1565, 1581, 1597, 1605, 1609, 1617, 1633, 1641, 1649, 1653, 1669, 1685, 1693, 1701, 1709, 1725, 1733, 1741, 1749, 1757, 1765, 1781, 1789, 1793, 1801, 1813, 1829, 1837, 1853, 1861, 1869, 1877, 1885, 1893, 1901, 1917, 1925, 1933, 1941, 1961, 1969, 1977, 1993, 2001, 2009, 2017, 2025, 2029, 2053, 2061, 2069, 2077, 2085, 2093, 2101, 2109, 2121, 2129, 2145, 2161, 2177, 2185, 2201, 2209, 2217, 2225, 2233, 2241, 2249, 2253, 2261, 2285, 2289, 2305, 2313, 2321, 2337, 2353, 2361, 2377, 2385, 2393, 2409, 2417, 2425, 2433, 2441, 2449, 2453, 2469, 2477, 2493, 2501, 2509, 2521, 2529, 2537, 2545, 2553, 2561, 2569, 2585, 2593, 2601, 2609, 2617, 2629, 2637, 2661, 2669, 2693, 2701, 2709, 2725, 2733, 2741, 2749, 2757, 2765, 2769, 2785, 2801, 2809, 2812, 2837, 2845, 2861, 2869, 2877, 2885, 2893, 2917, 2925, 2933, 2941, 2949, 2957, 2965, 2981, 2989, 2997, 3001, 3017, 3025, 3041, 3045, 3061, 3069, 3077, 3085, 3093, 3109, 3125, 3133, 3149]
Phadkeb_square = interp1d(_square_Ns, _square_C1s, copy=False, kind='zero')


def Ntubes_Phadkeb(DBundle=None, Ntp=None, do=None, pitch=None, angle=30):
    r'''Using tabulated values and correction factors for number of passes,
    the highly accurate method of [1]_ is used to obtain the tube count
    of a given tube bundle outer diameter for a given tube size and pitch.

    Parameters
    ----------
    DBundle : float
        Outer diameter of tube bundle, [m]
    Ntp : float
        Number of tube passes, [-]
    do : float
        Tube outer diameter, [m]
    pitch : float
        Pitch; distance between two orthogonal tube centers, [m]
    angle : float
        The angle the tubes are positioned; 30, 45, 60 or 90

    Returns
    -------
        Nt : float
            Number of tubes, [-]

    Notes
    -----
    This function will fail when there are more than several thousand tubes.
    This is due to a limitation in tabulated values presented in [1]_.

    Examples
    --------
    >>> [Ntubes_Phadkeb(DBundle=1.200-.008*2, do=.028, pitch=.036, Ntp=i, angle=45.) for i in [1,2,4,6,8]]
    [805, 782, 760, 698, 680]
    >>> [Ntubes_Phadkeb(DBundle=1.200-.008*2, do=.028, pitch=.035, Ntp=i, angle=45.) for i in [1,2,4,6,8]]
    [861, 838, 816, 750, 732]

    References
    ----------
    .. [1] Phadke, P. S., Determining tube counts for shell and tube
       exchangers, Chem. Eng., September, 91, 65-68 (1984).
    '''
    if Ntp == 6:
        e = 0.265
    elif Ntp == 8:
        e = 0.404
    else:
        e = 0

    r = 0.5*(DBundle-do)/pitch
    s = r**2
    Ns, Nr = floor(s), floor(r)
    if angle == 30 or angle == 60:
        C1 = Phadkeb_triangular(Ns)
    elif angle == 45 or angle == 90:
        C1 = Phadkeb_square(Ns)

    Cx = 2*Nr + 1

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
            if Nv %2 == 0:
                z = (s-u**2)**0.5
            else:
                z = (s-u**2)**0.5 - 0.5
            Nz = floor(z)
            if Ntp == 6:
                C6 = C1 - Cy - 4*Nz - 1
            else:
                C8 = C4 - 4*Nz
        else: # rotated triangular
            v = 2*e*r
            Nv = floor(v)
            u1 = 0.5*Nv
            z = (s-u1**2)**0.5
            w1 = 2**2**0.5
            u2 = 0.5*(Nv + 1)
            zs = (s-u2**2)**0.5
            w2 = 2*zs/3**0.5
            if Nv%2 == 0:
                z1 = 0.5*w1
                z2 = 0.5*(w2+1)
            else:
                z1 = 0.5*(w1+1)
                z2 = 0.5*w2
            Nz1 = floor(z1)
            Nz2 = floor(z2)
            if Ntp == 6:
                C6 = C1 - Cx - 4*(Nz1 + Nz2)
            else: # 8
                C8 = C4-4*(Nz1 + Nz2)

    if (angle == 45 or angle == 90):
        if angle == 90:
            Cy = Cx - 1
            # eq 6 or 8 for c2 or c4
            if Ntp == 2:
                C2 = C1 - Cx
            else: # 4 passes, or 8; this value is needed
                C4 = C1 - Cx - Cy
        else: # rotated square
            w = r/2**0.5
            Nw = floor(w)
            Cx = 2*Nw + 1
            Cy = Cx - 1
            if Ntp == 2:
                C2 = C1 - Cx
            else: # 4 passes, or 8; this value is needed
                C4 = C1 - Cx - Cy

    if (angle == 45 or angle == 90) and (Ntp == 6 or Ntp == 8):
        if angle == 90:
            v = e*r + 0.5
            Nv = floor(v)
            z = (s - Nv**2)**0.5
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
            z = (s-u1**2)**0.5
            w1 = 2**0.5*z
            u2 = (Nv + 1)/2**0.5
            zs = (s-u2**2)**0.5
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
                C8 = C4-4*(Nz1 + Nz2)


    if Ntp == 1:
        return int(C1)
    elif Ntp == 2:
        return int(C2)
    elif Ntp == 4:
        return int(C4)
    elif Ntp == 6:
        return int(C6)
    else:
        return int(C8)


#print Ntubes_Phadkeb(Dshell=1.00, do=.0135, pitch=.025, Ntp=1, angle=30.), 'good'
#print [Ntubes_Phadkeb(Dshell=1.200-.008*2, do=.028, pitch=.035, Ntp=i, angle=90.) for i in [1,2,4,6,8]]
#print [Ntubes_Phadkeb(DBundle=1.200-.008*2, do=.028, pitch=.036, Ntp=i, angle=45.) for i in [1,2,4,6,8]]



#print [[Ntubes_Phadkeb(DBundle=1.200-.008*2, do=.028, pitch=.028*1.25, Ntp=i, angle=j) for i in [1,2,4,6,8]] for j in [30, 45, 60, 90]]


def Ntubes_HEDH(DBundle=None, do=None, pitch=None, angle=30):
    r'''A rough equation presented in the HEDH for estimating
    the number of tubes in a tube bundle of differing geometries and tube
    sizes. No accuracy estimation given. Only 1 pass is supported.

    Parameters
    ----------
    DBundle : float
        Outer diameter of tube bundle, [m]
    do : float
        Tube outer diameter, [m]
    pitch : float
        Pitch; distance between two orthogonal tube centers, [m]
    angle : float
        The angle the tubes are positioned; 30, 45, 60 or 90

    Returns
    -------
    Nt : float
        Number of tubes, [-]

    Notes
    -----
    Seems highly accurate.

    Examples
    --------
    >>> [Ntubes_HEDH(DBundle=1.200-.008*2, do=.028, pitch=.036, angle=i) for i in [30, 45, 60, 90]]
    [928, 804, 928, 804]

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
    Dctl = DBundle-do
    Nt = 0.78*Dctl**2/C1/pitch**2
    Nt = int(Nt)
    return Nt


def Ntubes(DBundle=None, Ntp=1, do=None, pitch=None, angle=30, pitch_ratio=1.25, AvailableMethods=False, Method=None):
    '''Function to calculate the number of tubes which can fit in a given tube
    bundle outer diameter.

    >>> Ntubes(DBundle=1.2, do=0.025)
    1285
    '''
    def list_methods():
        methods = []
        methods.append('Phadkeb')
        if Ntp == 1:
            methods.append('HEDH')
        if Ntp == 1 or Ntp == 2 or Ntp == 4 or Ntp == 8:
            methods.append('VDI')
        if Ntp == 1 or Ntp == 2 or Ntp == 4 or Ntp == 6: # Also restricted to 1.25 pitch ratio but not hard coded
            methods.append('Perry\'s')
        methods.append('None')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if pitch_ratio and not pitch:
        pitch = pitch_ratio*do
    if Method == 'Phadkeb':
        N = Ntubes_Phadkeb(DBundle=DBundle, Ntp=Ntp, do=do, pitch=pitch, angle=angle)
    elif Method == 'HEDH':
        N = Ntubes_HEDH(DBundle=DBundle, do=do, pitch=pitch, angle=angle)
    elif Method == 'VDI':
        N = Ntubes_VDI(DBundle=DBundle, Ntp=Ntp, do=do, pitch=pitch, angle=angle)
    elif Method == 'Perry\'s':
        N = Ntubes_Perrys(DBundle=DBundle, do=do, Ntp=Ntp, angle=angle)
    elif Method == 'None':
        return None
    else:
        raise Exception('Failure in in function')
    return N


def D_for_Ntubes_VDI(Nt=None, Ntp=None, do=None, pitch=None, angle=30):
    r'''A rough equation presented in the VDI Heat Atlas for estimating
    the size of a tube bundle from a given number of tubes, number of tube
    passes, outer tube diameter, pitch, and arrangement.
    No accuracy estimation given.

    .. math::
        OTL = \sqrt{f_1 z t^2 + f_2 t \sqrt{z} - d_o}

    Parameters
    ----------
    Nt : float
        Number of tubes, [-]
    Ntp : float
        Number of tube passes, [-]
    do : float
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
    >>> D_for_Ntubes_VDI(Nt=970, Ntp=2., do=0.00735, pitch=0.015, angle=30.)
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
    do, pitch = do*1000, pitch*1000 # convert to mm, equation is dimensional.
    Dshell = (f1*Nt*pitch**2 + f2*Nt**0.5*pitch +do)**0.5
    Dshell = Dshell/1000.
    return Dshell


_heads = {'A': 'Removable Channel and Cover', 'B': 'Bonnet (Integral Cover)', 'C': 'Integral With Tubesheet Removable Cover', 'N': 'Channel Integral With Tubesheet and Removable Cover', 'D': 'Special High-Pressure Closures'}
_shells = {'E': 'One-Pass Shell', 'F': 'Two-Pass Shell with Longitudinal Baffle', 'G': 'Split Flow', 'H': 'Double Split Flow', 'J': 'Divided Flow', 'K': 'Kettle-Type Reboiler',  'X': 'Cross Flow'}
_rears = {'L': 'Fixed Tube Sheet; Like "A" Stationary Head', 'M': 'Fixed Tube Sheet; Like "B" Stationary Head', 'N': 'Fixed Tube Sheet; Like "C" Stationary Head', 'P': 'Outside Packed Floating Head', 'S': 'Floating Head with Backing Device', 'T': 'Pull-Through Floating Head', 'U': 'U-Tube Bundle', 'W': 'Externally Sealed Floating Tubesheet'}
_services = {'B': 'Chemical', 'R': 'Refinery', 'C': 'General'}
_baffle_types = ['segmental', 'double segmental', 'triple segmental', 'disk and doughnut', 'no tubes in window', 'orifice', 'rod']
