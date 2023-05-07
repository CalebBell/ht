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

import os
from math import e, exp

from fluids.constants import c, h, k, sigma
from fluids.numerics import numpy as np

__all__ = ['blackbody_spectral_radiance', 'q_rad', 'grey_transmittance',
           'solar_spectrum']


def blackbody_spectral_radiance(T, wavelength):
    r'''Returns the spectral radiance, in units of W/m^2/sr/µm.

    .. math::
        I_{\lambda,blackbody,e}(\lambda,T)=\frac{2hc_o^2}
        {\lambda^5[\exp(hc_o/\lambda k T)-1]}

    Parameters
    ----------
    T : float
        Temperature of the surface, [K]
    wavelength : float
        Length of the wave to be considered, [m]

    Returns
    -------
    I : float
        Spectral radiance [W/(m^2*sr*m)]

    Notes
    -----
    Can be used to derive the Stefan-Boltzman law, or determine the maximum
    radiant frequency for a given temperature.

    Examples
    --------
    Checked with Spectral-calc.com, at [2]_.

    >>> blackbody_spectral_radiance(800., 4E-6)
    1311694129.7430933

    Calculation of power from the sun (earth occupies 6.8E-5 steradian of the
    sun):

    >>> from scipy.integrate import quad
    >>> rad = lambda l: blackbody_spectral_radiance(5778., l)*6.8E-5
    >>> quad(rad, 1E-10, 1E-4)[0]
    1367.9827067638964

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    .. [2] Spectral-calc.com. Blackbody Calculator, 2015.
       http://www.spectralcalc.com/blackbody_calculator/blackbody.php
    '''
    to_exp = h*c/(wavelength*T*k)
    if to_exp > 709.7:
        return 0.0
    else:
        exp_term = exp(to_exp)
    return 2.*h*c*c*wavelength**-5/(exp_term - 1.0)


def q_rad(emissivity, T, T2=0):
    r'''Returns the radiant heat flux of a surface, optionally including
    assuming radiant heat transfer back to the surface.

    .. math::
        q = \epsilon \sigma (T_1^4 - T_2^4)

    Parameters
    ----------
    emissivity : float
        Fraction of black-body radiation which is emitted, [-]
    T : float
        Temperature of the surface, [K]
    T2 : float, optional
        Temperature of the surrounding material of the surface [K]

    Returns
    -------
    q : float
        Heat exchange [W/m^2]

    Notes
    -----
    Emissivity must be less than 1. T2 may be larger than T.

    Examples
    --------
    >>> q_rad(emissivity=1, T=400)
    1451.613952

    >>> q_rad(.85, T=400, T2=305.)
    816.7821722650002

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    T_T = T*T
    T2_T2 = T2*T2
    return sigma*emissivity*(T_T*T_T - T2_T2*T2_T2)


def grey_transmittance(extinction_coefficient, molar_density, length, base=e):
    r'''Calculates the transmittance of a grey body, given the extinction
    coefficient of the material, its molar density, and the path length of the
    radiation.

    .. math::
        \tau = base^{(-\epsilon \cdot l\cdot \rho_m )}

    Parameters
    ----------
    extinction_coefficient : float
        The extinction coefficient of the material the radiation is passing at
        the modeled frequency, [m^2/mol]
    molar_density : float
        The molar density of the material the radiation is passing through,
        [mol/m^3]
    length : float
        The length of the body the radiation is transmitted through, [m]
    base : float, optional
        The exponent used in calculations; `e` is more theoretically sound but
        10 is often used as a base by chemists, [-]

    Returns
    -------
    transmittance : float
        The fraction of spectral radiance which is transmitted through a grey
        body (can be liquid, gas, or even solid ex. in the case of glasses) [-]

    Notes
    -----
    For extinction coefficients, see the HITRAN database. They are temperature
    and pressure dependent for each chemical and phase.

    Examples
    --------
    Overall transmission loss through 1 cm of precipitable water equivalent
    atmospheric water vapor at a frequency of 1.3 um [2]_:

    >>> grey_transmittance(3.8e-4, molar_density=55300, length=1e-2)
    0.8104707721191062

    References
    ----------
    .. [1] Modest, Michael F. Radiative Heat Transfer, Third Edition. 3rd
       edition. New York: Academic Press, 2013.
    .. [2] Eldridge, Ralph G. "Water Vapor Absorption of Visible and Near
       Infrared Radiation." Applied Optics 6, no. 4 (April 1, 1967): 709-13.
       https://doi.org/10.1364/AO.6.000709.
    '''
    transmittance = molar_density*extinction_coefficient*length
    return base**(-transmittance)


def solar_spectrum(model='SOLAR-ISS'):
    r'''Returns the solar spectrum of the sun according to the specified model.
    Only the 'SOLAR-ISS' model is supported.

    Parameters
    ----------
    model : str, optional
        The model to use; 'SOLAR-ISS' is the only model available, [-]

    Returns
    -------
    wavelengths : ndarray
        The wavelengths of the solar spectra, [m]
    SSI : ndarray
        The solar spectral irradiance of the sun, [W/(m^2*m)]
    uncertainties : ndarray
        The estimated absolute uncertainty of the measured spectral irradiance
        of the sun, [W/(m^2*m)]

    Notes
    -----
    The power of the sun changes as the earth gets closer or further away.

    In [1]_, the UV and VIS data come from observations in 2008; the IR comes
    from measurements made from 2010-2016. There is a further 28 W/m^2 for the
    3 micrometer to 160 micrometer range, not included in this model. All data
    was corrected to a standard distance of one astronomical unit from the Sun,
    as is the resultant spectrum.

    The variation of the spectrum as a function of distance from the sun should
    alter only the absolute magnitudes.

    [2]_ contains another dataset.

    99.9% of the time this function takes is to read in the solar data from
    disk. This could be reduced by using pandas.

    Examples
    --------
    >>> wavelengths, SSI, uncertainties = solar_spectrum()

    Calculate the minimum and maximum values of the wavelengths (0.5 nm/3000nm)
    and SSI:

    >>> min(wavelengths), max(wavelengths), min(SSI), max(SSI)
    (5e-10, 2.9999e-06, 1330.0, 2256817820.0)

    Integration - calculate the solar constant, in untis of W/m^2 hitting
    earth's atmosphere.

    >>> import numpy as np
    >>> np.trapz(SSI, wavelengths)
    1344.802978

    References
    ----------
    .. [1] Meftah, M., L. Damé, D. Bolsée, A. Hauchecorne, N. Pereira, D.
       Sluse, G. Cessateur, et al. "SOLAR-ISS: A New Reference Spectrum Based
       on SOLAR/SOLSPEC Observations." Astronomy & Astrophysics 611 (March 1,
       2018): A1. https://doi.org/10.1051/0004-6361/201731316.
    .. [2] Woods Thomas N., Chamberlin Phillip C., Harder Jerald W., Hock
       Rachel A., Snow Martin, Eparvier Francis G., Fontenla Juan, McClintock
       William E., and Richard Erik C. "Solar Irradiance Reference Spectra
       (SIRS) for the 2008 Whole Heliosphere Interval (WHI)." Geophysical
       Research Letters 36, no. 1 (January 1, 2009).
       https://doi.org/10.1029/2008GL036373.
    '''
    if model == 'SOLAR-ISS':
        folder = os.path.join(os.path.dirname(__file__), 'data')
        pth = os.path.join(folder, 'solar_iss_2018_spectrum.dat')
        data = np.genfromtxt(pth, dtype=np.float64, delimiter=' ')
        wavelengths, SSI, uncertainties = data[:, 0], data[:, 1], data[:, 2]

        wavelengths *= 1E-9
        SSI *= 1E9

        # Convert -1 uncertainties to nans
        uncertainties[uncertainties == -1] = np.nan

        uncertainties *= 1E9
    return wavelengths, SSI, uncertainties

