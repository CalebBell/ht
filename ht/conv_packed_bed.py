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


__all__ = ['Nu_packed_bed_Gnielinski', 'Nu_Wakao_Kagei', 'Nu_Achenbach',
           'Nu_KTA']

def Nu_packed_bed_Gnielinski(dp, voidage, vs, rho, mu, Pr, fa=None):
    r'''Calculates Nusselt number of a fluid passing over a bed of particles
    using a correlation shown in [3]_ and cited as from [1]_ and [2]_. Likely
    the best available model as the author of [1]_ is the same as [2]_ and
    [3]_.

    .. math::
        Nu = f_a Nu_{sphere}

    .. math::
        Nu_{sphere} = 2 + \sqrt{Nu_{m,lam}^2 + Nu_{m,turb}^2}

    .. math::
        Nu_{m,lam} = 0.664Re^{0.5} Pr^{1/3}

    .. math::
        Nu_{m,turb} = \frac{0.037Re^{0.8} Pr}{1 + 2.443Re^{-0.1}(Pr^{2/3} -1)}

    .. math::
        Re = \frac{\rho v_s d_p}{\mu \epsilon}

    Parameters
    ----------
    dp : float
        Equivalent spherical particle diameter of packing [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid [m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    Pr : float
        Prandtl number of the fluid []
    fa : float, optional
        Fator increasing heat transfer []

    Returns
    -------
    Nu : float
        Nusselt number for heat transfer to the packed bed [-]

    Notes
    -----
    `fa` is a factor relating how much more heat transfer happens than would
    normally, around one sphere. For spheres of the same size,
    :math:`f_a = 1 + 1.5(1-\epsilon)`. For cylinders with l/d ratio of
    0.24 < l/d < 1.2 use fa = 1.6. For cubes, use fa = 1.6 For Raschig rings,
    use `fa` = 2.1 For Berl saddles, use `fa` = 2.3. fa is calculated with
    the relationship for spheres if not provided.

    Confirmed with experimental data for a range of :math:`1E-1 < Re <1,000`
    and :math:`0.4 < Pr < 1000` for spheres. Limits are smaller for other
    shapes.

    Examples
    --------
    >>> Nu_packed_bed_Gnielinski(dp=8E-4, voidage=0.4, vs=1, rho=1E3, mu=1E-3, Pr=0.7)
    61.37823202546954

    References
    ----------
    .. [1] Gnielinski, V. (1981) "Equations for the calculation of heat and
       mass transfer during flow through stationary spherical packings at
       moderate and high Peclet numbers". International Chemical Engineering
       21 (3): 378-383
    .. [2] Gnielinski, V. (1982) "Berechnung des Warmeund Stoffaustauschs in
       durchstomten ruhenden Schuttungen". Verfahrenstechnik 16(1): 36-39
    .. [3] Gnielinski, V. in G esellschaft, V. D. I., ed. VDI Heat Atlas.
       2nd ed. 2010 edition. Berlin; New York: Springer, 2010.
    '''
    Re = rho*vs*dp/mu/voidage
    Nu_lam = 0.664*Re**0.5*Pr**(1/3.)
    Nu_turb = 0.037*Re**0.8*Pr/(1 + 2.443*Re**-0.1*(Pr**(2/3.)-1))
    Nu_sphere = 2 + (Nu_lam**2 + Nu_turb**2)**0.5
    if fa is None:
        fa = 1.0 + 1.5*(1.0 - voidage)
    return fa*Nu_sphere


def Nu_Wakao_Kagei(Re, Pr):
    r'''Calculates Nusselt number of a fluid passing over a bed of particles
    using a correlation shown in [1]_ and also cited in the review of [2]_.
    Relatively rough, as it has no dependence on voidage.

    .. math::
        Nu = 2 + 1.1Pr^{1/3}Re^{0.6}

    Parameters
    ----------
    Re : float
        Reynolds number with pebble diameter as characteristic dimension, [-]
    Pr : float
        Prandtl number of the fluid []

    Returns
    -------
    Nu : float
        Nusselt number for heat transfer to the packed bed [-]

    Notes
    -----
    Fit for Re from 3 to 3000; claimed reasonableness of fit to to 1E6.

    Examples
    --------
    >>> Nu_Wakao_Kagei(2000, 0.7)
    95.40641328041248

    References
    ----------
    .. [1] Wakao, Noriaki, and Seiichirō Kagei. Heat and Mass Transfer in
       Packed Beds. Taylor & Francis, 1982.
    .. [2] Abdulmohsin, Rahman S., and Muthanna H. Al-Dahhan. "Characteristics
       of Convective Heat Transport in a Packed Pebble-Bed Reactor." Nuclear
       Engineering and Design 284 (April 1, 2015): 143-52.
       doi:10.1016/j.nucengdes.2014.11.041.
    '''
    return 2 + 1.1*Pr**(1/3.)*Re**0.6


def Nu_Achenbach(Re, Pr, voidage):
    r'''Calculates Nusselt number of a fluid passing over a bed of particles
    using a correlation shown in [1]_ and also cited in the review of [2]_.

    .. math::
        Nu = [(1.18Re^{0.58})^4 + (0.23\left(\frac{Re}{1-\epsilon}
        \right)^{0.75})^4]^{0.25}

    Parameters
    ----------
    Re : float
        Reynolds number with pebble diameter as characteristic dimension, [-]
    Pr : float
        Prandtl number of the fluid []
    voidage : float
        Void fraction of bed packing [-]

    Returns
    -------
    Nu : float
        Nusselt number for heat transfer to the packed bed [-]

    Notes
    -----
    Claimed value for Re/ε < 7.7E5
    Developed with tests performed in a wind tunnel at conditions up to 30 bar.

    Examples
    --------
    >>> Nu_Achenbach(2000, 0.7, 0.4)
    117.70343608599121

    References
    ----------
    .. [1] Achenbach, E. "Heat and Flow Characteristics of Packed Beds."
       Experimental Thermal and Fluid Science 10, no. 1 (January 1, 1995):
       17-27. doi:10.1016/0894-1777(94)00077-L.
    .. [2] Abdulmohsin, Rahman S., and Muthanna H. Al-Dahhan. "Characteristics
       of Convective Heat Transport in a Packed Pebble-Bed Reactor." Nuclear
       Engineering and Design 284 (April 1, 2015): 143-52.
       doi:10.1016/j.nucengdes.2014.11.041.
    '''
    return ((1.18*Re**0.58)**4 + (0.23*(Re/(1-voidage))**0.75)**4)**0.25


def Nu_KTA(Re, Pr, voidage):
    r'''Calculates Nusselt number of a fluid passing over a bed of particles
    using a correlation shown in [1]_ and also cited in the review of [2]_.

    .. math::
        Nu = 1.27\frac{Pr^{1/3}}{\epsilon^{1.18}}Re^{0.36}
        + 0.033\frac{Pr^{0.5}}{\epsilon^{1.07}}Re^{0.86}

    Parameters
    ----------
    Re : float
        Reynolds number with pebble diameter as characteristic dimension, [-]
    Pr : float
        Prandtl number of the fluid [-]
    voidage : float
        Void fraction of bed packing [-]

    Returns
    -------
    Nu : float
        Nusselt number for heat transfer to the packed bed [-]

    Notes
    -----
    100 < Re < 1E5;
    0.36 < ε < 0.42;
    D/d > 20 with D as bed diameter, d as particle diameter;
    H > 4d with H as bed height.

    Examples
    --------
    >>> Nu_KTA(2000, 0.7, 0.4)
    102.08516480718129

    References
    ----------
    .. [1] Reactor Core Design of High-Temperature Gas-Cooled Reactors Part 2:
       Heat Transfer in Spherical Fuel Elements (June 1983).
       http://www.kta-gs.de/e/standards/3100/3102_2_engl_1983_06.pdf
    .. [2] Abdulmohsin, Rahman S., and Muthanna H. Al-Dahhan. "Characteristics
       of Convective Heat Transport in a Packed Pebble-Bed Reactor." Nuclear
       Engineering and Design 284 (April 1, 2015): 143-52.
       doi:10.1016/j.nucengdes.2014.11.041.
    '''
    return (1.27*Pr**(1/3.)*Re**0.36/voidage**1.18
            + 0.033*Pr**0.5/voidage**1.07*Re**0.86)
