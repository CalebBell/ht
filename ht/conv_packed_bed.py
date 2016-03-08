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

__all__ = ['Nu_packed_bed_Gnielinski']

def Nu_packed_bed_Gnielinski(dp, voidage, vs, rho, mu, Pr, fa=None):
    r'''Calculates Nusselt number of a fluid passing over a bed of particles
    using a correlation shown in [3]_ and cited as from [1]_ and [2]_. Likely
    the best available model as the author of [1]_ is the sams as [2]_ and
    [3]_.

    .. math::
        Nu = f_a Nu_{sphere}

        Nu_{sphere} = 2 + \sqrt{Nu_{m,lam}^2 + Nu_{m,turb}^2}

        Nu_{m,lam} = 0.664Re^{0.5} Pr^{1/3}

        Nu_{m,turb} = \frac{0.037Re^{0.8} Pr}{1 + 2.443Re^{-0.1}(Pr^{2/3} -1)}

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
        Viscosity of the fluid, [Pa*S]
    Pr : float
        Prandtl number of the fluid []
    fa : float, optional
        Fator increasing heat transfer []

    Returns
    -------
    Nu : float
        Nusselt number for heat transfer to the packed bed [Pa]

    Notes
    -----
    `fa` is a factor relating how much more heat transfer happens than would
    normally, around one sphere. For spheres of the same size,
    :math:`f_a = 1 + 1.5(1-\epsilon)`. For cylinders with l/d ratio of
    0.24 < l/d < 1.2 use fa = 1.6. For cubes, use fa = 1.6 For Raschig rings,
    use `fa` = 2.1 For Berl saddles, use `fa` = 2.3. fa is calculated with
    the relationship for spheres if not provided.

    Confirmed with experiemental data for a range of :math:`1E-1 < Re <1,000`
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
    if not fa:
        fa = 1 + 1.5*(1-voidage)
    Nu = fa*Nu_sphere
    return Nu

