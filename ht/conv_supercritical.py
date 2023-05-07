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

from math import log10

__all__ = ['Nu_McAdams', 'Nu_Shitsman', 'Nu_Griem', 'Nu_Jackson', 'Nu_Gupta',
           'Nu_Swenson', 'Nu_Xu', 'Nu_Mokry', 'Nu_Bringer_Smith',
           'Nu_Ornatsky', 'Nu_Gorban', 'Nu_Zhu', 'Nu_Bishop', 'Nu_Yamagata',
           'Nu_Kitoh', 'Nu_Krasnoshchekov_Protopopov', 'Nu_Petukhov',
           'Nu_Krasnoshchekov']

### Vertical upflow only

def Nu_McAdams(Re, Pr):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_.

    Found in [2]_ to fit the enhanced heat transfer regime with a MAD of 10.3%
    which was better than and of the other reviewed correlations.

    .. math::
        Nu_b = 0.0243Re_b^{0.8}Pr_b^{0.4}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties, [-]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    This has also been one of the forms of the Dittus-Boelter correlations.
    Claimed to fit data for high pressures and low heat fluxes.


    Examples
    --------
    >>> Nu_McAdams(1E5, 1.2)
    261.3838629346147

    References
    ----------
    .. [1] Mac Adams, William H. Heat Transmission. New York and London, 1942.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    '''
    return 0.0243*Re**0.8*Pr**0.4


def Nu_Shitsman(Re, Pr_b, Pr_w):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_ and
    [2] as shown in both [3]_ and [4]_.

    .. math::
        Nu_b = 0.023 Re_b^{0.8}(min(Pr_b, Pr_w))^{0.8}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr_b : float
        Prandtl number with bulk fluid properties, [-]
    Pr_w : float
        Prandtl number with wall fluid properties, [-]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    [3]_ states this correlation was developed with D = 7.8 and 8.2 mm and with
    a `Pr` approximately 1. [3]_ ranked it third in the enhanced heat transfer
    category, with a MAD as 11.5%

    [4]_ cites a [1]_ as the source of the correlation. Neither have been
    reviewed, and both are in Russian. [4]_ lists this as third most accurate
    of the 14 correlations reviewed from a database of all regimes.

    Examples
    --------
    >>> Nu_Shitsman(1E5, 1.2, 1.6)
    266.1171311047253

    References
    ----------
    .. [1] M. E Shitsman, Impairment of the heat transmission at supercritical
       pressures, High. Temperature, 1963, 1(2): 237-244
    .. [2] Miropol`skiy ZL, Shitsman ME (1957). Heat transfer to water and
       steam at variable specific heat. J Tech Phys XXVII(10): 2359-2372
    .. [3] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [4] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009):
       757-66. doi:10.1007/s00231-008-0475-4.
    '''
    return 0.023*Re**0.8*min(Pr_b, Pr_w)**0.8


def Nu_Griem(Re, Pr, H=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_,
    also shown in [2]_, [3]_ and [4]_. Has complicated rules regarding where
    properties should be evaluated.

    .. math::
        Nu_m = 0.0169Re_b^{0.8356} Pr_{sel}^{0.432}\omega

    Parameters
    ----------
    Re : float
        Reynolds number as explained below, [-]
    Pr : float
        Prandtl number as explained below, [-]
    H : float, optional
        Enthalpy of water (if the fluid is water), [J/kg]

    Returns
    -------
    Nu : float
        Nusselt number as explained below, [-]

    Notes
    -----
    w is calculated as follows, for water only, with a reference point from
    the 1967-IFC formulation. It is set to 1 if H is not provided:
    if Hb < 1.54E6 J/kg, w = 0.82; if Hb > 1.74E6 J/kg, w = 1; otherwise
    w = 0.82 + 9E-7*(Hb-1.54E6).

    To determine heat capacity to be used, Cp should be calculated at 5 points,
    and the lowest three points should be averaged.
    The five points are: Tw, (Tw+Tf)/2, Tf, (Tb+Tf)/2, Tb.

    Viscosity should be the bulk viscosity.
    Thermal conductivity should be the average of the bulk and wall values.
    Density should be the bulk density.

    [2]_ states this correlation was developed with D = 10, 14, and 20 mm,
    P from 22 to 27 MPa, G from 300 to 2500 kg/m^2/s, and q from
    200 to 700 kW/m^2. It was ranked 6th among the 14 correlations reviewed for
    enhanced heat transfer, with a MAD of 13.8%, and 6th overall for the three
    heat transfer conditions with a overall MAD of 14.8%. [3]_ ranked it 8th
    of 14 correlations for the three heat transfer conditions.

    [2]_ has an almost complete description of the model; both [3]_ and [4]_
    simplify it.

    Examples
    --------
    >>> Nu_Griem(1E5, 1.2)
    275.4818576600527

    References
    ----------
    .. [1] Griem, H. "A New Procedure for the Prediction of Forced Convection
       Heat Transfer at near- and Supercritical Pressure." Heat and Mass
       Transfer 31, no. 5 (1996): 301-5. doi:10.1007/BF02184042.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [3] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009):
       757-66. doi:10.1007/s00231-008-0475-4.
    .. [4] Jäger, Wadim, Victor Hugo Sánchez Espinoza, and Antonio Hurtado.
       "Review and Proposal for Heat Transfer Predictions at Supercritical
       Water Conditions Using Existing Correlations and Experiments." Nuclear
       Engineering and Design, (W3MDM) University of Leeds International
       Symposium: What Where When? Multi-dimensional Advances for Industrial
       Process Monitoring, 241, no. 6 (June 2011): 2184-2203.
       doi:10.1016/j.nucengdes.2011.03.022.
    '''
    if H is not None:
        if H < 1.54E6:
            w = 0.82
        elif H > 1.74E6:
            w = 1.0
        else:
            w = 0.82 + 9E-7*(H - 1.54E6)
    else:
        w = 1.0
    Nu = 0.0169*Re**0.8356*Pr**0.432*w
    return Nu


def Nu_Jackson(Re, Pr, rho_w=None, rho_b=None, Cp_avg=None, Cp_b=None, T_b=None,
               T_w=None, T_pc=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_.

    .. math::
        Nu_b = 0.0183 Re_b^{0.82} Pr^{0.5}
        \left(\frac{\rho_w}{\rho_b}\right)^{0.3}
        \left(\frac{\bar C_p}{C_{p,b}}\right)^n

    .. math::
        n = 0.4 \text{ for } T_b < T_w < T_{pc} \text{ or } 1.2T_{pc} < T_b < T_w

    .. math::
        n = 0.4 + 0.2(T_w/T_{pc} - 1)[1 - 5(T_b/T_{pc}-1)]
        \text{ for } T_{pc} < T_b < 1.2T_{pc} \text{ and } T_b < T_w

    .. math::
        n = 0.4 + 0.2(T_w/T_{pc} - 1) \text{ for } T_b < T_{pc} < T_w

    .. math::
        \bar{Cp} = \frac{H_w-H_b}{T_w-T_b}



    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties, [-]
    rho_w : float, optional
        Density at the wall temperature, [kg/m^3]
    rho_b : float, optional
        Density at the bulk temperature, [kg/m^3]
    Cp_avg : float, optional
        Average heat capacity between the wall and bulk temperatures, [J/kg/K]
    Cp_b : float, optional
        Heat capacity at the bulk temperature, [J/kg/K]
    T_b : float
        Bulk temperature, [K]
    T_w : float
        Wall temperature, [K]
    T_pc : float
        Pseudocritical temperature, i.e. temperature at P where Cp is at a
        maximum, [K]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    The range of examined parameters is as follows:
    P from 23.4 to 29.3 MPa; G from 700-3600 kg/m^2/s;
    q from 46 to 2600 kW/m^2; Re from 8E4 to 5E5; D from 1.6 to 20 mm.

    For enhanced heat transfer database in [2]_, this correlation was the
    second best with a MAD of 11.5%. In the database in [3]_, the correlation
    was the second best as well.

    This is sometimes called the Jackson-Hall correlation.
    If the extra information is not provided, the correlation will be used
    without the corrections.

    Examples
    --------
    >>> Nu_Jackson(1E5, 1.2)
    252.37231572974918

    References
    ----------
    .. [1] Jackson, J. D. "Consideration of the Heat Transfer Properties of
       Supercritical Pressure Water in Connection with the Cooling of Advanced
       Nuclear Reactors", 2002.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [3] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009):
       757-66. doi:10.1007/s00231-008-0475-4.
    .. [4] Jäger, Wadim, Victor Hugo Sánchez Espinoza, and Antonio Hurtado.
       "Review and Proposal for Heat Transfer Predictions at Supercritical
       Water Conditions Using Existing Correlations and Experiments." Nuclear
       Engineering and Design, (W3MDM) University of Leeds International
       Symposium: What Where When? Multi-dimensional Advances for Industrial
       Process Monitoring, 241, no. 6 (June 2011): 2184-2203.
       doi:10.1016/j.nucengdes.2011.03.022.
    '''
    if T_b is not None and T_w is not None and T_pc is not None:
        if T_b < T_w < T_pc or 1.2*T_pc < T_b < T_w:
            n = 0.4
        elif T_b < T_pc < T_w:
            n = 0.4 + 0.2*(T_w/T_pc - 1)
        else:
            n = 0.4 + 0.2*(T_w/T_pc - 1)*(1 - 5*(T_b/T_pc - 1))
    else:
        n = 0.4
    Nu = 0.0183*Re**0.82*Pr**0.5
    if rho_w is not None and rho_b is not None:
        Nu *= (rho_w/rho_b)**0.3
    if Cp_avg is not None and Cp_b is not None:
        Nu *= (Cp_avg/Cp_b)**n
    return Nu


def Nu_Gupta(Re, Pr, rho_w=None, rho_b=None, mu_w=None, mu_b=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_.

    .. math::
        Nu_w = 0.004 Re_w^{0.923} \bar{Pr}_w^{0.773}
        \left(\frac{\rho_w}{\rho_b}\right)^{0.186}
        \left(\frac{\mu_w}{\mu_b}\right)^{0.366}

    .. math::
        \bar{Cp} = \frac{H_w-H_b}{T_w-T_b}

    Parameters
    ----------
    Re : float
        Reynolds number with wall fluid properties, [-]
    Pr : float
        Prandtl number with wall fluid properties and an average heat capacity
        between the wall and bulk temperatures [-]
    rho_w : float, optional
        Density at the wall temperature, [kg/m^3]
    rho_b : float, optional
        Density at the bulk temperature, [kg/m^3]
    mu_w : float, optional
        Viscosity at the wall temperature, [Pa*s]
    mu_b : float, optional
        Viscosity at the bulk temperature, [Pa*s]

    Returns
    -------
    Nu : float
        Nusselt number with wall fluid properties, [-]

    Notes
    -----
    For the data used to develop the correlation, P was set at 24 MPa, and D
    was 10 mm. G varied from 200-1500 kg/m^2/s and q varied from 0 to 1250
    kW/m^2.

    Cp used in the calculation of Prandtl number should be the average value
    of those at the wall and the bulk temperatures.

    For deteriorated heat transfer, this was the most accurate correlation in
    [2]_ with a MAD of 18.1%.

    If the extra density and viscosity information is not provided, it will
    not be used.

    Examples
    --------
    >>> Nu_Gupta(1E5, 1.2, 330, 290., 8e-4, 9e-4)
    186.20135477175126

    References
    ----------
    .. [1] Gupta, Sahil, Amjad Farah, Krysten King, Sarah Mokry, and Igor
       Pioro. "Developing New Heat-Transfer Correlation for SuperCritical-Water
       Flow in Vertical Bare Tubes," January 1, 2010, 809-17.
       doi:10.1115/ICONE18-30024.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    '''
    Nu = 0.004*Re**0.923*Pr**0.773
    if rho_w is not None and rho_b is not None:
        Nu *= (rho_w/rho_b)**0.186
    if mu_w is not None and mu_b is not None:
        Nu *= (mu_w/mu_b)**0.366
    return Nu



def Nu_Swenson(Re, Pr, rho_w=None, rho_b=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_.

    .. math::
        Nu_w = 0.00459 Re_w^{0.923} Pr_w^{0.613}
        \left(\frac{\rho_w}{\rho_b}\right)^{0.231}

    .. math::
        \bar{Cp} = \frac{H_w-H_b}{T_w-T_b}

    Parameters
    ----------
    Re : float
        Reynolds number with wall fluid properties, [-]
    Pr : float
        Prandtl number with wall fluid properties and an average heat capacity
        between the wall and bulk temperatures [-]
    rho_w : float, optional
        Density at the wall temperature, [kg/m^3]
    rho_b : float, optional
        Density at the bulk temperature, [kg/m^3]

    Returns
    -------
    Nu : float
        Nusselt number with wall fluid properties, [-]

    Notes
    -----
    The range of examined parameters is as follows:
    P from 22.8 to 27.6 MPa; G from 542-2150 kg/m^2/s;
    Re from 7.5E4 to 3.16E6; T_b from 75 to 576 degrees Celsius and T_w from
    93 to 649 degrees Celsius.

    Cp used in the calculation of Prandtl number should be the average value
    of those at the wall and the bulk temperatures.

    For deteriorated heat transfer, this was the most accurate correlation in
    [2]_ with a MAD of 18.4%. On the overall database in [3]_, it was the
    9th most accurate correlation.

    If the extra density information is not provided, it will not be used.

    Examples
    --------
    >>> Nu_Swenson(1E5, 1.2, 330, 290.)
    217.92827034803668

    References
    ----------
    .. [1] Swenson, H. S., J. R. Carver, and C. R. Kakarala. "Heat Transfer to
       Supercritical Water in Smooth-Bore Tubes." Journal of Heat Transfer 87,
       no. 4 (November 1, 1965): 477-83. doi:10.1115/1.3689139.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [3] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009):
       757-66. doi:10.1007/s00231-008-0475-4.
    .. [4] Jäger, Wadim, Victor Hugo Sánchez Espinoza, and Antonio Hurtado.
       "Review and Proposal for Heat Transfer Predictions at Supercritical
       Water Conditions Using Existing Correlations and Experiments." Nuclear
       Engineering and Design, (W3MDM) University of Leeds International
       Symposium: What Where When? Multi-dimensional Advances for Industrial
       Process Monitoring, 241, no. 6 (June 2011): 2184-2203.
       doi:10.1016/j.nucengdes.2011.03.022.
    '''
    Nu = 0.00459*Re**0.923*Pr**0.613
    if rho_w is not None and rho_b is not None:
        Nu *= (rho_w/rho_b)**0.231
    return Nu


def Nu_Xu(Re, Pr, rho_w=None, rho_b=None, mu_w=None, mu_b=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_.

    .. math::
        Nu_b = 0.02269 Re_b^{0.8079} \bar{Pr}_b^{0.9213}
        \left(\frac{\rho_w}{\rho_b}\right)^{0.6638}
        \left(\frac{\mu_w}{\mu_b}\right)^{0.8687}

    .. math::
        \bar{Cp} = \frac{H_w-H_b}{T_w-T_b}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties and an average heat capacity
        between the wall and bulk temperatures [-]
    rho_w : float, optional
        Density at the wall temperature, [kg/m^3]
    rho_b : float, optional
        Density at the bulk temperature, [kg/m^3]
    mu_w : float, optional
        Viscosity at the wall temperature, [Pa*s]
    mu_b : float, optional
        Viscosity at the bulk temperature, [Pa*s]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    For the data used to develop the correlation, P varied from 23 to 30 MPa,
    and D was 12 mm. G varied from 600-1200 kg/m^2/s and q varied from 100 to
    600 kW/m^2.

    Cp used in the calculation of Prandtl number should be the average value
    of those at the wall and the bulk temperatures.

    For deteriorated heat transfer, this was the third most accurate
    correlation in [2]_ with a MAD of 20.5%.

    If the extra density and viscosity information is not provided, it will
    not be used.

    Examples
    --------
    >>> Nu_Xu(1E5, 1.2, 330, 290., 8e-4, 9e-4)
    289.133054256742

    References
    ----------
    .. [1] Xu, F., Guo, L.J., Mao, Y.F., Jiang, X.E., 2005. "Experimental
       investigation to the heat transfer characteristics of water in vertical
       pipes under supercritical pressure". J. Xi'an Jiaotong University 39,
       468-471.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    '''
    Nu = 0.02269*Re**0.8079*Pr**0.9213
    if rho_w is not None and rho_b is not None:
        Nu *= (rho_w/rho_b)**0.6638
    if mu_w is not None and mu_b is not None:
        Nu *= (mu_w/mu_b)**0.8687
    return Nu


def Nu_Mokry(Re, Pr, rho_w=None, rho_b=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_,
    and reviewed in [2]_.

    .. math::
        Nu_b = 0.0061 Re_b^{0.904} \bar{Pr}_b^{0.684}
        \left(\frac{\rho_w}{\rho_b}\right)^{0.564}

    .. math::
        \bar{Cp} = \frac{H_w-H_b}{T_w-T_b}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties and an average heat capacity
        between the wall and bulk temperatures [-]
    rho_w : float, optional
        Density at the wall temperature, [kg/m^3]
    rho_b : float, optional
        Density at the bulk temperature, [kg/m^3]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    For the data used to develop the correlation, P was set at 20 MPa, and D
    was 10 mm. G varied from 200-1500 kg/m^2/s and q varied from 0 to 1250
    kW/m^2.

    Cp used in the calculation of Prandtl number should be the average value
    of those at the wall and the bulk temperatures.

    For deteriorated heat transfer, this was the four most accurate correlation
    in [2]_ with a MAD of 24.0%. It was also the 7th most accurate against
    enhanced heat transfer, with a MAD of 14.7%, and the most accurate for the
    normal heat transfer database as well as the top correlation in all
    categories combined.

    If the extra density information is not provided, it will not be used.

    Examples
    --------
    >>> Nu_Mokry(1E5, 1.2, 330, 290.)
    246.1156319156

    References
    ----------
    .. [1] Mokry, Sarah, Igor Pioro, Amjad Farah, Krysten King, Sahil Gupta,
       Wargha Peiman, and Pavel Kirillov. "Development of Supercritical Water
       Heat-Transfer Correlation for Vertical Bare Tubes." Nuclear Engineering
       and Design, International Conference on Nuclear Energy for New Europe
       2009, 241, no. 4 (April 2011): 1126-36.
       doi:10.1016/j.nucengdes.2010.06.012.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    '''
    Nu = 0.0061*Re**0.904*Pr**0.684
    if rho_w is not None and rho_b is not None:
        Nu *= (rho_w/rho_b)**0.564
    return Nu


def Nu_Bringer_Smith(Re, Pr):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under near-supercritical conditions according to
    [1]_ and as shown in [2]_ and [3]_.

    .. math::
        Nu_x = 0.0266Re_x^{0.77}Pr_w^{0.55}

    Parameters
    ----------
    Re : float
        Reynolds number with fluid properties at T_ref, [-]
    Pr : float
        Prandtl number with wall fluid properties, [-]

    Returns
    -------
    Nu : float
        Nusselt number with fluid properties at T_ref, [-]

    Notes
    -----
    Fit to data somewhat distant from the critical and pseudo-critical regions.
    Found to fit the data in [3]_ fourth best; in [2]_ however, it was ranked
    so low that no ranking was given.

    Tref and the properties therein should be evaluated as follows:

    .. math::
        T_{ref} = T_b \text{ if } \frac{T_{pc}-T_b}{T_w-T_b} < 0

    .. math::
        T_{ref} = T_{pc} \text{ if } 0 < \frac{T_{pc}-T_b}{T_w-T_b}  < 1

    .. math::
        T_{ref} = T_w \text{ if } \frac{T_{pc}-T_b}{T_w-T_b} > 1

    Examples
    --------
    >>> Nu_Bringer_Smith(1E5, 1.2)
    208.1763175327

    References
    ----------
    .. [1] Bringer, R. P., and J. M. Smith. "Heat Transfer in the Critical
       Region." AIChE Journal 3, no. 1 (March 1, 1957): 49-55.
       doi:10.1002/aic.690030110.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [3] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009):
       757-66. doi:10.1007/s00231-008-0475-4.
    '''
    return 0.0266*Re**0.77*Pr**0.55


def Nu_Ornatsky(Re, Pr_b, Pr_w, rho_w=None, rho_b=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_ as
    shown in both [2]_ and [3]_.

    .. math::
        Nu_b = 0.023Re_b^{0.8}(\min(Pr_b, Pr_w))^{0.8}
        \left(\frac{\rho_w}{\rho_b}\right)^{0.3}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr_b : float
        Prandtl number with bulk fluid properties, [-]
    Pr_w : float
        Prandtl number with wall fluid properties, [-]
    rho_w : float, optional
        Density at the wall temperature, [kg/m^3]
    rho_b : float, optional
        Density at the bulk temperature, [kg/m^3]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    [2]_ ranked it thirteenth in the enhanced heat transfer
    category, with a MAD of 19.8% and 11th in the normal heat transfer with a
    MAD of 17.6%. [3]_ ranked it seventh on a combined database.

    If the extra density information is not provided, it will not be used.

    Examples
    --------
    >>> Nu_Ornatsky(1E5, 1.2, 1.5, 330, 290.)
    276.6353115083

    References
    ----------
    .. [1] Ornatsky A.P., Glushchenko, L.P., Siomin, E.T. (1970). The research
       of temperature conditions of small diameter parallel tubes cooled by
       water under supercritical pressures. In: Proceedings of the 4th
       international heat transfer conference, Paris-Versailles, France.
       Elsevier, Amsterdam, vol VI, Paper no. B, 8 November 1970
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [3] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009):
       757-66. doi:10.1007/s00231-008-0475-4.
    '''
    Nu = 0.023*Re**0.8*min(Pr_b, Pr_w)**0.8
    if rho_w is not None and rho_b is not None:
        Nu *= (rho_w/rho_b)**0.3
    return Nu


def Nu_Gorban(Re, Pr):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_.
    Not recommended.

    .. math::
        Nu_b = 0.0059Re_b^{0.90}Pr_b^{-0.12}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties, [-]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    Reviewed in [2]_ and [3]_; [2]_ did not even rank it, and [3]_ ranked it
    12th of 14.

    Examples
    --------
    >>> Nu_Gorban(1E5, 1.2)
    182.536728273

    References
    ----------
    .. [1] Gorban LM, Pomet`ko RS, Khryaschev OA (1990) Modeling of water heat
       transfer with Freon of supercritical pressure, 1766, Institute of
       Physics and Power Engineering, Obninsk
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [3] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009):
       757-66. doi:10.1007/s00231-008-0475-4.
    '''
    return 0.0059*Re**0.90*Pr**-0.12


def Nu_Zhu(Re, Pr, rho_w=None, rho_b=None, k_w=None, k_b=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_.

    .. math::
        Nu_b = 0.0068 Re_b^{0.9} \bar{Pr}_b^{0.63}
        \left(\frac{\rho_w}{\rho_b}\right)^{0.17}
        \left(\frac{k_w}{k_b}\right)^{0.29}

    .. math::
        \bar{Cp} = \frac{H_w-H_b}{T_w-T_b}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties and an average heat capacity
        between the wall and bulk temperatures [-]
    rho_w : float, optional
        Density at the wall temperature, [kg/m^3]
    rho_b : float, optional
        Density at the bulk temperature, [kg/m^3]
    k_w : float, optional
        Thermal conductivity at the wall temperature, [W/m/K]
    k_b : float, optional
        Thermal conductivity at the bulk temperature, [W/m/K]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    For the data used to develop the correlation, P varied from 22 to 30 MPa,
    and D was 26 mm. G varied from 600-1200 kg/m^2/s and q varied from 200 to
    600 kW/m^2.

    Cp used in the calculation of Prandtl number should be the average value
    of those at the wall and the bulk temperatures.

    On the overall database in [2]_, this was the 8th most accurate
    correlation,and ninth most accurate against normal heat transfer.

    If the extra density and thermal conductivity information is not provided,
    it will not be used.

    Examples
    --------
    >>> Nu_Zhu(1E5, 1.2, 330, 290., 0.63, 0.69)
    240.145985449

    References
    ----------
    .. [1] Zhu, Xiaojing, Qincheng Bi, Dong Yang, and Tingkuan Chen. "An
       Investigation on Heat Transfer Characteristics of Different Pressure
       Steam-Water in Vertical Upward Tube." Nuclear Engineering and Design
       239, no. 2 (February 2009): 381-88. doi:10.1016/j.nucengdes.2008.10.026.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    '''
    Nu = 0.0068*Re**0.9*Pr**0.63
    if rho_w is not None and rho_b is not None:
        Nu *= (rho_w/rho_b)**0.17
    if k_w is not None and k_b is not None:
        Nu *= (k_w/k_b)**0.29
    return Nu


def Nu_Bishop(Re, Pr, rho_w=None, rho_b=None, D=None, x=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_.
    Correlation includes an adjustment for the thermal entry length.
    One of the most common correlations for supercritical convection.

    .. math::
        Nu_b = 0.0069 Re_b^{0.9} \bar Pr_b^{0.66}
        \left(\frac{\rho_w}{\rho_b}\right)^{0.43}(1+2.4D/x)

    .. math::
        \bar{Cp} = \frac{H_w-H_b}{T_w-T_b}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties and an average heat capacity
        between the wall and bulk temperatures [-]
    rho_w : float, optional
        Density at the wall temperature, [kg/m^3]
    rho_b : float, optional
        Density at the bulk temperature, [kg/m^3]
    D : float, optional
        Diameter of tube, [m]
    x : float, optional
        Axial distance along the tube, [m]

    Returns
    -------
    Nu : float
        Nusselt number with wall fluid properties, [-]

    Notes
    -----
    For the data used to develop the correlation, P varied from 22.8 to 27.6
    MPa, and D was x/D varied from 30-365. G varied from 651-3662 kg/m^2/s and
    q varied from 310 to 3460 kW/m^2. T_b varied from 282 to 527 degrees
    Celsius.

    Cp used in the calculation of Prandtl number should be the average value
    of those at the wall and the bulk temperatures.

    For enhanced heat transfer, this was the 11th most accurate correlation in
    [2]_ with a MAD of 19.0%. On the overall database in [3]_, it was the
    most accurate correlation however.

    If the extra density information is not provided, it will not be used.
    If both diameter and axial distance are not provided, the entrance
    correction is not used.

    Examples
    --------
    >>> Nu_Bishop(1E5, 1.2, 330, 290., .01, 1.2)
    265.362005007

    References
    ----------
    .. [1] Bishop A.A., Sandberg R.O., Tong L.S. (1965) Forced convection heat
       transfer to water at near-critical temperature and supercritical
       pressures. In: AIChE J. Chemical engineering symposium series, no. 2.
       Institute of Chemical Engineers, London
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [3] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009):
       757-66. doi:10.1007/s00231-008-0475-4.
    .. [4] Jäger, Wadim, Victor Hugo Sánchez Espinoza, and Antonio Hurtado.
       "Review and Proposal for Heat Transfer Predictions at Supercritical
       Water Conditions Using Existing Correlations and Experiments." Nuclear
       Engineering and Design, (W3MDM) University of Leeds International
       Symposium: What Where When? Multi-dimensional Advances for Industrial
       Process Monitoring, 241, no. 6 (June 2011): 2184-2203.
       doi:10.1016/j.nucengdes.2011.03.022.
    '''
    Nu = 0.0069*Re**0.9*Pr**0.66
    if rho_w is not None and rho_b is not None:
        Nu *= (rho_w/rho_b)**0.43
    if D is not None and x is not None:
        Nu *= (1 + 2.4*D/x)
    return Nu


def Nu_Yamagata(Re, Pr, Pr_pc=None, Cp_avg=None, Cp_b=None, T_b=None,
               T_w=None, T_pc=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_.

    .. math::
        Nu_b = 0.0138 Re_b^{0.85}Pr_b^{0.8}F

    .. math::
        F = \left(\frac{\bar C_p}{C_{p,b}}\right)^{n_2} \text{ if }
        \frac{T_{pc}-T_b}{T_w-T_b} < 0

    .. math::
        F = 0.67Pr_{pc}^{-0.05} \left(\frac{\bar C_p}{C_{p,b}}\right)^{n_1}
        \text{ if } 0 < \frac{T_{pc}-T_b}{T_w-T_b}  < 1

    .. math::
        F = 1\text{ if } \frac{T_{pc}-T_b}{T_w-T_b} > 1

    .. math::
        n_1 = -0.77(1 + 1/Pr_{pc}) + 1.49

    .. math::
        n_2 = 1.44(1 + 1/Pr_{pc}) - 0.53

    .. math::
        \bar{Cp} = \frac{H_w-H_b}{T_w-T_b}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties, [-]
    Pr_pc : float, optional
        Prandtl number at the pseudocritical temperature, [-]
    Cp_avg : float, optional
        Average heat capacity between the wall and bulk temperatures, [J/kg/K]
    Cp_b : float, optional
        Heat capacity at the bulk temperature, [J/kg/K]
    T_b : float
        Bulk temperature, [K]
    T_w : float
        Wall temperature, [K]
    T_pc : float
        Pseudocritical temperature, i.e. temperature at P where Cp is at a
        maximum, [K]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    For the data used to develop the correlation, P varied from 22.6 to 29.4
    MPa, and D was 7.5 and 10 mm. G varied from 310-1830 kg/m^2/s, q varied
    from 116 to 930 kW/m^2, and bulk temperature varied from 230 to 540 decrees
    Celsius.

    In the database in [3]_, the correlation was considered but not tested.
    In [2]_, the correlation was considered but no results were reported.

    For enhanced heat transfer database in [2]_, this correlation was the
    second best with a MAD of 11.5%. In the database in [3]_, the correlation
    was the second best as well.

    If the extra information is not provided, the correlation will be used
    without the corrections.

    Examples
    --------
    >>> Nu_Yamagata(Re=1E5, Pr=1.2, Pr_pc=1.5, Cp_avg=2080.845, Cp_b=2048.621, T_b=650, T_w=700, T_pc=600.0)
    292.347342800

    References
    ----------
    .. [1] Yamagata, K, K Nishikawa, S Hasegawa, T Fujii, and S Yoshida.
       "Forced Convective Heat Transfer to Supercritical Water Flowing in
       Tubes." International Journal of Heat and Mass Transfer 15, no. 12
       (December 1, 1972): 2575-93. doi:10.1016/0017-9310(72)90148-2.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [3] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009):
       757-66. doi:10.1007/s00231-008-0475-4.
    .. [4] Jäger, Wadim, Victor Hugo Sánchez Espinoza, and Antonio Hurtado.
       "Review and Proposal for Heat Transfer Predictions at Supercritical
       Water Conditions Using Existing Correlations and Experiments." Nuclear
       Engineering and Design, (W3MDM) University of Leeds International
       Symposium: What Where When? Multi-dimensional Advances for Industrial
       Process Monitoring, 241, no. 6 (June 2011): 2184-2203.
       doi:10.1016/j.nucengdes.2011.03.022.
    '''
    F = 1.0
    if (T_b is not None and T_w is not None and T_pc is not None
        and Pr_pc is not None and Cp_avg is not None and Cp_b is not None):
        E = (T_pc - T_b)/(T_w - T_b)
        if E < 0.0:
            n2 = 1.44*(1 + 1/Pr_pc) - 0.53
            F = (Cp_avg/Cp_b)**n2
        elif E < 1.0:
            n1 = -0.77*(1 + 1/Pr_pc) + 1.49
            F = 0.67*Pr_pc**-0.05*(Cp_avg/Cp_b)**n1
    return 0.0138*Re**0.85*Pr**0.8*F


def Nu_Kitoh(Re, Pr, H=None, G=None, q=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_,
    also shown in [2]_, [3]_ and [4]_. Depends on fluid enthalpy, mass flux,
    and heat flux.

    .. math::
        Nu_b = 0.015Re_b^{0.85} Pr_b^m

    .. math::
        m = 0.69 - \frac{81000}{q_{dht}} + f_cq

    .. math::
        q_{dht} = 200 G^{1.2}

    .. math::
        f_c = 2.9\times10^{-8} + \frac{0.11}{q_{dht}} \text{ for }
        H_b < 1500 \text{ kJ/kg}

    .. math::
        f_c = -8.7\times10^{-8} - \frac{0.65}{q_{dht}} \text{ for }
        1500 \text{ kJ/kg} < H_b < 3300 \text{ kJ/kg}

    .. math::
        f_c = -9.7\times10^{-7} + \frac{1.3}{q_{dht}} \text{ for }
        H_b > 3300 \text{ kJ/kg}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties, [-]
    H : float, optional
        Enthalpy of water (if the fluid is water), [J/kg]
    G : float, optional
        Mass flux of the fluid, [kg/m^2/s]
    q : float, optional
        Heat flux to wall, [W/m^2]

    Returns
    -------
    Nu : float
        Nusselt number as explained below, [-]

    Notes
    -----
    The reference point for the enthalpy values is not stated in [1]_. The
    upper and lower enthalpy limits for this correlation are 4000 kJ/kg and
    0 kJ/kg, but these are not enforced in this function.

    If not all of H, G, and q are provided, the correlation is used without
    the correction.

    This correlation was ranked 6th best in [3]_, and found 4th best for
    enhanced heat transfer in [2]_ with a MAD of 12.3%.

    For the data used to develop the correlation, G varied from 100-1750
    kg/m^2/s, q varied from 0 to 1800 kW/m^2, and bulk temperature varied from
    20 to 550 decrees Celsius.

    This correlation does not have realistic behavior for values outside those
    used in the study, and should not be used.

    Examples
    --------
    >>> Nu_Kitoh(1E5, 1.2, 1.3E6, 1500, 5E6)
    331.8023413959

    References
    ----------
    .. [1] Kitoh, Kazuaki, Seiichi Koshizuka, and Yoshiaki Oka. "Refinement of
       Transient Criteria and Safety Analysis for a High-Temperature Reactor
       Cooled by Supercritical Water." Nuclear Technology 135, no. 3
       (September 1, 2001): 252-64.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [3] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009):
       757-66. doi:10.1007/s00231-008-0475-4.
    .. [4] Jäger, Wadim, Victor Hugo Sánchez Espinoza, and Antonio Hurtado.
       "Review and Proposal for Heat Transfer Predictions at Supercritical
       Water Conditions Using Existing Correlations and Experiments." Nuclear
       Engineering and Design, (W3MDM) University of Leeds International
       Symposium: What Where When? Multi-dimensional Advances for Industrial
       Process Monitoring, 241, no. 6 (June 2011): 2184-2203.
       doi:10.1016/j.nucengdes.2011.03.022.
    '''
    if H is not None and G is not None and q is not None:
        qht = 200.*G**1.2
        if H < 1.5E6:
            fc = 2.9E-8 + 0.11/qht
        elif 1.5E6 <= H <= 3.3E6:
            fc = -8.7E-8 - 0.65/qht
        else:
            fc = -9.7E-7 + 1.3/qht
        m = 0.69 - 81000./qht + fc*q
    else:
        m = 0.69
    return 0.015*Re**0.85*Pr**m


def Nu_Krasnoshchekov_Protopopov(Re, Pr, Cp_avg=None, Cp_b=None, k_w=None,
                                 k_b=None, mu_w=None, mu_b=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_.

    .. math::
        Nu_b = Nu_0\left(\frac{\mu_w}{\mu_b}\right)^{0.11}\left(\frac{k_b}{k_w}
        \right)^{-0.33}\left(\frac{\bar C_p}{C_{p,b}}\right)^{0.35}

    .. math::
        Nu_0 = \frac{(f/8)Re_b \bar Pr_b}{1.07+12.7(f/8)^{1/2}
        (\bar Pr_b)^{2/3}-1)}

    .. math::
        fd = [1.82\log_{10}(Re_b) - 1.64]^{-2}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties [-]
    Cp_avg : float, optional
        Average heat capacity between the wall and bulk temperatures, [J/kg/K]
    Cp_b : float, optional
        Heat capacity at the bulk temperature, [J/kg/K]
    k_w : float, optional
        Thermal conductivity at the wall temperature, [W/m/K]
    k_b : float, optional
        Thermal conductivity at the bulk temperature, [W/m/K]
    mu_w : float, optional
        Viscosity at the wall temperature, [Pa*s]
    mu_b : float, optional
        Viscosity at the bulk temperature, [Pa*s]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    For the data used to develop the correlation, P varied from 22.3 to 32 MPa,
    Re varied from 2E4 to 8.6E6, Pr from 0.86-86, viscosity ration from 0.9 to
    3.6, thermal conductivity ratio from 1 to 6, and heat capacity ratio from
    0.07 to 4.5.

    For the heat transfer database in [3]_, this correlation was 14th most
    accurate.

    If the extra heat capacity, viscosity, and thermal conductivity
    information is not provided, it will not be used.

    Examples
    --------
    >>> Nu_Krasnoshchekov_Protopopov(1E5, 1.2, 330, 290., 0.62, 0.52, 8e-4, 9e-4)
    228.8529673740

    References
    ----------
    .. [1] Krasnoshchekov EA, Protopopov VS (1959) Heat transfer at
       supercritical region in flow of carbon dioxide and water in tubes.
       Therm Eng 12:26-30
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [3] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009):
       757-66. doi:10.1007/s00231-008-0475-4.
    .. [4] Jäger, Wadim, Victor Hugo Sánchez Espinoza, and Antonio Hurtado.
       "Review and Proposal for Heat Transfer Predictions at Supercritical
       Water Conditions Using Existing Correlations and Experiments." Nuclear
       Engineering and Design, (W3MDM) University of Leeds International
       Symposium: What Where When? Multi-dimensional Advances for Industrial
       Process Monitoring, 241, no. 6 (June 2011): 2184-2203.
       doi:10.1016/j.nucengdes.2011.03.022.
    '''
    fd = (1.82*log10(Re) - 1.64)**-2
    Nu = (fd/8.)*Re*Pr/(1.07 + 12.7*(fd/8.)**0.5*(Pr**(2/3.)-1))
    if mu_w is not None and mu_b is not None:
        Nu *= (mu_w/mu_b)**0.11
    if k_w is not None and k_b is not None:
        Nu *= (k_w/k_b)**-0.33
    if Cp_avg is not None and Cp_b is not None:
        Nu *= (Cp_avg/Cp_b)**0.35
    return Nu


def Nu_Petukhov(Re, Pr, rho_w=None, rho_b=None, mu_w=None, mu_b=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_.

    .. math::
        Nu_b = \frac{(f/8)Re_b \bar Pr_b}{1+900/Re_b+12.7(f/8)^{1/2}
        (\bar Pr_b)^{2/3}-1)}

    .. math::
        f = f_d\left(\frac{\rho_w}{\rho_b}\right)^{0.4}
        \left(\frac{\mu_w}{\mu_b}\right)^{0.2}

    .. math::
        f_d = [1.82\log_{10}(Re_b) - 1.64]^{-2}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties [-]
    rho_w : float, optional
        Density at the wall temperature, [kg/m^3]
    rho_b : float, optional
        Density at the bulk temperature, [kg/m^3]
    mu_w : float, optional
        Viscosity at the wall temperature, [Pa*s]
    mu_b : float, optional
        Viscosity at the bulk temperature, [Pa*s]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    For the heat transfer database in [2]_, this correlation was 5th most
    accurate in the enhanced heat transfer category, and second in the normal
    heat transfer category with MADs of 13.8% and 12.0% respectively.

    If the extra viscosity and density information is not provided, it will not
    be used.

    Examples
    --------
    >>> Nu_Petukhov(1E5, 1.2, 330, 290., 8e-4, 9e-4)
    254.825859846

    References
    ----------
    .. [1] Petukhov, B.S., V.A. Kurganov, and V.B. Ankudinov. "HEAT TRANSFER
       AND FLOW RESISTANCE IN THE TURBULENT PIPE FLOW OF A FLUID WITH
       NEAR-CRITICAL STATE PARAMETERS." High Temperature 21, no. 1 (1983):
       81-89.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    '''
    fd = (1.82*log10(Re) - 1.64)**-2
    if rho_w is not None and rho_b is not None:
        fd *= (rho_w/rho_b)**0.4
    if mu_w is not None and mu_b is not None:
        fd *= (mu_w/mu_b)**0.2
    return (fd/8.)*Re*Pr/(1 + 900./Re + 12.7*(fd/8.)**0.5*(Pr**(2/3.)-1))


def Nu_Krasnoshchekov(Re, Pr, rho_w=None, rho_b=None, Cp_avg=None, Cp_b=None,
                      T_b=None, T_w=None, T_pc=None):
    r'''Calculates internal convection Nusselt number for turbulent vertical
    upward flow in a pipe under supercritical conditions according to [1]_.

    .. math::
        Nu_b = Nu_0\left(\frac{\rho_w}{\rho_b}\right)^{0.3}\left(
        \frac{\bar C_p}{C_{p,b}}\right)^{n}

    .. math::
        Nu_0 = \frac{(f/8)Re_b \bar Pr_b}{1.07+12.7(f/8)^{1/2}
        (\bar Pr_b^{2/3}-1)}

    .. math::
        f_d = [1.82\log_{10}(Re_b) - 1.64]^{-2}

    .. math::
        n = 0.4 \text{ for } T_b < T_w < T_{pc} \text{ or }
        1.2T_{pc} < T_b < T_w

    .. math::
        n = n_1 = 0.22 + 0.18T_w/T_{pc} \text{ for } 1 < T_w/T_{pc} < 2.5

    .. math::
        n = n_1 + (5n_1 - 2)(1 - T_b/T_{pc}) \text{ for } T_{pc} < T_b <
        1.2T_{pc} \text{ and } T_{b} < T_w

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties, [-]
    rho_w : float, optional
        Density at the wall temperature, [kg/m^3]
    rho_b : float, optional
        Density at the bulk temperature, [kg/m^3]
    Cp_avg : float, optional
        Average heat capacity between the wall and bulk temperatures, [J/kg/K]
    Cp_b : float, optional
        Heat capacity at the bulk temperature, [J/kg/K]
    T_b : float
        Bulk temperature, [K]
    T_w : float
        Wall temperature, [K]
    T_pc : float
        Pseudocritical temperature, i.e. temperature at P where Cp is at a
        maximum, [K]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    The range of examined parameters is as follows:
    P from 23.4 to 29.3 MPa; G from 700-3600 kg/m^2/s;
    q from 46 to 2600 kW/m^2; Re from 8E4 to 5E5; D from 1.6 to 20 mm.

    If the extra information is not provided, the correlation will be used
    without the corrections.

    Examples
    --------
    >>> Nu_Krasnoshchekov(1E5, 1.2)
    234.8285518561

    References
    ----------
    .. [1] Krasnoshchekov, E.A., Protopopov, V.S., Van Fen, Kuraeva, I.V.,
       1967. Experimental investigation of heat transfer for carbon dioxide in
       the supercritical region. In Proceedings of the Second All-Soviet Union
       Conference on Heat and Mass Transfer, Minsk, Belarus, May.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015):
       451-60. doi:10.1016/j.anucene.2014.10.027.
    '''
    if T_b is not None and T_w is not None and T_pc is not None:
        n1 = 0.22 + 0.18*T_w/T_pc
        if T_b < T_w < T_pc or 1.2*T_pc < T_b < T_w:
            n = 0.4
        elif 1.0 < T_w/T_pc < 2.5:
            n = n1
        else:
            n = n1 + (5.0*n1 - 2.0)*(1.0 - T_b/T_pc)
    else:
        n = 0.4
    fd = (1.82*log10(Re) - 1.64)**-2
    Nu = (fd/8.)*Re*Pr/(1.07 + 12.7*(fd/8.)**0.5*(Pr**(2/3.)-1.0))
    if rho_w is not None and rho_b is not None:
        Nu *= (rho_w/rho_b)**0.3
    if Cp_avg is not None and Cp_b is not None:
        Nu *= (Cp_avg/Cp_b)**n
    return Nu
