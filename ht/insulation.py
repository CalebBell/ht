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

from scipy.interpolate import interp1d

# Format: density, thermal conductivity, heat capacity
# kg/m^3, W/m/K, and J/kg/K
# A roughly room-teperature value is attached to all values

building_materials = {'Asphalt': (2100, 0.7, 1000),
'Bitumen, pure': (1050, 0.17, 1000),
'Bitumen, felt or sheet': (1100, 0.23, 1000),
'Concrete, medium density 1800 kg/m^3': (1800, 1.15, 1000),
'Concrete, medium density 2000 kg/m^3': (2000, 1.35, 1000),
'Concrete, medium density 2200 kg/m^3': (2200, 1.65, 1000),
'Concrete, high density': (2400, 2, 1000),
'Concrete, reinforced with 1% steel': (2300, 2.3, 1000),
'Concrete, reinforced with 2% steel': (2400, 2.5, 1000),
'Floor covering, rubber': (1200, 0.17, 1400),
'Floor covering, plastic': (1700, 0.25, 1400),
'Floor covering, underlay, cellular or plastic': (270, 0.1, 1400),
'Floor covering, underlay, felt': (120, 0.05, 1300),
'Floor covering, underlay, wool': (200, 0.06, 1300),
'Floor covering, underlay, cork': (200, 0.05, 1500),
'Floor covering, tiles, cork': (400, 65, 1500),
'Floor covering, carpet / textile flooring': (200, 0.06, 1300),
'Floor covering, Linoleum': (1200, 0.17, 1400),
'Gases, air': (1.23, 25, 1008),
'Gases, carbon dioxide': (1.95, 14, 820),
'Gases, argon': (1.7, 17, 519),
'Gases, sulphur hexafluoride': (6.36, 13, 614),
'Gases, krypton': (3.56, 0.009, 245),
'Gases, xenon': (5.68, 0.0054, 160),
'Glass, soda lime': (2500, 1, 750),
'Glass, quartz': (2200, 1.4, 750),
'Glass, glass mosaic': (2000, 1.2, 750),
'Water, ice at -10 °C': (920, 2.3, 2000),
'Water, ice at 0 °C': (900, 2.2, 2000),
'Water, snow, freshly fallen (<30 mm)': (100, 0.05, 2000),
'Water, snow, soft (30 mm to 70 mm)': (200, 0.12, 2000),
'Water, snow, slightly compacted (70 mm to 100 mm)': (300, 0.23, 2000),
'Water, snow, compacted (<200 mm)': (500, 0.6, 2000),
'Water at 10°C': (1000, 0.6, 4190),
'Water at 40°C': (990, 0.63, 4190),
'Water at 80°C': (970, 0.67, 4190),
'Metals, aluminium alloys': (2800, 160, 880),
'Metals, bronze': (8700, 65, 380),
'Metals, brass': (8400, 120, 380),
'Metals, copper': (8900, 380, 380),
'Metals, iron, cast': (7500, 50, 450),
'Metals, lead': (11300, 35, 130),
'Metals, steel': (7800, 50, 450),
'Metals, stainless steel': (7900, 17, 460),
'Metals, zinc': (7200, 110, 380),
'Plastics, acrylic': (1050, 0.2, 1500),
'Plastics, polycarbonates': (1200, 0.2, 1200),
'Plastics, polytetrafluoroethylene (PTFE)': (2200, 0.25, 1000),
'Plastics, Polyvinylchloride (PVC)': (1390, 0.17, 900),
'Plastics, polymethylmethacrylate (PMMA)': (1180, 0.18, 1500),
'Plastics, polyacetate': (1410, 0.3, 1400),
'Plastics, polyamide (nylon)': (1150, 0.25, 1600),
'Plastics, polyamide 6.6 with 25% glass fibre': (1450, 0.3, 1600),
'Plastics, polyethylene / polythene, high density': (980, 0.5, 1800),
'Plastics, polyethylene / polythene, low density': (920, 0.33, 2200),
'Plastics, polystyrene': (1050, 0.16, 1300),
'Plastics, polypropylene': (910, 0.22, 1800),
'Plastics, polypropylene with 25% glass fibre': (1200, 0.25, 1800),
'Plastics, polyurethane (PU)': (1200, 0.25, 1800),
'Plastics, epoxy resin': (1200, 0.2, 1400),
'Plastics, phenolic resin': (1300, 0.3, 1700),
'Plastics, polyester resin': (1400, 0.19, 1200),
'Rubber, natural': (910, 0.13, 1100),
'Rubber, neoprene (polychloroprene)': (1240, 0.23, 2140),
'Rubber, butyl (isobutene), solid melt': (1200, 0.24, 1400),
'Rubber, foam rubber': (70, 0.06, 1500),
'Rubber, hard rubber (ebonite), solid': (1200, 0.17, 1400),
'Rubber, ethylene propylene diene monomer (EPDM)': (1150, 0.25, 1000),
'Rubber, polyisobutylene': (930, 0.2, 1100),
'Rubber, polysulfide': (1700, 0.4, 1000),
'Rubber, butadiene': (980, 0.25, 1000),
'Sealant, silica gel (dessicant)': (720, 0.13, 1000),
'Sealant, silicone, pure': (1200, 0.35, 1000),
'Sealant, silicone, filled': (1450, 0.5, 1000),
'Sealant, silicone foam': (750, 0.12, 1000),
'Sealant, urethane / polyurethane (thermal break)': (1300, 0.21, 1800),
'Sealant, polyvinylchloride (PVC) flexible, with 40% softner': (1200, 0.14, 1000),
'Sealant, elastomeric foam, flexible': (70, 0.05, 1500),
'Sealant, polyurethane (PU) foam': (70, 0.05, 1500),
'Sealant, polyethylene foam': (70, 0.05, 2300),
'Gypsum, 600 kg/m^3': (600, 0.18, 1000),
'Gypsum, 900 kg/m^3': (900, 0.3, 1000),
'Gypsum, 1200 kg/m^3': (1200, 0.43, 1000),
'Gypsum, 1500 kg/m^3': (1500, 0.56, 1000),
'Gypsum, plasterboard': (900, 0.25, 1000),
'Plasters and renders, gypsum insulating plaster': (600, 0.18, 1000),
'Plasters and renders, gypsum plastering, 1000 kg/m^3': (1000, 0.4, 1000),
'Plasters and renders, gypsum plastering, 1300 kg/m^3': (1300, 0.57, 1000),
'Plasters and renders, lime sand': (1600, 0.8, 1000),
'Plasters and renders, cement sand': (1600, 0.8, 1000),
'Plasters and renders, gypsum sand': (1800, 1, 1000),
'Solids, clay or silt': (1500, 1.5, 2085),
'Solids, sand and gravel': (1950, 2, 1045),
'Stone, natural, crystalline rock': (2800, 3.5, 1000),
'Stone, natural, sedimentary rock': (2600, 2.3, 1000),
'Stone, natural, sedimentary rock, light': (1500, 0.85, 1000),
'Stone, natural, porous': (1600, 0.55, 1000),
'Stone, basalt': (2850, 3.5, 1000),
'Stone, gneiss': (2550, 3.5, 1000),
'Stone, granite': (2600, 2.8, 1000),
'Stone, marble': (2800, 3.5, 1000),
'Stone, slate': (2400, 2.2, 1000),
'Stone, limestone, extra soft': (1600, 0.85, 1000),
'Stone, limestone, soft': (1800, 1.1, 1000),
'Stone, limestone, semi-hard': (2000, 1.4, 1000),
'Stone, limestone, hard': (2200, 1.7, 1000),
'Stone, limestone, extra hard': (2600, 2.3, 1000),
'Stone, sandstone (silica)': (2600, 2.3, 1000),
'Stone, natural pumice': (400, 0.12, 1000),
'Stone, artificial stone': (1750, 1.3, 1000),
'Tiles, clay': (2000, 1, 800),
'Tiles, concrete': (2100, 1.5, 1000),
'Tiles, ceramic or porcelain': (2300, 1.3, 840),
'Tiles, plastic': (1000, 0.2, 1000),
'Timber, 500 kg/m^3': (500, 0.13, 1600),
'Timber, 700 kg/m^3': (700, 0.18, 1600),
'Wood, plywood 300 kg/m^3': (300, 0.09, 1600),
'Wood, plywood 500 kg/m^3': (500, 0.13, 1600),
'Wood, plywood 700 kg/m^3': (700, 0.17, 1600),
'Wood, plywood 1000 kg/m^3': (1000, 0.24, 1600),
'Wood, cement-bonded particleboard': (1200, 0.23, 1500),
'Wood, particleboard, 300 kg/m^3': (300, 0.1, 1700),
'Wood, particleboard, 600 kg/m^3': (600, 0.14, 1700),
'Wood, particleboard, 900 kg/m^3': (900, 0.18, 1700),
'Wood, oriented strand board': (650, 0.13, 1700),
'Wood, fibreboard, 250 kg/m^3': (250, 0.07, 1700),
'Wood, fibreboard, 400 kg/m^3': (400, 0.1, 1700),
'Wood, fibreboard, 600 kg/m^3': (600, 0.14, 1700),
'Wood, fibreboard, 800 kg/m^3': (800, 0.18, 1700)}


_refractory_Ts = [673.15, 873.15, 1073.15, 1273.15, 1473.15]

refractories = {'Silica': [1820, (1.2, 1.36, 1.51, 1.64, 1.76), (915, 944, 961, 969, 979)],
'Silica special': [1910, (1.55, 1.76, 1.95, 2.12, 2.28), (915, 944, 961, 970, 980)],
'Fused silica': [1940, (1.44, 1.53, 1.61, 1.67, 1.73), (917, 946, 963, 972, 982)],
'Fireclay': [2150, (1.05, 1.1, 1.15, 1.18, 1.22), (956, 997, 1021, 1037, 1054)],
'High-duty fireclay': [2320, (1.2, 1.27, 1.33, 1.38, 1.42), (958, 999, 1024, 1040, 1058)],
'Sillimanite': [2530, (1.66, 1.76, 1.84, 1.92, 1.98), (978, 1024, 1052, 1072, 1093)],
'Mullite': [2540, (1.45, 1.52, 1.58, 1.63, 1.67), (987, 1035, 1065, 1087, 1109)],
'Corundum 90%': [2830, (2, 2.1, 2.19, 2.27, 2.33), (993, 1043, 1072, 1095, 1118)],
'Bauxite': [2760, (2.06, 2.03, 2.02, 2, 1.99), (994, 1045, 1077, 1100, 1124)],
'Corundum 99%': [2830, (4.97, 4.36, 3.93, 3.6, 3.35), (1011, 1066, 1099, 1124, 1150)],
'Corundum Spinel': [3100, (3.01, 3.02, 3.03, 3.04, 3.05), (1013, 1067, 1100, 1126, 1152)],
'ACr 90': [3180, (4.2, 3.81, 3.52, 3.3, 3.12), (782, 794, 806, 816, 825)],
'ACrZ 20': [3780, (2.4, 2.33, 2.27, 2.22, 2.18), (772, 789, 804, 814, 825)],
'ACrZ 60': [3200, (3.8, 3.4, 3.11, 2.89, 2.71), (905, 945, 970, 990, 1010)],
'Magnesite Chrome': [3060, (3.5, 3.27, 3.1, 2.96, 2.85), (1004, 1043, 1079, 1110, 1138)],
'Magnesia': [3000, (7.5, 6.23, 5.37, 4.75, 4.28), (1047, 1088, 1125, 1158, 1188)],
'Magnesite Spinel': [2850, (3.8, 3.44, 3.18, 2.98, 2.82), (1050, 1093, 1131, 1164, 1194)],
'Magnesite Graphite H15': [2980, (9.96, 8.46, 7.44, 6.68, 6.1), (1061, 1117, 1168, 1215, 1258)],
'Dolomite P10': [2970, (4.17, 3.99, 3.92, 3.75, 3.66), (950, 988, 1022, 1051, 1078)],
'Sillimanite P5': [2740, (1.5, 1.5, 1.5, 1.5, 1.5), (986, 1037, 1070, 1095, 1120)],
'Bauxite P5': [2830, (2.9, 2.67, 2.49, 2.36, 2.25), (1000, 1056, 1092, 1121, 1149)],
'Corundum P10': [3020, (5.49, 5.19, 4.96, 4.78, 4.62), (1020, 1083, 1126, 1160, 1195)],
'Magnesite P5': [2920, (5.05, 4.53, 4.15, 3.86, 3.63), (1050, 1097, 1139, 1177, 1211)],
'Zirconia': [4950, (1.63, 1.54, 1.48, 1.43, 1.38), (624, 668, 698, 718, 737)],
'Zircon': [3940, (2.67, 2.49, 2.35, 2.24, 2.15), (708, 747, 773, 788, 804)],
'AZS 41': [4000, (4.55, 4.17, 4.25, 4.85, 5.4), (831, 878, 908, 929, 950)],
'AZS 33': [3720, (5.17, 4.42, 4, 4.45, 5.4), (861, 908, 938, 958, 980)],
'a/b-Alumina': [3200, (4.78, 4.45, 4.3, 5, 6.05), (989, 1044, 1080, 1107, 1133)],
'SIC 40%': [2400, (4.2, 4.41, 4.58, 4.73, 4.86), (993, 1043, 1072, 1095, 1118)],
'SIC 70%': [2600, (7, 6.81, 6.67, 6.55, 6.45), (998, 1049, 1079, 1103, 1126)],
'SIC 90%': [2680, (18.6, 17.55, 16.76, 16.14, 15.62), (1005, 1058, 1090, 1115, 1140)],
'L1260': [490, (0.14, 0.16, 0.18, 0.2, 0.22), (942, 979, 1002, 1017, 1033)],
'L1400': [790, (0.27, 0.3, 0.32, 0.34, 0.36), (954, 994, 1018, 1034, 1050)],
'L1540': [890, (0.32, 0.35, 0.38, 0.41, 0.43), (979, 1026, 1054, 1075, 1096)],
'L1760': [1270, (0.45, 0.47, 0.49, 0.51, 0.53), (991, 1040, 1070, 1092, 1114)],
'L1870': [1440, (1.5, 1.34, 1.23, 1.14, 1.07), (1011, 1066, 1099, 1124, 1150)],
'Carbon, anthracite': [1540, (7, 8.51, 9.95, 11.33, 12.65), (1106, 1240, 1362, 1474, 1581)],
'Carbon, graphite': [1550, (67, 60.67, 56.06, 52.01, 49.46), (1108, 1244, 1366, 1479, 1588)]}

def refractory_VDI_rho(ID):
    r'''Returns density of a refractory material from a table in [1]_.
    Here, density is not a function of either temperature or porosity, both of
    which can affect it.

    Parameters
    ----------
    ID : str
        ID conresponding to a material in the dictionary `refractories`

    Returns
    -------
    rho : float
        Density of the refractory material, [kg/m^3]

    Examples
    --------
    >>> refractory_VDI_rho('Fused silica')
    1940

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    rho = refractories[ID][0]
    return rho


def refractory_VDI_k(ID, T=None):
    r'''Returns thermal conductivity of a refractory material from a table in
    [1]_. Here, thermal conductivity is a function of temperature between
    673.15 K and 1473.15 K according to linear interpolation among 5
    equally-spaced points. Here, thermal conductivity is not a function of
    porosity, which can affect it. If T is outside the acceptible range, it is
    rounded to the nearest limit. If T is not provied, the lowest temperature's
    value is provided.

    Parameters
    ----------
    ID : str
        ID conresponding to a material in the dictionary `refractories`
    T : float, optional
        Temperature of the refractory material, [K]

    Returns
    -------
    k : float
        Thermal conductivity of the refractory material, [W/m/K]

    Examples
    --------
    >>> [refractory_VDI_k('Fused silica', i) for i in [None, 200, 1000, 1500]]
    [1.44, 1.44, 1.58074, 1.73]

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    if ID not in refractories:
        raise Exception('ID provided is not in the table')
    if T is None:
        k = float(refractories[ID][1][0])
    else:
        ks = refractories[ID][1]
        to_interp = interp1d(_refractory_Ts, ks)
        if T < _refractory_Ts[0]:
            T = _refractory_Ts[0]
        elif T > _refractory_Ts[-1]:
            T = _refractory_Ts[-1]
        k = float(to_interp(T))
    return k


def refractory_VDI_Cp(ID, T=None):
    r'''Returns heat capacity of a refractory material from a table in
    [1]_. Here, heat capacity is a function of temperature between
    673.15 K and 1473.15 K according to linear interpolation among 5
    equally-spaced points. Here, heat capacity is not a function of
    porosity, affects it. If T is outside the acceptible range, it is
    rounded to the nearest limit. If T is not provied, the lowest temperature's
    value is provided.

    Parameters
    ----------
    ID : str
        ID conresponding to a material in the dictionary `refractories`
    T : float, optional
        Temperature of the refractory material, [K]

    Returns
    -------
    Cp : float
        Heat capacity of the refractory material, [W/m/K]

    Examples
    --------
    >>> [refractory_VDI_Cp('Fused silica', i) for i in [None, 200, 1000, 1500]]
    [917.0, 917.0, 956.78225, 982.0]

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    if ID not in refractories:
        raise Exception('ID provided is not in the table')
    if T is None:
        Cp = float(refractories[ID][2][0])
    else:
        Cps = refractories[ID][2]
        to_interp = interp1d(_refractory_Ts, Cps)
        if T < _refractory_Ts[0]:
            T = _refractory_Ts[0]
        elif T > _refractory_Ts[-1]:
            T = _refractory_Ts[-1]
        Cp = float(to_interp(T))
    return Cp

