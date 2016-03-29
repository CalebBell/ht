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

import difflib
from scipy.interpolate import interp1d
from ht.conduction import R_to_k

__all__ = ['nearest_material', 'k_material', 'rho_material', 'Cp_material',
           'building_materials', 'refractories', 'ASHRAE', 'ASHRAE_k',
           'refractory_VDI_k', 'refractory_VDI_Cp', 'materials_dict']

# building_materials In VDI Heat Atlas; full table in DIN EN 12524-2000 which
# is used here
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



# Format for ASHRAE strings: [rho, Cp, k, R, t]
# [kg/m^3, J/kg/K, W/m/K, m^2*K/W, mm] only t is in non-SI units

ASHRAE_board_siding = {'Board, Asbestos/cement': [1900, 1000, 0.57, None, None],
'Board, Cement': [1150, 840, 0.25, None, None],
'Board, Fiber/cement, 1400 kg/m^3': [1400, 840, 0.25, None, None],
'Board, Fiber/cement, 1000 kg/m^3': [1000, 840, 0.19, None, None],
'Board, Fiber/cement, 400 kg/m^3': [400, 1880, 0.07, None, None],
'Board, Fiber/cement, 300 kg/m^3': [300, 1150, 0.06, None, None],
'Gypsum or plaster board': [640, 1880, 0.16, None, None],
'Oriented strand board (OSB)': [650, 1880, None, 0.12, 12.7],
'Plywood (douglas fir)': [460, 1880, None, 0.14, 12.7],
'Plywood/wood panels': [450, 1880, None, 0.19, 19],
'Vegetable fiber board, Sheathing, regular density': [290, 1300, None, 0.23, 12.7],
'Vegetable fiber board, Sheathing, intermediate density': [350, 1300, None, 0.19, 12.7],
'Vegetable fiber board, Nail-base sheathing': [400, 1300, None, 0.19, 12.7],
'Vegetable fiber board, Shingle backer': [290, 1300, None, 0.17, 9.5],
'Vegetable fiber board, Sound deadening board': [240, 1260, None, 0.24, 12.7],
'Vegetable fiber board, Tile and lay-in panels, plain or acoustic': [290, 590, 0.058, None, None],
'Vegetable fiber board, Laminated paperboard': [480, 1380, 0.072, None, None],
'Vegetable fiber board, Homogeneous board from repulped paper': [480, 1170, 0.072, None, None],
'Hardboard, medium density': [800, 1300, 0.105, None, None],
'Hardboard, high density, service-tempered grade and service grade': [880, 1340, 0.12, None, None],
'Hardboard, high density, standard-tempered grade': [1010, 1340, 0.144, None, None],
'Particleboard, low density': [590, 1300, 0.102, None, None],
'Particleboard, medium density': [800, 1300, 0.135, None, None],
'Particleboard, high density': [1000, 1300, 1.18, None, None],
'Particleboard, underlayment': [640, 1210, None, 1.22, 15.9],
'Waferboard': [700, 1880, 0.072, None, None],
'Shingles, Asbestos/cement': [1900, 1000, None, 0.037, 6.4],
'Shingles, Wood, 400 mm, 190 mm exposure': [None, 1300, None, 0.15, 6],
'Shingles, Wood, double, 400 mm, 300 mm exposure': [None, 1170, None, 0.21, 12],
'Shingles, Wood, plus ins. backer board': [None, 1300, None, 0.25, 8],
'Shingles, Siding, Asbestos/cement, lapped': [None, 1010, None, 0.037, 6.4],
'Shingles, Siding, Asphalt roll siding': [None, 1470, None, 0.026, 2],
'Siding, Asphalt insulating siding': [None, 1470, None, 0.26, 12.7],
'Siding, Hardboard siding': [None, 1170, None, 0.12, 11],
'Siding, Wood, drop, 200 mm': [None, 1170, None, 0.14, 25],
'Siding, Wood, bevel, 200 mm, lapped': [None, 1170, None, 0.14, 13],
'Siding, Wood, bevel, 250 mm, lapped': [None, 1170, None, 0.18, 19],
'Siding, Wood, plywood, lapped': [None, 1220, None, 0.1, 9.5],
'Siding, Aluminum, steel, or vinyl, over sheathing, hollow-backed': [None, 1220, None, 0.11, 0.6],
'Siding, Aluminum, steel, or vinyl, over sheathing, insulating-board-backed': [None, 1340, None, 0.32, 9.5],
'Siding, Aluminum, steel, or vinyl, over sheathing foil-backed': [None, None, None, 0.52, 9.5],
'Siding, Architectural (soda-lime float) glass': [2500, 840, 1, None, None]}


ASHRAE_flooring = {'Carpet and rebounded urethane pad': [110, None, None, 0.42, 19],
'Carpet and rubber pad, one-piece': [320, None, None, 0.12, 9.5],
'Pile carpet with rubber pad': [290, None, None, 0.28, 11],
'Linoleum/cork tile': [465, None, None, 0.09, 6.4],
'PVC/Rubber floor covering, Rubber tile': [1900, None, 0.4, 0.06, 25],
'PVC/Rubber floor covering, Terrazzo': [None, 800, 0.4, 0.014, 25]}

ASHRAE_insulation = {'Glass-fiber batts, 90 mm': [12, 840, 0.043, None, 90],
'Glass-fiber batts, 50 mm': [10.5, 840, 0.0465, None, 50],
'Mineral fiber': [30, 840, 0.036, None, 140],
'Mineral wool, felted, 32 kg/m^3': [32, 840, 0.04, None, None],
'Mineral wool, felted, 100 kg/m^3': [97.5, 840, 0.035, None, None],
'Slag wool, 120 kg/m^3': [120, 950, 0.038, None, None],
'Slag wool, 255 kg/m^3': [255, 950, 0.04, None, None],
'Slag wool, 305 kg/m^3': [305, 950, 0.043, None, None],
'Slag wool, 350 kg/m^3': [350, 950, 0.048, None, None],
'Slag wool, 400 kg/m^3': [400, 950, 0.05, None, None],
'Cellular glass': [130, 750, 0.048, None, None],
'Cement fiber slabs, shredded wood, with Portland cement binder': [415, 1300, 0.074, None, None],
'Cement fiber slabs, shredded wood, with magnesia oxysulfide binder': [350, 1300, 0.082, None, None],
'Glass fiber board': [160, 840, 0.036, None, None],
'Expanded rubber': [70, 1670, 0.032, None, None],
'Expanded polystyrene, extruded': [32.5, 1470, 0.026, None, None],
'Expanded polystyrene, molded beads': [20, 1470, 0.0355, None, None],
'Mineral fiberboard, wet felted': [160, 840, 0.038, None, None],
'Mineral fiberboard, wet felted, core or roof insulation': [262.5, 840, 0.049, None, None],
'Mineral fiberboard, wet felted, acoustical tile, 290 kg/m^3': [290, 800, 0.05, None, None],
'Mineral fiberboard, wet felted, acoustical tile, 335 kg/m^3': [335, None, 0.053, None, None],
'Mineral fiberboard, wet-molded, acoustical tile': [370, 590, 0.061, None, None],
'Perlite board': [160, None, 0.052, None, None],
'Polyisocyanurate, aged, unfaced': [30, None, 0.0235, None, None],
'Polyisocyanurate, aged, with facers': [65, 1470, 0.019, None, None],
'Phenolic foam board with facers, aged': [65, None, 0.019, None, None],
'Loose fill, Cellulosic': [42.5, 1380, 0.042, None, None],
'Loose fill, Perlite, expanded, 50 kg/m^3': [50, 1090, 0.042, None, None],
'Loose fill, Perlite, expanded, 100 kg/m^3': [100, None, 0.0485, None, None],
'Loose fill, Perlite, expanded, 150 kg/m^3': [150, None, 0.0565, None, None],
'Loose fill, Mineral fiber, 95 to 130 mm': [20, 710, None, 1.92, 112.5],
'Loose fill, Mineral fiber, 170 to 220 mm': [20, None, None, 3.33, 195],
'Loose fill, Mineral fiber, 190 to 250 mm': [20, None, None, 3.85, 220],
'Loose fill, Mineral fiber, 260 to 350 mm': [20, None, None, 5.26, 305],
'Loose fill, Mineral fiber, 90 mm': [42.5, None, None, 2.3, 90],
'Loose fill, Vermiculite, exfoliated, 120 kg/m^3': [120, 1340, 0.068, None, None],
'Loose fill, Vermiculite, exfoliated, 80 kg/m^3': [80, None, 0.063, None, None],
'Spray-applied Cellulosic fiber': [75, None, 0.0455, None, None],
'Spray-applied Glass fiber': [62.5, None, 0.0385, None, None],
'Spray-applied Polyurethane foam, 7 kg/m^3': [7, 1470, 0.042, None, None],
'Spray-applied Polyurethane foam, 40 kg/m^3': [40, 1470, 0.026, None, None],
'Spray-applied Polyurethane foam, aged and dry, 40 mm': [30, 1470, None, 1.6, 40],
'Spray-applied Polyurethane foam, aged and dry, 50 mm': [55, 1470, None, 1.92, 50],
'Spray-applied Polyurethane foam, aged and dry, 120 mm': [30, None, None, 3.69, 120],
'Spray-applied Ureaformaldehyde foam, dry': [14, None, 0.031, None, None]}

ASHRAE_roofing = {'Asbestos/cement shingles': [1120, 1000, None, 0.037, 6],
'Asphalt (bitumen with inert fill), 1600 kg/m^3': [1600, None, 0.43, None, None],
'Asphalt (bitumen with inert fill), 1900 kg/m^3': [1900, None, 0.58, None, None],
'Asphalt (bitumen with inert fill), 2300 kg/m^3': [2300, None, 1.15, None, None],
'Asphalt roll roofing': [920, 1510, None, 0.027, 2],
'Asphalt shingles': [920, 1260, None, 0.078, 12],
'Built-up roofing': [920, 1470, None, 0.059, 10],
'Mastic asphalt (heavy, 20% grit)': [950, None, 0.19, None, None],
'Reed thatch': [270, None, 0.09, None, None],
'Roofing felt': [2250, None, 1.2, None, None],
'Slate': [None, 1260, None, 0.009, 13],
'Straw thatch': [240, None, 0.07, None, None],
'Wood shingles, plain and plastic-film-faced': [None, 1300, None, 0.166, 10]}

ASHRAE_plastering = {'Cement plaster, sand aggregate': [1860, 840, 0.72, None, None],
'Sand aggregate': [None, 840, None, 0.013, 10],
'Gypsum plaster, 1120 kg/m^3': [1120, None, 0.38, None, None],
'Gypsum plaster, 1280 kg/m^3': [1280, None, 0.46, None, None],
'Lightweight aggregate': [720, None, None, 0.056, 13],
'Perlite aggregate': [720, 1340, 0.22, None, None],
'Sand aggregate': [1680, 840, 0.81, None, None],
'Vermiculite aggregate, 480 kg/m^3': [480, None, 0.14, None, None],
'Vermiculite aggregate, 600 kg/m^3': [600, None, 0.2, None, None],
'Vermiculite aggregate, 720 kg/m^3': [720, None, 0.25, None, None],
'Vermiculite aggregate, 840 kg/m^3': [840, None, 0.26, None, None],
'Vermiculite aggregate, 960 kg/m^3': [960, None, 0.3, None, None],
'Perlite plaster, 400 kg/m^3': [400, None, 0.08, None, None],
'Perlite plaster, 600 kg/m^3': [600, None, 0.19, None, None],
'Pulpboard or paper plaster': [600, None, 0.07, None, None],
'Sand/cement plaster, conditioned': [1560, None, 0.63, None, None],
'Sand/cement/lime plaster, conditioned': [1440, None, 0.48, None, None],
'Sand/gypsum (3:1) plaster, conditioned': [1550, None, 0.65, None, None]}

ASHRAE_masonry = {'Brick, fired clay, 2400 kg/m^3': [2400, 800, 1.34, None, None],
'Brick, fired clay, 2240 kg/m^3': [2240, 800, 1.185, None, None],
'Brick, fired clay, 2080 kg/m^3': [2080, 800, 1.02, None, None],
'Brick, fired clay, 1920 kg/m^3': [1920, 800, 0.895, None, None],
'Brick, fired clay, 1760 kg/m^3': [1760, 800, 0.78, None, None],
'Brick, fired clay, 1600 kg/m^3': [1600, 800, 0.675, None, None],
'Brick, fired clay, 1440 kg/m^3': [1440, 800, 0.57, None, None],
'Brick, fired clay, 1280 kg/m^3': [1280, 800, 0.48, None, None],
'Brick, fired clay, 1120 kg/m^3': [1120, 800, 0.405, None, None],
'Clay tile, hollow, 1 cell deep': [None, 880, None, 0.14, 75],
'Clay tile, hollow, 2 cells deep': [None, 880, None, 0.27, 150],
'Clay tile, hollow, 3 cells deep': [None, 880, None, 0.44, 300],
'Lightweight brick, 800 kg/m^3': [800, None, 0.2, None, None],
'Lightweight brick, 770 kg/m^3': [770, None, 0.22, None, None],
'Concrete blocks, Limestone aggregate, 200 mm, 16 kg, 2200 kg/m^3, 2 cores with perlite-filled cores': [None, None, None, 0.37, 200],
'Concrete blocks, Limestone aggregate, 300 mm, 25 kg, 2200 kg/m^3, 2 cores with perlite-filled cores.': [None, None, None, 0.65, 300],
'Concrete blocks, normal-weight aggregate, 300 mm, 23 kg, 2100 kg/m^3, 2 or 3 cores': [None, 920, None, 0.185, 200],
'Concrete blocks, normal-weight aggregate, 300 mm, 23 kg, 2100 kg/m^3, with perlite-filled cores': [None, None, None, 0.35, 200],
'Concrete blocks, normal-weight aggregate, 300 mm, 23 kg, 2100 kg/m^3, with vermiculite-filled cores': [None, None, None, 0.29, 200],
'Concrete blocks, normal-weight aggregate, 300 mm, 23 kg, 2000 kg/m^3, 2 cores': [None, 920, None, 0.217, 300],
'Concrete blocks, medium-weight aggregate, 200 mm, 13 kg, 1650 kg/m^3, 2 or 3 cores': [1650, None, None, 0.26, 200],
'Concrete blocks, medium-weight aggregate, 200 mm, 13 kg, 1650 kg/m^3, with perlite-filled cores': [1650, None, None, 0.53, 200],
'Concrete blocks, medium-weight aggregate, 200 mm, 13 kg, 1650 kg/m^3, with vermiculite-filled cores': [None, None, None, 0.58, 200],
'Concrete blocks, medium-weight aggregate, 200 mm, 13 kg, 1650 kg/m^3, with molded-EPS-filled cores': [1650, None, None, 0.56, 200],
'Concrete blocks, medium-weight aggregate, 200 mm, 13 kg, 1650 kg/m^3, with molded EPS inserts in cores': [1650, None, None, 0.47, 200],
'Concrete blocks, low-mass aggregate, 150 mm, 7.5 kg, 1400 kg/m^3, 2 or 3 cores': [None, None, None, 0.315, 150],
'Concrete blocks, low-mass aggregate, 150 mm, 7.5 kg, 1400 kg/m^3, with perlite-filled cores': [None, None, None, 0.74, 150],
'Concrete blocks, low-mass aggregate, 150 mm, 7.5 kg, 1400 kg/m^3, with vermiculite-filled cores': [None, None, None, 0.53, 150],
'Concrete blocks, low-mass aggregate, 200 mm, 9 kg, 1250 kg/m^3': [None, 880, None, 0.445, 200],
'Concrete blocks, low-mass aggregate, 200 mm, 9 kg, 1250 kg/m^3, with perlite-filled cores': [None, 880, None, 0.985, 200],
'Concrete blocks, low-mass aggregate, 200 mm, 9 kg, 1250 kg/m^3, with vermiculite-filled cores': [None, 880, None, 0.81, 200],
'Concrete blocks, low-mass aggregate, 200 mm, 9 kg, 1250 kg/m^3, with molded-EPS-filled cores': [None, 880, None, 0.85, 200],
'Concrete blocks, low-mass aggregate, 200 mm, 9 kg, 1250 kg/m^3, with UF foam-filled cores': [None, 880, None, 0.79, 200],
'Concrete blocks, low-mass aggregate, 200 mm, 9 kg, 1250 kg/m^3, with molded EPS inserts in cores': [None, 880, None, 0.62, 200],
'Concrete blocks, low-mass aggregate, 300 mm, 16 kg, 1400 kg/m^3, 2 or 3 cores': [None, None, None, 0.43, 300],
'Concrete blocks, low-mass aggregate, 300 mm, 16 kg, 1400 kg/m^3, with perlite-filled cores': [None, None, None, 1.35, 300],
'Concrete blocks, low-mass aggregate, 300 mm, 16 kg, 1400 kg/m^3, with vermiculite-filled cores': [None, None, None, 1, 300],
'Stone, lime, or sand': [2880, None, 10.4, None, None],
'Quartzitic and sandstone, 2560 kg/m^3': [2560, None, 6.2, None, None],
'Quartzitic and sandstone, 2240 kg/m^3': [2240, None, 3.46, None, None],
'Quartzitic and sandstone, 1920 kg/m^3': [1920, 880, 1.88, None, None],
'Calcitic, dolomitic, limestone, marble, and granite, 2880 kg/m^3': [2880, None, 4.33, None, None],
'Calcitic, dolomitic, limestone, marble, and granite, 2560 kg/m^3': [2560, None, 3.17, None, None],
'Calcitic, dolomitic, limestone, marble, and granite, 2240 kg/m^3': [2240, None, 2.31, None, None],
'Calcitic, dolomitic, limestone, marble, and granite, 1920 kg/m^3': [1920, 880, 1.59, None, None],
'Calcitic, dolomitic, limestone, marble, and granite, 1600 kg/m^3': [1600, None, 1.15, None, None],
'Gypsum partition tile, 75 by 300 by 760 mm, solid, 3 cells': [None, 790, None, 0.222, 75],
'Gypsum partition tile, 75 by 300 by 760 mm, with 4 cells': [None, None, None, 0.238, 75],
'Gypsum partition tile, 100 by 300 by 760 mm, 3 cells': [None, None, None, 0.294, 100],
'Limestone, 2400 kg/m^3': [2400, 840, 0.57, None, None],
'Limestone, 2600 kg/m^3': [2600, 840, 0.93, None, None],
'Concrete, Sand and gravel or stone aggregate concretes, 2400 kg/m^3': [2400, None, 2.15, None, None],
'Concrete, Sand and gravel or stone aggregate concretes, 2240 kg/m^3': [2240, 900, 1.95, None, None],
'Concrete, Sand and gravel or stone aggregate concretes, 2080 kg/m^3': [2080, None, 1.45, None, None],
'Concrete, Low-mass aggregate or limestone': [1920, None, 1.1, None, None],
'Concrete, Expanded shale, clay, or slate; expanded slags; cinders; pumice; scoria; 1600 kg/m^3': [1600, 840, 0.785, None, None],
'Concrete, Expanded shale, clay, or slate; expanded slags; cinders; pumice; scoria; 1280 kg/m^3': [1280, 840, 0.535, None, None],
'Concrete, Expanded shale, clay, or slate; expanded slags; cinders; pumice; scoria; 960 kg/m^3': [960, None, 0.33, None, None],
'Concrete, Expanded shale, clay, or slate; expanded slags; cinders; pumice; scoria; 640 kg/m^3': [640, None, 0.18, None, None],
'Concrete, Gypsum/fiber concrete (87.5% gypsum, 12.5% wood chips)': [800, 840, 0.24, None, None],
'Concrete, Cement/lime, mortar, and stucco, 1920 kg/m^3': [1920, None, 1.4, None, None],
'Concrete, Cement/lime, mortar, and stucco, 1600 kg/m^3': [1600, None, 0.97, None, None],
'Concrete, Cement/lime, mortar, and stucco, 1280 kg/m^3': [1280, None, 0.65, None, None],
'Concrete, Perlite, vermiculite, and polystyrene beads, 800 kg/m^3': [800, None, 0.265, None, None],
'Concrete, Perlite, vermiculite, and polystyrene beads, 640 kg/m^3': [640, 795, 0.21, None, None],
'Concrete, Perlite, vermiculite, and polystyrene beads, 480 kg/m^3': [480, None, 0.16, None, None],
'Concrete, Perlite, vermiculite, and polystyrene beads, 320 kg/m^3': [320, None, 0.12, None, None],
'Concrete, Foam concretes, 1920 kg/m^3': [1920, None, 0.75, None, None],
'Concrete, Foam concretes, 1600 kg/m^3': [1600, None, 0.6, None, None],
'Concrete, Foam concretes, 1280 kg/m^3': [1280, None, 0.44, None, None],
'Concrete, Foam concretes, 1120 kg/m^3': [1120, None, 0.36, None, None],
'Concrete, Foam concretes and cellular concretes, 960 kg/m^3': [960, None, 0.3, None, None],
'Concrete, Foam concretes and cellular concretes, 640 kg/m^3': [640, None, 0.2, None, None],
'Concrete, Foam concretes and cellular concretes, 320 kg/m^3': [320, None, 0.12, None, None],
'Concrete, Aerated concrete (oven-dried)': [615, 840, 0.2, None, None],
'Concrete, Polystyrene concrete (oven-dried)': [527.5, 840, 0.37, None, None],
'Concrete, Polymer concrete, 1950 kg/m^3': [1950, None, 1.64, None, None],
'Concrete, Polymer concrete, 2200 kg/m^3': [2200, None, 1.03, None, None],
'Concrete, Polymer cement': [1870, None, 0.78, None, None],
'Concrete, Slag concrete, 960 kg/m^3': [960, None, 0.22, None, None],
'Concrete, Slag concrete, 1280 kg/m^3': [1280, None, 0.32, None, None],
'Concrete, Slag concrete, 1600 kg/m^3': [1600, None, 0.43, None, None],
'Concrete, Slag concrete, 2000 kg/m^3': [2000, None, 1.23, None, None]}

ASHRAE_woods = {'Oak': [705, 1630, 0.17, None, None],
'Birch': [702.5, 1630, 0.175, None, None],
'Maple': [667.5, 1630, 0.165, None, None],
'Ash': [642.5, 1630, 0.155, None, None],
'Southern pine': [615, 1630, 0.15, None, None],
'Southern yellow pine': [500, 1630, 0.13, None, None],
'Eastern white pine': [400, 1630, 0.1, None, None],
'Douglas fir/larch': [557.5, 1630, 0.145, None, None],
'Southern cypress': [507.5, 1630, 0.13, None, None],
'Hem/fir, spruce/pine/fir': [445, 1630, 0.12, None, None],
'Spruce': [400, 1630, 0.09, None, None],
'Western red cedar': [350, 1630, 0.09, None, None],
'West coast woods, cedars': [425, 1630, 0.115, None, None],
'Eastern white cedar': [360, 1630, 0.1, None, None],
'California redwood': [420, 1630, 0.115, None, None],
'Pine (oven-dried)': [370, 1880, 0.092, None, None],
'Spruce (oven-dried)': [395, 1880, 0.1, None, None]}

ASHRAE = {}
for i in [ASHRAE_board_siding, ASHRAE_flooring, ASHRAE_insulation,
          ASHRAE_roofing, ASHRAE_plastering, ASHRAE_masonry, ASHRAE_woods]:
    ASHRAE.update(i)



def ASHRAE_k(ID):
    r'''Returns thermal conductivity of a building or insulating material
    from a table in [1]_. Thermal conductivity is independent of temperature
    here. Many entries in the table are listed for varying densities, but the
    appropriate ID from the table must be selected to account for that.

    Parameters
    ----------
    ID : str
        ID conresponding to a material in the dictionary `ASHRAE`

    Returns
    -------
    k : float
        Thermal conductivity of the material, [W/m/K]

    Examples
    --------
    >>> ASHRAE_k(ID='Mineral fiber')
    0.036

    References
    ----------
    .. [1] ASHRAE Handbook: Fundamentals. American Society of Heating,
       Refrigerating and Air-Conditioning Engineers, Incorporated, 2013.
    '''
    values = ASHRAE[ID]
    if values[2]:
        k = values[2]
    else:
        R = values[3]
        t = values[4]/1000. # mm to m
        k = R_to_k(R, t)
    return k

#print [ASHRAE_k(ID='Mineral fiber')]
#print [sum([ASHRAE_k(ID) for ID in ASHRAE])]
#print [len([ASHRAE_k(ID) for ID in ASHRAE])]

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


materials_dict = {}
for mat_dict, reference in [(refractories, 1), (ASHRAE, 2), (building_materials, 3)]:
    for key in mat_dict.keys():
        materials_dict[key] = reference


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


def nearest_material(name, complete=False):
    r'''Returns the nearest hit to a given name from from dictionaries of
    building, insulating, or refractory material from tables in [1]_, [2]_,
    and [3]_. Function will pick the closest match based on a fuzzy search.
    if `complete` is True, will only return hits with all three of density,
    heat capacity, and thermal conductivity available.

    Parameters
    ----------
    name : str
        Search keywords to be used by difflib function

    Returns
    -------
    ID : str
        A key to one of the dictionaries mentioned above

    Examples
    --------
    >>> nearest_material('stainless steel')
    'Metals, stainless steel'

    References
    ----------
    .. [1] ASHRAE Handbook: Fundamentals. American Society of Heating,
       Refrigerating and Air-Conditioning Engineers, Incorporated, 2013.
    .. [2] DIN EN 12524 (2000-07) Building Materials and Products
       Hygrothermal Properties - Tabulated Design Values; English Version of
       DIN EN 12524.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    if complete:
        hits = difflib.get_close_matches(name, materials_dict.keys(), n=1000, cutoff=0)
        for hit in hits:
            if materials_dict[hit] == 1 or materials_dict[hit]==3 or (ASHRAE[hit][0] and ASHRAE[hit][1]):
                return hit
    else:
        ID = difflib.get_close_matches(name, materials_dict.keys(), n=1, cutoff=0.6)
        if not ID:
            ID = difflib.get_close_matches(name, materials_dict.keys(), n=1, cutoff=0.3)
        if not ID:
            ID = difflib.get_close_matches(name, materials_dict.keys(), n=1, cutoff=0)
        return ID[0]


def k_material(ID, T=298.15):
    r'''Returns thermal conductivity of a building, insulating, or refractory
    material from tables  in [1]_, [2]_, and [3]_. Thermal conductivity may or
    may not be dependent on temperature depending on the source used. Function
    must be provided with either a key to one of the dictionaries
    `refractories`, `ASHRAE`, or `building_materials` - or a search term which
    will pick the closest match based on a fuzzy search. To determine which
    source the fuzzy search will pick, use the function `nearest_material`.
    Fuzzy searches are slow; it is preferable to call this function with a
    material key directly.

    Parameters
    ----------
    ID : str
        String as described above
    T : float, optional
        Temperature of the material, [K]

    Returns
    -------
    k : float
        Thermal conductivity of the material, [W/m/K]

    Examples
    --------
    >>> k_material('Mineral fiber')
    0.036

    References
    ----------
    .. [1] ASHRAE Handbook: Fundamentals. American Society of Heating,
       Refrigerating and Air-Conditioning Engineers, Incorporated, 2013.
    .. [2] DIN EN 12524 (2000-07) Building Materials and Products
       Hygrothermal Properties - Tabulated Design Values; English Version of
       DIN EN 12524.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    if ID not in materials_dict:
        ID = nearest_material(ID)
    if ID in refractories:
        k = refractory_VDI_k(ID, T)
    elif ID in ASHRAE:
        k = ASHRAE_k(ID)
    else:
        k = float(building_materials[ID][1])
    return k


def rho_material(ID):
    r'''Returns the density of a building, insulating, or refractory
    material from tables  in [1]_, [2]_, and [3]_. No temperature dependence is
    available. Function must be provided with either a key to one of the
    dictionaries `refractories`, `ASHRAE`, or `building_materials` - or a
    search term which will pick the closest match based on a fuzzy search. To
    determine which source the fuzzy search will pick, use the function
    `nearest_material`. Fuzzy searches are slow; it is preferable to call this
    function with a material key directly.

    Parameters
    ----------
    ID : str
        String as described above

    Returns
    -------
    rho : float
        Density of the material, [kg/m^3]

    Examples
    --------
    >>> rho_material('Board, Asbestos/cement')
    1900.0

    References
    ----------
    .. [1] ASHRAE Handbook: Fundamentals. American Society of Heating,
       Refrigerating and Air-Conditioning Engineers, Incorporated, 2013.
    .. [2] DIN EN 12524 (2000-07) Building Materials and Products
       Hygrothermal Properties - Tabulated Design Values; English Version of
       DIN EN 12524.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    if ID not in materials_dict:
        ID = nearest_material(ID)
    if ID in refractories:
        rho = float(refractories[ID][0]) # Density available for all hits
    elif ID in building_materials:
        rho = float(building_materials[ID][0]) # Density available for all hits
    else:
        rho = ASHRAE[ID][0]
        if rho is None:
            raise Exception('Density is not available for this material')
        else:
            rho = float(rho)
    return rho


def Cp_material(ID, T=298.15):
    r'''Returns heat capacity of a building, insulating, or refractory
    material from tables  in [1]_, [2]_, and [3]_. Heat capacity may or
    may not be dependent on temperature depending on the source used. Function
    must be provided with either a key to one of the dictionaries
    `refractories`, `ASHRAE`, or `building_materials` - or a search term which
    will pick the closest match based on a fuzzy search. To determine which
    source the fuzzy search will pick, use the function `nearest_material`.
    Fuzzy searches are slow; it is preferable to call this function with a
    material key directly.

    Parameters
    ----------
    ID : str
        String as described above
    T : float, optional
        Temperature of the material, [K]

    Returns
    -------
    Cp : float
        Heat capacity of the material, [W/m/K]

    Examples
    --------
    >>> Cp_material('Mineral fiber')
    840.0

    References
    ----------
    .. [1] ASHRAE Handbook: Fundamentals. American Society of Heating,
       Refrigerating and Air-Conditioning Engineers, Incorporated, 2013.
    .. [2] DIN EN 12524 (2000-07) Building Materials and Products
       Hygrothermal Properties - Tabulated Design Values; English Version of
       DIN EN 12524.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    if ID not in materials_dict:
        ID = nearest_material(ID)
    if ID in refractories:
        Cp = refractory_VDI_Cp(ID, T)
    elif ID in building_materials:
        Cp = float(building_materials[ID][2]) # Density available for all hits
    else:
        Cp = ASHRAE[ID][1]
        if Cp is None:
            raise Exception('Heat capacity is not available for this material')
        else:
            Cp = float(Cp)
    return Cp

