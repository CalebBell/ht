# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020, 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import fluids

if not fluids.numerics.is_micropython:

    from . import core
    from . import hx
    from . import conv_internal
    from . import boiling_flow
    from . import boiling_nucleic
    from . import conv_tube_bank
    from . import air_cooler
    from . import radiation
    from . import condensation
    from . import conduction
    from . import conv_jacket
    from . import insulation
    from . import conv_free_immersed
    from . import conv_free_enclosed
    from . import conv_packed_bed
    from . import conv_external
    from . import conv_supercritical
    from . import conv_two_phase
    from . import conv_plate
    from . import boiling_plate
    
    
    
    from .core import *
    from .hx import *
    from .conv_internal import *
    from .conv_plate import *
    from .boiling_flow import *
    from .boiling_nucleic import *
    from .air_cooler import *
    from .radiation import *
    from .condensation import *
    from .conduction import *
    from .conv_jacket import *
    from .insulation import *
    from .conv_free_immersed import *
    from .conv_free_enclosed import *
    from .conv_tube_bank import *
    from .conv_packed_bed import *
    from .conv_external import *
    from .conv_supercritical import *
    from .conv_two_phase import *
    from .boiling_plate import *
    
    __all__ = ['core', 'hx', 'conv_internal', 'boiling_nucleic', 'air_cooler',
    'radiation', 'condensation', 'conduction', 'conv_jacket', 'conv_free_immersed',
    'conv_tube_bank', 'insulation', 'conv_packed_bed', 'conv_external',
    'conv_supercritical', 'conv_two_phase', 'boiling_flow', 'boiling_plate',
    'conv_plate', 'conv_free_enclosed']
    
    
    __all__.extend(core.__all__)
    __all__.extend(hx.__all__)
    __all__.extend(conv_internal.__all__)
    __all__.extend(boiling_flow.__all__)
    __all__.extend(boiling_nucleic.__all__)
    __all__.extend(air_cooler.__all__)
    __all__.extend(radiation.__all__)
    __all__.extend(condensation.__all__)
    __all__.extend(conduction.__all__)
    __all__.extend(conv_jacket.__all__)
    __all__.extend(conv_free_immersed.__all__)
    __all__.extend(conv_tube_bank.__all__)
    __all__.extend(insulation.__all__)
    __all__.extend(conv_packed_bed.__all__)
    __all__.extend(conv_external.__all__)
    __all__.extend(conv_supercritical.__all__)
    __all__.extend(conv_two_phase.__all__)
    __all__.extend(boiling_plate.__all__)
    __all__.extend(conv_plate.__all__)
    __all__.extend(conv_free_enclosed.__all__)
    
    submodules = (core, hx, conv_internal, boiling_flow, boiling_nucleic, conv_tube_bank,
                  air_cooler, radiation, condensation, conduction, conv_jacket, insulation,
                  conv_free_immersed, conv_free_enclosed, conv_packed_bed, conv_external,
                  conv_supercritical, conv_two_phase, conv_plate, boiling_plate)
    
    global vectorized, numba, units, numba_vectorized
    if fluids.numerics.PY37:
        def __getattr__(name):
            global vectorized, numba, units, numba_vectorized
            if name == 'vectorized':
                import ht.vectorized as vectorized
                return vectorized
            if name == 'numba':
                import ht.numba as numba
                return numba
            if name == 'units':
                import ht.units as units
                return units
            if name == 'numba_vectorized':
                import ht.numba_vectorized as numba_vectorized
                return numba_vectorized
            raise AttributeError("module %s has no attribute %s" %(__name__, name))
    else:
        from . import vectorized
    
__version__ = '1.0.5'

