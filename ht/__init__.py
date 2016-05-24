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


from . import core
from . import hx
from . import conv_internal
from . import boiling_nucleic
from . import air_cooler
from . import radiation
from . import condensation
from . import conduction
from . import conv_jacket
from . import insulation
from . import conv_free_immersed
from . import conv_tube_bank
from . import conv_packed_bed
from . import conv_external

from .core import *
from .hx import *
from .conv_internal import *
from .boiling_nucleic import *
from .air_cooler import *
from .radiation import *
from .condensation import *
from .conduction import *
from .conv_jacket import *
from .insulation import *
from .conv_free_immersed import *
from .conv_tube_bank import *
from .conv_packed_bed import *
from .conv_external import *

__all__ = ['core', 'hx', 'conv_internal', 'boiling_nucleic', 'air_cooler',
'radiation', 'condensation', 'conduction', 'conv_jacket', 'conv_free_immersed',
'conv_tube_bank', 'insulation', 'conv_packed_bed', 'conv_external']


__all__.extend(core.__all__)
__all__.extend(hx.__all__)
__all__.extend(conv_internal.__all__)
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


__version__ = '0.1.36'

