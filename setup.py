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

from distutils.core import setup

classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Manufacturing',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Natural Language :: English',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: BSD',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Unix',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: Implementation :: CPython',
    'Topic :: Education',
    'Topic :: Scientific/Engineering :: Atmospheric Science',
    'Topic :: Scientific/Engineering :: Chemistry',
    'Topic :: Scientific/Engineering :: Physics',
]





setup(
  name = 'ht',
  packages = ['ht'],
  license='GPL3',
  version = '0.1.46',
  description = 'Heat transfer component of Chemical Engineering Design Library (ChEDL)',
  author = 'Caleb Bell',
  long_description = open('README.rst').read(),
  platforms=["Windows", "Linux", "Mac OS", "Unix"],
  author_email = 'Caleb.Andrew.Bell@gmail.com',
  url = 'https://github.com/CalebBell/ht',
  download_url = 'https://github.com/CalebBell/ht/tarball/0.1.46',
  keywords = ['chemical engineering', 'heat transfer', 'mechanical engineering'],
  classifiers = classifiers,
  install_requires=['fluids'],
  extras_require = {
      'Coverage documentation':  ['wsgiref>=0.1.2', 'coverage>=4.0.3']
  },
)
