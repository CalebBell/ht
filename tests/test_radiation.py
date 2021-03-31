# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from __future__ import division
from ht import *
import numpy as np
import pytest
from fluids.numerics import assert_close, assert_close1d, assert_close2d


def test_radiation():
    assert_close(q_rad(1., 400), 1451.613952, rtol=1e-05)
    assert_close(q_rad(.85, 400, 305.), 816.7821722650002, rtol=1e-05)

    assert 0.0 == blackbody_spectral_radiance(5500., 5E-10)


    assert_close(blackbody_spectral_radiance(800., 4E-6), 1311692056.2430143, rtol=1e-05)


@pytest.mark.slow
def test_solar_spectrum():
    wavelengths, SSI, uncertainties = solar_spectrum()

    min_maxes = [min(wavelengths), max(wavelengths), min(SSI), max(SSI)]
    min_maxes_expect = [5.0000000000000003e-10, 2.9999000000000003e-06, 1330.0, 2256817820.0]
    assert_close1d(min_maxes, min_maxes_expect)

    assert_close(np.trapz(SSI, wavelengths), 1344.8029782379999)

def test_grey_transmittance():
    tau =  grey_transmittance(3.8e-4, molar_density=55300, length=1e-2)
    assert_close(tau, 0.8104707721191062)