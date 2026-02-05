# Copyright 2026 The HRAP Authors.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Authors: Thomas A. Scott

"""Provide unit conversions to and from SI units.

Internally, SI is used with the exception of kmol (1000 moles) being used in chemical contexts.
"""

from dataclasses import dataclass

@dataclass
class Affine:
    offset: float
    scale:  float
    
    def __call__(self, x):
        return self.scale*x + self.offset

_mm = 1E-3
_cm = 1E-2
_in = 0.0254
_ft = _in*12
_m = 1.0

_mm2 = _mm**2
_in2 = _in**2

_cc = _cm**3
_L = 1E-3
_m3 = 1.0
_gal = 0.00378541

_Pa = 1.0
_kPa = 1E3
_atm = 101325
_psi = 6895.0

_g = 1E-3
_kg = 1.0
_lbm = 0.4536

_N = 1.0
_kN = 1E3
_lbf  = 4.448

_C = Affine(273.15, 1.0)
_F = Affine(273.15-32*5/9, 5/9)

# Subset of unit conversions (from various common units to SI) exposed to GUI, truncation makes certain units unreasonable like m^3
unit_conversions = {
    'length':      {'mm': _mm, 'cm': _cm, 'm': _m, 'in': _in, 'ft': _ft},
    'area':        { 'mm2': _mm2, 'in2': _in2 },
    'volume':      {'cc': _cc, 'L': _L, 'gal': _gal},
    'pressure':    {'kPa': _kPa, 'atm': _atm, 'psi': _psi},
    'mass':        {'g': _g, 'kg': _kg, 'lbm': _lbm},
    'force':       {'N': _N, 'kN': _kN, 'lbf': _lbf},
    'temperature': { 'K': 1.0, 'C': _C, 'F': _F },
}
# Corresponding conversions from SI to other units
inv_unit_conversions = { k: {} for k in unit_conversions }

for unit_type, units in unit_conversions.items():
    for unit, val in units.items():
        if not unit in inv_unit_conversions[unit_type]:
            if isinstance(val, float):
                inv_unit_conversions[unit_type][unit] = lambda x, f=val: x/f
            elif isinstance(val, Affine):
                # y = ax+b   =>   x = (y-b)/a = (1/a)y + (-b/a)
                inv_unit_conversions[unit_type][unit] = lambda x, f=Affine(-val.offset/val.scale, 1/val.scale): f(x)
            
for unit_type, units in unit_conversions.items():
    for unit, val in units.items():
        if isinstance(val, float):
            unit_conversions[unit_type][unit] = lambda x, f=val: x*f
        elif isinstance(val, Affine):
            unit_conversions[unit_type][unit] = lambda x, f=val: f(x)

def get_unit_type(unit):
    for unit_type, units in unit_conversions.items():
        if unit in units: return unit_type
    return None
