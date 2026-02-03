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

"""Provide unit conversions to and from SI units, where SI is used internally"""

_mm = 1E-3
_cm = 1E-2
_in = 0.0254
_ft = _in*12
_m = 1.0

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

# Conversions from various common units to SI
unit_conversions = {
    'length':  {'mm': _mm, 'cm': _cm, 'm': _m, 'in': _in, 'ft': _ft},
    'volume':  {'cc': _cc, 'L': _L, 'm^3': _m3, 'gal': _gal},
    'pressure': {'kPa': _kPa, 'atm': _atm, 'psi': _psi},
    'mass': {'g': _g, 'kg': _kg, 'lbm': _lbm},
    'force': {'N': _N, 'kN': _kN, 'lbf': _lbf},
    # 'temperature': {'deg C': , 'K': 1.0, 'deg F': }
}
# Conversions from SI to other units
inv_unit_conversions = {
    'length':  {},
    'volume':  {},
    'pressure': {},
    'mass': {},
    'force': {},
}
for unit_type, units in unit_conversions.items():
    for unit, val in units.items():
        # print(unit, isinstance(val, float), type(val), unit in inv_unit_conversions[unit_type])
        if isinstance(val, float) and not unit in inv_unit_conversions[unit_type]:
            inv_unit_conversions[unit_type][unit] = lambda x, f=val: x/f
for unit_type, units in unit_conversions.items():
    for unit, val in units.items():
        if isinstance(val, float):
            unit_conversions[unit_type][unit] = lambda x, f=val: x*f

def get_unit_type(unit):
    for unit_type, units in unit_conversions.items():
        if unit in units: return unit_type
    return None
