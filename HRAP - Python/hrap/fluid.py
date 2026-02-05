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

"""Provide JAX-compilable fluid property tables using CoolProp outputs"""

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import CoolProp.CoolProp as CP
# import interpax
from tracept import tclass, tmethod

@tclass
class SatFluid:
    # TODO: use lin, need auto way of tabulating to a given tolerance
    T:     jax.Array #: temperature (the input)
    P:     jax.Array #: pressure
    Zv:    jax.Array #: vapor compressability factor
    Cpv:   jax.Array #: vapor constant-pressure specific heat
    He:    jax.Array #: vapor specific enthalpy of evaporation
    Sv:    jax.Array #: vapor specific entropy
    rho_v: jax.Array #: vapor density
    rho_l: jax.Array #: liquid density

    # TODO: this needs custom deriv and it should give 0 out of bounds since clipped
    @tmethod
    def __call__(self, T):
        """On-demand linearly interpolated fluid properties

        Args:
            T: temperature to evaluate at
        Returns:
            interpolator view of this object
        """
        return self.lerp(T, self.T)

@tclass(static_attrnames=['in1', 'in2'])
class Gas:
    in1: str #: name of first input property
    in2: str #: name of second second input property
    T:     jax.Array #: temperature
    P:    jax.Array #: pressure
    rho: jax.Array #: density
    Z:     jax.Array #: compressability
    # Cp:    jax.Array #: constant-pressure specific heat
    H:    jax.Array #: enthalpy
    S:    jax.Array #: entropy

    @tmethod
    def __call__(self, **kwargs):
        """On-demand bilinearly interpolated gas properties

        Despite in1, in2 being fixed after initialization, this takes kwargs instead of vargs to force
        the user to be explicit about the properties they are trying to evaluate, avoiding accidental conflicts.
    
        Args:
            **kwargs: the two values, variable names specified by attr:in1 and attr:in2, to evaluate at 
        Returns:
            interpolator view of this object,
            note that accessing either of the two input properties through the resulting interpolator will result in an error
        """
        # TODO: Convention? can two lerp calls already do it?
        return self.lerp((kwargs[self.in1], kwargs[self.in2]), getattr(self,self.in1), getattr(self,self.in2))

# Wrapper for coolprop to provide JAX compilable property tables
def bake_sat_coolprop(fluid, T_eval):
    Pv, Cpv, Zv, He, Sv, rho_l, rho_v = [np.zeros_like(T_eval) for i in range(7)]
    for i, T in enumerate(T_eval):
        Pv   [i] = CP.PropsSI('P', 'T', T, 'Q', 0, fluid)
        Zv   [i] = CP.PropsSI('Z', 'T', T, 'Q', 1, fluid)
        Cpv  [i] = CP.PropsSI('CPMASS', 'T', T, 'Q', 1, fluid)
        He   [i] = CP.PropsSI('H', 'T', T, 'Q', 1, fluid) - CP.PropsSI('H', 'T', T, 'Q', 0, fluid)
        Sv   [i] = CP.PropsSI('S', 'T', T, 'Q', 1, fluid)
        rho_v[i] = CP.PropsSI('D', 'T', T, 'Q', 1, fluid)
        rho_l[i] = CP.PropsSI('D', 'T', T, 'Q', 0, fluid)
    # print(_Pv)
    # print(_Hv)
    
    # TODO: enable extrap? gives NaNs when off
    # Construct monotomic cubic splines to interpolate
    # Pv, rho_l, rho_v, Hv, Cp, Z = [interpax.PchipInterpolator(T_eval, props) for props in [_Pv, _rho_l, _rho_v, _Hv, _Cp, _Z]]
    
    return SatFluid(T_eval, Pv, Zv, Cpv, He, Sv, rho_v, rho_l)
    # # Staticly supply interpolators to a new sat props function
    # def get_my_sat_props(T, Pv=Pv, rho_l=rho_l, rho_v=rho_v, Hv=Hv, Cp=Cp, Z=Z):
    #     return { 'Pv': Pv(T), 'rho_l': rho_l(T), 'rho_v': rho_v(T), 'Hv': Hv(T), 'Cp': Cp(T), 'Z': Z(T) }

    # return get_my_sat_props

def bake_S_rho_gas_coolprop(fluid, S_eval, rho_eval):
    T, P, Z, H = [np.zeros((S_eval.size, rho_eval.size)) for i in range(4)]
    for i, S in enumerate(S_eval):
        for j, rho in enumerate(rho_eval):
            T[i,j] = CP.PropsSI('T', 'S', S, 'D', rho, fluid)
            P[i,j] = CP.PropsSI('P', 'S', S, 'D', rho, fluid)
            Z[i,j] = CP.PropsSI('Z', 'S', S, 'D', rho, fluid)
            H[i,j] = CP.PropsSI('H', 'S', S, 'D', rho, fluid)

    return Gas('S', 'rho', T, P, rho_eval, Z, H, S_eval)
