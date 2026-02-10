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

# Authors: Drew Nickel, Thomas A. Scott

"""Provide chamber dynamics from tabulated chemical equilibrum properties"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.lax import cond
from tracept import tclass, tmethod, Dynamic, Derivative

from hrap.chem import Rhat

@tclass(static_attrnames=['comb_props'])
class Chamber:
    # Fixed variables
    comb_props: Callable #: function from OF, Pc to combustion HP equilibrium combustion k,M,T
    V0:        float #: empty volume
    cstar_eff: float #: characteristic velocity efficiency
    # Integrated variables
    P:      Dynamic #: combustion pressure
    m_g:    Dynamic #: stored gas mass
    Pdot:   Dynamic = Derivative('P') #: combustion pressure rate
    mdot_g: Dynamic = Derivative('mdot') #: stored gass mass rate
    # Dependent variables
    k:     Dynamic = None #: specific heat ratio
    T:     Dynamic = None #: temperature
    cstar: Dynamic = None #: characteristic velocity
    OF:    Dynamic = None #: oxidizer/fuel mass ratio, handled by grain file

    @classmethod
    def new(cls, comb_props: Callable, V0: float = -1, cstar_eff: float = 1.0, P0: float = -1, m_g0: float = -1):
        """Create a new tank filled with saturated fluid.

        Args:
          comb_props: function from OF, Pc to combustion HP equilibrium combustion k,M,T
          V0: empty volume, specify -1 to assume 1.2*(grn.OD)*(grn.L) for hybrids during prep()
          cstar_eff: characteristic velocity efficiency
          P0: initial pressure, specify -1 to use env.Pa during prep()
          m_g0: initial stored gas mass, specify -1 to determine automatically during prep()
        Returns:
          the new chamber
        """

        return cls(V0=V0, cstar_eff=cstar_eff, P=P0, m_g=m_g0)

    @tmethod
    def __call__(self, tnk, noz, env, grn=None):
        """Update dependent states."""
        cmbr, inj = self, tnk.inj # Aliases

        # Chamber stored mass derivative
        cmbr.mdot_g = inj.mdot - noz.mdot
        if grn is not None:
            cmbr.mdot_g = cmbr.mdot_g + grn.mdot
        cmbr.mdot_g = cond(
            (cmbr.m_g <= 0.0) & (cmbr.mdot_g < 0.0),
            lambda: 0.0,
            lambda: cmbr.mdot_g,
        )

        # Chamber pressure derivative
        cmbr.Pdot = Pc*mdot_g/m_g
        if grn is not None:
            cmbr.Pdot = cmbr.Pdot - Pc*grn.dV/(cmbr.V0 - grn.V)
        cmbr.Pdot = cond(
            ((cmbr.P <= env.Pa) & (cmbr.Pdot < 0.0)) | (cmbr.m_g <= 0.0),
            lambda: 0.0,
            lambda: cmbr.Pdot,
        )
        
        # Get chamber properties and update cstar
        # interp_point = jnp.array([[OF, Pc]])
        # TODO: Need an error if out of bounds?
        # ig = s['chem_interp_k'].grid # interp grid
        # ip = jnp.array([[OF, Pc]]) # interp point
        # for i in range(2):
        #     ip = ip.at[:,i].set(jnp.minimum(jnp.maximum(ip[:,i], ig[i][0]), ig[i][-1]))
        # k = s['chem_interp_k'](ip)[0]
        # M = s['chem_interp_M'](ip)[0]
        # T = s['chem_interp_T'](ip)[0]
        cmbr.k, M, cmbr.T = cmbr.comb_props(OF, Pc)
        # jax.debug.print('cmbr, k={a}, M={b}, T={c} from OF={d}, Pc={e}', a=k, b=M, c=T, d=OF, e=Pc)
        
        k = cmbr.k
        cmbr.cstar = cmbr.cstar_eff * jnp.sqrt((Rhat/M*cmbr.T)/(k*(2/(k+1))**((k+1)/(k-1))))

    @tmethod
    def prep(self, env, grn=None):
        """Automatically deterime dependent defaults requested on initialization.
    
        This is one of the few functions that is incompatible with JIT compilation (since V0 is fixed).
        """
        cmbr = self # Aliases

        if cmbr.V0 == -1:
            cmbr.V0 = grn.L*(jnp.pi/4*grn.OD**2)
        if cmbr.P == -1:
            cmbr.P = env.Pa
        if cmbr.m_g == -1:
            cmbr.m_g = (cmbr.P*29/Rhat/294) * V0 # Ideal gas, p = rho R T

    @tmethod
    def increment(self, env):
        """Limit stored and gas and pressure to tenable values."""

        cmbr = self # Aliases
        
        cmbr.m_g = jnp.maximum(cmbr.m_g,    0.0)
        cmbr.P   = jnp.maximum(cmbr.P,   env.Pa)
