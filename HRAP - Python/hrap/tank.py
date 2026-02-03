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

"""
Model the blowdown of fluids in a tank
"""

from collections.abc import Callable #, Iterable, Iterator, Sequence
from functools import partial
from dataclasses import field

import jax
import jax.numpy as jnp
from jax.lax import cond
from tracept import tclass, tmethod, Placeholder, Dynamic, Derivative

from hrap.core import store_x, make_part, StaticVar

# Assume that pressure derivative is historical average
def avg_liq_blowdown(tnk, dP_dT):
    Pdot = tnk.Pdot_sum / tnk.Pdot_N
    Tdot = Pdot / dP_dT
    
    return Tdot, Pdot

# Evaporation based liquid blowdown that falls back to average pressure drop if condensation is indicated
def liq_blowdown(tnk, fld, drho_v__dT, drho_l__dT):
    m_liq, m_vap = x[xmap['tnk_m_ox_liq']], x[xmap['tnk_m_ox_vap']]
    rho_v, rho_l = ox['rho_v'], ox['rho_l']
    
    # Find evap cooling rate, using total oxidizer mass since we consider thermal equilibrium
    A = (m_liq*drho_l__dT/(rho_l**2) + m_vap*drho_v__dT/(rho_v**2)) / (1/rho_v-1/rho_l)
    B = -mdot_ox/(rho_l/rho_v-1)
    C = -ox['Hv'] / ((m_liq+m_vap)*ox['Cp'])
    Tdot = B*C / (1-A*C)
    mdot_evap = Tdot*A + B
    
    Tdot, Pdot = cond(
        mdot_evap > 0.0,
        lambda: (Tdot, Tdot * dP_dT),
        lambda: avg_liq_blowdown(x, xmap, dP_dT),
        x, xmap, Tdot, dP_dT
    )
    
    return Tdot, Pdot

# Assume vapor remains along saturation line, which has been experimentally validated for nitrous oxide
def sat_vap_blowdown(T, m_ox, mdot_ox, ox, dP_dT, get_sat_props):
    delta = 1E-7 # FD step
    m_2__m_1 = (m_ox + mdot_ox*delta) / m_ox
    
    # TODO: precompute high order curve-fit using complex step results?
    def Z_body(args):
        eps, Z_i, Z_1, T_i, T_1, m_2__m_1 = args
        
        T_new = T_1*pow(Z_i/Z_1 * m_2__m_1, 0.3)
        Z_new = get_sat_props(T_new)['Z']
        
        # Get error and force convergence
        eps = jnp.abs(Z_i - Z_new)
        Z_new = (Z_i + Z_new) / 2
        
        return eps, Z_new, Z_1, T_new, T_1, m_2__m_1
    
    res = jax.lax.while_loop(lambda val: jnp.abs(val[0]) > 1E-9, Z_body, (1.0, ox['Z'], ox['Z'], T, T, m_2__m_1))
    # jax.debug.print('{a}', a=res[0])
    T_2 = res[3]
    
    Tdot = (T_2 - T) / delta
    Pdot = Tdot * dP_dT
    # jax.debug.print("TANK Debug {x} {y} {c} {d} {e}", x=Tdot, y=Pdot, c=m_ox, d=mdot_ox, e=dP_dT)
    
    return Tdot, Pdot

def u_sat_tank(s, x, xmap):
    x = store_x(x, xmap,
        # Limit to reasonable values
        tnk_m_ox = jnp.maximum(x[xmap['tnk_m_ox']], 0.0),
        # Record needed for avg pressure drop blowdown model
        tnk_Pdot_sum = x[xmap['tnk_Pdot_sum']] + x[xmap['tnk_Pdot']],
        tnk_Pdot_N = x[xmap['tnk_Pdot_N']] + 1,
    )

    return x


    """Initialize tank filled with saturated fluid.
    
    Args:
      get_sat_props: fluid-specific saturated property function
      V: tank volume
      vnt_S: vent mode
      vnt_CdA: coefficient of discharge times flow rate area for vent
      inj_CdA: coefficient of discharge times flow rate area for a single injector
      inj_N: number of injectors
      inj_vap_model: injector masss flow rate model for vapor phase, StaticVar('Real Gas') (default) or StaticVar('Incompressible')
      inj_liq_model: injector masss flow rate model for liquid phase, StaticVar('Incompressible')
      T: initial temperature
      m_ox: initial total oxidizer rate
    Returns:
      the tank
    """

@tclass(static_attrnames=['vap_model', 'liq_model'])
class Injector:
    # Fixed variables
    CdA: float
    N: int = 1
    vap_model: str = 'Incompressible'
    liq_model: str = 'Incompressible'
    # Dependent variables
    mdot: jax.Array = Dynamic()

@tclass(static_attrnames=['S'])
class Vent:
    # Fixed variables
    S: int = 0
    CdA: float = 0.0
    # Dependent variables
    mdot: jax.Array = Dynamic()

# TODO: custom new to make 
@tclass(static_attrnames=['get_sat_props'])
class SatTank:
    # Fixed variables
    V: float
    get_sat_props: Callable
    inj: Injector
    vnt: Vent = field(default_factory=Vent)
    # TODO: dynamics should be able to take defaults
    # Integrated variables
    T:    jax.Array = Placeholder()
    Tdot: jax.Array = Derivative('T')
    m:    jax.Array = Placeholder()
    mdot: jax.Array = Derivative('m')
    # Dependent variables
    m_liq:    jax.Array = Dynamic() #: Mass of liquid
    m_vap:    jax.Array = Dynamic()
    rho_liq:  jax.Array = Dynamic()
    rho_vap:  jax.Array = Dynamic()
    P:        jax.Array = Dynamic()
    Pdot:     jax.Array = Dynamic()
    Pdot_sum: jax.Array = Dynamic()
    Pdot_N:   jax.Array = Dynamic()

    @classmethod
    def new(cls, get_sat_props: Callable, V: float, T0: float, m0: float, inj: Injector, vnt: Vent):
        """Create a new tank filled with saturated fluid.

        Args:
            T0: initial temperature
        """
        return cls(V=V, get_sat_props=get_sat_props, inj=inj, vnt=vnt, T=Dynamic(T0), m=Dynamic(m0))

    @tmethod
    def __call__(self, cmbr, env):
        _sat_vap_blowdown = partial(sat_vap_blowdown, get_sat_props=self.get_sat_props)
        tnk, inj, vnt = self, self.inj, self.vnt # Aliases
        
        # Find fluid thermophysical properties
        fld = tnk.get_sat_props(T)
        tnk.rho_vap, tnk.rho_liq, tnk.P = fld.rho_vap, fld.rho_liq, fld.Pv
        
        # Use analytical derivative to get saturation pressure derivative
        dP_dT = jax.grad(lambda T: tnk.get_sat_props(T).Pv)(T)
        # Get saturation density derivatives w.r.t. temperature
        drho_v__dT = jax.grad(lambda T: tnk.get_sat_props(T).rho_vap)(T)
        drho_l__dT = jax.grad(lambda T: tnk.get_sat_props(T).rho_liq)(T)
        # dox_dT = jax.grad(get_sat_props)(T)
        
        # Get mass of fluid currently in the phases
        tnk.m_liq = jnp.maximum((tnk.V - (tnk.m/rho_v))/ ((1/fld.rho_liq)-(1/fld.rho_vap)), 0.0)
        tnk.m_vap = tnk.m - tnk.m_liq
        
        Pc, Pa = cmbr.P, env.Pa        
        dP = tnk.P - Pc
        
        Mcc  = jnp.sqrt(fld.Z*1.31*188.91*T*(Pc/tnk.P)**(0.31/1.31))
        Matm = jnp.sqrt(fld.Z*1.31*188.91*T*(Pa/tnk.P)**(0.31/1.31))
        
        Mcc  = jnp.minimum(Mcc,  1)
        Matm = jnp.minimum(Matm, 1)
        dP   = jnp.maximum(dP,   0)
        
        # Get vented vapor mass flow rate, mode selection at compile time
        if vnt.S == 0:
            vnt.mdot = 0.0
        else:
            vnt.mdot = (vnt.CdA*Pt/jnp.sqrt(tnk.T))*jnp.sqrt(1.31/(fld.Z*188.91))*Matm*(1+(0.31)/2*Matm**2)**(-2.31/0.62)
        
        # Select injector models at compile time
        def inj_liq_model():
            if inj.liq_model == 'Incompressible':
                return inj.N*inj.CdA*jnp.sqrt(2*fld.rho_liq*dP)
        def inj_vap_model():
            if inj.vap_model == 'Incompressible':
                return inj.N*inj.CdA*jnp.sqrt(2*fld.rho_vap*dP)
            elif inj.vap_model == 'Real Gas':
                return (inj.N*inj.CdA*tnk.P/jnp.sqrt(tnk.T))*jnp.sqrt(1.31/(fld.Z*188.91))*Mcc*(1+(0.31)/2*Mcc**2)**(-2.31/0.62)

        # Get injected vapor or liquid mass flow rate
        inj.mdot = cond(
            inj.m_liq <= 1E-3,
            lambda: cond(
                inj.m_vap <= 0.0,
                lambda: 0.0,
                inj_vap_model,
            ),
            inj_liq_model,
        )

        # Total loss rate of fluid = base injected rate + vent rate
        tnk.mdot = -(mdot_inj + mdot_vnt)
        
        # Add vent flow rate to injector rate if plumbed to chamber
        if vnt.S == 2:
            inj.mdot = inj.mdot + vnt.mdot
        
        # Get temperature and pressure rates at various stages of blowdown
        tnk.Tdot, tnk.Pdot = cond(tnk.m <= 0.0, lambda: (0.0, 0.0),
            lambda: cond(tnk.m_liq > 0.0,
                lambda: liq_blowdown(mdot_ox, x, xmap, ox, dP_dT, drho_v__dT, drho_l__dT),
                lambda: _sat_vap_blowdown(T, m_ox, mdot_ox, ox, dP_dT),
            ),
        )
        
        # TODO: should we?
        # Set temperature rate to 0 if outside supported range
