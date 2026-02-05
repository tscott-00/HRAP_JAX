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

"""Model the blowdown of fluids in a tank"""

from collections.abc import Callable #, Iterable, Iterator, Sequence
from functools import partial
from dataclasses import field

import jax
import jax.numpy as jnp
from jax.lax import cond
from tracept import tclass, tmethod, Placeholder, Dynamic, Derivative

from hrap.fluid import SatFluid, Gas

def sat_liq_blowdown(tnk, fld, dP_dT, drho_v__dT, drho_l__dT):
    """Equilibrium (i.e. saturated) based liquid blowdown that falls back to average pressure drop if condensation is indicated"""

    # TODO: if we keep this, would be better to use (current - initial)/time in case of variable time step
    # When modeling the pressure derivative as the historical average
    def avg_liq_blowdown(tnk, dP_dT):
        Pdot = tnk.Pdot_sum / tnk.Pdot_N
        Tdot = Pdot / dP_dT

        return Tdot, Pdot

    # Find evap cooling rate, using total oxidizer mass since we consider thermal equilibrium
    A = (tnk.m_l*drho_l__dT/(rho_l**2) + tnk.m_v*drho_v__dT/(fld.rho_v**2)) / (1/fld.rho_v-1/fld.rho_l)
    B = -tnk.mdot/(fld.rho_l/fld.rho_v-1)
    C = -fld.He / ((tnk.m_l+tnk.m_v)*fld.Cpv)
    Tdot = B*C / (1-A*C)
    mdot_evap = Tdot*A + B
    
    Tdot, Pdot = cond(
        mdot_evap > 0.0,
        lambda: (Tdot, Tdot * dP_dT),
        lambda: avg_liq_blowdown(tnk, dP_dT),
    )
    
    return Tdot, Pdot

def sat_vap_blowdown(tnk, fld, dP_dT):
    """Assume vapor remains along saturation line, which has been experimentally validated for nitrous oxide"""

    delta = 1E-7 # FD step
    m_2__m_1 = (tnk.m + tnk.mdot*delta) / tnk.m
    
    # TODO: precompute high order curve-fit using complex step results?
    def Z_body(args):
        eps, Z_i, Z_1, T_i, T_1, m_2__m_1 = args
        
        T_new = T_1*pow(Z_i/Z_1 * m_2__m_1, 0.3)
        Z_new = tnk.get_sat_props(T_new).Zv
        
        # Get error and force convergence
        eps = jnp.abs(Z_i - Z_new)
        Z_new = (Z_i + Z_new) / 2
        
        return eps, Z_new, Z_1, T_new, T_1, m_2__m_1
    
    res = jax.lax.while_loop(lambda val: jnp.abs(val[0]) > 1E-9, Z_body, (1.0, fld.Zv, fld.Zv, tnk.T, tnk.T, m_2__m_1))
    T_2 = res[3]
    
    Tdot = (T_2 - tnk.T) / delta
    Pdot = Tdot * dP_dT
    # jax.debug.print("TANK Debug {x} {y} {c} {d} {e}", x=Tdot, y=Pdot, c=m_ox, d=mdot_ox, e=dP_dT)
    
    return Tdot, Pdot

def isentropic_gas_blowdown(tnk, gas, S, gas_props: Gas):
    """Assume gas entropy is constant"""

    if { 'S', 'rho' } != { gas_props.in1, gas_props.in2 }:
        raise TypeError('gas_props must take S and rho')

    rhodot = tnk.mdot / tnk.V
    dT__drho = jax.grad(lambda rho: gas_props(S=S, rho=rho).T)(tnk.rho_v)
    dP__drho = jax.grad(lambda rho: gas_props(S=S, rho=rho).P)(tnk.rho_v)
    Tdot, Pdot = dT__drho*rhodot, dP__drho*rhodot

    return Tdot, Pdot

@tclass(static_attrnames=['vap_model', 'liq_model'])
class Injector:
    # Fixed variables
    CdA:       float                        #: coefficient of discharge times flow area for a single injector
    N:         int       = 1                #: number of injectors
    vap_model: str       = 'Real Gas'       #: masss flow rate model for vapor phase, 'Real Gas' (default) or 'Incompressible'
    liq_model: str       = 'Incompressible' #: masss flow rate model for liquid phase, currently only 'Incompressible' is supported
    # Dependent variables
    mdot:      jax.Array = Dynamic()        #: mass flow rate

    @tmethod
    def __call__(self, tnk, fld, cmbr):
        """Update mass flow rate"""
        inj = self # Aliases

        # Select injector models at compile time
        dP = jnp.maximum(tnk.P - cmbr.P, 0)
        def inj_liq_model():
            if inj.liq_model == 'Incompressible':
                return inj.N*inj.CdA*jnp.sqrt(2*fld.rho_l*dP)
        def inj_vap_model():
            if inj.vap_model == 'Incompressible':
                return inj.N*inj.CdA*jnp.sqrt(2*fld.rho_v*dP)
            elif inj.vap_model == 'Real Gas':
                Mcc = jnp.sqrt(fld.Zv*1.31*188.91*T*(cmbr.P/tnk.P)**(0.31/1.31))
                Mcc = jnp.minimum(Mcc, 1)
                return (inj.N*inj.CdA*tnk.P/jnp.sqrt(tnk.T))*jnp.sqrt(1.31/(fld.Zv*188.91))*Mcc*(1+(0.31)/2*Mcc**2)**(-2.31/0.62)

        # Get injected vapor or liquid mass flow rate
        inj.mdot = cond(
            inj.m_l <= 1E-3,
            lambda: cond(
                inj.m_v <= 0.0,
                lambda: 0.0,
                inj_vap_model,
            ),
            inj_liq_model,
        )

@tclass(static_attrnames=['S'])
class Vent:
    # Fixed variables
    S:    int       = 0         #: vent mode, 0 for no vent, 1 for externally vented, 2 for internally plumbed
    CdA:  float     = 0.0       #: coefficient of discharge times total flow area
    # Dependent variables
    mdot: jax.Array = Dynamic() #: mass flow rate

    @tmethod
    def __call__(self, tnk, fld, env):
        """Update mass flow rate"""
        vnt = self # Aliases

        # Get vented vapor mass flow rate, mode selection at compile time
        if vnt.S == 0:
            vnt.mdot = 0.0
        else:
            Matm = jnp.sqrt(fld.Zv*1.31*188.91*T*(env.Pa/tnk.P)**(0.31/1.31))
            Matm = jnp.minimum(Matm, 1)
            vnt.mdot = (vnt.CdA*tnk.P/jnp.sqrt(tnk.T))*jnp.sqrt(1.31/(fld.Zv*188.91))*Matm*(1+(0.31)/2*Matm**2)**(-2.31/0.62)

@tclass(static_attrnames=['get_sat_props'])
class SatTank:
    # Fixed variables
    get_sat_props: Callable #: fluid-specific saturated property function
    V: float #: tank volume
    inj: Injector
    vnt: Vent
    # TODO: dynamics should be able to take defaults
    # Integrated variables
    T:    jax.Array = Placeholder()   #: temperature
    Tdot: jax.Array = Derivative('T') #: temperature rate
    m:    jax.Array = Placeholder()   #: mass
    mdot: jax.Array = Derivative('m') #: mass rate
    # Dependent variables
    m_l:      jax.Array = Dynamic() #: mass of liquid
    m_v:      jax.Array = Dynamic() #: mass of vapor
    rho_l:    jax.Array = Dynamic() #: density of liquid
    rho_v:    jax.Array = Dynamic() #: density of vapor
    P:        jax.Array = Dynamic() #: pressure
    Pdot:     jax.Array = Dynamic() #: pressure rate
    Pdot_sum: jax.Array = Dynamic() #: sum of historical pressure rates
    Pdot_N:   jax.Array = Dynamic() #: number of historical pressure rates included in sum

    @classmethod
    def new(cls, get_sat_props: Callable, V: float, T0: float, m0: float, inj: Injector, vnt: Vent = Vent()):
        """Create a new tank filled with saturated fluid.

        Args:
          get_sat_props: saturated property function defining the contained fluid
          V: tank volume
          T0: initial temperature
          m0: initial mass
          inj: injector(s) definition
          vent: vent definition, default is no vent
        Returns:
          the new tank
        """
        return cls(V=V, get_sat_props=get_sat_props, inj=inj, vnt=vnt, T=Dynamic(T0), m=Dynamic(m0))

    @tmethod
    def __call__(self, cmbr, env):
        """Update dependent states"""
        _sat_vap_blowdown = partial(sat_vap_blowdown, get_sat_props=self.get_sat_props)
        tnk, inj, vnt = self, self.inj, self.vnt # Aliases
        
        # Find fluid thermophysical properties
        fld = tnk.get_sat_props(T)
        tnk.rho_v, tnk.rho_l, tnk.P = fld.rho_v, fld.rho_l, fld.Pv
        
        # Use analytical derivative to get saturation pressure and densities derivative w.r.t. temperature
        dP_dT      = jax.grad(lambda T: tnk.get_sat_props(T).Pv     )(T)
        drho_v__dT = jax.grad(lambda T: tnk.get_sat_props(T).rho_v)(T)
        drho_l__dT = jax.grad(lambda T: tnk.get_sat_props(T).rho_l)(T)
        
        # Get mass of fluid currently in the phases
        tnk.m_l = jnp.maximum((tnk.V - (tnk.m/fld.rho_v))/ ((1/fld.rho_l)-(1/fld.rho_v)), 0.0)
        tnk.m_v = tnk.m - tnk.m_l
        
        # Update injector and vent flow rates
        vnt(tnk, fld, env)
        inj(tnk, fld, cmbr)

        # Total loss rate of fluid = base injected rate + vent rate
        tnk.mdot = -(inj.mdot + vnt.mdot)
        
        # Add vent flow rate to injector rate if plumbed to chamber
        if vnt.S == 2:
            inj.mdot = inj.mdot + vnt.mdot
        
        # Get temperature and pressure rates at various stages of blowdown
        tnk.Tdot, tnk.Pdot = cond(tnk.m <= 0.0, lambda: (0.0, 0.0),
            lambda: cond(tnk.m_l > 0.0,
                lambda: sat_liq_blowdown (tnk, fld, dP_dT, drho_v__dT, drho_l__dT),
                lambda: _sat_vap_blowdown(tnk, fld, dP_dT),
            ),
        )
        
        # TODO: should we?
        # Set temperature rate to 0 if outside supported range

    @tmethod
    def increment(self):
        """Postprocessing, including limiting, after a full time step"""
        tnk = self # Aliases

        # Limit mass to a tenable value
        tnk.m = jnp.maximum(tnk.m, 0.0)
        # Record needed for avg pressure drop blowdown model
        tnk.Pdot_sum = tnk.Pdot_sum + tnk.Pdot
        tnk.Pdot_N = tnk.Pdot_N + 1
