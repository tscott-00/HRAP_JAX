# Purpose: Provide throat mass flow rate and nozzle exit conditions (using frozen equilibrium from chamber)
# Authors: Drew Nickel, Thomas A. Scott

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import cond
from tracept import tclass, tmethod, Dynamic, Derivative

from hrap.core import store_x, make_part

@tclass
class FrozenCDNozzle:
    # Fixed variables
    thrt:       float                        #: throat diameter
    ER:       float                        #: exit area to throat area ratio
    Cd:       float                        #: coefficient of discharge
    eff:       float                        #: efficiency
    # Dependent variables
    mdot:      Dynamic  = None       #: mass flow rate
    Me:      Dynamic  = None       #: exit Mach
    Pe:      Dynamic  = None       #: exit pressure
    thrust:      Dynamic  = None       #: total thrust
    
    @classmethod
    def new(cls, ):
        """ """

    @tmethod
    def __call__(self, cmbr, env):
        """Update dependent states"""
        noz, k = self, cmbr.k # Aliases

        # Nozzle throat area
        A_thrt = np.pi/4 * noz.thrt**2

        # Nozzle mass flow rate
        mdot = cond(
            cmbr.P <= env.Pa,
            lambda: 0.0,
            lambda: cmbr.P*noz.Cd * A_thrt/cmbr.cstar,
        )
        
        # Exit Mach
        noz.Me = FrozenCDNozzle.M_solve(cmbr.k, noz.ER)

        # Exit pressure
        noz.Pe = cmbr.P*(1+0.5*(k-1)*noz.Me**2)**(-k/(k-1))
        
        # TODO: Speed of sound is involved here and specific heat ratio is not correct to use, rather isentropic exponent, but they barely differ so perhaps not worth it
        Cf = jnp.sqrt(((2*k**2)/(k-1))*(2/(k+1))**((k+1)/ \
            (k-1))*(1-(noz.Pe/cmbr.P)**((k-1)/k)))+ \
            ((noz.Pe-env.Pa)*(A_thrt*noz.ER))/ \
            (cmbr.P*A_thrt)
        
        # Thrust
        noz.thrust = noz.eff*Cf*A_thrt*cmbr.P*noz.Cd

    @tmethod
    def increment(self):
        """Limit thrust to positive"""
        noz = self # Aliases

        noz.thrust = jnp.maximum(noz.thrust, 0.0)

    @staticmethod
    def M_solve(k, ER):
        """Solve for exit mach.

        Args:
          k: specific heat ratio
          ER: exit to throat area ratio
        Returns:
          exit mach number
        """
        def get_error(Me, k, ER):
            error = ((k+1)/2)**(-(k+1)/ \
                (2*(k-1)))*(1+(k-1)/2*Me**2)** \
                ((k+1)/(2*(k-1)))/Me- \
                ER
            
            return error
        get_derror_dMe = jax.grad(get_error)
        
        def newton_step(val):
            _, Me, k, ER, i = val
            error = get_error(Me, k, ER)
            Me -= error / get_derror_dMe(Me, k, ER)
            return (error, Me, k, ER, i+1)
        result = jax.lax.while_loop(lambda val: (jnp.abs(val[0]) > 1E-8) & (val[4]<10), newton_step, (1.0, 3.0, k, ER, 0))
        
        return result[1]
