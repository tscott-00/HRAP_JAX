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

"""Provide classes for common rocket motor types in amateur rocketry"""

import jax
import jax.numpy as jnp
# from jax.lax import cond
from tracept import tclass, tmethod, Placeholder, Dynamic, Derivative

from hrap.tank import SatTank
from hrap.chamber import Chamber

@tclass
class SelfPressurizedHybrid:
    # Individual components
    tnk: SatTank
    cmbr: Chamber

    @tmethod
    def __call__(self):
        """Update dependent states of all components."""

        self.tnk(self.cmbr, self.env)
        self.cmbr(self.tnk, self.noz, self.env, self.grn)

    @tmethod
    def prep(self):
        """Automatically deterime any dependent defaults across all components requested on initialization.
    
        This is one of the few functions that is incompatible with JIT compilation (modifies some fixed values).
        """

        self.cmbr.prep(self.env, self.grn)

    @tmethod
    def increment(self):
        """Postprocessing, including limiting to tenable values, after a full time step"""

        self.tnk.increment()
        self.cmbr.increment(env)
