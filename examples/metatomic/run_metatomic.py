"""Run a pet-jax checkpoint as a metatomic model (metatomic-core ``JaxModel``).

    python run_metatomic.py ../../tests/assets/checkpoints/pet-mad-xs

The model is driven the way an engine drives any metatomic model: it says which
pair list it needs, the engine builds the systems with that list, and
``execute_model`` returns the energy with its gradients.
"""

import numpy as np
import jax

import sys

from ase.build import bulk
from metatomic import Quantity, execute_model

from petjax.metatomic import UPETModel, system_from_atoms

jax.config.update("jax_enable_x64", True)

model = UPETModel.from_checkpoint(sys.argv[1], dtype="float64")
capabilities = model.capabilities()
print(f"interaction range {capabilities.interaction_range} Å, dtype {capabilities.dtype}")

atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
atoms.rattle(0.05, seed=0)
(options,) = model.requested_pair_lists(capabilities.outputs)
system = system_from_atoms(atoms, options)

request = Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions", "strain"]
)
(energy,) = execute_model(model, [system], [request], check_consistency=True)
block = energy.block()

forces = -np.asarray(block.gradient("positions").values).reshape(-1, 3)
virial = np.asarray(block.gradient("strain").values).reshape(3, 3)
print(f"energy        {float(block.values[0, 0]):.6f} eV")
print(f"max |force|   {np.abs(forces).max():.6f} eV/Å")
print(f"stress (diag) {np.diag(virial) / atoms.get_volume()} eV/Å³")
