"""UPETModel (metatomic JaxModel) against UPETCalculator on the mini sets."""

import numpy as np
import jax

import ase.io
import pytest

pytest.importorskip("metatomic._model._jax", reason="metatomic-core without JaxModel")

from metatensor import Labels  # noqa: E402
from metatomic import Quantity, execute_model  # noqa: E402

from petjax import UPETCalculator  # noqa: E402
from petjax.metatomic import UPETModel, system_from_atoms  # noqa: E402

ENERGY = Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions", "strain"]
)
FORCE = Quantity(name="non_conservative_force", unit="eV/Angstrom", sample_kind="atom")
STRESS = Quantity(
    name="non_conservative_stress", unit="eV/Angstrom^3", sample_kind="system"
)


@pytest.fixture(scope="module")
def setup(mini_release):
    checkpoint, structures, _, _ = mini_release
    jax.config.update("jax_enable_x64", True)
    model = UPETModel.from_checkpoint(checkpoint, dtype="float64")
    structures = ase.io.read(structures, ":")
    return checkpoint, model, structures


def systems(model, structures):
    (options,) = model.requested_pair_lists([])
    inputs = model.requested_inputs([])
    return [system_from_atoms(atoms, options, inputs) for atoms in structures]


def test_conservative(setup):
    checkpoint, model, structures = setup
    calculator = UPETCalculator.from_checkpoint(checkpoint, default_dtype="float64")
    for atoms, system in zip(structures, systems(model, structures)):
        (energy,) = execute_model(model, [system], [ENERGY], check_consistency=True)
        block = energy.block()

        atoms.calc = calculator
        np.testing.assert_allclose(
            block.values[0, 0], atoms.get_potential_energy(), rtol=1e-10
        )
        forces = -np.asarray(block.gradient("positions").values).reshape(-1, 3)
        np.testing.assert_allclose(forces, atoms.get_forces(), atol=1e-8)
        if atoms.pbc.all():
            stress = np.asarray(block.gradient("strain").values).reshape(3, 3)
            # The calculator reports the symmetrised (Voigt) stress.
            stress = (stress + stress.T) / (2 * atoms.get_volume())
            np.testing.assert_allclose(stress, atoms.get_stress(voigt=False), atol=1e-8)


def test_direct_heads(setup):
    checkpoint, model, structures = setup
    has_stress = model._model.direct_stress
    calculator = UPETCalculator.from_checkpoint(
        checkpoint, direct_forces=True, direct_stress=has_stress, default_dtype="float64"
    )
    for atoms, system in zip(structures, systems(model, structures)):
        periodic = atoms.pbc.all()
        requests = [FORCE] + ([STRESS] if has_stress and periodic else [])
        outputs = execute_model(model, [system], requests, check_consistency=True)

        atoms.calc = calculator
        forces = np.asarray(outputs[0].block().values).reshape(-1, 3)
        np.testing.assert_allclose(forces, atoms.get_forces(), atol=1e-8)
        if len(outputs) == 2:
            stress = np.asarray(outputs[1].block().values).reshape(3, 3)
            np.testing.assert_allclose(stress, atoms.get_stress(voigt=False), atol=1e-8)


def test_per_atom_and_selected_atoms(setup):
    _, model, structures = setup
    batch = systems(model, structures[:2])

    per_system = Quantity(name="energy", unit="eV", sample_kind="system")
    per_atom = Quantity(name="energy", unit="eV", sample_kind="atom")
    total, atomic = execute_model(model, batch, [per_system, per_atom])
    atomic = atomic.block()
    owner = np.asarray(atomic.samples.column("system"))
    sums = [np.asarray(atomic.values)[owner == s].sum() for s in range(len(batch))]
    np.testing.assert_allclose(np.asarray(total.block().values)[:, 0], sums, rtol=1e-10)

    selected = Labels(["system", "atom"], np.array([[0, 0], [1, 1]], dtype=np.int32))
    (partial,) = execute_model(model, batch, [per_atom], selected_atoms=selected)
    assert partial.block().samples == selected
