"""UPETModel (metatomic JaxModel) against UPETCalculator on the mini set."""

import numpy as np
import jax

import ase.io
import pytest

metatomic = pytest.importorskip("metatomic")
pytest.importorskip("metatomic._model._jax", reason="metatomic-core without JaxModel")

from metatensor import Labels  # noqa: E402
from metatomic import Quantity, execute_model  # noqa: E402

from petjax import UPETCalculator  # noqa: E402
from petjax.metatomic import UPETModel, system_from_atoms  # noqa: E402

CHECKPOINT = "pet-mad-xs"


@pytest.fixture(scope="module")
def setup(assets_dir):
    folder = assets_dir / "checkpoints" / CHECKPOINT
    if not folder.exists():
        pytest.skip("run `tox -e fetch-checkpoints` first")
    jax.config.update("jax_enable_x64", True)
    model = UPETModel.from_checkpoint(folder, dtype="float64")
    calculator = UPETCalculator.from_checkpoint(folder, default_dtype="float64")
    structures = ase.io.read(assets_dir / "test_mini.xyz", ":")
    return model, calculator, structures


def test_matches_calculator(setup):
    model, calculator, structures = setup
    (options,) = model.requested_pair_lists(model.capabilities().outputs)
    request = Quantity(
        name="energy", unit="eV", sample_kind="system", gradients=["positions", "strain"]
    )
    for atoms in structures:
        system = system_from_atoms(atoms, options)
        (energy,) = execute_model(model, [system], [request], check_consistency=True)
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


def test_per_atom_and_selected_atoms(setup):
    model, _, structures = setup
    (options,) = model.requested_pair_lists(model.capabilities().outputs)
    systems = [system_from_atoms(atoms, options) for atoms in structures[:2]]

    per_system = Quantity(name="energy", unit="eV", sample_kind="system")
    per_atom = Quantity(name="energy", unit="eV", sample_kind="atom")
    total, atomic = execute_model(model, systems, [per_system, per_atom])
    atomic = atomic.block()
    sums = [
        np.asarray(atomic.values)[np.asarray(atomic.samples.column("system")) == s].sum()
        for s in range(len(systems))
    ]
    np.testing.assert_allclose(np.asarray(total.block().values)[:, 0], sums, rtol=1e-10)

    selected = Labels(["system", "atom"], np.array([[0, 0], [1, 1]], dtype=np.int32))
    (partial,) = execute_model(model, systems, [per_atom], selected_atoms=selected)
    assert partial.block().samples == selected
