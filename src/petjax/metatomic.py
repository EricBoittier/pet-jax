"""uPET as a metatomic model, through ``metatomic.JaxModel`` (metatomic-core).

``UPETModel`` runs the same forward as ``UPETCalculator`` (adaptive selection,
pack, ``UPET``), but on the systems an engine hands to a metatomic model: the
engine builds the full pair list at the trained cutoff, the model reads the pair
vectors off it, and ``JaxModel`` takes the forces and stress by autodiff of the
energy through those vectors. Any metatomic engine can then drive a pet-jax
checkpoint. The outputs are the metatomic ones: ``energy`` (per system or per
atom, with ``positions`` / ``strain`` gradients), and the direct heads as
``non_conservative_force`` / ``non_conservative_stress``. Conditioned
checkpoints read the ``charge`` and ``spin_multiplicity`` inputs.

Requires ``metatomic-core`` with ``JaxModel``; not imported by the package.
"""

import numpy as np
import jax
import jax.numpy as jnp

from metatensor import Labels
from metatomic import (
    JaxBlock,
    JaxModel,
    JaxTensorMap,
    ModelCapabilities,
    ModelMetadata,
    PairListOptions,
    Quantity,
)

from .select import select_edges, truncate_edges
from .structure import _bucket_or, _get_corresponding_edges
from .utils import cast_floats

ENERGY = Quantity(name="energy", unit="eV", sample_kind="system")
FORCE = Quantity(name="non_conservative_force", unit="eV/Angstrom", sample_kind="atom")
STRESS = Quantity(
    name="non_conservative_stress", unit="eV/Angstrom^3", sample_kind="system"
)
CHARGE = Quantity(name="charge", unit="e", sample_kind="system")
SPIN = Quantity(name="spin_multiplicity", unit="", sample_kind="system")


class UPETModel(JaxModel):
    """A pet-jax ``UPET`` as a metatomic model.

    ``bucket_strategy`` snaps the pair axis and ``k_sel`` to buckets (as in
    ``UPETCalculator``), so the jitted forward only retraces when a bucket
    changes. ``k_sel`` is sized on the host from each call's geometry, so the
    selection never overflows it; ``extra_neighbors`` pads it so that a few new
    neighbours do not move it to the next bucket.

    The direct heads of ``model`` (``direct_forces`` / ``direct_stress``) are
    offered as the non-conservative outputs, and only run when one of them is
    requested; ``from_checkpoint`` turns on the heads the checkpoint has.
    """

    def __init__(
        self, model, params, metadata, dtype="float32", bucket_strategy="multiples",
        extra_neighbors=4,
    ):  # fmt: skip
        if dtype == "float64" and not jax.config.read("jax_enable_x64"):
            raise ValueError(
                "UPETModel: dtype='float64' needs "
                "`jax.config.update('jax_enable_x64', True)` first"
            )
        self._model = model
        self._dtype = dtype
        self._params = cast_floats(params, jnp.dtype(dtype))
        self._config = metadata["config"]
        self._shifts = {int(z): float(v) for z, v in metadata["shifts"].items()}
        self._bucket_strategy = bucket_strategy
        self._extra_neighbors = extra_neighbors
        self._pairs = PairListOptions(
            cutoff=float(model.cutoff), full_list=True, strict=True
        )
        # One jitted forward per set of heads that runs, built on first use.
        self._forward_fns = {}

    @classmethod
    def from_checkpoint(cls, folder, **kwargs):
        from .convert import load_checkpoint
        from .model import UPET

        params, metadata = load_checkpoint(folder)
        heads = params["params"] if "params" in params else params
        model = UPET(
            **metadata["config"],
            direct_forces="forces_head" in heads,
            direct_stress="stress_head" in heads,
        )
        return cls(model, params, metadata, **kwargs)

    def capabilities(self):
        heads = [(self._model.direct_forces, FORCE), (self._model.direct_stress, STRESS)]
        return ModelCapabilities(
            atomic_types=sorted(self._shifts),
            # Each GNN layer reaches one cutoff further.
            interaction_range=self._config["num_gnn_layers"] * self._config["cutoff"],
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype=self._dtype,
            outputs=[ENERGY] + [quantity for has, quantity in heads if has],
        )

    def metadata(self):
        return ModelMetadata(name="pet-jax uPET")

    def requested_pair_lists(self, requested_outputs):
        return [self._pairs]

    def requested_inputs(self, requested_outputs):
        return [CHARGE, SPIN] if self._model.system_conditioning else []

    def forward(self, systems, selected_atoms, outputs):
        names = {request.name for request in outputs}
        unknown = names - {ENERGY.name, FORCE.name, STRESS.name}
        if unknown:
            raise ValueError(f"UPETModel: unknown outputs {sorted(unknown)}")
        heads = (FORCE.name in names, STRESS.name in names)
        predictions = [self._predict(system, heads) for system in systems]
        masks = [_selected(selected_atoms, s, len(p["energy"])) for s, p in enumerate(predictions)]  # fmt: skip
        return [_output(request, predictions, masks) for request in outputs]

    def _predict(self, system, heads):
        """Per-atom ``energy`` (composition shifts included) and the requested
        heads of one system: ``forces`` without their net force, as the
        calculator does, and the per-atom ``stress`` terms divided by the
        volume, so their sum over atoms is the stress."""
        types = np.asarray(system.types)
        try:
            shifts = np.array([self._shifts[int(z)] for z in types])
        except KeyError as exc:
            raise ValueError(
                f"UPETModel: system contains atomic number {exc.args[0]}, "
                "which the model was not trained on"
            ) from exc

        if heads not in self._forward_fns:
            forces, stress = heads
            model = self._model.clone(direct_forces=forces, direct_stress=stress)
            self._forward_fns[heads] = get_forward_fn(model)
        out, overflow = self._forward_fns[heads](self._params, **self._inputs(system))
        # k_sel is sized from this geometry, so this is a bug, not a resize.
        if bool(overflow):
            raise RuntimeError("UPETModel: adaptive selection overflowed k_sel")

        n = len(types)
        energy = out["energy"][:n] + jnp.asarray(shifts, dtype=out["energy"].dtype)
        result = {"energy": energy}
        if heads[0]:
            forces = out["forces"][:n]
            result["forces"] = forces - forces.mean(axis=0)
        if heads[1]:
            if not np.all(np.asarray(system.pbc)):
                raise ValueError("UPETModel: the stress needs a fully periodic system")
            result["stress"] = out["stress"][:n] / jnp.abs(jnp.linalg.det(system.cell))
        return result

    def _inputs(self, system):
        """The padded flat pair list of ``system`` (``to_structure``'s layout,
        with the engine's pair vectors in place of the cell shifts), the
        ``k_sel_sizer`` carrier and the conditioning inputs."""
        n_atoms = len(system)
        block = system.pairs(self._pairs)
        samples = np.asarray(block.samples.values)
        # Grouped by center, as truncate_edges requires.
        order = np.argsort(samples[:, 0], kind="stable")
        i, j, S = samples[order, 0], samples[order, 1], samples[order, 2:5]
        vectors = block.values.reshape(-1, 3)[order]

        N_padded = n_atoms + 1
        dummy_atom = N_padded - 1
        n_pair_raw = len(i)
        # +1 keeps the sentinel slot at n_pair_padded - 1 empty (see to_structure).
        n_pair_padded = _bucket_or(n_pair_raw + 1, self._bucket_strategy)
        n_pad = n_pair_padded - n_pair_raw

        def pad(array, value):
            return np.concatenate([array, np.full(n_pad, value, dtype=array.dtype)])

        reverse = np.arange(n_pair_padded, dtype=np.int32)
        reverse[:n_pair_raw] = _get_corresponding_edges(i, j, S)
        atomic_numbers = np.zeros(N_padded, dtype=np.int32)
        atomic_numbers[:n_atoms] = np.asarray(system.types)
        inputs = {
            "R_ij": jnp.concatenate([vectors, jnp.zeros((n_pad, 3), vectors.dtype)]),
            "centers": pad(i.astype(np.int32), dummy_atom),
            "others": pad(j.astype(np.int32), dummy_atom),
            "reverse": reverse,
            "pair_mask": np.arange(n_pair_padded) < n_pair_raw,
            "atomic_numbers": atomic_numbers,
            "atom_mask": np.arange(N_padded) < n_atoms,
        }
        inputs["k_sel_sizer"] = np.zeros(self._k_sel(inputs), dtype=bool)
        if self._model.system_conditioning:
            inputs.update(self._conditioning(system, N_padded))
        return inputs

    def _k_sel(self, inputs):
        """``k_sel`` for this geometry: the largest selected count (from
        ``select_edges`` on the host), plus ``extra_neighbors``, bucketed."""
        # The values under JaxModel's autodiff are concrete: it runs eagerly.
        R_ij = np.asarray(jax.lax.stop_gradient(inputs["R_ij"]))
        # select_edges derives R_ij from (positions, cell, cell_shifts); with
        # zero positions and an identity cell, R_ij as the shifts reproduces
        # the engine's vectors exactly.
        N_padded = len(inputs["atom_mask"])
        selection = select_edges(
            {
                "positions": np.zeros((N_padded, 3), dtype=R_ij.dtype),
                "cell": np.eye(3, dtype=R_ij.dtype),
                "centers": inputs["centers"],
                "others": inputs["others"],
                "cell_shifts": R_ij,
                "pair_mask": inputs["pair_mask"],
            },
            self._model.num_neighbors_adaptive,
            self._model.cutoff,
            self._model.cutoff_width_adaptive,
            method=self._model.adaptive_cutoff_method,
        )
        k_sel_actual = max(int(selection.counts.max(initial=0)), 1)
        # T = k_sel edge tokens + 1 central-atom token, bucketed as in the calculator.
        tokens = _bucket_or(k_sel_actual + 1 + self._extra_neighbors, self._bucket_strategy)
        return tokens - 1

    def _conditioning(self, system, N_padded):
        """Per-atom ``charge`` / ``spin_multiplicity`` from the system inputs,
        checked against the embedding tables (inside JIT an out-of-range row
        would give NaN, not an error)."""
        limits = {
            "charge": (-self._model.max_charge, self._model.max_charge),
            "spin_multiplicity": (1, self._model.max_spin_multiplicity),
        }
        inputs = {}
        for name, (low, high) in limits.items():
            value = float(np.asarray(system.custom_data(name).block().values).item())
            if value != int(value) or not low <= value <= high:
                raise ValueError(f"UPETModel: {name}={value} outside [{low}, {high}]")
            inputs[name] = np.full(N_padded, int(value), dtype=np.int32)
        return inputs


def get_forward_fn(model):
    """Build the JIT-compiled ``forward_fn(params, R_ij, centers, ...) ->
    (out, overflow)`` of ``UPETModel``: ``truncate_edges`` and the model, on
    precomputed pair vectors. ``out`` holds the per-atom ``energy`` ``[N_padded]``
    (scaled, masked, without composition shifts) and the outputs of the
    direct heads ``model`` has on; ``k_sel`` is read off the ``k_sel_sizer``
    carrier."""

    @jax.jit
    def forward_fn(
        params, R_ij, centers, others, reverse, pair_mask, atomic_numbers, atom_mask,
        k_sel_sizer, charge=None, spin_multiplicity=None,
    ):  # fmt: skip
        truncated, overflow = truncate_edges(
            R_ij,
            centers,
            others,
            reverse,
            pair_mask,
            atomic_numbers,
            atom_mask,
            k_sel_sizer.shape[-1],
            model.num_neighbors_adaptive,
            model.cutoff,
            model.cutoff_width_adaptive,
            method=model.adaptive_cutoff_method,
        )
        out = model.apply(
            params, **truncated, charge=charge, spin_multiplicity=spin_multiplicity
        )
        return (out if isinstance(out, dict) else {"energy": out}), overflow

    return forward_fn


def system_from_atoms(atoms, options, inputs=(), dtype=np.float64):
    """A metatomic ``System`` for ASE ``atoms``, with the pair list ``options``
    (from ``model.requested_pair_lists``) computed by ``vesin`` and the
    ``charge`` / ``spin_multiplicity`` ``inputs`` (from
    ``model.requested_inputs``) read off ``atoms.info``: what an engine does
    before calling the model."""
    from metatensor import TensorBlock, TensorMap
    from metatomic import System
    from vesin import NeighborList

    from .structure import read_conditioning

    system = System(
        "angstrom",
        atoms.numbers.astype(np.int32),
        atoms.positions.astype(dtype),
        # metatomic wants the cell vectors of non-periodic directions zeroed.
        np.where(atoms.pbc[:, None], atoms.cell[:], 0.0).astype(dtype),
        atoms.pbc.copy(),
    )
    nl = NeighborList(cutoff=options.cutoff, full_list=options.full_list)
    i, j, S, D = nl.compute(
        points=atoms.positions,
        box=atoms.cell[:],
        periodic=atoms.pbc.tolist(),
        quantities="ijSD",
    )
    pairs = TensorBlock(
        values=D.astype(dtype).reshape(-1, 3, 1),
        samples=Labels(
            ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            np.column_stack([i, j, S]).astype(np.int32),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    system.add_pairs(options, pairs)

    values = dict(zip(("charge", "spin_multiplicity"), read_conditioning(atoms)))
    for quantity in inputs:
        block = TensorBlock(
            values=np.array([[values[quantity.name]]], dtype=dtype),
            samples=Labels.range("system", 1),
            components=[],
            properties=Labels.range(quantity.name, 1),
        )
        system.add_custom_data(quantity.name, TensorMap(Labels.range("_", 1), [block]))
    return system


# -- outputs --


def _selected(selected_atoms, system, n_atoms):
    """Mask of the atoms of ``system`` in ``selected_atoms`` (all if ``None``)."""
    if selected_atoms is None:
        return np.ones(n_atoms, dtype=bool)
    values = np.asarray(selected_atoms.values)
    return np.isin(np.arange(n_atoms), values[values[:, 0] == system, 1])


def _output(request, predictions, masks):
    """The ``request``-ed ``TensorMap`` from the per-system predictions,
    restricted to the selected atoms."""
    per_atom = request.sample_kind == "atom"
    atom_samples = Labels(
        ["system", "atom"],
        np.array(
            [[s, a] for s, m in enumerate(masks) for a in np.flatnonzero(m)],
            dtype=np.int32,
        ).reshape(-1, 2),
    )
    system_samples = Labels.range("system", len(predictions))

    if request.name == FORCE.name:
        values = jnp.concatenate([p["forces"][m] for p, m in zip(predictions, masks)])
        components = [Labels.range("xyz", 3)]
        values, samples = values[..., None], atom_samples
    elif request.name == STRESS.name:
        values = jnp.stack([p["stress"][m].sum(0) for p, m in zip(predictions, masks)])
        components = [Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)]
        values, samples = values[..., None], system_samples
    elif per_atom:
        values = jnp.concatenate([p["energy"][m] for p, m in zip(predictions, masks)])
        components, samples = [], atom_samples
        values = values[:, None]
    else:
        values = jnp.stack([p["energy"][m].sum() for p, m in zip(predictions, masks)])
        components, samples = [], system_samples
        values = values[:, None]

    block = JaxBlock(
        values=values,
        samples=samples,
        components=components,
        properties=Labels.range(request.name, 1),
    )
    return JaxTensorMap(Labels.range("_", 1), [block])
