"""uPET as a metatomic model, through ``metatomic.JaxModel`` (metatomic-core).

``UPETModel`` runs the same forward as ``UPETCalculator`` (adaptive selection,
pack, ``UPET``), but on the systems an engine hands to a metatomic model: the
engine builds the full pair list at the trained cutoff, the model reads the pair
vectors off it, and ``JaxModel`` takes the forces and stress by autodiff of the
energy through those vectors. Any metatomic engine can then drive a pet-jax
checkpoint, and the outputs are the metatomic ones (``energy``, per system or
per atom, with ``positions`` / ``strain`` gradients).

Requires ``metatomic-core`` with ``JaxModel``; lazy-imported by the package.
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

from .express import Express
from .select import select_edges, truncate_edges
from .structure import _bucket_or, _get_corresponding_edges
from .utils import cast_floats

# Express switches that need inputs or sizing this adapter does not provide yet.
_UNSUPPORTED_EXPRESS = {
    "neighbor_list": "host",
    "seed_cutoffs": False,
    "tokens": "padded",
    "atom_block": None,
}


class UPETModel(JaxModel):
    """A pet-jax ``UPET`` as a metatomic model.

    ``bucket_strategy`` snaps the pair axis and ``k_sel`` to buckets (as in
    ``UPETCalculator``), so the jitted forward only retraces when a bucket
    changes. ``k_sel`` is sized on the host from each call's geometry, so the
    selection never overflows it; ``extra_neighbors`` pads it so that a few new
    neighbours do not move it to the next bucket. ``express`` takes the
    performance switches that need nothing beyond the pair list (solver
    variants, ``reverse_unique``, ``remat``, ...).
    """

    def __init__(
        self,
        model,
        params,
        metadata,
        dtype="float32",
        bucket_strategy="multiples",
        extra_neighbors=4,
        express=None,
    ):
        if model.direct_forces or model.direct_stress:
            raise ValueError("UPETModel: direct force/stress heads are not supported")
        if model.system_conditioning:
            raise ValueError("UPETModel: charge/spin conditioned models are not supported")

        self.express = Express.from_any(express)
        for name, stock in _UNSUPPORTED_EXPRESS.items():
            if getattr(self.express, name) != stock:
                raise ValueError(f"UPETModel: express {name} is not supported")
        updates = self.express.model_updates()
        self._model = model.clone(**updates) if updates else model

        if dtype == "float64" and not jax.config.read("jax_enable_x64"):
            raise ValueError(
                "UPETModel: dtype='float64' needs "
                "`jax.config.update('jax_enable_x64', True)` first"
            )
        self._dtype = dtype
        self._params = cast_floats(params, jnp.dtype(dtype))
        self._config = metadata["config"]
        self._shifts = {int(z): float(v) for z, v in metadata["shifts"].items()}
        self._bucket_strategy = bucket_strategy
        self._extra_neighbors = extra_neighbors
        self._pairs = PairListOptions(
            cutoff=float(model.cutoff), full_list=True, strict=True
        )
        self._energy_fn = get_atomic_energies_fn(self._model, express=self.express)

    @classmethod
    def from_checkpoint(cls, folder, **kwargs):
        from .convert import load_checkpoint
        from .model import UPET

        params, metadata = load_checkpoint(folder)
        return cls(UPET(**metadata["config"]), params, metadata, **kwargs)

    def capabilities(self):
        return ModelCapabilities(
            atomic_types=sorted(self._shifts),
            # Each GNN layer reaches one cutoff further.
            interaction_range=self._config["num_gnn_layers"] * self._config["cutoff"],
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype=self._dtype,
            outputs=[Quantity(name="energy", unit="eV", sample_kind="system")],
        )

    def metadata(self):
        return ModelMetadata(name="pet-jax uPET")

    def requested_pair_lists(self, requested_outputs):
        return [self._pairs]

    def forward(self, systems, selected_atoms, outputs):
        energies = [self._atomic_energies(system) for system in systems]

        results = []
        for request in outputs:
            if request.name != "energy":
                raise ValueError(f"UPETModel: unknown output '{request.name}'")
            results.append(_energy_tensor(energies, selected_atoms, request))
        return results

    def _atomic_energies(self, system):
        """Per-atom energies ``[n_atoms]`` of one system, composition shifts
        included."""
        types = np.asarray(system.types)
        try:
            shifts = np.array([self._shifts[int(z)] for z in types])
        except KeyError as exc:
            raise ValueError(
                f"UPETModel: system contains atomic number {exc.args[0]}, "
                "which the model was not trained on"
            ) from exc

        inputs = self._inputs(system)
        energies, overflow = self._energy_fn(self._params, **inputs)
        # k_sel is sized from this geometry, so this is a bug, not a resize.
        if bool(overflow):
            raise RuntimeError("UPETModel: adaptive selection overflowed k_sel")
        return energies[: len(types)] + jnp.asarray(shifts, dtype=energies.dtype)

    def _inputs(self, system):
        """The padded flat pair list of ``system`` (``to_structure``'s layout,
        with the engine's pair vectors in place of the cell shifts) and the
        ``k_sel_sizer`` carrier."""
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

        centers = pad(i.astype(np.int32), dummy_atom)
        others = pad(j.astype(np.int32), dummy_atom)
        reverse = np.arange(n_pair_padded, dtype=np.int32)
        reverse[:n_pair_raw] = _get_corresponding_edges(i, j, S)
        pair_mask = np.arange(n_pair_padded) < n_pair_raw
        atomic_numbers = np.zeros(N_padded, dtype=np.int32)
        atomic_numbers[:n_atoms] = np.asarray(system.types)
        atom_mask = np.arange(N_padded) < n_atoms
        R_ij = jnp.concatenate([vectors, jnp.zeros((n_pad, 3), dtype=vectors.dtype)])

        inputs = {
            "R_ij": R_ij,
            "centers": centers,
            "others": others,
            "reverse": reverse,
            "pair_mask": pair_mask,
            "atomic_numbers": atomic_numbers,
            "atom_mask": atom_mask,
        }
        inputs["k_sel_sizer"] = np.zeros(self._k_sel(inputs, S), dtype=bool)
        return inputs

    def _k_sel(self, inputs, S):
        """``k_sel`` for this geometry: the largest selected count (from
        ``select_edges`` on the host), plus ``extra_neighbors``, bucketed."""
        # The values under JaxModel's autodiff are concrete: it runs eagerly.
        R_ij = np.asarray(jax.lax.stop_gradient(inputs["R_ij"]))
        n_pair_padded = len(R_ij)
        cell_shifts = np.zeros((n_pair_padded, 3))
        cell_shifts[: len(S)] = S
        # select_edges derives R_ij from (positions, cell, cell_shifts); with
        # zero positions and an identity cell, putting R_ij in the shifts
        # reproduces the engine's vectors exactly.
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
            express=self.express,
        )
        k_sel_actual = max(int(selection.counts.max(initial=0)), 1)
        # T = k_sel edge tokens + 1 central-atom token, bucketed as in the calculator.
        tokens = _bucket_or(k_sel_actual + 1 + self._extra_neighbors, self._bucket_strategy)
        return tokens - 1


def get_atomic_energies_fn(model, express=None):
    """Build the JIT-compiled ``energies_fn(params, R_ij, centers, ...) ->
    (energies, overflow)`` of ``UPETModel``: ``truncate_edges`` and the model,
    on precomputed pair vectors. ``energies`` are the per-atom energies
    ``[N_padded]`` (scaled, masked, without composition shifts); ``k_sel`` is
    read off the ``k_sel_sizer`` carrier."""
    express = Express.from_any(express)

    @jax.jit
    def energies_fn(
        params, R_ij, centers, others, reverse, pair_mask, atomic_numbers, atom_mask,
        k_sel_sizer,
    ):  # fmt: skip
        truncated, truncation = truncate_edges(
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
            express=express,
        )
        return model.apply(params, **truncated), truncation.overflow

    return energies_fn


def system_from_atoms(atoms, options, dtype=np.float64):
    """A metatomic ``System`` for ASE ``atoms``, with the pair list ``options``
    (from ``model.requested_pair_lists``) computed by ``vesin``: what an engine
    does before calling the model."""
    from metatensor import TensorBlock
    from metatomic import System
    from vesin import NeighborList

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
    return system


def _energy_tensor(energies, selected_atoms, request):
    """The ``energy`` output from the per-atom energies of each system,
    restricted to ``selected_atoms``, per system or per atom."""
    masks = []
    for system, e in enumerate(energies):
        mask = np.ones(e.shape[0], dtype=bool)
        if selected_atoms is not None:
            values = np.asarray(selected_atoms.values)
            atoms = values[values[:, 0] == system, 1]
            mask = np.isin(np.arange(e.shape[0]), atoms)
        masks.append(mask)

    if request.sample_kind == "atom":
        values = jnp.concatenate([e[m] for e, m in zip(energies, masks)])
        samples = Labels(
            ["system", "atom"],
            np.array(
                [[s, a] for s, m in enumerate(masks) for a in np.flatnonzero(m)],
                dtype=np.int32,
            ).reshape(-1, 2),
        )
    else:
        values = jnp.stack([jnp.sum(e[m]) for e, m in zip(energies, masks)])
        samples = Labels.range("system", len(energies))

    block = JaxBlock(
        values=values.reshape(-1, 1),
        samples=samples,
        components=[],
        properties=Labels.range("energy", 1),
    )
    return JaxTensorMap(Labels.range("_", 1), [block])
