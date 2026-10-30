# pet-jax as a metatomic model

`petjax.metatomic.UPETModel` wraps a pet-jax checkpoint as a `metatomic.JaxModel`, so any metatomic engine can drive it. It needs `metatomic-core` with `JaxModel` (currently [EricBoittier/metatomic#15](https://github.com/EricBoittier/metatomic/pull/15)), and `metatensor-core` with JAX array support ([EricBoittier/metatensor#1](https://github.com/EricBoittier/metatensor/pull/1)) to keep the outputs as JAX arrays; without it they are copied to NumPy.

```bash
python run_metatomic.py ../../tests/assets/checkpoints/pet-mad-xs
```

The model asks for one full pair list at the trained cutoff. Inside `forward` it sorts and pads that list into the layout `UPETCalculator` uses, sizes `k_sel` on the host from the current geometry (so the selection never overflows), and runs the adaptive selection and `UPET` on the engine's pair vectors in one jitted call. `JaxModel` differentiates through the pair vectors for the `positions` and `strain` gradients.

Outputs: `energy` (per system or per atom, with `positions` / `strain` gradients from autodiff), and for checkpoints with direct heads `non_conservative_force` and `non_conservative_stress` (fully periodic systems only). Conditioned checkpoints (PET-OMol) request the `charge` and `spin_multiplicity` inputs; `system_from_atoms` reads them off `atoms.info`, as `UPETCalculator` does.
