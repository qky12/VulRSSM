# Reproducibility guide

## Code-to-paper map

| Paper component | Location | Notes |
| --- | --- | --- |
| Dreamer V1 training | `implementations/dreamer-v1/main.py` | Uses YAML files under `dreamer/configs/`. |
| VulRSSM / rollout attack (V1) | `implementations/dreamer-v1/mainattack.py` | `R_IAP` is the active rollout-based attack implementation. |
| Experimental attack variants (V1) | `implementations/dreamer-v1/model_attack.py` | Includes FGSM, PGD, TRAP-RSSM, ValRSSM, and rollout objectives retained from experimentation. |
| Dreamer V2 training | `implementations/dreamer-v2/main.py` | Uses the same EasyDreamer-style layout. |
| VulRSSM / rollout attacks (V2) | `implementations/dreamer-v2/mainattack.py` | `R_IAP` and `R_IAP2` cover continuous and discrete latent variants. |
| Policy-level baselines | V1/V2 `mainattack.py` | Value-based FGSM and PGD implementations. |
| Dreamer V3 foundation | `implementations/dreamer-v3/` | Preserved PyTorch implementation; original provenance and instructions are retained. |

The experimental names in source files reflect the research process. The manuscript name **VulRSSM** refers to the imagination-rollout objective implemented by the `R_IAP` family.

## Evaluated tasks

The manuscript reports experiments on the following DeepMind Control Suite tasks:

- `walker_walk`
- `walker_stand`
- `walker_run`
- `cheetah_run`
- `finger_spin`
- `ball_in_cup_catch`
- `reacher_easy`
- `cartpole_swingup`
- `hopper_stand`

EasyDreamer configuration names use hyphens (for example, `dmc-walker-walk.yml`), while Dreamer V3 command-line tasks use underscores (for example, `dmc_walker_walk`).

## Environment isolation

Do not install all three requirement files into one Python environment. They pin incompatible PyTorch, Gym, NumPy, and environment versions. Create one virtual or Conda environment per implementation.

GPU-enabled PyTorch wheels are platform-specific. If the pinned version in a requirements file does not match the local CUDA runtime, install the appropriate PyTorch build first and then install the remaining dependencies.

## Before running an evaluation

1. Select the task configuration.
2. Set the trained checkpoint path expected by the attack script.
3. Confirm which attack block is enabled in `mainattack.py`.
4. Set the L-infinity budget `epsilon`, optimization steps `S`, and imagination horizon `K` to the paper setting being reproduced.
5. Confirm observations are normalized and clipped to `[-0.5, 0.5]`.
6. Record random seeds, checkpoint identity, task, attack parameters, and episode count.
7. Write outputs to an ignored directory such as `runs/` or `results/`.

The principal paper setting for VulRSSM uses iterative optimization and studies sensitivity to the attack horizon. The defense table uses `K=15` and `S=5`; consult the manuscript for the exact budget sweep and task-level reporting protocol.

## Repository hygiene

Checkpoints, TensorBoard events, training logs, generated frames, and local caches are excluded from this repository. Store large public artifacts in a release, an archival dataset, or Git LFS, and document their checksums and download locations.

