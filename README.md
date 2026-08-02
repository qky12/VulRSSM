# VulRSSM


VulRSSM is a white-box adversarial evaluation method for RSSM-based world models. Instead of only changing the current policy output, it optimizes a bounded observation perturbation over imagined trajectories to expose temporal error propagation in latent dynamics. The repository also contains policy-level FGSM/PGD baselines and experimental code for the Dreamer family.

## Highlights

- RSSM-aware attacks that target multi-step imagination.
- Policy-level FGSM and PGD baselines under matched L-infinity budgets.
- Experiments across Dreamer V1, V2, and V3 codebases.
- DeepMind Control Suite configurations for the tasks used in the study.
- Clean repository layout without checkpoints, logs, rendered frames, or local caches.

## Repository layout

```text
vulrssm/
├── implementations/
│   ├── dreamer-v1/       # Dreamer V1 experiments and VulRSSM attack code
│   ├── dreamer-v2/       # Dreamer V2 experiments and attack baselines
│   └── dreamer-v3/       # PyTorch Dreamer V3 implementation used by the study
├── docs/
│   └── REPRODUCIBILITY.md
├── CITATION.cff
├── CONTRIBUTING.md
└── README.md
```

Each implementation is self-contained because the three variants require different dependency versions. Install and run them in separate environments.

## Quick start

### Dreamer V1

```bash
cd implementations/dreamer-v1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train a clean agent.
python main.py --config dmc-walker-walk

# Run the configured adversarial evaluation.
python mainattack.py --config dmc-walker-walk
```

The main VulRSSM implementation for this variant is in `mainattack.py` (`R_IAP`). Additional experimental attack variants are retained in `model_attack.py` for reproducibility.

### Dreamer V2

```bash
cd implementations/dreamer-v2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py --config dmc-walker-walk
python mainattack.py --config dmc-walker-walk
```

The attack entry point includes continuous- and discrete-latent rollout variants (`R_IAP` and `R_IAP2`) as well as value-based FGSM/PGD baselines.

### Dreamer V3

```bash
cd implementations/dreamer-v3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python dreamer.py \
  --configs dmc_vision \
  --task dmc_walker_walk \
  --logdir ./logdir/dmc_walker_walk
```

Run VulRSSM against a trained DreamerV3 checkpoint:

```bash
python attack.py \
  --configs dmc_vision \
  --task dmc_walker_walk \
  --checkpoint ./logdir/dmc_walker_walk/latest.pt \
  --attack vulrssm \
  --epsilon 0.03 \
  --attack-steps 5 \
  --attack-horizon 15 \
  --episodes 5
```

The attack entry point also supports `--attack policy-fgsm`, `--attack policy-pgd`, and `--attack none`. Perturbation budgets are measured in normalized `[0,1]` pixel space. JSON summaries are written under `results/attack/` by default.

## Reproducing the experiments

The paper evaluates nine DeepMind Control Suite tasks and studies attack budget, attack imagination horizon, cross-variant transfer, and imagination-consistency defense. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the code-to-paper map, environment notes, and a pre-run checklist.

> [!IMPORTANT]
> Model checkpoint paths and active attack choices are currently configured inside the experimental scripts. Review the selected block in `mainattack.py` and set the checkpoint path before launching a run. This repository intentionally does not include trained weights or generated results.

## Citation

If this repository is useful in your research, please cite the manuscript using the metadata in [`CITATION.cff`](CITATION.cff). Update the author list and publication metadata there before creating a public release if the manuscript record changes.

## Acknowledgments and licenses

The implementations build on open-source Dreamer projects. Their original READMEs, license files, and acknowledgments are preserved inside each implementation directory. In particular, `implementations/dreamer-v3` is based on [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch). Please follow the license terms in each subdirectory.
