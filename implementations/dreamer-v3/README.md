# dreamerv3-torch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). DreamerV3 is a scalable algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

## Instructions

### Method 1: Manual

Get dependencies with python 3.11:
```
pip install -r requirements.txt
```
Run training on DMC Vision:
```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```
Monitor results:
```
tensorboard --logdir ./logdir
```

### Adversarial evaluation

`attack.py` evaluates a trained checkpoint with the RSSM-aware VulRSSM objective or matched policy-level baselines. The attack is online: each image is perturbed inside an L-infinity ball before posterior inference, and the adversarial posterior is carried into the next environment step.

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

Available attack modes are `vulrssm`, `policy-pgd`, `policy-fgsm`, and `none`. Use `none` to obtain a clean return with the same evaluation loop. Budgets use normalized `[0,1]` pixels, so `0.03` is approximately `8/255`. Results are saved as JSON under `results/attack/` unless `--output` is specified.

For a deterministic PGD start, add `--no-random-start`. The four VulRSSM loss components can be adjusted with `--w-decoder`, `--w-reward`, `--w-policy`, and `--w-latent`.

To set up Atari or Minecraft environments, please check the scripts located in [env/setup_scripts](https://github.com/NM512/dreamerv3-torch/tree/main/envs/setup_scripts).

### Method 2: Docker

Please refer to the Dockerfile for the instructions, as they are included within.

## Benchmarks
So far, the following benchmarks can be used for testing.
| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [Atari 100k](https://github.com/openai/atari-py) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Minecraft](https://github.com/minerllabs/minerl) | Image and State |Discrete |100M| Vast 3D open world.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|

## Results
#### DMC Proprio
![dmcproprio](imgs/dmcproprio.png)
#### DMC Vision
![dmcvision](imgs/dmcvision.png)
#### Atari 100k
![atari100k](imgs/atari100k.png)

#### Crafter
<img src="https://github.com/NM512/dreamerv3-torch/assets/70328564/a0626038-53f6-4300-a622-7ac257f4c290" width="300" height="150" />

## Acknowledgments
This code is heavily inspired by the following works:
- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
- danijar's Dreamer-v2 tensorflow implementation: https://github.com/danijar/dreamerv2
- jsikyoon's Dreamer-v2 pytorch implementation: https://github.com/jsikyoon/dreamer-torch
- RajGhugare19's Dreamer-v2 pytorch implementation: https://github.com/RajGhugare19/dreamerv2
- denisyarats's DrQ-v2 original implementation: https://github.com/facebookresearch/drqv2
