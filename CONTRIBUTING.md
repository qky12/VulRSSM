# Contributing

Thank you for helping improve VulRSSM.

## Development guidelines

1. Keep Dreamer V1, V2, and V3 dependencies isolated.
2. Do not commit checkpoints, training logs, rendered episodes, datasets, credentials, or machine-specific paths.
3. Place generated artifacts in an ignored output directory.
4. Document the task, random seed, checkpoint, attack budget, optimization steps, and imagination horizon for experiment changes.
5. Preserve upstream license and attribution files when modifying a bundled implementation.

Before opening a pull request, run a Python syntax check on changed files and confirm `git status` contains only intentional source or documentation changes.

