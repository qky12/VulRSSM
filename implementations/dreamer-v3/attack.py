"""Online adversarial evaluation for the bundled PyTorch DreamerV3 agent.

The VulRSSM objective maximizes disagreement between clean and perturbed
imagination rollouts while keeping every attacked image inside an L-infinity
ball around the normalized input. Policy-FGSM/PGD baselines minimize the
current value estimate under the same observation budget.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
import ruamel.yaml as yaml

import dreamer
import tools


def _clone_state(state):
    if state is None:
        return None
    return {key: value.clone() for key, value in state.items()}


def _state_distance(left, right):
    keys = ("logit",) if "logit" in left else ("mean", "std")
    return sum(F.mse_loss(left[key], right[key]) for key in keys)


class OnlineAttacker:
    """Craft one bounded image perturbation per environment step."""

    def __init__(
        self,
        agent,
        method="vulrssm",
        epsilon=0.03,
        attack_steps=5,
        attack_horizon=15,
        step_size=None,
        random_start=True,
        w_decoder=1.0,
        w_reward=1.0,
        w_policy=1.0,
        w_latent=1.0,
    ):
        self.agent = agent
        self.wm = agent._wm
        self.behavior = agent._task_behavior
        self.method = method
        self.epsilon = float(epsilon)
        self.steps = 1 if method == "policy-fgsm" else int(attack_steps)
        self.horizon = int(attack_horizon)
        self.step_size = (
            float(step_size)
            if step_size is not None
            else self.epsilon / max(1, self.steps)
        )
        self.random_start = bool(random_start) and method not in ("none", "policy-fgsm")
        self.weights = dict(
            decoder=float(w_decoder),
            reward=float(w_reward),
            policy=float(w_policy),
            latent=float(w_latent),
        )

    def _posterior(self, data, previous):
        latent, action = (None, None) if previous is None else previous
        embed = self.wm.encoder(data)
        return self.wm.dynamics.obs_step(
            _clone_state(latent),
            None if action is None else action.clone(),
            embed,
            data["is_first"].clone(),
            sample=False,
        )[0]

    def _rollout(self, initial, detach=False):
        state = initial
        trajectory = []
        for _ in range(self.horizon):
            feat = self.wm.dynamics.get_feat(state)
            action = self.behavior.actor(feat).mode()
            reward = self.wm.heads["reward"](feat).mode()
            decoded = self.wm.heads["decoder"](feat)
            image = decoded["image"].mode() if "image" in decoded else None
            trajectory.append((state, feat, action, reward, image))
            state = self.wm.dynamics.img_step(state, action, sample=False)
        if not detach:
            return trajectory
        return [
            (
                {key: value.detach() for key, value in state.items()},
                feat.detach(),
                action.detach(),
                reward.detach(),
                None if image is None else image.detach(),
            )
            for state, feat, action, reward, image in trajectory
        ]

    def _vulrssm_loss(self, adversarial, clean):
        loss = adversarial[0][1].new_zeros(())
        for adv_step, clean_step in zip(adversarial, clean):
            adv_state, _, adv_action, adv_reward, adv_image = adv_step
            clean_state, _, clean_action, clean_reward, clean_image = clean_step
            loss = loss + self.weights["latent"] * _state_distance(
                adv_state, clean_state
            )
            loss = loss + self.weights["policy"] * F.mse_loss(
                adv_action, clean_action
            )
            loss = loss + self.weights["reward"] * F.mse_loss(
                adv_reward, clean_reward
            )
            if adv_image is not None and clean_image is not None:
                loss = loss + self.weights["decoder"] * F.mse_loss(
                    adv_image, clean_image
                )
        return loss / max(1, len(adversarial))

    def _objective(self, data, previous, clean_rollout):
        posterior = self._posterior(data, previous)
        if self.method == "vulrssm":
            return self._vulrssm_loss(self._rollout(posterior), clean_rollout)
        feat = self.wm.dynamics.get_feat(posterior)
        # Gradient ascent on negative value is equivalent to minimizing value.
        return -self.behavior.value(feat).mode().mean()

    def act(self, observation, previous=None):
        data = self.wm.preprocess(observation)
        clean_image = data["image"].detach()

        with torch.no_grad():
            clean_posterior = self._posterior(data, previous)
            clean_rollout = self._rollout(clean_posterior, detach=True)

        if self.method == "none" or self.epsilon <= 0:
            adversarial_image = clean_image
            objective = 0.0
        else:
            delta = torch.zeros_like(clean_image)
            if self.random_start:
                delta.uniform_(-self.epsilon, self.epsilon)
                delta.copy_((clean_image + delta).clamp(0.0, 1.0) - clean_image)

            objective = 0.0
            for _ in range(self.steps):
                delta = delta.detach().requires_grad_(True)
                attacked = dict(data)
                attacked["image"] = (clean_image + delta).clamp(0.0, 1.0)
                loss = self._objective(attacked, previous, clean_rollout)
                gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]
                objective = float(loss.detach().cpu())
                with torch.no_grad():
                    delta = delta + self.step_size * gradient.sign()
                    delta.clamp_(-self.epsilon, self.epsilon)
                    delta.copy_((clean_image + delta).clamp(0.0, 1.0) - clean_image)
            adversarial_image = (clean_image + delta.detach()).clamp(0.0, 1.0)

        attacked = dict(data)
        attacked["image"] = adversarial_image
        with torch.no_grad():
            posterior = self._posterior(attacked, previous)
            feat = self.wm.dynamics.get_feat(posterior)
            action = self.behavior.actor(feat).mode()
            if self.agent._config.actor["dist"] == "onehot_gumble":
                action = torch.one_hot(
                    torch.argmax(action, dim=-1),
                    self.agent._config.num_actions,
                )

        metrics = {
            "objective": objective,
            "linf": float((adversarial_image - clean_image).abs().max().cpu()),
        }
        next_state = (
            {key: value.detach() for key, value in posterior.items()},
            action.detach(),
        )
        return action.detach(), next_state, metrics


def _load_config(known, remaining):
    configs = yaml.safe_load(
        (pathlib.Path(__file__).parent / "configs.yaml").read_text()
    )

    def update(base, extra):
        for key, value in extra.items():
            if isinstance(value, dict) and key in base:
                update(base[key], value)
            else:
                base[key] = value

    defaults = {}
    for name in ["defaults", *(known.configs or [])]:
        update(defaults, configs[name])
    parser = argparse.ArgumentParser(add_help=False)
    for key, value in sorted(defaults.items()):
        kind = tools.args_type(value)
        parser.add_argument(f"--{key}", type=kind, default=kind(value))
    config, unknown = parser.parse_known_args(remaining)
    if unknown:
        raise SystemExit(f"Unknown Dreamer arguments: {' '.join(unknown)}")
    return config


def _batched(observation):
    return {key: np.expand_dims(value, 0) for key, value in observation.items()}


def evaluate(args, config):
    tools.set_seed_everywhere(config.seed)
    config.time_limit //= config.action_repeat
    config.compile = False
    output = pathlib.Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    env = dreamer.make_env(config, "eval", 0)
    action_space = env.action_space
    config.num_actions = (
        action_space.n if hasattr(action_space, "n") else action_space.shape[0]
    )
    logger = tools.Logger(output, 0)
    agent = dreamer.Dreamer(
        env.observation_space,
        action_space,
        config,
        logger,
        iter(()),
    ).to(config.device)
    checkpoint = torch.load(
        pathlib.Path(args.checkpoint).expanduser(),
        map_location=config.device,
    )
    state_dict = checkpoint.get("agent_state_dict", checkpoint)
    agent.load_state_dict(state_dict)
    agent.requires_grad_(False)
    agent.eval()

    attacker = OnlineAttacker(
        agent,
        method=args.attack,
        epsilon=args.epsilon,
        attack_steps=args.attack_steps,
        attack_horizon=args.attack_horizon,
        step_size=args.step_size,
        random_start=not args.no_random_start,
        w_decoder=args.w_decoder,
        w_reward=args.w_reward,
        w_policy=args.w_policy,
        w_latent=args.w_latent,
    )

    returns = []
    linf_values = []
    objective_values = []
    try:
        for episode in range(args.episodes):
            observation = env.reset()
            previous = None
            done = False
            episode_return = 0.0
            while not done:
                action, previous, metrics = attacker.act(
                    _batched(observation), previous
                )
                env_action = {"action": action[0].cpu().numpy()}
                observation, reward, done, _ = env.step(env_action)
                episode_return += float(reward)
                linf_values.append(metrics["linf"])
                objective_values.append(metrics["objective"])
            returns.append(episode_return)
            print(f"episode={episode + 1} return={episode_return:.3f}")
    finally:
        env.close()

    summary = {
        "task": config.task,
        "attack": args.attack,
        "episodes": args.episodes,
        "epsilon": args.epsilon,
        "attack_steps": attacker.steps,
        "attack_horizon": args.attack_horizon,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "max_linf": float(max(linf_values, default=0.0)),
        "mean_objective": float(np.mean(objective_values)),
        "returns": returns,
    }
    destination = output / f"attack-{args.attack}-{config.task}.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {destination}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DreamerV3 with VulRSSM or policy-level attacks."
    )
    parser.add_argument("--configs", nargs="+", default=["dmc_vision"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--attack",
        choices=("vulrssm", "policy-pgd", "policy-fgsm", "none"),
        default="vulrssm",
    )
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--attack-steps", type=int, default=5)
    parser.add_argument("--attack-horizon", type=int, default=15)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", default="results/attack")
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--w-decoder", type=float, default=1.0)
    parser.add_argument("--w-reward", type=float, default=1.0)
    parser.add_argument("--w-policy", type=float, default=1.0)
    parser.add_argument("--w-latent", type=float, default=1.0)
    args, remaining = parser.parse_known_args()
    config = _load_config(args, remaining)
    evaluate(args, config)


if __name__ == "__main__":
    main()
