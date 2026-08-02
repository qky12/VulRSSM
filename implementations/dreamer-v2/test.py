import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
from datetime import datetime
from dreamer.algorithms.dreamerv2 import DreamerV2

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.utils.utils import load_config, get_base_directory
from dreamer.envs.envs import make_dmc_env, make_atari_env, get_env_infos


def load_model_weights(agent, model_load_path: str, strict: bool = True):
    device = agent.device if hasattr(agent, "device") else torch.device("cpu")
    ckpt = torch.load(model_load_path, map_location=device)

    def _try_load(name: str):
        if hasattr(agent, name) and name in ckpt and ckpt[name] is not None:
            getattr(agent, name).load_state_dict(ckpt[name], strict=strict)
            getattr(agent, name).eval()

    for k in ["encoder", "decoder", "rssm", "reward_predictor", "continue_predictor", "actor", "critic", "target_critic"]:
        _try_load(k)

    if hasattr(agent, "target_critic") and ("target_critic" not in ckpt or ckpt.get("target_critic", None) is None):
        if hasattr(agent, "critic"):
            agent.target_critic.load_state_dict(agent.critic.state_dict())
            agent.target_critic.eval()

    print(f"[load_model_weights] loaded: {model_load_path}")


@torch.no_grad()
def test_score_only(config_file):
    config = load_config(config_file)

    log_dir = os.path.join(
        get_base_directory(),
        "test",
        f"{config.operation.log_dir}",
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # env
    if config.environment.benchmark == "atari":
        env = make_atari_env(
            task_name=config.environment.task_name,
            seed=config.environment.seed,
            height=config.environment.height,
            width=config.environment.width,
            skip_frame=config.environment.frame_skip,
            pixel_norm=config.environment.pixel_norm,
        )
    elif config.environment.benchmark == "dmc":
        env = make_dmc_env(
            domain_name=config.environment.domain_name,
            task_name=config.environment.task_name,
            seed=config.environment.seed,
            visualize_reward=config.environment.visualize_reward,
            from_pixels=config.environment.from_pixels,
            height=config.environment.height,
            width=config.environment.width,
            frame_skip=config.environment.frame_skip,
            pixel_norm=config.environment.pixel_norm,
        )
    else:
        raise ValueError(f"Unknown benchmark: {config.environment.benchmark}")

    obs_shape, discrete_action_bool, action_size = get_env_infos(env)
    device = config.operation.device

    # agent
    if config.algorithm == "dreamer-v1":
        agent = Dreamer(obs_shape, discrete_action_bool, action_size, writer, device, config)
    elif config.algorithm == "plan2explore":
        agent = Plan2Explore(obs_shape, discrete_action_bool, action_size, writer, device, config)
    elif config.algorithm == 'dreamer-v2':
        from attrdict import AttrDict
        config_dict = dict(config)
        config_dict['parameters']['dreamer']['stochastic_size'] = config_dict['parameters']['dreamer']['categorical_head'] * config_dict['parameters']['dreamer']['categorical_size']
        config = AttrDict(config_dict)
        agent = DreamerV2(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
    # load weights（改成你的路径）
    model_path = "checkpoints/walker-walk/final_model_weights.pth"
    load_model_weights(agent, model_path)

    # 兼容 actor 返回 action 或 (action, log_prob, entropy)
    def get_action(posterior, deterministic):
        out = agent.actor(posterior, deterministic)
        return out[0] if isinstance(out, (tuple, list)) else out

    num_episodes = 5
    scores = []

    for epi in range(num_episodes):
        posterior, deterministic = agent.rssm.recurrent_model_input_init(1)
        action = torch.zeros(1, agent.action_size).to(agent.device)

        observation = env.reset()
        embedded_observation = agent.encoder(torch.from_numpy(observation).float().to(agent.device))

        done = False
        score = 0.0

        while not done:
            deterministic = agent.rssm.recurrent_model(posterior, action, deterministic)

            emb = embedded_observation.reshape(1, -1)
            _, posterior = agent.rssm.representation_model(emb, deterministic)

            action = get_action(posterior, deterministic).detach()

            if agent.discrete_action_bool:
                env_action = action.cpu().numpy().argmax()
            else:
                env_action = action.cpu().numpy()[0]

            next_observation, reward, done, info = env.step(env_action)
            score += float(reward)

            embedded_observation = agent.encoder(torch.from_numpy(next_observation).float().to(agent.device))

        scores.append(score)
        writer.add_scalar("test/score", score, epi)
        print(f"[epi {epi}] score = {score:.2f}")

    print("\n===== Score Only Summary =====")
    print(f"avg score over {num_episodes} episodes: {np.mean(scores):.2f}")
    print(f"log_dir: {log_dir}")



parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="dmc-walker-walk_dreamerv2.yml",
    help="config file to run(default: dmc-walker-walk_dreamerv2.yml)",
)
test_score_only(parser.parse_args().config)

