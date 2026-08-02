import os

os.environ["MUJOCO_GL"] = "egl"
import torch
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.algorithms.dreamerv2 import DreamerV2
from dreamer.utils.utils import load_config, get_base_directory
from dreamer.envs.envs import make_dmc_env, make_atari_env, get_env_infos


def main(config_file):
    config = load_config(config_file)

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
    obs_shape, discrete_action_bool, action_size = get_env_infos(env)

    log_dir = (
        get_base_directory()
        + "/runs/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + config.operation.log_dir
    )
    writer = SummaryWriter(log_dir)
    device = config.operation.device

    if config.algorithm == "dreamer-v1":
        agent = Dreamer(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
    elif config.algorithm == "plan2explore":
        agent = Plan2Explore(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
    elif config.algorithm == 'dreamer-v2':
        from attrdict import AttrDict
        config_dict = dict(config)
        config_dict['parameters']['dreamer']['stochastic_size'] = config_dict['parameters']['dreamer']['categorical_head'] * config_dict['parameters']['dreamer']['categorical_size']
        config = AttrDict(config_dict)
        agent = DreamerV2(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
    agent.train(env)
    model_save_path = os.path.join(log_dir, "final_model_weights.pth")

    ckpt = {
        "encoder": agent.encoder.state_dict(),
        "decoder": agent.decoder.state_dict(),
        "rssm": agent.rssm.state_dict(),
        "reward_predictor": agent.reward_predictor.state_dict(),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
    }

    # 有 continue 才存
    if hasattr(agent, "continue_predictor"):
        ckpt["continue_predictor"] = agent.continue_predictor.state_dict()

    # 有 target critic 也存（推荐）
    if hasattr(agent, "target_critic"):
        ckpt["target_critic"] = agent.target_critic.state_dict()

    torch.save(ckpt, model_save_path)
    print(f"Model weights saved to {model_save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="dmc-cartpole-swingup_dreamerv2.yml",
        help="config file to run(default: dmc-cartpole-swingup_dreamerv2.yml)",
    )
    main(parser.parse_args().config)
