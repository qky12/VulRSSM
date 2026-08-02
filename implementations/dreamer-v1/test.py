import os

import numpy as np
import torch
from PIL import Image

os.environ["MUJOCO_GL"] = "egl"

import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.utils.utils import load_config, get_base_directory
from dreamer.envs.envs import make_dmc_env, make_atari_env, get_env_infos



def load_model_weights(agent, model_load_path):
    if os.path.exists(model_load_path):
        checkpoint = torch.load(model_load_path)
        agent.encoder.load_state_dict(checkpoint['encoder'])
        agent.decoder.load_state_dict(checkpoint['decoder'])
        agent.rssm.load_state_dict(checkpoint['rssm'])
        agent.reward_predictor.load_state_dict(checkpoint['reward_predictor'])
        if checkpoint['continue_predictor'] is not None and hasattr(agent, 'continue_predictor'):
            agent.continue_predictor.load_state_dict(checkpoint['continue_predictor'])
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        print(f"Model weights loaded from {model_load_path}")
    else:
        print("No saved model weights found. Starting with random initialization.")

def testmodel(config_file,weight):
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

    load_model_weights(agent,weight)
    
    save_dir = (
        get_base_directory()
        + "/rendered_images/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + config.operation.log_dir
    )
    os.makedirs(save_dir, exist_ok=True)
    for i in range(10):
        posterior, deterministic = agent.rssm.recurrent_model_input_init(1)
        action = torch.zeros(1, agent.action_size).to(agent.device)
        observation = env.reset()
        print(obs_shape)
        score = 0
        done = False
        step =0
        while not done:
            # img = env.render(mode='rgb_array')
            # if img is not None:
            #     img = Image.fromarray(img)
            #     img.save(os.path.join(save_dir, f"step_{step:04d}.png"))
            #     print(f"Saved image for step {step}")
            # else:
            #     print(f"No image rendered for step {step}")
            embedded_observation = agent.encoder(
            torch.from_numpy(observation).float().to(agent.device)
        )
            print(embedded_observation.shape)
            return
            deterministic = agent.rssm.recurrent_model(
                posterior, action, deterministic
            )
            embedded_observation = embedded_observation.reshape(1, -1)
            _, posterior = agent.rssm.representation_model(
                embedded_observation, deterministic
            )
            action = agent.actor(posterior, deterministic).detach()

            if agent.discrete_action_bool:
                buffer_action = action.cpu().numpy()
                env_action = buffer_action.argmax()

            else:
                buffer_action = action.cpu().numpy()[0]
                env_action = buffer_action

            next_observation, reward, done, info = env.step(env_action)
            score += reward
            observation = next_observation
            step +=1
        print(score)
        print(step)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="dmc-walker-walk.yml",
    help="config file to run(default: dmc-walker-walk.yml)",
)

testmodel(parser.parse_args().config, "checkpoints/walker-stand/final_model_weights.pth")
