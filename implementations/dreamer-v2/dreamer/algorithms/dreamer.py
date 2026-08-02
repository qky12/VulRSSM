import torch
import torch.nn as nn
import numpy as np
import os

from dreamer.modules.model import RSSM, RewardModel, ContinueModel
from dreamer.modules.encoder import Encoder
from dreamer.modules.decoder import Decoder
from dreamer.modules.actor import Actor
from dreamer.modules.critic import Critic

from dreamer.utils.utils import (
    compute_lambda_values,
    create_normal_dist,
    DynamicInfos,
)
from dreamer.utils.buffer import ReplayBuffer
import random
from datetime import datetime
import cv2
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Dreamer:
    def __init__(
        self,
        observation_shape,
        discrete_action_bool,
        action_size,
        writer,
        device,
        config,
    ):
        self.device = device
        self.action_size = action_size
        self.discrete_action_bool = discrete_action_bool

        self.encoder = Encoder(observation_shape, config).to(self.device)
        self.decoder = Decoder(observation_shape, config).to(self.device)  # p(o_t|s_t, h_t)
        self.rssm = RSSM(action_size, config).to(self.device)
        self.reward_predictor = RewardModel(config).to(self.device)  # p(r_t|s_t, h_t)
        if config.parameters.dreamer.use_continue_flag:
            self.continue_predictor = ContinueModel(config).to(self.device)
        self.actor = Actor(discrete_action_bool, action_size, config).to(self.device)  # p(a_t|s_t, h_t)
        self.critic = Critic(config).to(self.device)  # p(v_t|s_t, h_t)

        self.buffer = ReplayBuffer(observation_shape, action_size, self.device, config)

        self.config = config.parameters.dreamer

        # optimizer
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
        )
        if self.config.use_continue_flag:
            self.model_params += list(self.continue_predictor.parameters())

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )

        self.continue_criterion = nn.BCELoss()

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)

        self.writer = writer
        self.num_total_episode = 0

    def train(self, env):
        if len(self.buffer) < 1:
            self.environment_interaction(env, self.config.seed_episodes)

        for iteration in range(self.config.train_iterations):
            for collect_interval in range(self.config.collect_interval):
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )  # Select data from the replay buffer, according to the batch_size and batch_length
                posteriors, deterministics = self.dynamic_learning(data)
                self.behavior_learning(posteriors, deterministics)

            self.environment_interaction(env, self.config.num_interaction_episodes)
            self.evaluate(env)

    def evaluate(self, env):
        self.environment_interaction(env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))  # 获取初始的先验s_t和h_t

        data.embedded_observation = self.encoder(data.observation)

        for t in range(1, self.config.batch_length):  # 将整个batch_size输入到RSSM
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1], deterministic
            )  # h_t+1 = p(s_t, h_t, a_t)
            prior_dist, prior = self.rssm.transition_model(deterministic)  # p(s_t|h_t) 先验
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )  # p(s_t|h_t, o_t) 后验

            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()  # 将列表中的数据按照dim=1堆叠起来
        self._model_update(data, infos)
        return infos.posteriors.detach(), infos.deterministics.detach()

    def _model_update(self, data, posterior_info):
        reconstructed_observation_dist = self.decoder(
            posterior_info.posteriors, posterior_info.deterministics
        )  # 把最后的三维，也就是[batch_size, time_length, channel, height, width] 的三个channel, height, width的出现作为一整个概率
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data.observation[:, 1:]
        )  # 计算在重构的概率分布之下，生成data.observation的概率是怎么样的
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            continue_loss = self.continue_criterion(
                continue_dist.probs, 1 - data.done[:, 1:]
            )

        reward_dist = self.reward_predictor(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reward_loss = reward_dist.log_prob(data.reward[:, 1:]) # This computes the log-probability of the actual reward under the predicted reward distribution.

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        )
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(self.device), kl_divergence_loss
        )
        model_loss = (
            self.config.kl_divergence_scale * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )  # 一张图片，一个KL散度，一个奖励预测问题的损失
        if self.config.use_continue_flag:
            model_loss += continue_loss.mean()

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step()

    def behavior_learning(self, states, deterministics):
        """
        #TODO : last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        state = states.reshape(-1, self.config.stochastic_size)
        deterministic = deterministics.reshape(-1, self.config.deterministic_size)  # [batch_size, time_length, dim] -> [batch_size * time_length, dim]

        # continue_predictor reinit
        for t in range(self.config.horizon_length):
            action = self.actor(state, deterministic)  # 当前的batch_size与time_length对未来要执行动作的预测
            deterministic = self.rssm.recurrent_model(state, action, deterministic)  # p(h_t+1|s_t, a_t, h_t)
            _, state = self.rssm.transition_model(deterministic)  # p(s_t+1|h_t+1)
            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic
            )

        self._agent_update(self.behavior_learning_infos.get_stacked())

    def _agent_update(self, behavior_learning_infos):
        predicted_rewards = self.reward_predictor(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean
        values = self.critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean

        if self.config.use_continue_flag:
            continues = self.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )

        actor_loss = -torch.mean(lambda_values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.actor_optimizer.step()

        value_dist = self.critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.critic_optimizer.step()

    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_episodes, train=True, render=False, starting_point=None, record_path=None):
        score_lst = np.array([])
        train_score_list = np.array([])
        for epi in range(num_interaction_episodes):
            current_timestep = 0
            if render:
                evaluation_list = []
                real_obs = []
                imagined_obs = []
                if starting_point is None:
                    imagined_length = 100
                    starting_point = random.randint(1, env.spec.max_episode_steps - imagined_length - 1)
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)  # 初始的s_0与h_0
            action = torch.zeros(1, self.action_size).to(self.device)
            observation = env.reset()
            embedded_observation = self.encoder(
                torch.from_numpy(observation).float().to(self.device)
            )

            score = 0
            done = False

            while not done:
                deterministic = self.rssm.recurrent_model(
                    posterior, action, deterministic
                )
                embedded_observation = embedded_observation.reshape(1, -1)
                _, posterior = self.rssm.representation_model(
                    embedded_observation, deterministic
                )
                action = self.actor(posterior, deterministic).detach()

                if self.discrete_action_bool:
                    buffer_action = action.cpu().numpy()
                    env_action = buffer_action.argmax()
                else:
                    buffer_action = action.cpu().numpy()[0]
                    env_action = buffer_action
                if render:
                    frame = env.render(mode="rgb_array")
                    evaluation_list.append(frame)
                    if current_timestep >= starting_point and current_timestep < starting_point + imagined_length:
                        real_obs.append(frame)
                        if current_timestep == starting_point:
                            start_hidden_state = deterministic
                            start_state = posterior
                        start_hidden_state = self.rssm.recurrent_model(
                            start_state, action, start_hidden_state
                        )
                        _, start_state = self.rssm.transition_model(start_hidden_state)
                        reconstructed_observation_dist = self.decoder(start_state, start_hidden_state) # p(o_t|s_t, h_t)
                        reconstructed_obs = reconstructed_observation_dist.mean.cpu().numpy().squeeze().transpose((1, 2, 0))
                        restored_reconstructed_obs = np.clip((reconstructed_obs + 0.5) * 255, 0, 255).astype(np.uint8)
                        imagined_obs.append(restored_reconstructed_obs)
                next_observation, reward, done, info = env.step(env_action)
                current_timestep += 1
                if train:
                    self.buffer.add(
                        observation, buffer_action, reward, next_observation, done
                    )
                score += reward
                embedded_observation = self.encoder(
                    torch.from_numpy(next_observation).float().to(self.device)
                )
                observation = next_observation
                if done:
                    if train:
                        self.num_total_episode += 1
                        self.writer.add_scalar(
                            "training score", score, self.num_total_episode
                        )
                        train_score_list = np.append(train_score_list, score)
                    else:
                        if render:
                            render = False
                            self.save_imagined_observations(imagined_obs,
                                                            os.path.join(record_path, "imagined_obs_images"))
                            self.save_imagined_observations(real_obs,
                                                            os.path.join(record_path, 'real_obs_images'))
                            self.save_imagined_vs_real_gif(np.stack(imagined_obs, axis=0),
                                                           np.stack(real_obs, axis=0),
                                                           os.path.join(record_path, 'imagination_comparison.gif'))
                            self.save_gif_file(evaluation_list, gif_path=os.path.join(record_path, 'evaluation.gif'))
                        score_lst = np.append(score_lst, score)
                    break
        if not train:
            evaluate_score = score_lst.mean()
            print("evaluate score : ", evaluate_score)
            if self.writer is not None:
                self.writer.add_scalar("test score", evaluate_score, self.num_total_episode)
        else:
            print(f"[INFO] Interaction Done!!!!, training score: {train_score_list.mean()}")


    def save_model(self, log_dir):
        torch.save(self.encoder.state_dict(), os.path.join(log_dir, 'encoder.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(log_dir, 'decoder.pth'))
        torch.save(self.reward_predictor.state_dict(), os.path.join(log_dir, 'reward.pth'))
        torch.save(self.actor.state_dict(), os.path.join(log_dir, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(log_dir, 'critic.pth'))
        torch.save(self.rssm.recurrent_model.state_dict(), os.path.join(log_dir, 'recurrent.pth'))
        torch.save(self.rssm.transition_model.state_dict(), os.path.join(log_dir, 'transition.pth'))
        torch.save(self.rssm.representation_model.state_dict(), os.path.join(log_dir, 'representation.pth'))
        print(f"[INFO] model saved at: {log_dir}")

    def load_model(self, log_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(log_dir, 'encoder.pth')))
        self.decoder.load_state_dict(torch.load(os.path.join(log_dir, 'decoder.pth')))
        self.reward_predictor.load_state_dict(torch.load(os.path.join(log_dir, 'reward.pth')))
        self.actor.load_state_dict(torch.load(os.path.join(log_dir, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(log_dir, 'critic.pth')))
        self.rssm.recurrent_model.load_state_dict(torch.load(os.path.join(log_dir, 'recurrent.pth')))
        self.rssm.transition_model.load_state_dict(torch.load(os.path.join(log_dir, 'transition.pth')))
        self.rssm.representation_model.load_state_dict(torch.load(os.path.join(log_dir, 'representation.pth')))
        print(f"[INFO] model loaded successfully")

    def save_imagined_observations(self, imagined_obs_sequence, folder_name="imagined_obs_images"):
        """
        Save each image in the imagined observation sequence to a specified folder.

        Parameters:
            imagined_obs_sequence (list): List of numpy arrays representing imagined observations.
            folder_name (str): The name of the folder to save the images.
        """
        # Create the folder if it doesn't exist
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for i, img in enumerate(imagined_obs_sequence):
            if not isinstance(img, np.ndarray) and isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            file_path = os.path.join(folder_name, f"frame_{i:04d}_{timestamp}.png")
            cv2.imwrite(file_path, img_bgr)

        print(f"All imagined observations have been saved to the folder: {folder_name}")

    def save_gif_file(self, frame, gif_path='output.gif', scale=6):
        upscaled_frames = [
        cv2.resize(f, (f.shape[1]*scale, f.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        for f in frame]
        imageio.mimsave(gif_path, upscaled_frames, duration=1/5)

    def save_imagined_vs_real_gif(self, imagined_obs_sequence, real_obs_sequence, gif_path='imagination_comparison.gif'):
        """
        Saves a GIF comparing imagined and real observations side-by-side.

        Args:
            imagined_obs_sequence: list of np.ndarray with shape [H, W, 3] in [0.0, 1.0]
            real_obs_sequence: list of np.ndarray with same shape and range as imagined
            gif_path: output file path
        """
        assert imagined_obs_sequence.shape[0] == real_obs_sequence.shape[0], "Sequence length mismatch"
        fig, ax = plt.subplots()
        combined_image = np.concatenate((real_obs_sequence[0], imagined_obs_sequence[0]), axis=1)
        patch = ax.imshow(combined_image)
        ax.axis('off')

        def animate(i):
            combined = np.concatenate((real_obs_sequence[i], imagined_obs_sequence[i]), axis=1)
            patch.set_data(combined)
            ax.set_title(f"Step {i}  |  Left: Real  |  Right: Imagined")

        anim = animation.FuncAnimation(fig, animate, frames=len(real_obs_sequence), interval=150)
        anim.save(gif_path, writer='pillow')
        print(f"[INFO] Saved GIF to: {gif_path}")



    
