import os
import numpy as np
import torch
from PIL import Image
import torch
import torch.nn.functional as F

os.environ["MUJOCO_GL"] = "egl"

import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.utils.utils import load_config, get_base_directory
from dreamer.envs.envs import make_dmc_env, make_atari_env, get_env_infos
import matplotlib.pylab as plt

import numpy as np

import numpy as np
from skimage.metrics import structural_similarity as ssim

from datetime import datetime

# 获取当前时间的时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def _actor_action(out):
    # DreamerV2: (action, log_prob, entropy)
    # DreamerV1: action
    return out[0] if isinstance(out, (tuple, list)) else out

def calculate_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """
    计算两个 64x64x3 图像之间的结构相似性指数（SSIM）。

    参数:
        img1 (np.ndarray): 第一张图像，形状为 (64, 64, 3)。
        img2 (np.ndarray): 第二张图像，形状为 (64, 64, 3)。
        data_range (float): 图像数据的范围。默认为 1.0（假设图像像素值在 [0, 1] 范围内）。

    返回:
        float: 两个图像之间的 SSIM 值。
    """
    # 确保图像数据类型为 float，并且在 [0, 1] 范围内
    img1 = img1.astype(np.float64) / 255.0 if img1.max() > 1 else img1.astype(np.float64)
    img2 = img2.astype(np.float64) / 255.0 if img2.max() > 1 else img2.astype(np.float64)

    # 计算 SSIM，指定 win_size=7，channel_axis=2，以及 data_range
    ssim_value = ssim(img1, img2, win_size=7, channel_axis=2, data_range=data_range)

    return ssim_value


def compute_action_difference(action1, action2):
    """
  
    
    参数：
    action1（numpy.ndarray 或 list）：第一个动作，6 维向量。
    action2（numpy.ndarray 或 list）：第二个动作，6 维向量。
    
    返回：
    float：两个动作之间的欧几里得距离。
    """
    # 确保输入是 NumPy 数组
    action1 = np.array(action1)
    action2 = np.array(action2)
    
  
    absolute_difference = np.sum(np.abs(action1 - action2))
    return absolute_difference


def value_adv_attack(observation, agent, posterior, deterministic, epsilon=0.03, attack_steps=3):

    obs_tensor = torch.from_numpy(observation).float().to(agent.device)
    obs_tensor.requires_grad = True
    
    # 创建临时计算图
    with torch.enable_grad():
        # 多步迭代攻击
        for _ in range(attack_steps):
            # 编码观测
            embedded = agent.encoder(obs_tensor.unsqueeze(0))  # [1, embed_dim]
            
            # 状态表征
            embedded = embedded.reshape(1, -1)
            _, new_posterior = agent.rssm.representation_model(
                embedded, 
                deterministic.detach()  
            )
            
            # 获取critic输出分布
            value_dist = agent.critic(
                new_posterior.unsqueeze(0),  # 添加时间维度 [1, 1, state_dim]
                deterministic.detach().unsqueeze(0)
            )
            
            # 构造虚假目标值（最大化value_loss）
            fake_lambda = torch.randn_like(value_dist.mean) * 5.0  # 随机大偏差
            
            # 计算对抗损失
            adv_loss = -value_dist.log_prob(fake_lambda).mean()  # 最大化分布偏差
            adv_loss += 0.1 * torch.norm(obs_tensor)  # 正则化项防止过度扰动
            
            # 梯度计算
            grad = torch.autograd.grad(adv_loss, obs_tensor, 
                                     retain_graph=False,
                                     create_graph=False)[0]
            
            # 投影梯度扰动
            perturbation = epsilon * grad.sign() / (attack_steps ** 0.5)
            obs_tensor = (obs_tensor + perturbation).detach().requires_grad_(True)
            
            # 像素值裁剪
            obs_tensor.data = torch.clamp(obs_tensor, -0.5, 0.5)
    
    return obs_tensor.detach().cpu().numpy().squeeze()

import torch

def value_adv_attack1(
    observation,
    agent,
    posterior,                 # 保留签名以兼容，未直接使用
    deterministic,
    epsilon=0.03,              # L∞ 约束半径
    attack_steps=10,           # 迭代步数
    alpha=None,                # 每步步长；默认按 ε/steps 设定
    rand_init=True,            # 是否随机初始化到 ε-球内
    clamp_range=(-0.5, 0.5),   # 像素/特征裁剪范围
    direction="min",           # 'min'：降低 value；'max'：提高 value
    loss_mode="value_mean"     # 'value_mean' 或 'nll_self'
):
    """
    标准PGD（L∞）：在输入空间对critic进行对抗攻击。
    - loss_mode='value_mean'：以critic输出的均值作为目标（默认）。
      direction='min' 会最小化 value（使状态看起来更差）。
    - loss_mode='nll_self'：对自身分布的负对数似然（更通用，但依赖分布形式）。
    """
    device = agent.device
    x0 = torch.from_numpy(observation).float().to(device)

    # 初始化对抗样本
    if rand_init:
        # 均匀随机扰动到 ε-球内
        delta = torch.empty_like(x0).uniform_(-epsilon, epsilon)
        x = torch.clamp(x0 + delta, clamp_range[0], clamp_range[1]).detach()
    else:
        x = x0.clone().detach()

    if alpha is None:
        alpha = epsilon / max(attack_steps, 1)  # 合理缺省

    for _ in range(attack_steps):
        x.requires_grad_(True)

        # ===== 前向：encoder -> rssm -> critic =====
        embedded = agent.encoder(x.unsqueeze(0))                 # [1, embed_dim]
        embedded = embedded.reshape(1, -1)
        _, new_posterior = agent.rssm.representation_model(
            embedded,
            deterministic.detach()
        )
        value_dist = agent.critic(
            new_posterior.unsqueeze(0),                          # [1,1,state_dim]
            deterministic.detach().unsqueeze(0)
        )

        # ===== 构造损失 =====
        if loss_mode == "value_mean":
            # 以均值为目标：direction='min' 使 value 降低；'max' 使 value 升高
            value_mean = value_dist.mean
            loss = value_mean.mean() if direction == "min" else (-value_mean.mean())
        elif loss_mode == "nll_self":
            # 对自身当前输出作为“目标”的负对数似然（不反传到目标）
            # 注：这会鼓励增大不确定度或拉偏均值，属于更通用的 untargeted 目标
            target = value_dist.mean.detach()
            loss = -value_dist.log_prob(target).mean()
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        # 反向求梯度
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]

        # ===== PGD 更新（L∞）=====
        step = alpha * grad.sign()
        # direction='min' 我们已在定义 loss 时体现方向，这里统一做 ascent
        x = x.detach() + step

        # 投影回 ε-球：clip(x - x0) 到 [-ε, ε]
        x = x0 + torch.clamp(x - x0, -epsilon, epsilon)

        # 全局有效范围裁剪
        x = torch.clamp(x, clamp_range[0], clamp_range[1]).detach()

    return x.cpu().numpy().squeeze()



def world_model_attack(observation, agent, posterior, deterministic, epsilon=0.05, attack_steps=3):
    
    adv_obs = torch.tensor(observation, dtype=torch.float32, device=agent.device).requires_grad_(True)
    
    for _ in range(attack_steps):

        
        embedded = agent.encoder(adv_obs)  
        
        # 获取transition输出（确保参数一致）
        prior_dist, prior = agent.rssm.transition_model(deterministic)
        
        # 重构损失计算
        recon_dist = agent.decoder(posterior, deterministic)
        recon_loss = -recon_dist.log_prob(adv_obs).mean()  
        
        # 奖励预测损失
        reward_dist = agent.reward_predictor(posterior,deterministic)
        dummy_reward = torch.zeros_like(reward_dist.mean)  
        reward_loss = -reward_dist.log_prob(dummy_reward).mean()
        
        # 反向传播优化
        total_loss = recon_loss + reward_loss
        # total_loss.backward()
        total_loss.backward(retain_graph=True)

        
        # FGSM扰动更新
        perturbation = epsilon * adv_obs.grad.sign()
        adv_obs = (adv_obs + perturbation).clamp(0, 1).detach_().requires_grad_(True)
    
    return adv_obs.detach().cpu().numpy()

def world_model_attack1(observation, agent, posterior,deterministic, prev_action, epsilon=0.1, attack_steps=5):
    # 观测格式处理及初始化
    original_obs = torch.tensor(observation, dtype=torch.float32, device=agent.device)
    adv_obs = original_obs.clone().requires_grad_(True)
    
    # 初始状态推断（需重新计算）
    with torch.no_grad():
        embedded_init = agent.encoder(original_obs.unsqueeze(0))
        deterministic_init = agent.rssm.recurrent_model(
            posterior,  # 零初始化后验
            prev_action, 
            deterministic
        ) 
        _, posterior_init = agent.rssm.representation_model(embedded_init, deterministic_init)
    
    # 动态更新隐状态和后验
    posterior = posterior_init.detach().clone()
    current_deterministic = deterministic_init.detach().clone()
    
    for _ in range(attack_steps):
        # ---- 状态重新推断 ----
        embedded = agent.encoder(adv_obs.unsqueeze(0))
        
        # 计算新后验和隐状态
        prior_dist, prior = agent.rssm.transition_model(current_deterministic)
        post_dist, posterior = agent.rssm.representation_model(embedded, current_deterministic)
        
        # ---- 对抗损失计算 ----
        # 1. 最大化重构误差（目标：原始观测）
        recon_dist = agent.decoder(posterior, current_deterministic)
        recon_loss = -recon_dist.log_prob(original_obs).mean()
        
        # 2. 误导奖励预测
        reward_dist = agent.reward_predictor(posterior, current_deterministic)
        target_reward = -1.0 * torch.ones_like(reward_dist.mean)  # 使模型预测错误低奖励
        reward_loss = -reward_dist.log_prob(target_reward).mean()
        
        total_loss = recon_loss + 0.5 * reward_loss  # 加权平衡
        
        # ---- 梯度计算与扰动更新 ----
        if adv_obs.grad is not None:
            adv_obs.grad.zero_()
        total_loss.backward(retain_graph=True)
        
        # 投影梯度符号扰动
        perturbation = epsilon * adv_obs.grad.sign() 
        adv_obs = (adv_obs + perturbation).clamp(-0.5, 0.5).detach_().requires_grad_(True)
        
        # 隐状态传递（避免梯度积累）
        current_deterministic = agent.rssm.recurrent_model(
            posterior, 
            prev_action, 
            current_deterministic
        ).detach()
    
    return adv_obs.detach().cpu().numpy().squeeze()

def world_model_PGD(observation, agent, posterior, deterministic, prev_action, epsilon, steps=5):
    # 初始图像处理
    # print(epsilon)
    original_obs = torch.tensor(observation, dtype=torch.float32, device=agent.device)
    adv_obs = original_obs.clone().detach() + torch.empty_like(original_obs).uniform_(-epsilon, epsilon)
    adv_obs = adv_obs.clamp(-0.5, 0.5).requires_grad_(True)
    alpha = epsilon/steps
    # 初始状态计算
    with torch.no_grad():
        embedded_init = agent.encoder(original_obs.unsqueeze(0))
        deterministic_init = agent.rssm.recurrent_model(posterior, prev_action, deterministic)
        _, posterior_init = agent.rssm.representation_model(embedded_init, deterministic_init)

    current_deterministic = deterministic_init.detach().clone()

    for _ in range(steps):
        embedded = agent.encoder(adv_obs.unsqueeze(0))
        prior_dist, prior = agent.rssm.transition_model(current_deterministic)
        post_dist, posterior = agent.rssm.representation_model(embedded, current_deterministic)

        recon_dist = agent.decoder(posterior, current_deterministic)
        recon_loss = -recon_dist.log_prob(original_obs).mean()

        reward_dist = agent.reward_predictor(posterior, current_deterministic)
        target_reward = -reward_dist.mean.detach()
        reward_loss = -reward_dist.log_prob(target_reward).mean()

        kl_loss = torch.distributions.kl.kl_divergence(post_dist, prior_dist).mean()
        total_loss = recon_loss + reward_loss + kl_loss

        # 梯度更新
        adv_obs.grad = None
        total_loss.backward()
        perturbation = alpha * adv_obs.grad.sign()

        # 投影到 epsilon 邻域内，同时限制观测范围
        adv_obs = (adv_obs + perturbation).clamp(original_obs - epsilon, original_obs + epsilon)
        adv_obs = adv_obs.clamp(-0.5, 0.5).detach().requires_grad_(True)

        # 更新 recurrent 状态（可选是否每步更新）
        current_deterministic = agent.rssm.recurrent_model(posterior, prev_action, current_deterministic).detach()

    return adv_obs.detach().cpu().numpy().squeeze()


def world_model_FGSM(observation, agent, posterior, deterministic, prev_action, epsilon):
    # 观测格式处理及初始化
    original_obs = torch.tensor(observation, dtype=torch.float32, device=agent.device)
    adv_obs = original_obs.clone().requires_grad_(True)
    
    # 初始状态推断
    with torch.no_grad():
        embedded_init = agent.encoder(original_obs.unsqueeze(0))
        deterministic_init = agent.rssm.recurrent_model(posterior, prev_action, deterministic)
        _, posterior_init = agent.rssm.representation_model(embedded_init, deterministic_init)

    posterior = posterior_init.detach().clone()
    current_deterministic = deterministic_init.detach().clone()
    
    # 单次攻击步骤
    embedded = agent.encoder(adv_obs.unsqueeze(0))
    prior_dist, prior = agent.rssm.transition_model(current_deterministic)
    post_dist, posterior = agent.rssm.representation_model(embedded, current_deterministic)
    
    # 对抗损失计算
    recon_dist = agent.decoder(posterior, current_deterministic)
    recon_loss = -recon_dist.log_prob(original_obs).mean()
    
    reward_dist = agent.reward_predictor(posterior, current_deterministic)
    target_reward = -reward_dist.mean.detach()
    reward_loss = -reward_dist.log_prob(target_reward).mean()

    # reward_dist = agent.reward_predictor(posterior, current_deterministic)
    # target_reward = -10 * torch.ones_like(reward_dist.mean)
    # # target_reward = (torch.rand_like(reward_dist.mean) * 20.0) - 10.0  # [-10, +10]

    # reward_loss = -reward_dist.log_prob(target_reward).mean()

    kl_loss = torch.distributions.kl.kl_divergence(post_dist, prior_dist).mean()
    total_loss = recon_loss + reward_loss+ kl_loss
    # total_loss = recon_loss + 0.5 * reward_loss
    # 梯度计算与扰动更新
    total_loss.backward()
    perturbation = epsilon * adv_obs.grad.sign()
    adv_obs = (adv_obs + perturbation).clamp(-0.5, 0.5).detach()
    
    return adv_obs.cpu().numpy().squeeze()

# def world_model_PGD(observation, agent, posterior, deterministic, prev_action, epsilon, attack_steps=5):
#     original_obs = torch.tensor(observation, dtype=torch.float32, device=agent.device)
#     # 随机初始化扰动
#     adv_obs = original_obs.clone() + torch.empty_like(original_obs).uniform_(-epsilon, epsilon)
#     adv_obs = adv_obs.clamp(-0.5, 0.5).requires_grad_(True)
    
#     # 初始状态推断
#     with torch.no_grad():
#         embedded_init = agent.encoder(adv_obs.unsqueeze(0))
#         deterministic_init = agent.rssm.recurrent_model(posterior, prev_action, deterministic)
#         _, posterior_init = agent.rssm.representation_model(embedded_init, deterministic_init)
    
#     alpha = epsilon / attack_steps
#     current_deterministic = deterministic_init.clone()
    
#     for _ in range(attack_steps):
#         # 状态重新推断
#         embedded = agent.encoder(adv_obs.unsqueeze(0))
#         prior_dist, prior = agent.rssm.transition_model(current_deterministic)
#         post_dist, posterior = agent.rssm.representation_model(embedded, current_deterministic)
        
#         # 对抗损失计算
#         recon_dist = agent.decoder(posterior, current_deterministic)
#         recon_loss = -recon_dist.log_prob(original_obs).mean()
#         reward_dist = agent.reward_predictor(posterior, current_deterministic)
#         target_reward = -reward_dist.mean.detach()
#         reward_loss = -reward_dist.log_prob(target_reward).mean()
#         kl_loss = torch.distributions.kl.kl_divergence(post_dist, prior_dist).mean()
#         total_loss = recon_loss +  reward_loss+ kl_loss
#         # total_loss = recon_loss + 0.5 * reward_loss
        
#         # 梯度更新
#         adv_obs.grad = None
#         total_loss.backward()
#         perturbation = alpha * adv_obs.grad.sign()
        
#         # 投影到epsilon邻域并保持输入范围
#         adv_obs = (adv_obs + perturbation).clamp(original_obs-epsilon, original_obs+epsilon)
#         adv_obs = adv_obs.clamp(-0.5, 0.5).detach().requires_grad_(True)
        
#         # 更新隐状态
#         current_deterministic = agent.rssm.recurrent_model(posterior, prev_action, current_deterministic).detach()
    
#     return adv_obs.detach().cpu().numpy().squeeze()

def world_model_MI_FGSM(observation, agent, posterior, deterministic, prev_action, epsilon=0.1, attack_steps=5, momentum=0.9):
    original_obs = torch.tensor(observation, dtype=torch.float32, device=agent.device)
    adv_obs = original_obs.clone().requires_grad_(True)
    
    # 初始状态推断
    with torch.no_grad():
        embedded_init = agent.encoder(original_obs.unsqueeze(0))
        deterministic_init = agent.rssm.recurrent_model(posterior, prev_action, deterministic)
        _, posterior_init = agent.rssm.representation_model(embedded_init, deterministic_init)
    
    current_deterministic = deterministic_init.clone()
    momentum_grad = 0
    alpha = epsilon / attack_steps
    
    for _ in range(attack_steps):
        # 状态推断
        embedded = agent.encoder(adv_obs.unsqueeze(0))
        prior_dist, prior = agent.rssm.transition_model(current_deterministic)
        post_dist, posterior = agent.rssm.representation_model(embedded, current_deterministic)
        
        # 对抗损失
        recon_dist = agent.decoder(posterior, current_deterministic)
        recon_loss = -recon_dist.log_prob(original_obs).mean()
        reward_dist = agent.reward_predictor(posterior, current_deterministic)
        target_reward = -1.0 * torch.ones_like(reward_dist.mean)
        reward_loss = -reward_dist.log_prob(target_reward).mean()
        kl_loss = torch.distributions.kl.kl_divergence(post_dist, prior_dist).mean()
        total_loss = recon_loss + 0.5 * reward_loss+ 0.2* kl_loss
        # total_loss = recon_loss + 0.5 * reward_loss
        
        # 动量梯度计算
        total_loss.backward()
        grad = adv_obs.grad / torch.norm(adv_obs.grad, p=1)
        momentum_grad = momentum * momentum_grad + grad
        
        # 扰动更新
        perturbation = alpha * momentum_grad.sign()
        adv_obs = (adv_obs + perturbation).clamp(original_obs-epsilon, original_obs+epsilon)
        adv_obs = adv_obs.clamp(-0.5, 0.5).detach().requires_grad_(True)
        
        # 更新隐状态
        current_deterministic = agent.rssm.recurrent_model(posterior, prev_action, current_deterministic).detach()
        adv_obs.grad = None
    
    return adv_obs.detach().cpu().numpy().squeeze()

def world_model_I_FGSM(observation, agent, posterior, deterministic, prev_action, epsilon=0.1, attack_steps=5):
    original_obs = torch.tensor(observation, dtype=torch.float32, device=agent.device)
    adv_obs = original_obs.clone().requires_grad_(True)
    
    # 初始状态推断
    with torch.no_grad():
        embedded_init = agent.encoder(original_obs.unsqueeze(0))
        deterministic_init = agent.rssm.recurrent_model(posterior, prev_action, deterministic)
        _, posterior_init = agent.rssm.representation_model(embedded_init, deterministic_init)
    
    current_deterministic = deterministic_init.clone()
    alpha = epsilon / attack_steps
    
    for _ in range(attack_steps):
        # 状态推断
        embedded = agent.encoder(adv_obs.unsqueeze(0))
        prior_dist, prior = agent.rssm.transition_model(current_deterministic)
        post_dist, posterior = agent.rssm.representation_model(embedded, current_deterministic)
        
        # 对抗损失
        recon_dist = agent.decoder(posterior, current_deterministic)
        recon_loss = -recon_dist.log_prob(original_obs).mean()
        reward_dist = agent.reward_predictor(posterior, current_deterministic)
        target_reward = -1.0 * torch.ones_like(reward_dist.mean)
        reward_loss = -reward_dist.log_prob(target_reward).mean()
        kl_loss = torch.distributions.kl.kl_divergence(post_dist, prior_dist).mean()
        total_loss = recon_loss + 0.5 * reward_loss+ 0.2* kl_loss
        # total_loss = recon_loss + 0.5 * reward_loss
        
        # 梯度更新
        adv_obs.grad = None
        total_loss.backward()
        perturbation = alpha * adv_obs.grad.sign()
        
        # 扰动投影
        adv_obs = (adv_obs + perturbation).clamp(original_obs-epsilon, original_obs+epsilon)
        adv_obs = adv_obs.clamp(-0.5, 0.5).detach().requires_grad_(True)
        
        # 隐状态更新
        current_deterministic = agent.rssm.recurrent_model(posterior, prev_action, current_deterministic).detach()
    
    return adv_obs.detach().cpu().numpy().squeeze()

def world_model_FGM(observation, agent, posterior, deterministic, prev_action, epsilon=0.1):
    original_obs = torch.tensor(observation, dtype=torch.float32, device=agent.device)
    adv_obs = original_obs.clone().requires_grad_(True)
    
    # 初始状态推断
    with torch.no_grad():
        embedded_init = agent.encoder(original_obs.unsqueeze(0))
        deterministic_init = agent.rssm.recurrent_model(posterior, prev_action, deterministic)
        _, posterior_init = agent.rssm.representation_model(embedded_init, deterministic_init)
    current_deterministic = deterministic_init.clone()
    # 单次攻击
    embedded = agent.encoder(adv_obs.unsqueeze(0))
    post_dist, posterior = agent.rssm.representation_model(embedded, deterministic_init)
    prior_dist, prior = agent.rssm.transition_model(current_deterministic)
    # 对抗损失
    recon_dist = agent.decoder(posterior, deterministic_init)
    recon_loss = -recon_dist.log_prob(original_obs).mean()
    reward_dist = agent.reward_predictor(posterior, deterministic_init)
    target_reward = -1.0 * torch.ones_like(reward_dist.mean)
    reward_loss = -reward_dist.log_prob(target_reward).mean()
    kl_loss = torch.distributions.kl.kl_divergence(post_dist, prior_dist).mean()
    total_loss = recon_loss + 0.5 * reward_loss+ 0.2* kl_loss
    # total_loss = recon_loss + 0.5 * reward_loss
    
    # 梯度计算
    total_loss.backward()
    grad = adv_obs.grad.data
    grad_norm = torch.norm(grad, p=2)
    # print(grad_norm)
    if grad_norm > 0:
        perturbation = epsilon * (grad / grad_norm)
    else:
        perturbation = torch.zeros_like(grad)
    
    adv_obs = (original_obs + perturbation).clamp(-0.5, 0.5)
    return adv_obs.detach().cpu().numpy().squeeze()
    

    # TRAP-RSSM (PGD-K)
import torch

def trap_rssm_pgd(
    observation,
    agent,
    posterior,
    deterministic,
    prev_action,
    *,
    epsilon: float = 0.10,
    steps: int = 7,
    K: int = 5,
    alpha: float | None = None,
    # 结构通道权重
    w_dec: float = 1.0,
    w_r: float = 0.5,
    w_kl: float = 0.2,
    w_pi: float = 0,
    # 像素范围（与你的训练保持一致）
    clip_min: float = -0.5,
    clip_max: float = 0.5,
    # 可选动量（MI-FGSM 风格）
    use_momentum: bool = False,
    mu: float = 0.9,
):
    """
    TRAP-RSSM（Temporal Rollout Amplifying Perturbation, PGD 版本）
    只对当前帧 observation 注入像素扰动 delta（||delta||_inf <= epsilon），
    以 RSSM 的 K 步想象轨迹为目标，最大化多步累积的结构破坏。

    参数
    ----
    observation : np.ndarray / torch.Tensor   # CxHxW，像素范围建议与训练一致（默认[-0.5, 0.5]）
    agent       : 你的 Dreamer agent（需有 encoder/decoder/rssm/actor/reward_predictor 接口）
    posterior   : torch.Tensor  # 上一时刻后验（或 recurrent_model_input_init 的初值）
    deterministic: torch.Tensor # 上一时刻确定性隐状态
    prev_action : torch.Tensor  # 上一动作（连续/离散 onehot 均可，需与 agent 接口一致）
    epsilon     : float         # L∞ 扰动半径
    steps       : int           # PGD 迭代步数
    K           : int           # 想象前滚步数
    alpha       : float|None    # PGD 步长，默认 epsilon/steps
    w_dec,w_r,w_kl,w_pi : float # 四项损失的权重
    clip_min/clip_max : float   # 像素裁剪
    use_momentum, mu : bool/float # 是否使用动量更新

    返回
    ----
    adv_obs : np.ndarray  # 对抗后的 observation，形状同输入
    """

    device = agent.device
    # 转 tensor
    if not torch.is_tensor(observation):
        o0 = torch.as_tensor(observation, dtype=torch.float32, device=device)
    else:
        o0 = observation.to(device).detach().float()
    # PGD 步长
    if alpha is None:
        alpha = float(epsilon) / float(max(1, steps))

    # ---------- 1) 预计算 teacher（无扰）参考轨迹 ----------
    with torch.no_grad():
        emb_ref = agent.encoder(o0.unsqueeze(0))
        h_ref   = agent.rssm.recurrent_model(posterior.detach(), prev_action, deterministic.detach())
        _, z_ref = agent.rssm.representation_model(emb_ref, h_ref)

        ref_traj = [(z_ref, h_ref)]  # [(z_i^ref, h_i^ref)]，长度 K+1
        for _ in range(K):
            pr_dist, pr = agent.rssm.transition_model(ref_traj[-1][1])
            h_next = agent.rssm.recurrent_model(pr, prev_action, ref_traj[-1][1])
            z_next = pr  # imagination 用 prior 作为 latent
            ref_traj.append((z_next, h_next))

    # 可选：动量缓冲
    mom = torch.zeros_like(o0, device=device) if use_momentum else None
    # 扰动初始化（从 0 开始；也可改为均匀随机初始化）
    delta = torch.zeros_like(o0, device=device, requires_grad=True)

    # ---------- 2) PGD 外循环（每轮重新构图 + 断图） ----------
    for _ in range(steps):
        # 外部状态在每轮先 detach，避免挂旧图
        posterior_t     = posterior.detach()
        deterministic_t = deterministic.detach()
        prev_action_t   = prev_action.detach() if torch.is_tensor(prev_action) else prev_action

        # 当前对抗观测（裁剪到像素范围）
        adv = (o0 + delta).clamp(clip_min, clip_max)

        # 第一步（i=0）：有像素输入 -> posterior
        emb_adv = agent.encoder(adv.unsqueeze(0))
        h_adv   = agent.rssm.recurrent_model(posterior_t, prev_action_t, deterministic_t)
        post_dist, z_adv = agent.rssm.representation_model(emb_adv, h_adv)

        loss = 0.0
        z_i, h_i = z_adv, h_adv
        post_i   = post_dist  # 仅 i=0 用 posterior，其余步用 prior

        # ---------- 累计 K+1 步的联合目标 ----------
        for i in range(K + 1):
            # (1) KL: posterior vs prior（i=0 用 posterior，其余步 prior vs prior => 0，保留写法方便扩展）
            prior_dist, prior = agent.rssm.transition_model(h_i)
            kl_i = torch.distributions.kl.kl_divergence(
                post_i if i == 0 else prior_dist, prior_dist
            ).mean()

            # (2) 解码重构：对齐 teacher 的解码均值（更稳）
            dec_dist = agent.decoder(z_i, h_i)
            with torch.no_grad():
                o_ref_dist = agent.decoder(ref_traj[i][0], ref_traj[i][1])
                o_ref_mean = o_ref_dist.mean
            nll_i = -dec_dist.log_prob(o_ref_mean).mean()

            # (3) 奖励预测偏移：对齐 teacher 奖励分布的均值
            rew_dist = agent.reward_predictor(z_i, h_i)
            with torch.no_grad():
                r_ref_mean = agent.reward_predictor(ref_traj[i][0], ref_traj[i][1]).mean
            r_loss = -rew_dist.log_prob(r_ref_mean).mean()

            # (4) 行为偏移：动作差异的 L1（离散可改 KL）
            with torch.no_grad():
                a_ref = agent.actor(ref_traj[i][0], ref_traj[i][1])
            a_adv = agent.actor(z_i, h_i)
            act_dev = (a_adv - a_ref).abs().mean()

            loss = loss + w_dec * nll_i + w_r * r_loss + w_kl * kl_i + w_pi * act_dev

            # 想象前滚到下一步
            if i < K:
                h_next = agent.rssm.recurrent_model(z_i, prev_action_t, h_i)
                pr_dist, pr = agent.rssm.transition_model(h_next)
                z_i, h_i = pr, h_next
                post_i = None  # i>0 不再用 posterior

        # ---------- 反向 + PGD（断图！） ----------
        # 清理 delta 的梯度（避免累积）
        if delta.grad is not None:
            delta.grad.zero_()
        # 双保险：清理模型参数的梯度，虽然我们不会用到它们
        for m in [agent.encoder, agent.decoder, agent.rssm, agent.actor, agent.reward_predictor]:
            if hasattr(m, "zero_grad"):
                m.zero_grad(set_to_none=True)

        # 反向
        loss.backward()

        with torch.no_grad():
            if use_momentum:
                # 归一化梯度，动量累积
                g = delta.grad
                g = g / (g.abs().sum() + 1e-8)
                mom.mul_(mu).add_(g)
                delta.add_(alpha * mom.sign())
            else:
                delta.add_(alpha * delta.grad.sign())

            # 投影到 L∞ 球，并裁剪像素范围
            delta.clamp_(-epsilon, epsilon)

        # 关键：用“新叶子”替换 delta，彻底断开本轮计算图
        delta = delta.detach().requires_grad_(True)

    # 最终对抗样本
    adv_final = (o0 + delta).clamp(clip_min, clip_max).detach()
    return adv_final.cpu().numpy().squeeze(),epsilon

def ValRSSM(
    observation,
    agent,
    posterior,
    deterministic,
    prev_action,
    *,
    epsilon: float = 0.10,
    steps: int = 5,
    K: int = 15,
    alpha: float | None = None,
    w_dec: float = 1.0,
    w_r: float = 1.0,
    w_kl: float = 1.0,
    w_pi: float = 1.0,
    clip_min: float = -0.5,
    clip_max: float = 0.5,
    use_amp: bool = True,
    random_start: bool = True,
    use_momentum: bool = False,
    mu: float = 0.9,
):
    """
    TRAP-RSSM (optimized):
    - teacher trajectory cached once
    - grad only wrt delta
    - rollout action is detached to save compute (but w_pi keeps gradient)
    - optional AMP
    """
    device = agent.device
    if alpha is None:
        alpha = float(epsilon) / float(max(1, steps))

    # --- obs to tensor + batchify ---
    if not torch.is_tensor(observation):
        o = torch.as_tensor(observation, dtype=torch.float32, device=device)
    else:
        o = observation.detach().to(device).float()

    added_batch = (o.dim() == 3)  # e.g. (C,H,W) or (H,W,C)
    if added_batch:
        o_b = o.unsqueeze(0)       # (1, ...)
    else:
        o_b = o                    # (B, ...)

    # detach external recurrent inputs
    posterior0 = posterior.detach()
    det0 = deterministic.detach()
    prev_action0 = prev_action.detach() if torch.is_tensor(prev_action) else prev_action

    # --- freeze model params grads (we only need grad wrt delta) ---
    modules = [agent.encoder, agent.decoder, agent.rssm, agent.actor, agent.reward_predictor]
    old_req = []
    for m in modules:
        for p in m.parameters():
            old_req.append(p.requires_grad)
            p.requires_grad_(False)

    autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

    # ============================================================
    # A) Teacher trajectory (no grad)
    # ============================================================
    with torch.no_grad():
        emb0 = agent.encoder(o_b).reshape(o_b.shape[0], -1)
        h0 = agent.rssm.recurrent_model(posterior0, prev_action0, det0)
        _, z0 = agent.rssm.representation_model(emb0, h0)

        ref_traj = [(z0, h0)]
        for _ in range(K):
            z_i, h_i = ref_traj[-1]
            a_i_ref = _actor_action(agent.actor(z_i, h_i))
            h_next = agent.rssm.recurrent_model(z_i, a_i_ref, h_i)
            _, z_next = agent.rssm.transition_model(h_next)
            ref_traj.append((z_next, h_next))

        # cache teacher targets on demand
        o_ref_means = [None] * (K + 1) if w_dec != 0 else None
        r_ref_means = [None] * (K + 1) if w_r != 0 else None
        a_ref_list  = [None] * (K + 1) if w_pi != 0 else None

        for i in range(K + 1):
            z_i, h_i = ref_traj[i]
            if w_dec != 0:
                o_ref_means[i] = agent.decoder(z_i, h_i).mean.detach()
            if w_r != 0:
                r_ref_means[i] = agent.reward_predictor(z_i, h_i).mean.detach()
            if w_pi != 0:
                a_ref_list[i]  = _actor_action(agent.actor(z_i, h_i)).detach()

    # ============================================================
    # B) PGD init
    # ============================================================
    if random_start:
        delta = torch.empty_like(o_b).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(o_b)

    delta = delta.clamp(-epsilon, epsilon).detach().requires_grad_(True)
    mom = torch.zeros_like(delta) if use_momentum else None

    # ============================================================
    # C) PGD loop
    # ============================================================
    for _ in range(steps):
        adv = (o_b + delta).clamp(clip_min, clip_max)

        with autocast_ctx:
            emb_adv = agent.encoder(adv).reshape(adv.shape[0], -1)
            h_adv = agent.rssm.recurrent_model(posterior0, prev_action0, det0)
            post_dist, z_adv = agent.rssm.representation_model(emb_adv, h_adv)

            loss = adv.new_zeros(())
            z_i, h_i = z_adv, h_adv

            # KL only needs prior at i==0
            if w_kl != 0:
                prior_dist0, _ = agent.rssm.transition_model(h_i)
                kl0 = torch.distributions.kl.kl_divergence(post_dist, prior_dist0).mean()
                loss = loss + w_kl * kl0

            for i in range(K + 1):
                # decoder deviation (MSE on mean)
                if w_dec != 0:
                    dec_mean = agent.decoder(z_i, h_i).mean
                    loss = loss + w_dec * (dec_mean - o_ref_means[i]).pow(2).mean()

                # reward deviation (MSE on mean)
                if w_r != 0:
                    r_mean = agent.reward_predictor(z_i, h_i).mean
                    loss = loss + w_r * (r_mean - r_ref_means[i]).pow(2).mean()

                # action deviation (keep gradient for w_pi term)
                if w_pi != 0:
                    a_adv = _actor_action(agent.actor(z_i, h_i))          # gradient flows to delta through z,h
                    loss = loss + w_pi * (a_adv - a_ref_list[i]).abs().mean()
                    a_roll = a_adv.detach()                               # BUT rollout uses detached action (faster)
                else:
                    a_roll = _actor_action(agent.actor(z_i, h_i)).detach()

                # rollout to next imagined step
                if i < K:
                    h_next = agent.rssm.recurrent_model(z_i, a_roll, h_i)
                    _, z_next = agent.rssm.transition_model(h_next)
                    z_i, h_i = z_next, h_next

        # grad only wrt delta
        g = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            if use_momentum:
                # normalize for stable MI-FGSM style momentum
                gg = g / (g.abs().mean() + 1e-8)
                mom.mul_(mu).add_(gg)
                delta.add_(alpha * mom.sign())
            else:
                delta.add_(alpha * g.sign())

            delta.clamp_(-epsilon, epsilon)

        delta = delta.detach().requires_grad_(True)

    adv_final = (o_b + delta).clamp(clip_min, clip_max).detach()

    # restore requires_grad
    idx = 0
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(old_req[idx])
            idx += 1

    # return shape matches input
    adv_np = adv_final[0].cpu().numpy() if added_batch else adv_final.cpu().numpy()
    return adv_np, epsilon





def r_iap_pgd(
    observation,
    agent,
    posterior,
    deterministic,
    prev_action,
    *,
    epsilon: float = 0.10,
    steps: int = 5,
    K: int = 15,
    alpha: float | None = None,
    # loss weights
    w_dec: float = 1.0,
    w_r: float = 1.0,
    w_kl: float = 1.0,      # KL(post||prior) at t
    w_pi: float = 1.0,      # action deviation (keep grad)
    w_prior: float = 0.0,   # optional: multi-step prior-drift (MSE on prior mean)
    # pixel range (after normalization)
    clip_min: float = -0.5,
    clip_max: float = 0.5,
    # speed / stability
    use_amp: bool = True,
    random_start: bool = True,
    deterministic_rollout: bool = True,  # use dist.mean for z to reduce gradient noise
    use_momentum: bool = False,
    mu: float = 0.9,
):
    """
    R-IAP (optimized & stable):

    - Teacher trajectory cached once (no grad)
    - Only compute grad wrt delta (model params frozen)
    - Reuse constant recurrent h_t (depends only on external states, not on delta)
    - Optional deterministic latent rollout (use dist.mean) for lower-variance gradients
    - Optional multi-step prior-drift term (w_prior)
    - Optional AMP (autocast) with float32 loss accumulation

    Returns:
        adv_np: adversarial observation (same shape as input)
        epsilon: epsilon used
    """
    device = agent.device
    if alpha is None:
        alpha = float(epsilon) / float(max(1, steps))

    # ---------------------------
    # Helpers
    # ---------------------------
    def _dist_mean_or_sample(dist, sample_fallback=None):
        """Prefer deterministic mean/mode if available; else fallback to provided sample."""
        if deterministic_rollout:
            if hasattr(dist, "mean") and dist.mean is not None:
                return dist.mean
            if hasattr(dist, "mode") and dist.mode is not None:
                return dist.mode
        # fallback
        if sample_fallback is not None:
            return sample_fallback
        # last resort
        if hasattr(dist, "rsample"):
            return dist.rsample()
        if hasattr(dist, "sample"):
            return dist.sample()
        raise RuntimeError("Distribution has no mean/mode/sample.")

    def _actor_action(actor_out):
        """
        actor_out can be:
        - torch Distribution: take mean/mode if deterministic_rollout else sample
        - tuple/list: common (dist, extra)
        - tensor action already
        """
        if isinstance(actor_out, (tuple, list)):
            actor_out = actor_out[0]

        if torch.is_tensor(actor_out):
            return actor_out

        # distribution-like
        if deterministic_rollout:
            if hasattr(actor_out, "mean") and actor_out.mean is not None:
                return actor_out.mean
            if hasattr(actor_out, "mode") and actor_out.mode is not None:
                return actor_out.mode

        if hasattr(actor_out, "rsample"):
            return actor_out.rsample()
        if hasattr(actor_out, "sample"):
            return actor_out.sample()

        raise RuntimeError("Unsupported actor output type.")

    def _mse(a, b):
        return (a - b).pow(2).mean()

    # ---------------------------
    # obs to tensor + batchify
    # ---------------------------
    if not torch.is_tensor(observation):
        o = torch.as_tensor(observation, dtype=torch.float32, device=device)
    else:
        o = observation.detach().to(device).float()

    added_batch = (o.dim() == 3)  # e.g. (C,H,W) or (H,W,C)
    o_b = o.unsqueeze(0) if added_batch else o
    B = o_b.shape[0]

    # detach external recurrent inputs
    posterior0 = posterior.detach()
    det0 = deterministic.detach()
    prev_action0 = prev_action.detach() if torch.is_tensor(prev_action) else prev_action

    # ---------------------------
    # Freeze model params grads (only need grad wrt delta)
    # ---------------------------
    modules = [agent.encoder, agent.decoder, agent.rssm, agent.actor, agent.reward_predictor]
    old_req = []
    for m in modules:
        for p in m.parameters():
            old_req.append(p.requires_grad)
            p.requires_grad_(False)

    autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

    # ============================================================
    # A) Cache constant recurrent state (does NOT depend on delta)
    # ============================================================
    with torch.no_grad():
        h_const = agent.rssm.recurrent_model(posterior0, prev_action0, det0)

    # ============================================================
    # B) Teacher trajectory (no grad)
    # ============================================================
    with torch.no_grad():
        emb0 = agent.encoder(o_b).reshape(B, -1)
        post0_dist, z0_samp = agent.rssm.representation_model(emb0, h_const)
        z0 = _dist_mean_or_sample(post0_dist, z0_samp)

        ref_traj = [(z0, h_const)]
        prior_ref_mean = [None] * (K + 1)  # prior at each step (optional)
        prior_ref_mean[0] = None

        # rollout in latent space (teacher)
        for i in range(K):
            z_i, h_i = ref_traj[-1]
            a_i = _actor_action(agent.actor(z_i, h_i))
            h_next = agent.rssm.recurrent_model(z_i, a_i, h_i)
            prior_dist, z_next_samp = agent.rssm.transition_model(h_next)
            z_next = _dist_mean_or_sample(prior_dist, z_next_samp)
            ref_traj.append((z_next, h_next))

            if w_prior != 0:
                if hasattr(prior_dist, "mean") and prior_dist.mean is not None:
                    prior_ref_mean[i + 1] = prior_dist.mean.detach()
                else:
                    prior_ref_mean[i + 1] = z_next.detach()

        # cache teacher targets
        o_ref_means = [None] * (K + 1) if w_dec != 0 else None
        r_ref_means = [None] * (K + 1) if w_r != 0 else None
        a_ref_list  = [None] * (K + 1) if w_pi != 0 else None

        for i in range(K + 1):
            z_i, h_i = ref_traj[i]
            if w_dec != 0:
                o_ref_means[i] = agent.decoder(z_i, h_i).mean.detach()
            if w_r != 0:
                r_ref_means[i] = agent.reward_predictor(z_i, h_i).mean.detach()
            if w_pi != 0:
                a_ref_list[i] = _actor_action(agent.actor(z_i, h_i)).detach()

    # ============================================================
    # C) PGD init
    # ============================================================
    if random_start:
        delta = torch.empty_like(o_b).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(o_b)

    delta = delta.clamp(-epsilon, epsilon).detach().requires_grad_(True)
    mom = torch.zeros_like(delta) if use_momentum else None

    # ============================================================
    # D) PGD loop
    # ============================================================
    for _ in range(steps):
        adv = (o_b + delta).clamp(clip_min, clip_max)

        # IMPORTANT: keep loss accumulation in float32 for stability
        loss32 = adv.new_zeros((), dtype=torch.float32)

        with autocast_ctx:
            emb_adv = agent.encoder(adv).reshape(B, -1)
            post_dist, z_adv_samp = agent.rssm.representation_model(emb_adv, h_const)
            z_adv = _dist_mean_or_sample(post_dist, z_adv_samp)

            # --- KL at t (posterior vs prior) ---
            if w_kl != 0:
                prior0_dist, _ = agent.rssm.transition_model(h_const)
                kl0 = torch.distributions.kl.kl_divergence(post_dist, prior0_dist)
                loss32 = loss32 + float(w_kl) * kl0.float().mean()

            z_i, h_i = z_adv, h_const

            for i in range(K + 1):
                # decoder deviation
                if w_dec != 0:
                    dec_mean = agent.decoder(z_i, h_i).mean
                    loss32 = loss32 + float(w_dec) * _mse(dec_mean.float(), o_ref_means[i].float())

                # reward deviation
                if w_r != 0:
                    r_mean = agent.reward_predictor(z_i, h_i).mean
                    loss32 = loss32 + float(w_r) * _mse(r_mean.float(), r_ref_means[i].float())

                # action deviation (keep gradient to delta through z_i/h_i)
                if w_pi != 0:
                    a_adv = _actor_action(agent.actor(z_i, h_i))
                    # L1 is often stabler for actions than MSE; keep your original choice
                    loss32 = loss32 + float(w_pi) * (a_adv.float() - a_ref_list[i].float()).abs().mean()
                    a_roll = a_adv.detach()  # rollout uses detached action for speed
                else:
                    a_roll = _actor_action(agent.actor(z_i, h_i)).detach()

                # rollout to next imagined step
                if i < K:
                    h_next = agent.rssm.recurrent_model(z_i, a_roll, h_i)
                    prior_dist, z_next_samp = agent.rssm.transition_model(h_next)
                    z_next = _dist_mean_or_sample(prior_dist, z_next_samp)

                    # optional: multi-step prior drift (align priors to teacher)
                    if w_prior != 0:
                        if hasattr(prior_dist, "mean") and prior_dist.mean is not None:
                            prior_mean_adv = prior_dist.mean
                        else:
                            prior_mean_adv = z_next
                        loss32 = loss32 + float(w_prior) * _mse(
                            prior_mean_adv.float(),
                            prior_ref_mean[i + 1].float()
                        )

                    z_i, h_i = z_next, h_next

        # grad only wrt delta
        g = torch.autograd.grad(loss32, delta, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            if use_momentum:
                # MI-FGSM style momentum (normalize to stabilize)
                gg = g / (g.abs().mean() + 1e-8)
                mom.mul_(mu).add_(gg)
                delta.add_(alpha * mom.sign())
            else:
                delta.add_(alpha * g.sign())

            delta.clamp_(-epsilon, epsilon)

        delta = delta.detach().requires_grad_(True)

    adv_final = (o_b + delta).clamp(clip_min, clip_max).detach()

    # ---------------------------
    # Restore requires_grad
    # ---------------------------
    idx = 0
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(old_req[idx])
            idx += 1

    # return shape matches input
    adv_np = adv_final[0].cpu().numpy() if added_batch else adv_final.cpu().numpy()
    return adv_np, epsilon










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


def attack(config_file):
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

    load_model_weights(agent, "checkpoints/walker-walk/final_model_weights.pth")
    
    # save_dir = (
    #     get_base_directory()
    #     + "/rendered_images/"
    #     + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     + "_"
    #     + config.operation.log_dir
    # )
   
    # attack_dir = (
    #     get_base_directory()
    #     + "/attack/"
    #     + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     + "_"
    #     + config.operation.log_dir
    # )
    # os.makedirs(attack_dir, exist_ok=True)
    # os.makedirs(save_dir, exist_ok=True)
    # for i in range(1):
    #     posterior, deterministic = agent.rssm.recurrent_model_input_init(1)
    #     action = torch.zeros(1, agent.action_size).to(agent.device)
    #     observation = env.reset()
    #     score = 0
    #     done = False
    #     step =0
    #     while not done:
    #         # img = env.render(mode='rgb_array')
    #         # if img is not None:
    #         #     img = Image.fromarray(img)
    #         #     img.save(os.path.join(save_dir, f"step_{step:04d}.png"))
    #         #     # print(f"Saved image for step {step}")
    #         # else:
    #         #     print(f"No image rendered for step {step}")
    #         embedded_observation = agent.encoder(
    #         torch.from_numpy(observation).float().to(agent.device)
    #     )
        
    #         deterministic = agent.rssm.recurrent_model(
    #             posterior, action, deterministic
    #         )
    #         embedded_observation = embedded_observation.reshape(1, -1)
    #         _, posterior = agent.rssm.representation_model(
    #             embedded_observation, deterministic
    #         )
    #         action = agent.actor(posterior, deterministic).detach()

    #         if agent.discrete_action_bool:
    #             buffer_action = action.cpu().numpy()
    #             env_action = buffer_action.argmax()

    #         else:
    #             buffer_action = action.cpu().numpy()[0]
    #             env_action = buffer_action

    #         next_observation, reward, done, info = env.step(env_action)
    #         score += reward
    #         observation = next_observation
    #         step +=1
    #     writer.add_scalar("world_model_attack_score", score, i)
        # print(score)
        # print(step)
    eps_list = [round(0.02 *i , 2) for i in range(10)] 
    for epl in eps_list:
        print(f"\n=== Testing epsilon = {epl:.2f} ===")  
        actionlosslist =[]
        SSIMlist = []
        scorelist = []
        for i in range(5):
            posterior, deterministic = agent.rssm.recurrent_model_input_init(1)
            action = torch.zeros(1, agent.action_size).to(agent.device)
            observation = env.reset()
            score = 0
            done = False
            step =0
            actionloss=0
            SSIM =0
            while not done:
                # img = env.render(mode='rgb_array')
                # if img is not None:
                #     img = Image.fromarray(img)
                #     img.save(os.path.join(attack_dir, f"stepbe_{step:04d}.png"))
                #     # print(f"Saved image for step {step}")
                # else:
                #     print(f"No image rendered for step {step}")

                # adv_obs = observation.copy()
                # adv_obs = world_model_FGSM(
                # observation, agent,posterior, deterministic, action, epsilon = epl
                # )
                

                # adv_obs, _ = ValRSSM(
                # observation, agent, posterior, deterministic, action,
                # epsilon=epl, steps=5, K=15,
                # w_dec=1, w_r=1, w_kl=1, w_pi=1,
                # clip_min=-0.5, clip_max=0.5,
                # use_amp=True)

                adv_obs, _ = r_iap_pgd(
                    observation, agent, posterior, deterministic, action,
                    epsilon=epl, steps=5, K=15,
                    w_dec=1, w_r=1, w_kl=1, w_pi=1,
                    clip_min=-0.5, clip_max=0.5,
                    use_amp=True)
                
                
                # adv_obs = value_adv_attack1(observation, agent, posterior, deterministic,
                #            epsilon=epl, attack_steps=5,
                #            direction="min", loss_mode="value_mean")


                # adv_obs = world_model_pgd_k(observation, agent, posterior, deterministic, action,
                #       epsilon=epl, steps=5, K=5, alpha=None, clip_min=-0.5, clip_max=0.5, use_momentum=False, mu=0.9)

                # adv_obs,epl=trap_rssm_pgd(
                #     observation, agent, posterior, deterministic, action,
                #     epsilon=epl, steps=5, K=15, alpha=None,
                #     w_dec=1.0, w_r=1, w_kl=1, w_pi=1,
                #     clip_min=-0.5, clip_max=0.5
                # )
                # print((observation+0.5).transpose(1, 2, 0).shape)
                if step ==0:
                    SSIM = calculate_ssim((observation+0.5).transpose(1, 2, 0),(adv_obs+0.5).transpose(1, 2, 0))
                # plt.figure()
                # plt.imshow((observation+0.5).transpose(1, 2, 0))
                # plt.axis('off')
                # # plt.savefig(f"rendered_images/{timestamp}/{step}.png", dpi=300)
                # plt.savefig("./5.png", dpi=300)
                # plt.close()
                # plt.figure()
                # plt.imshow(((adv_obs+0.5)).transpose(1, 2, 0))
                # plt.axis('off')
                # plt.savefig("./6.png", dpi=300)
                # plt.close()
                # print(1)
                # raise ValueError(1)
                

                deterministic = agent.rssm.recurrent_model(
                    posterior, action, deterministic
                )


                embedded_observation = agent.encoder(
                    torch.from_numpy(observation).float().to(agent.device)
                )
                embedded_observation = embedded_observation.reshape(1, -1)
                
                _, posterior = agent.rssm.representation_model(
                    embedded_observation, deterministic
                )
                action1 = agent.actor(posterior, deterministic).detach()

                if agent.discrete_action_bool:
                    buffer_action = action1.cpu().numpy()
                    obs_action = buffer_action.argmax()

                else:
                    buffer_action = action1.cpu().numpy()[0]
                    obs_action = buffer_action


                
                embedded_observation = agent.encoder(
                    torch.from_numpy(adv_obs).float().to(agent.device)
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
                

                actionloss += compute_action_difference(obs_action,env_action)
                # print(env_action)
                next_observation, reward, done, info = env.step(env_action)
                # print(reward)
                score += reward
                observation = next_observation
                step +=1
                writer.add_scalar("world_model_attack_score", score, int(i+10))
                # print(step)
            SSIMlist.append(SSIM)
            actionlosslist.append(actionloss/step)
            scorelist.append(score)
        # print(f"ssim:{SSIMlist}")
        # print(f"动作差异：{actionlosslist}")
        # print(f"得分：{scorelist}")
        print(f"平均 SSIM: {sum(SSIMlist) / len(SSIMlist):.4f}")
        print(f"平均动作差异: {sum(actionlosslist) / len(actionlosslist):.4f}")
        print(f"平均得分: {sum(scorelist) / len(scorelist):.2f}")
        # print(step)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="dmc-walker-walk.yml",
    help="config file to run(default: dmc-walker-walk.yml)",
)

attack(parser.parse_args().config)
