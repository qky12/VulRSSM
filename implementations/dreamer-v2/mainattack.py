import os
import numpy as np
import torch
from PIL import Image
from dreamer.algorithms.dreamerv2 import DreamerV2
os.environ["MUJOCO_GL"] = "egl"
import torch.nn.functional as F
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




def _gaussian_kernel2d(ks: int = 5, sigma: float = 1.0, device="cpu", dtype=torch.float32):
    """Create (1,1,ks,ks) gaussian kernel."""
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, ks, ks)

def gaussian_blur_obs(obs_np: np.ndarray, ks: int = 5, sigma: float = 0.8):
    """
    obs_np: (C,H,W) in [-0.5,0.5]
    return: blurred obs with same shape/range
    """
    x = torch.from_numpy(obs_np).float().unsqueeze(0)  # (1,C,H,W)
    device = x.device
    k = _gaussian_kernel2d(ks, sigma, device=device, dtype=x.dtype)
    # depthwise conv for 3 channels
    k = k.repeat(x.shape[1], 1, 1, 1)  # (C,1,ks,ks)
    x_pad = F.pad(x, (ks//2, ks//2, ks//2, ks//2), mode="reflect")
    y = F.conv2d(x_pad, k, groups=x.shape[1])
    y = y.squeeze(0).cpu().numpy()
    return np.clip(y, -0.5, 0.5)

def defend_observation(
    obs_np: np.ndarray,
    method: str = "blur",          # "blur" | "rand_smooth" | "blur+rand"
    *,
    clip_min: float = -0.5,
    clip_max: float = 0.5,
    # blur params
    ks: int = 5,
    sigma: float = 0.8,
    # random smoothing params
    rs_sigma: float = 0.01,        # noise std in normalized pixel space
):
    if method == "blur":
        return gaussian_blur_obs(obs_np, ks=ks, sigma=sigma)

    if method == "rand_smooth":
        noise = np.random.normal(0.0, rs_sigma, size=obs_np.shape).astype(np.float32)
        return np.clip(obs_np + noise, clip_min, clip_max)

    if method == "blur+rand":
        x = gaussian_blur_obs(obs_np, ks=ks, sigma=sigma)
        noise = np.random.normal(0.0, rs_sigma, size=x.shape).astype(np.float32)
        return np.clip(x + noise, clip_min, clip_max)

    raise ValueError(f"Unknown defense method: {method}")

@torch.no_grad()
def act_with_randomized_smoothing(
    agent,
    obs_np: np.ndarray,
    deterministic: torch.Tensor,
    *,
    M: int = 8,                    # number of samples
    rs_sigma: float = 0.01,
    clip_min: float = -0.5,
    clip_max: float = 0.5,
):
    """
    Return action by averaging actions over M noisy copies of obs.
    Works for continuous actions. For discrete, you can do voting (see below).
    """
    device = agent.device
    acts = []
    for _ in range(M):
        noise = np.random.normal(0.0, rs_sigma, size=obs_np.shape).astype(np.float32)
        o = np.clip(obs_np + noise, clip_min, clip_max)
        emb = agent.encoder(torch.from_numpy(o).float().to(device)).reshape(1, -1)
        _, post = agent.rssm.representation_model(emb, deterministic)
        a = agent.actor(post, deterministic)
        # DreamerV2 returns (action, log_prob, entropy)
        if isinstance(a, (tuple, list)):
            a = a[0]
        if (not torch.is_tensor(a)) and hasattr(a, "sample"):
            a = a.sample()
        acts.append(a.detach())

    a_mean = torch.stack(acts, dim=0).mean(dim=0)  # (1, action_dim)
    return a_mean




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




def _dist_to_latent(dist):
    """Deterministic latent for stable gradients: mean > mode > rsample > sample."""
    if dist is None:
        return None
    # continuous distributions usually have .mean
    if hasattr(dist, "mean") and dist.mean is not None:
        return dist.mean
    # some categorical / straight-through dists may expose .mode
    if hasattr(dist, "mode"):
        m = dist.mode
        if torch.is_tensor(m):
            return m
        if callable(m):
            return m()
    # fallback
    if hasattr(dist, "rsample"):
        return dist.rsample()
    return dist.sample()

def R_IAP2(
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
    R-IAP (DreamerV1/V2 compatible)

    - cache constant h_const once (depends only on posterior, prev_action, deterministic)
    - teacher trajectory cached once (deterministic latent: mean/mode)
    - attack rollout uses straight-through latent (differentiable for v2)
    - w_kl includes: initial KL(post||prior) + multi-step latent drift (MSE)
    - AMP-safe: KL and losses accumulated in float32
    """
    device = agent.device
    if alpha is None:
        alpha = float(epsilon) / float(max(1, steps))

    # -------------------------
    # helpers
    # -------------------------
    def _actor_action(out):
        # DreamerV2: (action, log_prob, entropy)
        # DreamerV1: action
        return out[0] if isinstance(out, (tuple, list)) else out

    def _is_onehot_categorical(dist):
        return dist.__class__.__name__ == "OneHotCategorical"

    def _dist_to_latent(dist, deterministic_latent: bool, straight_through: bool):
        """
        For v1 continuous: mean or rsample()
        For v2 discrete OneHotCategorical: mode(onehot argmax) or sample(), optionally straight-through
        Return shape: (B, stochastic_size)
        """
        if _is_onehot_categorical(dist):
            # dist.logits: (B, H, K)
            logits = dist.logits
            probs = F.softmax(logits, dim=-1)

            if deterministic_latent:
                idx = probs.argmax(dim=-1)  # (B, H)
                hard = F.one_hot(idx, num_classes=probs.shape[-1]).to(probs.dtype)  # (B,H,K)
            else:
                hard = dist.sample()  # (B,H,K) onehot

            if straight_through and (not deterministic_latent):
                st = hard + probs - probs.detach()  # ST estimator
            else:
                st = hard

            return st.reshape(st.shape[0], -1)  # (B, H*K)

        # continuous (Normal/Independent Normal)
        if deterministic_latent:
            z = dist.mean
        else:
            z = dist.rsample() if hasattr(dist, "rsample") else dist.sample()
        return z

    def _kl(post_dist, prior_dist):
        """
        Match training-style KL for v2: Independent(OneHotCategorical, 1)
        For v1: regular kl_divergence.
        Always returns float32 scalar mean.
        """
        if _is_onehot_categorical(post_dist) and _is_onehot_categorical(prior_dist):
            # compute in float32 for stability under AMP
            post = torch.distributions.OneHotCategorical(logits=post_dist.logits.float())
            prior = torch.distributions.OneHotCategorical(logits=prior_dist.logits.float())
            post = torch.distributions.Independent(post, 1)   # sum over head dimension
            prior = torch.distributions.Independent(prior, 1)
            return torch.distributions.kl.kl_divergence(post, prior).mean()

        # continuous
        return torch.distributions.kl.kl_divergence(post_dist, prior_dist).float().mean()

    # -------------------------
    # obs to tensor + batchify
    # -------------------------
    if not torch.is_tensor(observation):
        o = torch.as_tensor(observation, dtype=torch.float32, device=device)
    else:
        o = observation.detach().to(device).float()

    added_batch = (o.dim() == 3)  # (C,H,W) or (H,W,C)
    o_b = o.unsqueeze(0) if added_batch else o

    posterior0 = posterior.detach()
    det0 = deterministic.detach()

    if torch.is_tensor(prev_action):
        prev_action0 = prev_action.detach().to(device)
    else:
        prev_action0 = torch.as_tensor(prev_action, dtype=torch.float32, device=device)

    # -------------------------
    # freeze params grads (only need grad wrt delta)
    # -------------------------
    modules = [agent.encoder, agent.decoder, agent.rssm, agent.actor, agent.reward_predictor]
    old_req = []
    for m in modules:
        for p in m.parameters():
            old_req.append(p.requires_grad)
            p.requires_grad_(False)

    autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

    # ============================================================
    # Precompute constant deterministic state h_const (no grad)
    # ============================================================
    with torch.no_grad():
        h_const = agent.rssm.recurrent_model(posterior0, prev_action0, det0)

    # ============================================================
    # A) Teacher trajectory (no grad, deterministic latent)
    # ============================================================
    with torch.no_grad():
        emb0 = agent.encoder(o_b).reshape(o_b.shape[0], -1)
        post0_dist, _ = agent.rssm.representation_model(emb0, h_const)

        # teacher posterior uses deterministic latent (mean/mode)
        z0_ref = _dist_to_latent(post0_dist, deterministic_latent=True, straight_through=False)

        ref_traj = [(z0_ref, h_const)]
        for _ in range(K):
            z_i, h_i = ref_traj[-1]
            a_i_ref = _actor_action(agent.actor(z_i, h_i))

            h_next = agent.rssm.recurrent_model(z_i, a_i_ref, h_i)
            prior_dist, _ = agent.rssm.transition_model(h_next)

            # teacher prior uses deterministic latent (mean/mode)
            z_next_ref = _dist_to_latent(prior_dist, deterministic_latent=True, straight_through=False)
            ref_traj.append((z_next_ref, h_next))

        # cache teacher targets
        o_ref_means = [None] * (K + 1) if w_dec != 0 else None
        r_ref_means = [None] * (K + 1) if w_r != 0 else None
        a_ref_list  = [None] * (K + 1) if w_pi != 0 else None
        z_ref_list  = [None] * (K + 1) if w_kl != 0 else None

        for i in range(K + 1):
            z_i, h_i = ref_traj[i]
            if w_dec != 0:
                o_ref_means[i] = agent.decoder(z_i, h_i).mean.detach()
            if w_r != 0:
                r_ref_means[i] = agent.reward_predictor(z_i, h_i).mean.detach()
            if w_pi != 0:
                a_ref_list[i] = _actor_action(agent.actor(z_i, h_i)).detach()
            if w_kl != 0:
                z_ref_list[i] = z_i.detach()

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
            post_dist, _ = agent.rssm.representation_model(emb_adv, h_const)

            # attack latent must be differentiable for v2 -> straight-through sample
            z_adv = _dist_to_latent(post_dist, deterministic_latent=False, straight_through=True)

            # accumulate loss in float32 (AMP-safe)
            loss = adv.new_zeros((), dtype=torch.float32)

            # initial KL(post||prior) at k=0
            if w_kl != 0:
                prior0_dist, _ = agent.rssm.transition_model(h_const)
                loss = loss + w_kl * _kl(post_dist, prior0_dist)

            z_i, h_i = z_adv, h_const

            for k in range(K + 1):
                # decoder deviation (MSE on mean)
                if w_dec != 0:
                    dec_mean = agent.decoder(z_i, h_i).mean.float()
                    loss = loss + w_dec * (dec_mean - o_ref_means[k].float()).pow(2).mean()

                # reward deviation (MSE on mean)
                if w_r != 0:
                    r_mean = agent.reward_predictor(z_i, h_i).mean.float()
                    loss = loss + w_r * (r_mean - r_ref_means[k].float()).pow(2).mean()

                # action deviation (MAE)
                if w_pi != 0:
                    a_adv = _actor_action(agent.actor(z_i, h_i))
                    loss = loss + w_pi * (a_adv - a_ref_list[k]).abs().mean().float()
                    a_roll = a_adv.detach()  # speed
                else:
                    a_roll = _actor_action(agent.actor(z_i, h_i)).detach()

                # multi-step latent drift (MSE)
                if w_kl != 0 and z_ref_list is not None:
                    loss = loss + (w_kl * 0.1) * (z_i.float() - z_ref_list[k].float()).pow(2).mean()

                # rollout
                if k < K:
                    h_next = agent.rssm.recurrent_model(z_i, a_roll, h_i)
                    prior_dist, _ = agent.rssm.transition_model(h_next)
                    z_next = _dist_to_latent(prior_dist, deterministic_latent=False, straight_through=True)
                    z_i, h_i = z_next, h_next

        # grad only wrt delta
        g = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            if use_momentum:
                gg = g / (g.abs().mean() + 1e-8)
                mom.mul_(mu).add_(gg)
                delta.add_(alpha * mom.sign())
            else:
                delta.add_(alpha * g.sign())

            delta.clamp_(-epsilon, epsilon)

        delta = delta.detach().requires_grad_(True)

    adv_final = (o_b + delta).clamp(clip_min, clip_max).detach()

    # -------------------------
    # restore requires_grad
    # -------------------------
    idx = 0
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(old_req[idx])
            idx += 1

    adv_np = adv_final[0].cpu().numpy() if added_batch else adv_final.cpu().numpy()
    return adv_np, epsilon

def R_IAP(
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
    R-IAP (improved):
    - cache constant h_t once (depends only on external recurrent inputs)
    - teacher trajectory cached once (deterministic latent)
    - attack rollout uses deterministic latent for stable gradients
    - w_kl includes: initial KL(post||prior) + multi-step prior-mean drift accumulation
    - AMP-safe: KL and loss accumulation in float32
    """
    device = agent.device
    if alpha is None:
        alpha = float(epsilon) / float(max(1, steps))

    # --- obs to tensor + batchify ---
    if not torch.is_tensor(observation):
        o = torch.as_tensor(observation, dtype=torch.float32, device=device)
    else:
        o = observation.detach().to(device).float()

    added_batch = (o.dim() == 3)  # (C,H,W) or (H,W,C)
    o_b = o.unsqueeze(0) if added_batch else o  # (B, ...)

    # detach external recurrent inputs
    posterior0 = posterior.detach()
    det0 = deterministic.detach()

    if torch.is_tensor(prev_action):
        prev_action0 = prev_action.detach().to(device)
    else:
        prev_action0 = torch.as_tensor(prev_action, dtype=torch.float32, device=device)

    # --- freeze params grads (only need grad wrt delta) ---
    modules = [agent.encoder, agent.decoder, agent.rssm, agent.actor, agent.reward_predictor]
    old_req = []
    for m in modules:
        for p in m.parameters():
            old_req.append(p.requires_grad)
            p.requires_grad_(False)

    autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

    # ============================================================
    # Precompute constant deterministic state h_const (no grad)
    # ============================================================
    with torch.no_grad():
        h_const = agent.rssm.recurrent_model(posterior0, prev_action0, det0)

    # ============================================================
    # A) Teacher trajectory (no grad, deterministic latent)
    # ============================================================
    with torch.no_grad():
        emb0 = agent.encoder(o_b).reshape(o_b.shape[0], -1)
        # post0, _ = agent.rssm.representation_model(emb0, h_const)
        # z0 = _dist_to_latent(post0)
        _, z0 = agent.rssm.representation_model(emb0, h_const)


        ref_traj = [(z0, h_const)]
        for _ in range(K):
            z_i, h_i = ref_traj[-1]
            a_i_ref = _actor_action(agent.actor(z_i, h_i))
            h_next = agent.rssm.recurrent_model(z_i, a_i_ref, h_i)

            prior_i, _ = agent.rssm.transition_model(h_next)
            z_next = _dist_to_latent(prior_i)
            ref_traj.append((z_next, h_next))

        # cache teacher targets
        o_ref_means = [None] * (K + 1) if w_dec != 0 else None
        r_ref_means = [None] * (K + 1) if w_r != 0 else None
        a_ref_list  = [None] * (K + 1) if w_pi != 0 else None
        z_ref_list  = [None] * (K + 1) if w_kl != 0 else None  # for multi-step drift

        for i in range(K + 1):
            z_i, h_i = ref_traj[i]
            if w_dec != 0:
                o_ref_means[i] = agent.decoder(z_i, h_i).mean.detach()
            if w_r != 0:
                r_ref_means[i] = agent.reward_predictor(z_i, h_i).mean.detach()
            if w_pi != 0:
                a_ref_list[i] = _actor_action(agent.actor(z_i, h_i)).detach()
            if w_kl != 0:
                # store latent "anchor" for drift penalty
                z_ref_list[i] = z_i.detach()

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
            post_dist, _ = agent.rssm.representation_model(emb_adv, h_const)
            z_adv = _dist_to_latent(post_dist)

            # accumulate loss in float32 for stability under AMP
            loss = adv.new_zeros((), dtype=torch.float32)

            # initial KL(post||prior) at step 0
            if w_kl != 0:
                prior0, _ = agent.rssm.transition_model(h_const)
                kl0 = torch.distributions.kl.kl_divergence(post_dist, prior0).float().mean()
                loss = loss + w_kl * kl0

            z_i, h_i = z_adv, h_const

            for i in range(K + 1):
                # (optional) decoder deviation
                if w_dec != 0:
                    dec_mean = agent.decoder(z_i, h_i).mean.float()
                    loss = loss + w_dec * (dec_mean - o_ref_means[i].float()).pow(2).mean()

                # (optional) reward deviation
                if w_r != 0:
                    r_mean = agent.reward_predictor(z_i, h_i).mean.float()
                    loss = loss + w_r * (r_mean - r_ref_means[i].float()).pow(2).mean()

                # (optional) action deviation (keeps gradient to delta through z,h)
                if w_pi != 0:
                    a_adv = _actor_action(agent.actor(z_i, h_i))
                    loss = loss + w_pi * (a_adv - a_ref_list[i]).abs().mean().float()
                    a_roll = a_adv.detach()  # speed
                else:
                    a_roll = _actor_action(agent.actor(z_i, h_i)).detach()

                # multi-step latent drift (prior-mean drift) to encourage temporal accumulation
                if w_kl != 0 and z_ref_list is not None:
                    loss = loss + (w_kl * 0.1) * (z_i.float() - z_ref_list[i].float()).pow(2).mean()

                # rollout
                if i < K:
                    h_next = agent.rssm.recurrent_model(z_i, a_roll, h_i)
                    prior_dist, _ = agent.rssm.transition_model(h_next)
                    z_next = _dist_to_latent(prior_dist)
                    z_i, h_i = z_next, h_next

        # grad only wrt delta
        g = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            if use_momentum:
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

    adv_np = adv_final[0].cpu().numpy() if added_batch else adv_final.cpu().numpy()
    return adv_np, epsilon




def _get_value_dist(agent, obs_tensor, deterministic):
    """
    obs_tensor: torch.Tensor, shape 与你的 encoder 输入一致
    deterministic: torch.Tensor, Dreamer/RSSM 的 deterministic state
    return: value_dist (需要有 .mean 属性；若用 nll_mismatch 还需要 .log_prob)
    """
    embedded = agent.encoder(obs_tensor.unsqueeze(0))   # [1, embed_dim]
    embedded = embedded.reshape(1, -1)

    _, post = agent.rssm.representation_model(
        embedded,
        deterministic.detach()
    )

    # 你的代码里 critic 期望 [B,T,...]，这里保持一致
    value_dist = agent.critic(
        post.unsqueeze(0),  # [1,1,stoch_dim] (视实现而定)
        deterministic.detach().unsqueeze(0)
    )
    return value_dist


def value_fgsm_attack(
    observation,
    agent,
    posterior,       # 保留接口一致（这里不必须用）
    deterministic,
    epsilon=0.03,
    clip_min=-0.5,
    clip_max=0.5,
    objective="mse",         # "mse" | "min" | "nll_mismatch"
    mismatch_scale=5.0,      # objective="nll_mismatch" 时用
    l2_reg=0.0,
):
    """
    攻击 value 网络的 FGSM（单步，L_inf）。
    """
    device = agent.device
    obs = torch.as_tensor(observation, dtype=torch.float32, device=device)

    # 参考 value（只在 objective="mse" 时需要）
    with torch.no_grad():
        clean_vdist = _get_value_dist(agent, obs, deterministic)
        clean_vmean = getattr(clean_vdist, "mean", None)
        if clean_vmean is None:
            raise TypeError("critic 输出没有 .mean，无法做 value attack。")

    obs_adv = obs.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        vdist = _get_value_dist(agent, obs_adv, deterministic)
        vmean = vdist.mean

        if objective == "mse":
            # 让 value 预测尽量偏离干净输入下的 value
            adv_loss = (vmean - clean_vmean).pow(2).mean()

        elif objective == "min":
            # 让 value 尽量变小：最大化 (-value)
            adv_loss = (-vmean).mean()

        elif objective == "nll_mismatch":
            # 让 value 分布对“假目标”的 NLL 尽量大（打乱分布）
            if not hasattr(vdist, "log_prob"):
                raise TypeError("objective='nll_mismatch' 需要 critic 输出支持 .log_prob()。")
            fake_target = torch.randn_like(vmean) * mismatch_scale
            adv_loss = (-vdist.log_prob(fake_target)).mean()

        else:
            raise ValueError(f"Unknown objective: {objective}")

        if l2_reg > 0:
            adv_loss = adv_loss + l2_reg * torch.norm(obs_adv - obs, p=2)

        grad = torch.autograd.grad(adv_loss, obs_adv, retain_graph=False, create_graph=False)[0]

        obs_adv = obs_adv + epsilon * grad.sign()
        obs_adv = torch.clamp(obs_adv, clip_min, clip_max)

    return obs_adv.detach().cpu().numpy()


def value_pgd_attack(
    observation,
    agent,
    posterior,       # 保留接口一致（这里不必须用）
    deterministic,
    epsilon=0.03,
    attack_steps=5,
    alpha=None,
    random_start=True,
    clip_min=-0.5,
    clip_max=0.5,
    objective="mse",         # "mse" | "min" | "nll_mismatch"
    mismatch_scale=5.0,
    l2_reg=0.0,
):
    """
    攻击 value 网络的 PGD（多步迭代 + 投影到 L_inf ball）。
    """
    device = agent.device
    obs = torch.as_tensor(observation, dtype=torch.float32, device=device)

    if alpha is None:
        alpha = epsilon / max(1, attack_steps)

    # 参考 value（只在 objective="mse" 时需要）
    with torch.no_grad():
        clean_vdist = _get_value_dist(agent, obs, deterministic)
        clean_vmean = getattr(clean_vdist, "mean", None)
        if clean_vmean is None:
            raise TypeError("critic 输出没有 .mean，无法做 value attack。")

    # init delta
    if random_start:
        delta = torch.empty_like(obs).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(obs)

    for _ in range(attack_steps):
        delta = delta.detach().requires_grad_(True)
        obs_adv = torch.clamp(obs + delta, clip_min, clip_max)

        with torch.enable_grad():
            vdist = _get_value_dist(agent, obs_adv, deterministic)
            vmean = vdist.mean

            if objective == "mse":
                adv_loss = (vmean - clean_vmean).pow(2).mean()
            elif objective == "min":
                adv_loss = (-vmean).mean()
            elif objective == "nll_mismatch":
                if not hasattr(vdist, "log_prob"):
                    raise TypeError("objective='nll_mismatch' 需要 critic 输出支持 .log_prob()。")
                fake_target = torch.randn_like(vmean) * mismatch_scale
                adv_loss = (-vdist.log_prob(fake_target)).mean()
            else:
                raise ValueError(f"Unknown objective: {objective}")

            if l2_reg > 0:
                adv_loss = adv_loss + l2_reg * torch.norm(delta, p=2)

            grad = torch.autograd.grad(adv_loss, delta, retain_graph=False, create_graph=False)[0]

        # PGD step (梯度上升) + 投影
        delta = delta + alpha * grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)

        # 保证像素范围合法（通过 clamp(obs+delta) 间接实现）
        delta = torch.clamp(obs + delta, clip_min, clip_max) - obs

    obs_adv = torch.clamp(obs + delta, clip_min, clip_max)
    return obs_adv.detach().cpu().numpy()







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

def get_action_tensor(agent, posterior, deterministic):
    out = agent.actor(posterior, deterministic)

    # case 1: (action, something)
    if isinstance(out, tuple):
        act = out[0]
    else:
        act = out

    # case 2: distribution object
    if not torch.is_tensor(act) and hasattr(act, "sample"):
        act = act.sample()

    return act.detach()


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
    elif config.algorithm == 'dreamer-v2':
        from attrdict import AttrDict
        config_dict = dict(config)
        config_dict['parameters']['dreamer']['stochastic_size'] = config_dict['parameters']['dreamer']['categorical_head'] * config_dict['parameters']['dreamer']['categorical_size']
        config = AttrDict(config_dict)
        agent = DreamerV2(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )

    model_path = "checkpoints/cartpole-swingup/final_model_weights.pth"
    load_model_weights(agent, model_path)
    
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

                # adv_obs, _ = R_IAP2(
                #     observation=observation,
                #     agent=agent,
                #     posterior=posterior,
                #     deterministic=deterministic,
                #     prev_action=action,
                #     epsilon=epl,
                #     steps=5,
                #     K=15,
                #     w_dec=1.0, w_r=1.0, w_kl=1.0, w_pi=1.0,
                #     clip_min=-0.5, clip_max=0.5,
                #     use_amp=True, random_start=True,
                # )

                
                adv_obs = value_fgsm_attack(
                    observation=observation,          # np.ndarray，和环境给你的 obs 一样
                    agent=agent,
                    posterior=posterior,      # 你原来传什么就传什么（函数里其实不依赖）
                    deterministic=deterministic,  # torch.Tensor, RSSM deterministic state
                    epsilon=epl,
                    objective="mse",          # 推荐：破坏 value 预测（稳定）
                    clip_min=-0.5,
                    clip_max=0.5,
                )

                # adv_obs = value_pgd_attack(
                #     observation=observation,
                #     agent=agent,
                #     posterior=posterior,
                #     deterministic=deterministic,
                #     epsilon=epl,
                #     attack_steps=5,
                #     alpha=0.03/5,             # 不写也行，默认 epsilon/steps
                #     random_start=True,
                #     objective="mse",
                #     clip_min=-0.5,
                #     clip_max=0.5,
                # )

                # def_obs = defend_observation(
                #     adv_obs,
                #     method="blur+rand",
                #     ks=5, sigma=0.8,
                #     rs_sigma=0.01,
                #     clip_min=-0.5, clip_max=0.5,
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
                action1 = get_action_tensor(agent, posterior, deterministic)

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

                action  = get_action_tensor(agent, posterior, deterministic)

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
    default="dmc-cartpole-swingup_dreamerv2.yml",
    help="config file to run(default:dmc-cartpole-swingup_dreamerv2.yml)",
)

attack(parser.parse_args().config)
