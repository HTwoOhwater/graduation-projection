"""FoundIR 模型主体文件。

这里现在只保留两类内容：
1. 网络结构（如 Unet / UnetRes）
2. 扩散过程本身（ResidualDiffusion）

把 Trainer 和通用工具从这里拆出去的原因是：
模型文件应该尽量回答“模型是什么、前向和采样怎么做”，
而不是同时承担训练调度、日志保存、patch 推理等工程职责。
"""

import math
from collections import namedtuple
from functools import partial
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum, nn
from torch.optim import RAdam
from thop import profile
import importlib

from src.utils import (
    default,
    exists,
    identity,
    normalize_to_neg_one_to_one,
    num_to_groups,
    set_seed,
    tensor2img,
    unnormalize_to_zero_to_one,
)

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])
metric_module = importlib.import_module('metrics')

# small helper modules


class Residual(nn.Module):
    # 最基础的残差包装模块，后面多个 block 都复用这类思想。
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    # 原项目中的基础 U-Net 主干。
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=False,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.depth = len(dim_mults)
        input_channels = channels + channels * (1 if condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(self, x, time):
        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W].contiguous()
        return x


class UnetRes(nn.Module):
    # 面向恢复任务的主干封装。当前训练主线就是围绕它构建。
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        num_unet=1,
        condition=False,
        objective='pred_res_noise',
        test_res_or_noise="res_noise"
    ):
        super().__init__()
        self.condition = condition
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise
        # determine dimensions
    
        self.unet0 = Unet(dim,
                        init_dim=init_dim,
                        out_dim=out_dim,
                        dim_mults=dim_mults,
                        channels=channels,
                        resnet_block_groups=resnet_block_groups,
                        learned_variance=learned_variance,
                        learned_sinusoidal_cond=learned_sinusoidal_cond,
                        random_fourier_features=random_fourier_features,
                        learned_sinusoidal_dim=learned_sinusoidal_dim,
                        condition=condition)

    def forward(self, x, time):
        if self.objective == "pred_noise":
            time = time[1]
        elif self.objective == "pred_res":
            time = time[0]
        return [self.unet0(x, time)]

# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "lamda":
        x = np.linspace(0.0001, 0.02, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        alphas = 1 - y
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)

    return alphas*sum_scale

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class ResidualDiffusion(nn.Module):
    """条件扩散恢复的核心实现。

    如果你以后要继续改 FoundIR，优先在这里理解三件事：
    1. 训练时如何构造 q_sample
    2. 网络到底在预测什么（如 pred_res）
    3. 采样阶段如何从 condition 邻域逐步恢复
    """
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        delta_end = 1.5e-3,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_res_noise',
        ddim_sampling_eta=0.,
        condition=False,
        sum_scale=None,
        test_res_or_noise="None",
    ):
        super().__init__()
        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.test_res_or_noise = test_res_or_noise
        self.delta_end = delta_end

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            # ddim_sampling_eta = 0.25
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        beta_schedule = "linear"
        beta_start = 0.0001
        beta_end = 0.02
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32) # [0.0001, ... ,0.02]
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = (torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2)
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
            
        delta_start = 1e-6
        delta = torch.linspace(delta_start, self.delta_end, timesteps, dtype=torch.float32)
        delta_cumsum = delta.cumsum(dim=0).clip(0, 1)

        alphas = 1.0 - betas # [0.9999, ... ,0.98]
        alphas_cumprod = torch.cumprod(alphas, dim=0) # 累积 [1, 1x2, 1x2x3]
        alphas_cumsum = 1-alphas_cumprod ** 0.5 
        betas2_cumsum = 1-alphas_cumprod

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.) 
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        alphas = alphas_cumsum-alphas_cumsum_prev
        alphas[0] = 0
        betas2 = betas2_cumsum-betas2_cumsum_prev
        betas2[0] = 0
        # print(alphas_cumsum[-1])
        # print(delta_cumsum[-1])
        # raise
        betas_cumsum = torch.sqrt(betas2_cumsum) 

        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        # import pdb
        # pdb.set_trace()

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('delta', delta)
        register_buffer('delta_cumsum', delta_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                        alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def init(self):
        timesteps = 1000

        beta_schedule = "linear"
        beta_start = 0.0001
        beta_end = 0.02
        if beta_schedule == "linear":
            betas = torch.linspace(
                beta_start, beta_end, timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
                
        delta_start = 1e-6
        delta = torch.linspace(delta_start, self.delta_end, timesteps, dtype=torch.float32)
        delta_cumsum = delta.cumsum(dim=0).clip(0, 1)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumsum = 1-alphas_cumprod ** 0.5
        betas2_cumsum = 1-alphas_cumprod

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        alphas = alphas_cumsum-alphas_cumsum_prev
        alphas[0] = alphas[1]
        betas2 = betas2_cumsum-betas2_cumsum_prev
        betas2[0] = betas2[1]

        betas_cumsum = torch.sqrt(betas2_cumsum)
    
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.alphas = alphas
        self.alphas_cumsum = alphas_cumsum
        self.delta = delta
        self.delta_cumsum = delta_cumsum
        self.one_minus_alphas_cumsum = 1-alphas_cumsum
        self.betas2 = betas2
        self.betas = torch.sqrt(betas2)
        self.betas2_cumsum = betas2_cumsum
        self.betas_cumsum = betas_cumsum
        self.posterior_mean_coef1 = betas2_cumsum_prev/betas2_cumsum
        self.posterior_mean_coef2 = (
            betas2 * alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum
        self.posterior_mean_coef3 = betas2/betas2_cumsum
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res): 
        ##  self.predict_noise_from_res(x, t, x_input, pred_res) # x_input: LQ, x: 0.1 * x_input + 0.1 * torch.randn_like(x_input)
        return (
            (x_t - (1-extract(self.delta_cumsum,t,x_t.shape)) * x_input - (extract(self.alphas_cumsum, t, x_t.shape)-1)
             * pred_res)/extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise): 
        return (
            (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input -
             extract(self.betas_cumsum, t, x_t.shape) * noise)/extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise): 
        return (
            x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
            extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t): 
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t): 
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, task=None, clip_denoised=True): 
        if not self.condition: # x_input: LQ, x: img = 0.1 * x_input + 0.1 * torch.randn_like(x_input)
            x_in = x
        else:
            x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in,
                                  [self.alphas_cumsum[t]*self.num_timesteps,
                                      self.betas_cumsum[t]*self.num_timesteps])
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            if self.test_res_or_noise == "res_noise":
                pred_res = model_output[0]
                pred_noise = model_output[1]
                pred_res = maybe_clip(pred_res)
                x_start = self.predict_start_from_res_noise(
                    x, t, pred_res, pred_noise)
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "res":
                pred_res = model_output[0]
                pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(
                    x, t, x_input, pred_res)
                x_start = x_input - pred_res
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "noise":
                pred_noise = model_output[1]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                x_start = maybe_clip(x_start)
                pred_res = x_input - x_start
                pred_res = maybe_clip(pred_res)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input-model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res) # x_input: LQ, x: 0.1 * x_input + 0.1 * torch.randn_like(x_input)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t):
        preds = self.model_predictions(x_input, x, t)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x_input, x=x, t=batched_times)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        x_input = x_input[0]

        batch, device = shape[0], self.betas.device

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(x_input, img, t)

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True, task=None):
        x_input = x_input[0]

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)#[:num] # [-1, 332, 665, 999]

        times = list(reversed(times.int().tolist())) # [999, 665, 332, -1]
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:])) # [(999, 665), (665, 332), (332, -1)]
   
        if self.condition:
            img = (1-self.delta_cumsum[-1]) * x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device) # self.delta_cumsum: [1e-6, 3.80e-6, 0.003, ..., 0.9005]; img = 0.1 * x_input + 0.1 * torch.randn
            # img = (1-self.delta_cumsum[-1]) * x_input + self.betas_cumsum[-1] * torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"
        # type = "use_x_start"
        last=False

        if not last:
            img_list = []

        # import pdb 
        # pdb.set_trace()

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'): # (999, 665), (665, 332), (332, -1)
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)  # tensor([999]) 1x1
            preds = self.model_predictions(x_input, img, time_cond, task)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next
            delta_cumsum = self.delta_cumsum[time]
            delta_cumsum_next = self.delta_cumsum[time_next]
            delta = delta_cumsum-delta_cumsum_next
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum
            q = betas2_cumsum_next / betas2_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)
            if type == "use_pred_noise":
                img = img - alpha*pred_res + delta*x_input + sigma2.sqrt()*noise
            elif type == "use_x_start":
                # img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*img + \
                #     (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
                #     (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
                #     (delta_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum-delta_cumsum_next)*x_input + \
                #     sigma2.sqrt()*noise
                img = q*img + \
                    (1-q)*x_start + \
                    (alpha_cumsum_next-alpha_cumsum*q)*pred_res + \
                    (delta_cumsum*q-delta_cumsum_next)*x_input + \
                    sigma2.sqrt()*noise
            elif type == "special_eta_0":
                img = img - alpha*pred_res - \
                    (betas_cumsum-betas_cumsum_next)*pred_noise
            elif type == "special_eta_1":
                img = img - alpha*pred_res - betas2/betas_cumsum*pred_noise + \
                    betas*betas2_cumsum_next.sqrt()/betas_cumsum*noise
                    
            if not last:
                img_list.append(img)
    
        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
                # img_list = img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True, task=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample  # se;f.ddim_sample
        if self.condition:
            x_input = 2 * x_input - 1
            x_input = x_input.unsqueeze(0)

            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)
        return sample_fn(x_input, size, last=last, task=task)

    def q_sample(self, x_start, x_res, condition, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            x_start+extract(self.alphas_cumsum, t, x_start.shape) * x_res +
            extract(self.betas_cumsum, t, x_start.shape) * noise -
            extract(self.delta_cumsum, t, x_start.shape) * condition
        )

    @property
    def loss_fn(self, loss_type='l1'):
        if loss_type == 'l1':
            return F.l1_loss
        elif loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def p_losses(self, imgs, t, noise=None):
        if isinstance(imgs, list):  # Condition
            x_input = 2 * imgs[1] - 1
            x_start = 2 * imgs[0] - 1  #gt:imgs[0], cond:imgs[1]
            task = imgs[2][0]


        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start # cond-GT, x_input=LQ, x_start=GT

        b, c, h, w = x_start.shape

        # noise sample
        x = self.q_sample(x_start, x_res, x_input, t, noise=noise) # 

        # predict and take gradient step
        if not self.condition:
            x_in = x
        else:
            x_in = torch.cat((x, x_input), dim=1)

        model_out = self.model(x_in,
                               [self.alphas_cumsum[t]*self.num_timesteps,
                                self.betas_cumsum[t]*self.num_timesteps])

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)

            pred_res = x_input-model_out[0]
            pred_noise = model_out[1]
        elif self.objective == "pred_noise":
            target.append(noise)

            pred_noise = model_out[0]

        elif self.objective == "pred_res":
            target.append(x_res)
        
            pred_res = model_out[0]

        else:
            raise ValueError(f'unknown objective {self.objective}')

        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t)
            loss = 10000*self.loss_fn(x_u, u_gt, reduction='none')
            return [loss]
        else:
            loss_list = []
            for i in range(len(model_out)):
                loss = self.loss_fn(model_out[i], target[i], reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
                loss_list.append(loss)
            return loss_list

    def forward(self, img, *args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device, img_size, = * \
                img[0].shape, img[0].device, self.image_size
        else:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long() # b个0-timesteps int []

        return self.p_losses(img, t, *args, **kwargs)
    # def forward(self, x_input_sample, batches, last, file_):
    #     # profile
    #     return self.sample(x_input_sample, batch_size=batches, last=last, task=file_)
