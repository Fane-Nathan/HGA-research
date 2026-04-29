# === Trackmania =======================================================================================================


# standard library imports

# third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from math import floor, sqrt
from torch.nn import Conv2d, Module, ModuleList
# import torchvision

# local imports
from tmrl.util import prod
from tmrl.actor import TorchActorModule
import tmrl.config.config_constants as cfg


# SUPPORTED ============================================================================================================


# Spinup MLP: =======================================================
# Adapted from the SAC implementation of OpenAI Spinup


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20.0
EPSILON = 1e-7


def _compute_log_std_smooth(log_std_raw):
    """Tanh-based smooth log_std parametrization. Always has gradient."""
    return LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std_raw) + 1)


def _squashed_gaussian_logprob(pi_distribution, pi_action):
    """Compute per-dim and total log prob with tanh squashing correction."""
    logp_per_dim = pi_distribution.log_prob(pi_action)
    tanh_correction = 2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))
    logp_per_dim = logp_per_dim - tanh_correction

    # Keep gradients intact for entropy/actor updates (no hard clamp here).

    logp_total = logp_per_dim.sum(dim=-1)
    return logp_total, logp_per_dim


def _apply_sagr(pre_tanh_action, delta=0.05):
    """
    SAGR: Squash-Aware Gradient Rescaling (Novel — part of GRAC algorithm).

    Registers a backward hook on the pre-tanh action tensor that rescales
    gradients to prevent vanishing through tanh saturation.

    When |action| approaches 1, the tanh derivative (1 - a^2) approaches 0,
    killing the policy gradient. SAGR ensures the effective Jacobian never
    drops below `delta`, preserving gradient flow at action boundaries.

    Math:
        Standard gradient:  dL/du = dL/da * (1 - a^2)
        SAGR gradient:      dL/du = dL/da * max(1 - a^2, delta)

    Implementation:
        The hook receives grad_u (which already has (1-a^2) baked in from
        autograd). We undo the vanished Jacobian and replace with the floored
        version: grad_corrected = grad_u * max(1-a^2, delta) / (1-a^2 + eps)

    Args:
        pre_tanh_action: u (before tanh), must have requires_grad=True
        delta: minimum gradient floor (default 0.05 = 5% of max gradient)
    """
    if not pre_tanh_action.requires_grad:
        return

    # Compute squashed action for correction factor (detached from graph)
    a_detached = torch.tanh(pre_tanh_action).detach()

    def hook(grad):
        a_sq = a_detached ** 2
        tanh_jac = 1.0 - a_sq                # Standard Jacobian: (1 - a^2)
        safe_jac = tanh_jac.clamp(min=delta)  # Floored Jacobian: max(1-a^2, delta)
        correction = safe_jac / (tanh_jac + 1e-8)
        return grad * correction

    pre_tanh_action.register_hook(hook)


class SquashedGaussianMLPActor(TorchActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__(observation_space, action_space)
        try:
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            dim_obs = prod(observation_space.shape)
            self.tuple_obs = False
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        x = torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        # pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


class MLPQFunction(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        try:
            obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
            self.tuple_obs = True
        except TypeError:
            obs_dim = prod(obs_space.shape)
            self.tuple_obs = False
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1) if self.tuple_obs else torch.cat((torch.flatten(obs, start_dim=1), act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.  # FIXME: understand this


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


# REDQ MLP: =====================================================


class REDQMLPActorCritic(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU,
                 n=10):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.n = n
        self.qs = ModuleList([MLPQFunction(observation_space, action_space, hidden_sizes, activation) for _ in range(self.n)])

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# CNNs: ================================================================================================================

# EfficientNet =========================================================================================================

# EfficientNetV2 implementation adapted from https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
# We use the EfficientNetV2 structure for image features and we merge the TM2020 float features to linear layers


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.GroupNorm(num_groups=1, num_channels=oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.GroupNorm(num_groups=1, num_channels=oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, nb_channels_in=3, dim_output=1, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(nb_channels_in, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, dim_output)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


class SquashedGaussianEffNetActor(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        self.cnn = effnetv2_s(nb_channels_in=4, dim_output=247, width_mult=1.).float()
        self.net = mlp([256, 256], [nn.ReLU, nn.ReLU])
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        imgs_tensor = obs[3].float()
        float_tensors = (obs[0], obs[1], obs[2], *obs[4:])
        float_tensor = torch.cat(float_tensors, -1).float()
        cnn_out = self.cnn(imgs_tensor)
        mlp_in = torch.cat((cnn_out, float_tensor), -1)
        net_out = self.net(mlp_in)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        # pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        import sys
        size = sys.getsizeof(obs)
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()


class EffNetQFunction(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class EffNetActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# Vanilla CNN FOR GRAYSCALE IMAGES: ====================================================================================


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out


class VanillaCNN(Module):
    def __init__(self, q_net):
        super(VanillaCNN, self).__init__()
        self.q_net = q_net
        self.h_out, self.w_out = cfg.IMG_HEIGHT, cfg.IMG_WIDTH
        hist = cfg.IMG_HIST_LEN

        self.conv1 = Conv2d(hist, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels
        self.flat_features = self.out_channels * self.h_out * self.w_out
        self.mlp_input_features = self.flat_features + 12 if self.q_net else self.flat_features + 9
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        if self.q_net:
            speed, gear, rpm, images, act1, act2, act = x
        else:
            speed, gear, rpm, images, act1, act2 = x

        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, f"x.shape:{x.shape}, flat_features:{flat_features}, self.out_channels:{self.out_channels}, self.h_out:{self.h_out}, self.w_out:{self.w_out}"
        x = x.view(-1, flat_features)
        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)
        return x


class SquashedGaussianVanillaCNNActor(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = VanillaCNN(q_net=False)
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # NB: this is from Spinup:
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)  # FIXME: this formula is mathematically wrong, no idea why it seems to work
            # Whereas SB3 does this:
            # logp_pi -= torch.sum(torch.log(1 - torch.tanh(pi_action) ** 2 + EPSILON), dim=1)  # TODO: double check
            # # log_prob -= th.sum(th.log(1 - actions**2 + self.epsilon), dim=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        # pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()


class VanillaCNNQFunction(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = VanillaCNN(q_net=True)

    def forward(self, obs, act):
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class VanillaCNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        # build policy and value functions
        self.actor = SquashedGaussianVanillaCNNActor(observation_space, action_space)
        self.q1 = VanillaCNNQFunction(observation_space, action_space)
        self.q2 = VanillaCNNQFunction(observation_space, action_space)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# Vanilla CNN FOR COLOR IMAGES: ========================================================================================

def remove_colors(images):
    """
    We remove colors so that we can simply use the same structure as the grayscale model.

    The "color" default pipeline is mostly here for support, as our model effectively gets rid of 2 channels out of 3.
    If you actually want to use colors, do not use the default pipeline.
    Instead, you need to code a custom model that doesn't get rid of them.
    """
    images = images[:, :, :, :, 0]
    return images


class SquashedGaussianVanillaColorCNNActor(SquashedGaussianVanillaCNNActor):
    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, act1, act2 = obs
        images = remove_colors(images)
        obs = (speed, gear, rpm, images, act1, act2)
        return super().forward(obs, test=False, with_logprob=True)


class VanillaColorCNNQFunction(VanillaCNNQFunction):
    def forward(self, obs, act):
        speed, gear, rpm, images, act1, act2 = obs
        images = remove_colors(images)
        obs = (speed, gear, rpm, images, act1, act2)
        return super().forward(obs, act)


class VanillaColorCNNActorCritic(VanillaCNNActorCritic):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # build policy and value functions
        self.actor = SquashedGaussianVanillaColorCNNActor(observation_space, action_space)
        self.q1 = VanillaColorCNNQFunction(observation_space, action_space)
        self.q2 = VanillaColorCNNQFunction(observation_space, action_space)


# HybridNanoEffNet: VRAM-Efficient CNN with SE Attention ===============================================================
# Uses existing EffNetV2/MBConv blocks with a lightweight "Nano" configuration
# Benefits: ~90% smaller than ResNet18, SE attention for important features, SiLU for smooth RL gradients

# Nano config: [expansion_ratio, channels, num_layers, stride, use_se]
NANO_EFFNET_CFG = [
    [1, 24, 2, 1, 0],   # MBConv1: don't expand, keep 24 ch
    [4, 32, 2, 2, 0],   # MBConv6: expand 4x, downsample to 32 ch
    [4, 48, 2, 2, 1],   # MBConv6: expand 4x, downsample to 48 ch, USE SE (Attention)
    [4, 64, 3, 2, 1],   # MBConv6: expand 4x, downsample to 64 ch, USE SE (Attention)
]


class HybridNanoEffNetActor(TorchActorModule):
    """
    Optimized Actor: Uses EffNetV2 (MBConv) for images + Gated Fusion for LIDAR.
    High Quality (SE Attention) + Low VRAM (Depthwise Convs).
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        # --- 1. Smart Image Branch (EffNetV2) ---
        # We reuse the EffNetV2 class with our Nano config.
        # dim_output=128 means the CNN will output a vector of size 128.
        self.cnn = EffNetV2(cfgs=NANO_EFFNET_CFG, nb_channels_in=4, dim_output=128)

        # --- 2. LIDAR/Telemetry Branch ---
        # Speed(1) + Gear(1) + RPM(1) + Lidar(19) + PastActions(2*dim_act)
        self.dim_float_input = 1 + 1 + 1 + 19 + dim_act + dim_act

        # Float Encoder
        self.float_mlp = mlp([self.dim_float_input, 64, 64], nn.ReLU, nn.ReLU)

        # --- 3. Gated Fusion with LayerNorm ---
        # Visual features are 128, float features are 64.
        self.fusion_norm = nn.LayerNorm(128 + 64)
        self.net = mlp([128 + 64, 256, 256], SiLU, SiLU)  # SiLU (Swish) is better for RL

        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        # Unpack observation tuple
        speed, gear, rpm, images, lidar, act1, act2 = obs

        # 1. Images -> EffNet Features
        img_embed = self.cnn(images.float())  # Output: (Batch, 128)

        # 2. Floats -> MLP Features
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self.float_mlp(floats)  # Output: (Batch, 64)

        # 3. Fusion with LayerNorm for stable training
        combined = torch.cat((img_embed, float_embed), dim=-1)
        combined = self.fusion_norm(combined)

        # 4. Policy Head
        net_out = self.net(combined)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # SAC Logic
        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()


class HybridNanoEffNetQFunction(nn.Module):
    """Q-function using same EffNet architecture for observations."""
    def __init__(self, observation_space, action_space):
        super().__init__()
        dim_act = action_space.shape[0]

        # Re-use same architecture structure
        self.cnn = EffNetV2(cfgs=NANO_EFFNET_CFG, nb_channels_in=4, dim_output=128)
        self.dim_float_input = 1 + 1 + 1 + 19 + dim_act + dim_act
        self.float_mlp = mlp([self.dim_float_input, 64, 64], nn.ReLU, nn.ReLU)

        # Q-Function takes action as input too
        self.fusion_norm = nn.LayerNorm(128 + 64 + dim_act)
        self.q = mlp([128 + 64 + dim_act, 256, 256, 1], SiLU)

    def forward(self, obs, act):
        speed, gear, rpm, images, lidar, act1, act2 = obs

        img_embed = self.cnn(images.float())

        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self.float_mlp(floats)

        # Fuse Obs + Action
        combined = torch.cat((img_embed, float_embed, act), dim=-1)
        combined = self.fusion_norm(combined)

        q = self.q(combined)
        return torch.squeeze(q, -1)


class HybridNanoEffNetActorCritic(nn.Module):
    """Standard SAC Actor-Critic with HybridNanoEffNet."""
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.actor = HybridNanoEffNetActor(observation_space, action_space)
        self.q1 = HybridNanoEffNetQFunction(observation_space, action_space)
        self.q2 = HybridNanoEffNetQFunction(observation_space, action_space)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


class REDQHybridNanoEffNetActorCritic(nn.Module):
    """REDQ Actor-Critic with HybridNanoEffNet and N Q-networks."""
    def __init__(self, observation_space, action_space, n=10):
        super().__init__()
        self.actor = HybridNanoEffNetActor(observation_space, action_space)
        self.n = n
        self.qs = ModuleList([HybridNanoEffNetQFunction(observation_space, action_space) for _ in range(self.n)])

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# ========== VRAM-OPTIMIZED SHARED BACKBONE ARCHITECTURE ==========
# For 4GB VRAM GPUs: Share ONE CNN backbone across actor + all critics
# Reduces CNN count from 5 (actor + 2 critics + 2 targets) to 1


class SharedBackboneActorHead(nn.Module):
    """Actor head that takes pre-computed CNN features."""
    def __init__(self, action_space, feature_dim=192):  # 128 CNN + 64 float
        super().__init__()
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        
        self.net = mlp([feature_dim, 256, 256], SiLU, SiLU)
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def forward(self, features, test=False, with_logprob=True):
        net_out = self.net(features)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi


class SharedBackboneQHead(nn.Module):
    """Q-function head that takes pre-computed CNN features + action."""
    def __init__(self, action_space, feature_dim=192):  # 128 CNN + 64 float
        super().__init__()
        dim_act = action_space.shape[0]
        self.fusion_norm = nn.LayerNorm(feature_dim + dim_act)
        self.q = mlp([feature_dim + dim_act, 256, 256, 1], SiLU)

    def forward(self, features, act):
        combined = torch.cat((features, act), dim=-1)
        combined = self.fusion_norm(combined)
        q = self.q(combined)
        return torch.squeeze(q, -1)


class SharedBackboneEncoder(nn.Module):
    """Shared CNN + Float encoder backbone."""
    def __init__(self, observation_space, action_space):
        super().__init__()
        dim_act = action_space.shape[0]
        
        # Single shared CNN
        self.cnn = EffNetV2(cfgs=NANO_EFFNET_CFG, nb_channels_in=4, dim_output=128)
        
        # Float encoder: Speed(1) + Gear(1) + RPM(1) + Lidar(19) + PastActions(2*dim_act)
        self.dim_float_input = 1 + 1 + 1 + 19 + dim_act + dim_act
        self.float_mlp = mlp([self.dim_float_input, 64, 64], nn.ReLU, nn.ReLU)
        
        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(128 + 64)

    def forward(self, obs):
        speed, gear, rpm, images, lidar, act1, act2 = obs
        
        # CNN features (computed ONCE)
        img_embed = self.cnn(images.float())
        
        # Float features
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self.float_mlp(floats)
        
        # Fused features
        combined = torch.cat((img_embed, float_embed), dim=-1)
        combined = self.fusion_norm(combined)
        return combined


class SharedBackboneHybridActor(TorchActorModule):
    """
    Standalone Actor for worker inference.
    Has its own encoder (weights synced from trainer).
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        # Own encoder for standalone inference
        self.cnn = EffNetV2(cfgs=NANO_EFFNET_CFG, nb_channels_in=4, dim_output=128)
        self.dim_float_input = 1 + 1 + 1 + 19 + dim_act + dim_act
        self.float_mlp = mlp([self.dim_float_input, 64, 64], nn.ReLU, nn.ReLU)
        self.fusion_norm = nn.LayerNorm(128 + 64)

        # Policy head
        self.net = mlp([128 + 64, 256, 256], SiLU, SiLU)
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.std_net = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.SiLU(),
            nn.Linear(64, dim_act)
        )
        # Fix 6: Initialize bias to 1.0 -> tanh(0.97) -> log_std ~ -0.7 -> std ~ 0.5
        nn.init.constant_(self.std_net[-1].bias, 1.0)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, lidar, act1, act2 = obs

        # Encode
        img_embed = self.cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self.float_mlp(floats)
        combined = torch.cat((img_embed, float_embed), dim=-1)
        features = self.fusion_norm(combined)

        # Policy head - DECOUPLED mu/std
        net_out = self.net(features)
        mu = self.mu_layer(net_out)
        log_std_raw = self.std_net(features)
        log_std = _compute_log_std_smooth(log_std_raw)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi, _ = _squashed_gaussian_logprob(pi_distribution, pi_action)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()


class SharedBackboneHybridActorCritic(nn.Module):
    """
    VRAM-Optimized Actor-Critic for 4GB GPUs.
    
    Key optimization: ONE shared CNN backbone for actor + all critics.
    The actor has the same structure as SharedBackboneHybridActor so weights sync properly.
    Internally, for training, we extract features once and reuse them.
    """
    def __init__(self, observation_space, action_space, n=2):
        super().__init__()
        self.n = n
        
        # Actor has its own encoder (same structure as standalone actor for weight sync)
        self.actor = SharedBackboneHybridActor(observation_space, action_space)
        
        # Reference to actor's encoder for shared feature extraction
        self._actor_cnn = self.actor.cnn
        self._actor_float_mlp = self.actor.float_mlp
        self._actor_fusion_norm = self.actor.fusion_norm
        
        # N Q-heads (lightweight MLPs only)
        self.qs = ModuleList([SharedBackboneQHead(action_space, feature_dim=192) for _ in range(n)])

    def forward_features(self, obs):
        """Extract features using actor's encoder (shared for training)."""
        speed, gear, rpm, images, lidar, act1, act2 = obs
        
        img_embed = self._actor_cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self._actor_float_mlp(floats)
        combined = torch.cat((img_embed, float_embed), dim=-1)
        features = self._actor_fusion_norm(combined)
        return features, None, None

    def actor_from_features(self, features, film=None, test=False, with_logprob=True):
        """Compute action from pre-extracted features."""
        del film
        features = features.detach()
        net_out = self.actor.net(features)
        mu = self.actor.mu_layer(net_out)
        log_std_raw = self.actor.std_net(features)
        log_std = _compute_log_std_smooth(log_std_raw)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_total, logp_per_dim = _squashed_gaussian_logprob(pi_distribution, pi_action)
        else:
            logp_total, logp_per_dim = None, None

        _apply_sagr(pi_action)  # GRAC: rescue vanishing gradients through tanh
        pi_action = torch.tanh(pi_action)
        pi_action = self.actor.act_limit * pi_action
        return pi_action, logp_total, logp_per_dim

    def q_from_features(self, features, act, q_idx=None):
        """Compute Q-values from pre-extracted features."""
        if q_idx is not None:
            return self.qs[q_idx](features, act)
        return [q(features, act) for q in self.qs]

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# ============== DroQ (Dropout Q-functions) ==============

class DroQQHead(nn.Module):
    """
    Q-head with Dropout and LayerNorm for DroQ.
    Dropout provides ensemble-like diversity with only 2 Q-networks.
    LayerNorm stabilizes training with high UTD ratios.
    """
    def __init__(self, action_space, feature_dim=192, dropout_rate=0.01):
        super().__init__()
        dim_act = action_space.shape[0]
        
        self.q = nn.Sequential(
            nn.Linear(feature_dim + dim_act, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 1)
        )

    def forward(self, features, act, film=None):
        """
        Args:
            features: Pre-extracted encoder features (B, feature_dim)
            act: Actions (B, dim_act)
            film: Ignored (for compatibility with FiLM loops)
        Returns:
            Q-values (B,)
        """
        combined = torch.cat((features, act), dim=-1)
        q = self.q(combined)
        return torch.squeeze(q, -1)


class DroQHybridActorCritic(nn.Module):
    """
    DroQ Actor-Critic for maximum sample efficiency.
    
    Key features:
    - Shared CNN backbone (VRAM efficient)
    - Only 2 Q-networks with Dropout+LayerNorm
    - Supports high UTD (Update-to-Data) ratios (20+)
    - Dropout provides ensemble-like diversity for uncertainty estimation
    """
    def __init__(self, observation_space, action_space, dropout_rate=0.01):
        super().__init__()
        self.n = 2  # DroQ uses exactly 2 Q-networks
        
        # Actor has its own encoder (same structure as standalone for weight sync)
        self.actor = SharedBackboneHybridActor(observation_space, action_space)
        
        # Reference to actor's encoder for shared feature extraction
        self._actor_cnn = self.actor.cnn
        self._actor_float_mlp = self.actor.float_mlp
        self._actor_fusion_norm = self.actor.fusion_norm
        
        # 2 DroQ Q-heads with Dropout + LayerNorm
        self.qs = ModuleList([
            DroQQHead(action_space, feature_dim=192, dropout_rate=dropout_rate)
            for _ in range(2)
        ])

    def forward_features(self, obs):
        """Extract features using actor's encoder (shared for training)."""
        speed, gear, rpm, images, lidar, act1, act2 = obs
        
        img_embed = self._actor_cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self._actor_float_mlp(floats)
        combined = torch.cat((img_embed, float_embed), dim=-1)
        features = self._actor_fusion_norm(combined)
        return features, None, None

    def actor_from_features(self, features, film=None, test=False, with_logprob=True):
        """Compute action from pre-extracted features."""
        del film
        net_out = self.actor.net(features)
        mu = self.actor.mu_layer(net_out)
        log_std_raw = self.actor.std_net(features)
        log_std = _compute_log_std_smooth(log_std_raw)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_total, logp_per_dim = _squashed_gaussian_logprob(pi_distribution, pi_action)
        else:
            logp_total, logp_per_dim = None, None

        _apply_sagr(pi_action)  # GRAC: rescue vanishing gradients through tanh
        pi_action = torch.tanh(pi_action)
        pi_action = self.actor.act_limit * pi_action
        return pi_action, logp_total, logp_per_dim

    def q_from_features(self, features, act, q_idx=None):
        """Compute Q-values from pre-extracted features."""
        if q_idx is not None:
            return self.qs[q_idx](features, act)
        return [q(features, act) for q in self.qs]

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# ============== Context-Based Meta-Learning + Fusion Gate ==============

CONTEXT_WINDOW_SIZE = 16  # K: number of recent transitions for context
CONTEXT_INPUT_DIM = 24    # speed(1) + lidar(19) + action(3) + reward(1)
CONTEXT_Z_DIM = 64        # latent context dimension (widened for richer context)
FUSED_DIM = 128           # output dimension of fusion gate (widened for capacity)
FILM_HIDDEN = 256         # hidden dimension for FiLM-modulated actor/critic MLPs
FILM_N_LAYERS = 2         # number of FiLM-modulated layers


class FusionGate(nn.Module):
    """
    Learnable Fusion Gate for CNN + Float sensor fusion.
    Replaces naive concatenation with a learned gate that dynamically
    balances CNN vs Float contributions per feature dimension.
    
    gate ∈ [0,1]^fused_dim:
        output = gate * img_proj + (1 - gate) * float_proj
    """
    def __init__(self, img_dim=128, float_dim=64, fused_dim=FUSED_DIM):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, fused_dim)
        self.float_proj = nn.Linear(float_dim, fused_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(img_dim + float_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, img_embed, float_embed):
        img_proj = self.img_proj(img_embed)
        float_proj = self.float_proj(float_embed)
        gate = self.gate_net(torch.cat((img_embed, float_embed), dim=-1))
        # Keep both modalities active to preserve gradient flow.
        gate = torch.clamp(gate, min=0.1, max=0.9)
        fused = gate * img_proj + (1.0 - gate) * float_proj
        return self.norm(fused)



class ContextEncoder(nn.Module):
    """
    ROBUST CAUSAL CONTEXT ENCODER (Patched for Stability)
    - Fixed Logic Error: Now uses Transformer during inference (Sliding Window).
    - Causal Masking: Prevents "cheating" by looking at future tokens.
    - Spectral Normalization: Prevents gradient explosions (Entropy Saver).
    """
    REWARD_HORIZONS = [1, 3, 5]

    def __init__(self, input_dim=CONTEXT_INPUT_DIM, z_dim=CONTEXT_Z_DIM,
                 fused_dim=FUSED_DIM, max_len=CONTEXT_WINDOW_SIZE):
        super().__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.max_len = max_len
        self.d_model = 128  # Fixed transformer dimension

        # === 1. Image Sequence CNN ===
        # Processes (B*K, 1, 64, 64) -> (B*K, 64) spatial features
        self.img_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.LayerNorm(64)
        )
        img_feat_dim = 64

        # === 2. Positional Encoding ===
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pe.unsqueeze(0))

        # === 3. Stabilized Projections (Spectral Norm) ===
        from torch.nn.utils import spectral_norm
        self.state_dim = 40 + img_feat_dim  # 40 (telemetry+deltas) + 64 (img) = 104
        self.action_dim = 6  # 3 (act) + 3 (deltas)
        self.state_proj = spectral_norm(nn.Linear(self.state_dim, self.d_model))
        self.action_proj = spectral_norm(nn.Linear(self.action_dim, self.d_model))

        # === 4. Causal Transformers ===
        state_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.state_transformer = nn.TransformerEncoder(state_layer, num_layers=2)

        action_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.action_transformer = nn.TransformerEncoder(action_layer, num_layers=2)

        self.stream_gate = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model), nn.Sigmoid())
        self.gru = nn.GRU(self.d_model, z_dim, num_layers=1, batch_first=True)

        # Cross Attention & Output
        self.obs_proj = spectral_norm(nn.Linear(fused_dim, z_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=z_dim, num_heads=4, batch_first=True)
        self.pool_queries = nn.Parameter(torch.randn(1, 4, z_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim=z_dim, num_heads=4, batch_first=True)
        self.compress = nn.Sequential(nn.Linear(4 * z_dim, z_dim), nn.LayerNorm(z_dim))
        self.norm = nn.LayerNorm(z_dim)

        # Variational inference: z ~ N(μ, σ²) with KL(q(z|c) || N(0,I))
        self.mu_layer = nn.Linear(z_dim, z_dim)
        self.log_var_layer = nn.Linear(z_dim, z_dim)
        # Initialize log_var to produce small variance initially (stable start)
        nn.init.zeros_(self.log_var_layer.weight)
        nn.init.constant_(self.log_var_layer.bias, -2.0)  # σ² ≈ 0.14
        self.last_kl_div = torch.tensor(0.0)  # stored for training loop access


        # Inference Buffer
        self._window_buffer = None

    def _compute_deltas(self, context_seq):
        deltas = context_seq[:, 1:, :] - context_seq[:, :-1, :]
        deltas = F.pad(deltas, (0, 0, 1, 0))
        return torch.cat([context_seq, deltas], dim=-1)

    def _generate_causal_mask(self, size, device):
        """Prevents attending to the future (Anti-Cheating)."""
        mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, context_seq, img_context=None, fused_obs=None):
        B, K, _ = context_seq.shape
        enriched = self._compute_deltas(context_seq)
        
        # Split Features
        # Telemetry: speed+lidar (0:20), actions (20:23)
        # enriched has length 46 (input_dim=23 -> raw+deltas)
        state_telemetry = torch.cat([enriched[:, :, :20], enriched[:, :, 23:43]], dim=-1)
        action_feats = torch.cat([enriched[:, :, 20:23], enriched[:, :, 43:46]], dim=-1)
        
        # Image Context Features
        if img_context is not None:
            B_img, K_img, C, H, W = img_context.shape
            # (B*K, C, H, W)
            img_flat = img_context.reshape(B_img * K_img, C, H, W)
            img_feats = self.img_cnn(img_flat).reshape(B, K, 64)
        else:
            img_feats = torch.zeros(B, K, 64, device=context_seq.device)
            
        state_feats = torch.cat([state_telemetry, img_feats], dim=-1)
        
        state_seq = self.state_proj(state_feats) + self.pos_enc[:, :K, :]
        action_seq = self.action_proj(action_feats) + self.pos_enc[:, :K, :]

        # === CAUSAL MASKING ===
        # Ensures step t can only see steps 0..t
        causal_mask = self._generate_causal_mask(K, context_seq.device)
        
        state_out = self.state_transformer(state_seq, mask=causal_mask)
        action_out = self.action_transformer(action_seq, mask=causal_mask)
        
        # Residual Skip Connection (Gradient Highway)
        state_out = state_out + state_seq
        action_out = action_out + action_seq

        gate = self.stream_gate(torch.cat([state_out, action_out], dim=-1))
        # Keep both streams active to preserve gradient flow.
        gate = torch.clamp(gate, min=0.1, max=0.9)
        transformed = gate * state_out + (1 - gate) * action_out

        self.gru.flatten_parameters()
        gru_out, h_n = self.gru(transformed)

        if fused_obs is not None:
            obs_query = self.obs_proj(fused_obs).unsqueeze(1)
            cross_out, _ = self.cross_attn(obs_query, gru_out, gru_out)
            cross_z = cross_out.squeeze(1)
        else:
            cross_z = h_n.squeeze(0)

        queries = self.pool_queries.expand(B, -1, -1)
        pooled, _ = self.pool_attn(queries, gru_out, gru_out)
        pool_z = self.compress(pooled.reshape(B, -1))
        z_pre = self.norm(cross_z + pool_z)

        # Variational: produce μ and log(σ²), sample z
        mu = self.mu_layer(z_pre)
        log_var = self.log_var_layer(z_pre).clamp(-6, 2)  # prevent extreme variance

        if self.training:
            # Reparameterization trick: z = μ + σ·ε, ε ~ N(0,I)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
            # KL divergence: KL(N(μ,σ²) || N(0,I))
            self.last_kl_div = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1))
        else:
            # Deterministic during inference (use mean)
            z = mu
            self.last_kl_div = torch.tensor(0.0, device=z.device)

        return z

    def get_initial_hidden(self, batch_size=1, device='cpu'):
        return torch.zeros(1, batch_size, self.z_dim, device=device)

    def reset_online_state(self):
        self._window_buffer = None
        self._img_window_buffer = None

    def step(self, context_step, hidden, img_context_step=None, fused_obs=None):
        """
        Sliding Window Inference:
        We now buffer the last K steps and run the FULL Transformer.
        This guarantees Mathematically Exact consistency between Training and Gameplay.
        """
        # 1. Initialize fixed-size zero window to match training-time left padding.
        if self._window_buffer is None:
            self._window_buffer = torch.zeros(
                (context_step.shape[0], self.max_len, context_step.shape[-1]),
                device=context_step.device,
                dtype=context_step.dtype,
            )
            if img_context_step is not None:
                self._img_window_buffer = torch.zeros(
                    (img_context_step.shape[0], self.max_len, *img_context_step.shape[1:]),
                    device=img_context_step.device,
                    dtype=img_context_step.dtype,
                )
            else:
                self._img_window_buffer = None

        # 2. Shift left and append latest context token.
        self._window_buffer = torch.roll(self._window_buffer, shifts=-1, dims=1)
        self._window_buffer[:, -1, :] = context_step
        
        if self._img_window_buffer is not None and img_context_step is not None:
            self._img_window_buffer = torch.roll(self._img_window_buffer, shifts=-1, dims=1)
            self._img_window_buffer[:, -1] = img_context_step

        # 3. Run Full Forward Pass (Robust)
        # We call forward() directly. It handles masking and deltas.
        z = self.forward(self._window_buffer, img_context=self._img_window_buffer, fused_obs=fused_obs)
        
        # We don't need to pass hidden state between steps manually anymore 

        # because the buffer + full forward pass handles the temporal state.
        return z, None
class FiLMGenerator(nn.Module):
    """
    Shared FiLM generator: takes context z and produces (γ, β) pairs
    for each FiLM layer. Used by both actor and critic.
    
    This is the ONLY module updated during deployment-time adaptation.
    """
    def __init__(self, z_dim=CONTEXT_Z_DIM, hidden_dim=FILM_HIDDEN, n_layers=FILM_N_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # z → compact intermediate → all γ/β pairs
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_layers * hidden_dim * 2)  # γ+β for each layer
        )
        # Initialize so FiLM starts as identity: γ=1, β=0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Bounded gamma transform maps zero raw output to identity (gamma = 1.0).

    def forward(self, z):
        """
        Args:
            z: (B, z_dim) context vector
        Returns:
            film_params: list of (γ, β) tuples, one per layer
                γ, β each have shape (B, hidden_dim)
        """
        raw = self.net(z)  # (B, n_layers * hidden_dim * 2)
        params = []
        for i in range(self.n_layers):
            start = i * self.hidden_dim * 2
            gamma_raw = raw[:, start:start + self.hidden_dim]
            beta = raw[:, start + self.hidden_dim:start + self.hidden_dim * 2]
            gamma = 1.0 + 0.5 * torch.tanh(gamma_raw)
            params.append((gamma, beta))
        return params


class FiLMActorMLP(nn.Module):
    """
    Actor policy MLP with FiLM modulation + residual connections.
    Input: fused features (FUSED_DIM). Modulated by (γ, β) from FiLMGenerator.
    """
    def __init__(self, input_dim=FUSED_DIM, hidden_dim=FILM_HIDDEN, n_layers=FILM_N_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()  # residual projections
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_d, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            # Skip projection when dims don't match
            if in_d != hidden_dim:
                self.skip_projs.append(nn.Linear(in_d, hidden_dim, bias=False))
            else:
                self.skip_projs.append(nn.Identity())

    def forward(self, x, film_params):
        for i, (layer, norm, skip) in enumerate(zip(self.layers, self.norms, self.skip_projs)):
            residual = skip(x)
            x = layer(x)
            x = norm(x)
            x = F.silu(x)
            gamma, beta = film_params[i]
            x = gamma * x + beta
            x = x + residual  # residual connection
        return x


class FiLMQHead(nn.Module):
    """
    Q-network with FiLM modulation, residual connections, spectral norm,
    and dropout (DroQ-compatible).
    """
    def __init__(self, action_space, input_dim=FUSED_DIM, hidden_dim=FILM_HIDDEN,
                 n_layers=FILM_N_LAYERS, dropout_rate=0.01):
        super().__init__()
        dim_act = action_space.shape[0]
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for i in range(n_layers):
            in_d = (input_dim + dim_act) if i == 0 else hidden_dim
            # Spectral normalization on critic linear layers
            self.layers.append(nn.utils.spectral_norm(nn.Linear(in_d, hidden_dim)))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            if in_d != hidden_dim:
                self.skip_projs.append(nn.Linear(in_d, hidden_dim, bias=False))
            else:
                self.skip_projs.append(nn.Identity())
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, features, act, film_params):
        x = torch.cat((features, act), dim=-1)
        for i, (layer, norm, drop, skip) in enumerate(
                zip(self.layers, self.norms, self.dropouts, self.skip_projs)):
            residual = skip(x)
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = drop(x)
            gamma, beta = film_params[i]
            x = gamma * x + beta
            x = x + residual  # residual connection
        q = self.out(x)
        return torch.squeeze(q, -1)



class ContextualSharedBackboneHybridActor(TorchActorModule):
    """
    Standalone Actor with Context Encoder + Fusion Gate for worker inference.
    Maintains a running GRU hidden state across steps within an episode.
    Weights are synced from the trainer's actor-critic.
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        # Encoder components
        self.cnn = EffNetV2(cfgs=NANO_EFFNET_CFG, nb_channels_in=4, dim_output=128)
        self.dim_float_input = 1 + 1 + 1 + 19 + dim_act + dim_act
        self.float_mlp = mlp([self.dim_float_input, 64, 64], nn.ReLU, nn.ReLU)

        # Fusion Gate
        self.fusion_gate = FusionGate(img_dim=128, float_dim=64, fused_dim=FUSED_DIM)

        # Context Encoder
        self.context_encoder = ContextEncoder(input_dim=CONTEXT_INPUT_DIM, z_dim=CONTEXT_Z_DIM)
        self._gru_hidden = None  # persistent hidden state for online inference

        # PEARL Policy MLP: Actor receives z for fast context-based adaptation
        self.net = mlp([FUSED_DIM + CONTEXT_Z_DIM, 256, 256], nn.ReLU, nn.ReLU)
        self.mu_layer = nn.Linear(256, dim_act)
        # Smart init: car drives forward from step 1 (steer=0, gas=0.3, brake=0)
        with torch.no_grad():
            nn.init.zeros_(self.mu_layer.weight)
            self.mu_layer.bias.copy_(torch.tensor([0.0, 2.0, -2.0]))
        
        self.std_net = nn.Sequential(
            nn.Linear(FUSED_DIM + CONTEXT_Z_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, dim_act)
        )
        # Initialize bias to 1.0 -> tanh(0.97) -> log_std ~ -0.7 -> std ~ 0.5
        nn.init.constant_(self.std_net[-1].bias, 1.0)
        self.act_limit = act_limit
        # Previous reward from worker-side rollout for online context alignment.
        self._online_prev_reward = 0.0

    def _build_context_step(self, obs):
        """Build a single context vector from current observation for online GRU stepping."""
        if len(obs) == 9:
            speed, gear, rpm, images, lidar, xyz, progress, act1, act2 = obs
        else:
            speed, gear, rpm, images, lidar, xyz, progress, crash, progress_gain, act1, act2 = obs
            
        # action = last action (act1 is the most recent in act_buf)
        ctx = torch.cat((speed, lidar, act1), dim=-1)  # (B, 23)
        # In online mode, images has shape (B, 4, 64, 64). We take the latest frame.
        img_ctx = images[:, -1:, :, :]  # (B, 1, 64, 64)
        return ctx, img_ctx

    def set_online_transition(self, reward=0.0, terminated=False, truncated=False, train_mode=True):
        """
        Update online context metadata from rollout transitions.
        Called by RolloutWorker after each environment step.
        """
        if terminated or truncated:
            self._online_prev_reward = 0.0
        else:
            self._online_prev_reward = float(reward)

    def forward(self, obs, test=False, with_logprob=True):
        if len(obs) == 9:
            speed, gear, rpm, images, lidar, xyz, progress, act1, act2 = obs
            B = images.shape[0] if isinstance(images, torch.Tensor) else 1
            crash = torch.zeros((B, 1), device=speed.device)
            progress_gain = torch.zeros((B, 1), device=speed.device)
        else:
            speed, gear, rpm, images, lidar, xyz, progress, crash, progress_gain, act1, act2 = obs

        # Encode
        img_embed = self.cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self.float_mlp(floats)

        # Fusion Gate
        fused = self.fusion_gate(img_embed, float_embed)  # (B, 128)

        # Context: step GRU with current observation
        ctx_input, img_ctx_input = self._build_context_step(obs)
        device = fused.device
        if self._gru_hidden is None or self._gru_hidden.device != device:
            self._gru_hidden = self.context_encoder.get_initial_hidden(
                batch_size=ctx_input.shape[0], device=device)
        z, self._gru_hidden = self.context_encoder.step(
            ctx_input, self._gru_hidden, img_context_step=img_ctx_input, fused_obs=fused)

        # PEARL: Actor is conditioned on context z for fast adaptation
        actor_input = torch.cat([fused, z], dim=-1)
        net_out = self.net(actor_input)

        # Policy head - context-conditioned mu/std
        mu = self.mu_layer(net_out)
        log_std_raw = self.std_net(actor_input)
        log_std = _compute_log_std_smooth(log_std_raw)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi, _ = _squashed_gaussian_logprob(pi_distribution, pi_action)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi

    def act(self, obs, test=False):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
        if was_training:
            self.train()
        return a.squeeze().cpu().numpy()

    def reset_context(self):
        """Call at the beginning of each episode to reset GRU state."""
        self._gru_hidden = None
        self._online_prev_reward = 0.0
        self.context_encoder.reset_online_state()


class ContextualDroQHybridActorCritic(nn.Module):
    """
    DroQ Actor-Critic with FiLM conditioning.

    Key features:
    - Fusion Gate: learned CNN vs Float balancing
    - Context Encoder: GRU for rapid adaptation
    - FiLM Generator: z → (γ, β) for multiplicative modulation
    - FiLMQHead: Q-networks with FiLM modulation + DroQ dropout
    - Shared FiLM generator for actor + critic (key for deployment adaptation)
    """
    def __init__(self, observation_space, action_space, dropout_rate=0.05):
        super().__init__()
        self.n = 2  # DroQ uses exactly 2 Q-networks

        # Actor has its own encoder (same structure for weight sync)
        self.actor = ContextualSharedBackboneHybridActor(observation_space, action_space)

        # References to actor's encoder for shared feature extraction
        self._actor_cnn = self.actor.cnn
        self._actor_float_mlp = self.actor.float_mlp
        self._actor_fusion_gate = self.actor.fusion_gate

        # Shared context encoder (used during training to provide z to the Critic)
        self.context_encoder = self.actor.context_encoder

        # 2 Q-heads with DroQ dropout
        # Critic is asymmetric: skips FusionGate, takes 34-dim numeric float vector + z
        self.qs = ModuleList([
            DroQQHead(action_space, feature_dim=34 + CONTEXT_Z_DIM, dropout_rate=dropout_rate)
            for _ in range(2)
        ])

    def forward_features(self, obs, context=None, img_context=None):
        """
        Extract features and FiLM params using actor's encoder.
        
        Args:
            obs: tuple of (speed, gear, rpm, images, lidar, xyz, progress, crash, progress_gain, act1, act2)
            context: (B, K, 23) tensor of recent transitions, or None
            img_context: (B, K, 1, 64, 64) tensor of recent images, or None
        Returns:
            fused: (B, FUSED_DIM) fused sensor features for the Actor
            critic_floats: (B, 34) float features for the Critic
            film_params: list of (γ, β) tuples for FiLM modulation
        """
        if len(obs) == 9:
            speed, gear, rpm, images, lidar, xyz, progress, act1, act2 = obs
            B = images.shape[0]
            crash = torch.zeros((B, 1), device=speed.device)
            progress_gain = torch.zeros((B, 1), device=speed.device)
        else:
            speed, gear, rpm, images, lidar, xyz, progress, crash, progress_gain, act1, act2 = obs
            B = images.shape[0]
        
        # Ensure all flat features are strictly 2D (Batch, Dim) regardless of how the replay buffer padded them
        speed = speed.view(B, -1)
        gear = gear.view(B, -1)
        rpm = rpm.view(B, -1)
        lidar = lidar.view(B, -1)
        xyz = xyz.view(B, -1)
        progress = progress.view(B, -1)
        crash = crash.view(B, -1)
        progress_gain = progress_gain.view(B, -1)
        act1 = act1.view(B, -1)
        act2 = act2.view(B, -1)

        img_embed = self._actor_cnn(images.float())
        actor_floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self._actor_float_mlp(actor_floats)

        # Fusion Gate for Actor
        fused = self._actor_fusion_gate(img_embed, float_embed)  # (B, 128)

        # Critic sees EVERYTHING except pixels (Asymmetric SAC)
        # Added crash (1) and progress_gain (1) to critic features
        critic_floats = torch.cat((speed, gear, rpm, lidar, xyz, progress, crash, progress_gain, act1, act2), dim=-1)

        # Context → latent belief `z`
        if context is not None:
            z = self.context_encoder(context, img_context=img_context, fused_obs=fused)  # (B, 64)
        else:
            z = torch.zeros(fused.shape[0], CONTEXT_Z_DIM, device=fused.device)

        # Asymmetric Critic concatenated belief
        critic_features = torch.cat([critic_floats, z], dim=-1) # (B, 32 + 64)

        # Removed FiLM params generation
        # film_params = self.film_generator(z)
        return fused, critic_features, None, z # Modified: film_params is now None

    def actor_from_features(self, fused, film_params=None, test=False, with_logprob=True, z=None, return_pretanh_mu=False):
        """Compute action from fused features + context z (PEARL)."""
        del film_params
        fused = fused.detach()
        if z is None:
            z = torch.zeros(fused.shape[0], CONTEXT_Z_DIM, device=fused.device)
        actor_input = torch.cat([fused, z.detach()], dim=-1)
        net_out = self.actor.net(actor_input)
        mu = self.actor.mu_layer(net_out)
        log_std_raw = self.actor.std_net(actor_input)
        log_std = _compute_log_std_smooth(log_std_raw)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_total, logp_per_dim = _squashed_gaussian_logprob(pi_distribution, pi_action)
        else:
            logp_total, logp_per_dim = None, None

        _apply_sagr(pi_action)  # GRAC: rescue vanishing gradients through tanh
        pi_action = torch.tanh(pi_action)
        pi_action = self.actor.act_limit * pi_action
        
        if return_pretanh_mu:
            return pi_action, logp_total, logp_per_dim, mu
        return pi_action, logp_total, logp_per_dim

    def q_from_features(self, critic_floats, act, film_params, q_idx=None):
        """Compute Q-values from critic float features + FiLM params."""
        if q_idx is not None:
            return self.qs[q_idx](critic_floats, act, film_params)
        return [q(critic_floats, act, film_params) for q in self.qs]

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# RNN: ==========================================================


def rnn(input_size, rnn_size, rnn_len):
    """
    sizes is ignored for now, expect first values and length
    """
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers, bias=True, batch_first=True, dropout=0, bidirectional=False)
    return gru


class SquashedGaussianRNNActor(nn.Module):
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        act_limit = act_space.high[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size] + list(mlp_sizes), activation, activation)
        self.mu_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.act_limit = act_limit
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, obs_seq, test=False, with_logprob=True, save_hidden=False):
        """
        obs: observation
        h: hidden state
        Returns:
            pi_action, log_pi, h
        """
        self.rnn.flatten_parameters()

        # sequence_len = obs_seq[0].shape[0]
        batch_size = obs_seq[0].shape[0]

        if not save_hidden or self.h is None:
            device = obs_seq[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        obs_seq_cat = torch.cat(obs_seq, -1)
        net_out, h = self.rnn(obs_seq_cat, h)
        net_out = net_out[:, -1]
        net_out = self.mlp(net_out)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        if save_hidden:
            self.h = h

        return pi_action, logp_pi

    def act(self, obs, test=False):
        obs_seq = tuple(o.view(1, *o.shape) for o in obs)  # artificially add sequence dimension
        with torch.no_grad():
            a, _ = self.forward(obs_seq=obs_seq, test=test, with_logprob=False, save_hidden=True)
            return a.squeeze().cpu().numpy()


class RNNQFunction(nn.Module):
    """
    The action is merged in the latent space after the RNN
    """
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size + dim_act] + list(mlp_sizes) + [1], activation)
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, obs_seq, act, save_hidden=False):
        """
        obs: observation
        h: hidden state
        Returns:
            pi_action, log_pi, h
        """
        self.rnn.flatten_parameters()

        # sequence_len = obs_seq[0].shape[0]
        batch_size = obs_seq[0].shape[0]

        if not save_hidden or self.h is None:
            device = obs_seq[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        # logging.debug(f"len(obs_seq):{len(obs_seq)}")
        # logging.debug(f"obs_seq[0].shape:{obs_seq[0].shape}")
        # logging.debug(f"obs_seq[1].shape:{obs_seq[1].shape}")
        # logging.debug(f"obs_seq[2].shape:{obs_seq[2].shape}")
        # logging.debug(f"obs_seq[3].shape:{obs_seq[3].shape}")

        obs_seq_cat = torch.cat(obs_seq, -1)

        # logging.debug(f"obs_seq_cat.shape:{obs_seq_cat.shape}")

        net_out, h = self.rnn(obs_seq_cat, h)
        # logging.debug(f"1 net_out.shape:{net_out.shape}")
        net_out = net_out[:, -1]
        # logging.debug(f"2 net_out.shape:{net_out.shape}")
        net_out = torch.cat((net_out, act), -1)
        # logging.debug(f"3 net_out.shape:{net_out.shape}")
        q = self.mlp(net_out)

        if save_hidden:
            self.h = h

        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class RNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()

        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianRNNActor(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
        self.q1 = RNNQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
        self.q2 = RNNQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)


# ═══════════════════════════════════════════════════════════════
#  LATENT-SPACE WORLD MODEL — RSSM (DreamerV3-style)
# ═══════════════════════════════════════════════════════════════
#
# Architecture:
#   Encoder:    obs(26) → z(latent_dim)
#   Prior:      GRU(h, a) → ẑ(t+1)       [predicted next latent]
#   Posterior:  h(t+1) + obs(t+1) → z(t+1) [grounded next latent]
#   Decoder:    z → obs_hat, reward_hat
#
# The KL(posterior || prior) loss trains the prior to match reality.
# At imagination time, only the prior is used (no observations needed).
# ═══════════════════════════════════════════════════════════════


class RSSMEncoder(nn.Module):
    """Encode raw 26-dim critic state into a compact latent vector z."""
    def __init__(self, state_dim=26, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        # Start with small variance for stability
        nn.init.zeros_(self.log_var.weight)
        nn.init.constant_(self.log_var.bias, -2.0)

    def forward(self, state):
        """
        Args:
            state: (B, state_dim)
        Returns:
            mu: (B, latent_dim)
            log_var: (B, latent_dim)
            z: (B, latent_dim) — reparameterized sample
        """
        h = self.net(state)
        mu = self.mu(h)
        log_var = self.log_var(h).clamp(-6, 2)
        if self.training:
            std = torch.exp(0.5 * log_var)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return mu, log_var, z


class RSSMPrior(nn.Module):
    """
    Transition model: predicts next latent from GRU hidden state.
    GRU input = concat(z_t, action_t, context_z), output h_{t+1}.
    Then h_{t+1} → (mu_prior, log_var_prior) for ẑ_{t+1}.
    
    context_z (from PEARL encoder) conditions the dynamics on the
    agent's belief about the current track/task context.
    """
    def __init__(self, latent_dim=32, action_dim=3, gru_dim=128, hidden_dim=256, context_z_dim=64):
        super().__init__()
        self.gru_dim = gru_dim
        self.context_z_dim = context_z_dim
        # Pre-process (z, action, context_z) before feeding to GRU
        self.pre_gru = nn.Sequential(
            nn.Linear(latent_dim + action_dim + context_z_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, gru_dim)
        # Map GRU hidden state to prior distribution parameters
        self.mu = nn.Linear(gru_dim, latent_dim)
        self.log_var = nn.Linear(gru_dim, latent_dim)
        nn.init.zeros_(self.log_var.weight)
        nn.init.constant_(self.log_var.bias, -2.0)

    def forward(self, h, z, action, context_z=None):
        """
        One-step transition.
        Args:
            h: (B, gru_dim) — previous GRU hidden state
            z: (B, latent_dim) — previous latent
            action: (B, action_dim)
            context_z: (B, context_z_dim) — PEARL context vector (optional)
        Returns:
            h_next: (B, gru_dim)
            mu_prior: (B, latent_dim)
            log_var_prior: (B, latent_dim)
        """
        if context_z is None:
            context_z = torch.zeros(z.shape[0], self.context_z_dim, device=z.device)
        x = self.pre_gru(torch.cat([z, action, context_z], dim=-1))
        h_next = self.gru(x, h)
        mu_prior = self.mu(h_next)
        log_var_prior = self.log_var(h_next).clamp(-6, 2)
        return h_next, mu_prior, log_var_prior

    def get_initial_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.gru_dim, device=device)


class RSSMPosterior(nn.Module):
    """
    Observation-conditioned refinement of prior prediction.
    Combines GRU hidden state h_{t+1} with the encoded observation
    to produce a more accurate posterior distribution for z_{t+1}.
    """
    def __init__(self, gru_dim=128, state_dim=26, latent_dim=32, hidden_dim=256):
        super().__init__()
        # Combine GRU hidden + raw observation for posterior
        self.net = nn.Sequential(
            nn.Linear(gru_dim + state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.log_var.weight)
        nn.init.constant_(self.log_var.bias, -2.0)

    def forward(self, h, obs_next):
        """
        Args:
            h: (B, gru_dim) — GRU hidden state AFTER transition
            obs_next: (B, state_dim) — actual next observation
        Returns:
            mu_post: (B, latent_dim)
            log_var_post: (B, latent_dim)
            z_post: (B, latent_dim) — reparameterized sample
        """
        x = self.net(torch.cat([h, obs_next], dim=-1))
        mu = self.mu(x)
        log_var = self.log_var(x).clamp(-6, 2)
        if self.training:
            std = torch.exp(0.5 * log_var)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return mu, log_var, z


class RSSMDecoder(nn.Module):
    """Decode latent z back to critic state + reward.

    Reward uncertainty via MC-Dropout: a single reward network with dropout
    is run N times with different dropout masks to estimate epistemic
    uncertainty. This CANNOT collapse because the variance comes from
    inference-time dropout randomness, not from weight diversity.
    """
    def __init__(self, latent_dim=32, gru_dim=128, state_dim=26, hidden_dim=256, num_reward_heads=5):
        super().__init__()
        self.num_reward_heads = num_reward_heads  # used as num_mc_samples for MC-Dropout
        self.num_mc_samples = num_reward_heads  # alias for clarity
        self.bootstrap_ratio = 0.5
        self.state_net = nn.Sequential(
            nn.Linear(latent_dim + gru_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Single reward network with heavier dropout for MC-Dropout uncertainty
        self.reward_net = nn.Sequential(
            nn.Linear(latent_dim + gru_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Keep reward_nets as alias pointing to a ModuleList wrapping the single net
        # for backward compatibility with checkpoint loading
        self.reward_nets = nn.ModuleList([self.reward_net])

    def forward(self, z, h):
        """
        Args:
            z: (B, latent_dim)
            h: (B, gru_dim)
        Returns:
            state_hat: (B, state_dim) — reconstructed critic state
            reward_hats: (num_mc_samples, B, 1) — MC-Dropout reward predictions
        """
        combined = torch.cat([z, h], dim=-1)
        state_hat = self.state_net(combined)

        # Always run reward_net with dropout ENABLED for MC-Dropout uncertainty
        # Force train mode on reward_net regardless of model mode
        was_training = self.reward_net.training
        self.reward_net.train()

        reward_hats = torch.stack(
            [self.reward_net(combined) for _ in range(self.num_mc_samples)],
            dim=0
        )  # (num_mc_samples, B, 1) — each pass has different dropout mask

        # Restore original mode
        if not was_training:
            self.reward_net.eval()

        return state_hat, reward_hats

    def generate_bootstrap_masks(self, batch_size, device):
        """Generate bootstrap masks for training.

        Returns: (num_mc_samples, B) boolean tensor
        """
        k = max(1, int(batch_size * self.bootstrap_ratio))
        masks = torch.zeros(self.num_mc_samples, batch_size, dtype=torch.bool, device=device)
        for i in range(self.num_mc_samples):
            idx = torch.randperm(batch_size, device=device)[:k]
            masks[i, idx] = True
        return masks


class ImaginationActor(nn.Module):
    """
    Lightweight MLP policy for imagination rollouts.

    Maps critic-state (28-dim: speed/gear/rpm/lidar/xyz/progress/crash/progress_gain)
    to actions (3-dim: steer/gas/brake) with tanh output.

    Trained to mimic the real actor's behavior from (critic_state, action) pairs
    collected during regular training. Used in _imagined_critic_update so that
    imagined rollouts follow realistic trajectories instead of random actions.
    """
    def __init__(self, state_dim=28, action_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, critic_state):
        """
        Args:
            critic_state: (B, state_dim) — raw telemetry vector
        Returns:
            action: (B, action_dim) — bounded in [-1, 1]
        """
        return self.net(critic_state)


class RunningMeanStd:
    """
    Welford's online algorithm for running mean/variance.

    Used to normalize curiosity surprise so it doesn't overwhelm the RL reward
    signal early in training (when the world model is bad and surprise is huge)
    or vanish late (when the model has converged and surprise is near zero).

    Not an nn.Module — just a state container. Compatible with pickle for
    checkpoint serialization.
    """
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4  # small epsilon to avoid division by zero

    def update(self, x):
        """Update running stats with a batch of values.
        Args:
            x: torch.Tensor of shape (B,) or scalar
        """
        if hasattr(x, 'mean'):
            batch_mean = float(x.mean().item())
            batch_var = float(x.var().item()) if x.numel() > 1 else 0.0
            batch_count = int(x.numel())
        else:
            batch_mean = float(x)
            batch_var = 0.0
            batch_count = 1

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        """Normalize values using running statistics.
        Args:
            x: torch.Tensor
        Returns:
            normalized x (zero-mean, unit-variance based on running stats)
        """
        return x / (self.var ** 0.5 + 1e-8)


class LatentWorldModel(nn.Module):
    """
    RSSM World Model (DreamerV3-style) for model-based RL.

    Training:
        Given sequences (s_t, a_t, s_{t+1}, r_t):
        1. Encode s_t → z_t
        2. Prior: GRU(h, z_t, a_t) → h_{t+1}, ẑ_{t+1}
        3. Posterior: h_{t+1} + s_{t+1} → z_{t+1} (ground truth)
        4. Loss = recon(decode(z_post) vs s_{t+1}) + KL(post || prior)

    Imagination:
        Roll out using prior only (no observations):
        1. Start from real state s → z via encoder
        2. Unroll: prior(h, z, a) → h', ẑ' → decode(ẑ') → (ŝ', r̂')
        3. Repeat for H steps
    """
    def __init__(self, state_dim=26, action_dim=3, latent_dim=32,
                 gru_dim=128, hidden_dim=256, kl_free_nats=1.0, num_reward_heads=5,
                 context_z_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.gru_dim = gru_dim
        self.kl_free_nats = kl_free_nats
        self.context_z_dim = context_z_dim

        self.encoder = RSSMEncoder(state_dim, latent_dim, hidden_dim)
        self.prior = RSSMPrior(latent_dim, action_dim, gru_dim, hidden_dim, context_z_dim)
        self.posterior = RSSMPosterior(gru_dim, state_dim, latent_dim, hidden_dim)
        self.decoder = RSSMDecoder(latent_dim, gru_dim, state_dim, hidden_dim, num_reward_heads)

    def train_step(self, state, action, next_state, reward, context_z=None):
        """
        Single training step on a batch of transitions.
        Args:
            state: (B, state_dim)
            action: (B, action_dim)
            next_state: (B, state_dim)
            reward: (B, 1)
            context_z: (B, context_z_dim) — PEARL context vector (optional)
        Returns:
            loss: scalar — total RSSM loss
            metrics: dict with component losses for logging
        """
        self.train()
        B = state.shape[0]

        # 1. Encode current state
        _, _, z_t = self.encoder(state)

        # 2. Prior: predict next latent from (h, z_t, action, context_z)
        h_t = self.prior.get_initial_hidden(B, state.device)
        h_next, mu_prior, log_var_prior = self.prior(h_t, z_t, action, context_z)

        # 3. Posterior: refine using actual next observation
        mu_post, log_var_post, z_post = self.posterior(h_next, next_state)

        # 4. Decode from posterior latent (training uses ground-truth-grounded z)
        state_hat, reward_hats = self.decoder(z_post, h_next)

        # === Losses ===
        # Reconstruction loss
        recon_state = F.mse_loss(state_hat, next_state)

        # Bootstrap reward loss: each head trained on its own random subset
        bootstrap_masks = self.decoder.generate_bootstrap_masks(B, state.device)  # (num_heads, B)
        reward_target = reward.unsqueeze(0).expand_as(reward_hats)  # (num_heads, B, 1)
        reward_errors = (reward_hats - reward_target).pow(2)  # (num_heads, B, 1)
        # Mask: only count errors for samples assigned to each head
        mask_expanded = bootstrap_masks.unsqueeze(-1).float()  # (num_heads, B, 1)
        recon_reward = (reward_errors * mask_expanded).sum() / mask_expanded.sum().clamp(min=1.0)

        loss_recon = recon_state + recon_reward

        # KL divergence: KL(posterior || prior), with free nats
        var_post = log_var_post.exp()
        var_prior = log_var_prior.exp()
        kl_per_dim = 0.5 * (
            log_var_prior - log_var_post
            + var_post / var_prior
            + (mu_post - mu_prior).pow(2) / var_prior
            - 1.0
        )
        kl_per_sample = kl_per_dim.sum(dim=-1)
        kl = kl_per_sample.mean()
        kl_clamped = torch.clamp(kl, min=self.kl_free_nats)

        loss = loss_recon + kl_clamped

        metrics = {
            "wm_recon_state": recon_state.item(),
            "wm_recon_reward": recon_reward.item(),
            "wm_kl": kl.item(),
            "wm_kl_clamped": kl_clamped.item(),
            "wm_total_loss": loss.item(),
        }

        with torch.no_grad():
            z_dim_std = z_t.detach().std(dim=0, unbiased=False)
            reward_error = reward_hats.detach() - reward_target.detach()
            metrics.update({
                "wm/z_t_mean": z_t.detach().mean().item(),
                "wm/z_t_std": z_t.detach().std(unbiased=False).item(),
                "wm/z_t_abs_mean": z_t.detach().abs().mean().item(),
                "wm/z_t_dead_dim_ratio_1e3": (z_dim_std < 1e-3).float().mean().item(),
                "wm/z_t_dim_std_min": z_dim_std.min().item(),
                "wm/z_t_dim_std_mean": z_dim_std.mean().item(),
                "wm/z_t_dim_std_max": z_dim_std.max().item(),
                "wm/prior_mu_mean": mu_prior.detach().mean().item(),
                "wm/prior_mu_std": mu_prior.detach().std(unbiased=False).item(),
                "wm/prior_logvar_mean": log_var_prior.detach().mean().item(),
                "wm/prior_logvar_std": log_var_prior.detach().std(unbiased=False).item(),
                "wm/post_mu_mean": mu_post.detach().mean().item(),
                "wm/post_mu_std": mu_post.detach().std(unbiased=False).item(),
                "wm/post_logvar_mean": log_var_post.detach().mean().item(),
                "wm/post_logvar_std": log_var_post.detach().std(unbiased=False).item(),
                "wm/kl_mean": kl_per_sample.detach().mean().item(),
                "wm/kl_std": kl_per_sample.detach().std(unbiased=False).item(),
                "wm/kl_min": kl_per_sample.detach().min().item(),
                "wm/kl_max": kl_per_sample.detach().max().item(),
                "wm/state_recon_loss": recon_state.detach().item(),
                "wm/reward_loss": recon_reward.detach().item(),
                "wm/reward_pred_mean": reward_hats.detach().mean().item(),
                "wm/reward_pred_std": reward_hats.detach().std(unbiased=False).item(),
                "wm/reward_target_mean": reward_target.detach().mean().item(),
                "wm/reward_target_std": reward_target.detach().std(unbiased=False).item(),
                "wm/reward_error_abs_mean": reward_error.abs().mean().item(),
            })
        return loss, metrics

    @staticmethod
    def _kl_divergence(mu_post, log_var_post, mu_prior, log_var_prior):
        """
        KL(N(mu_post, var_post) || N(mu_prior, var_prior))
        Averaged over batch and latent dims.
        """
        var_post = log_var_post.exp()
        var_prior = log_var_prior.exp()
        kl = 0.5 * (
            log_var_prior - log_var_post
            + var_post / var_prior
            + (mu_post - mu_prior).pow(2) / var_prior
            - 1.0
        )
        return kl.sum(dim=-1).mean()  # sum over latent, mean over batch

    @torch.no_grad()
    def imagine(self, state, policy_fn, horizon, gamma=0.99, context_z=None):
        """
        Imagine future trajectories in latent space using the prior only.
        Args:
            state: (B, state_dim) — real starting states
            policy_fn: callable(critic_state) → action — the current policy
            horizon: int — number of imagination steps
            gamma: float — discount for computing returns
            context_z: (B, context_z_dim) — PEARL context vector (optional)
        Returns:
            imagined_states: (H, B, state_dim) — decoded states
            imagined_rewards: (H, B, 1) — decoded mean rewards
            imagined_actions: (H, B, action_dim)
            imagined_uncertainty: (H, B, 1) — ensemble reward variance
        """
        self.eval()  # Disable BN/Dropout for core RSSM (reward_net dropout handled inside forward())
        B = state.shape[0]

        # Encode starting state
        _, _, z = self.encoder(state)
        h = self.prior.get_initial_hidden(B, state.device)

        all_states = []
        all_rewards = []
        all_actions = []
        all_uncertainties = []

        for t in range(horizon):
            # Decode current latent to get critic state for policy
            decoded_state, _ = self.decoder(z, h)
            all_states.append(decoded_state)

            # Get action from policy
            action = policy_fn(decoded_state)
            all_actions.append(action)

            # Step forward using prior (no observation needed, context_z conditions dynamics)
            h, mu_prior, log_var_prior = self.prior(h, z, action, context_z)
            # Use the mean (deterministic) during imagination for stability
            z = mu_prior

            # Decode reward at the new state
            _, reward_hats = self.decoder(z, h)
            mean_reward = reward_hats.mean(dim=0)
            # Use coefficient of variation (std / |mean|) for scale-invariant uncertainty
            # Raw variance collapses to ~0 when reward predictions are near-zero
            reward_std = reward_hats.std(dim=0)
            uncertainty = reward_std / (mean_reward.abs() + 1e-6)  # (B, 1)
            all_rewards.append(mean_reward)
            all_uncertainties.append(uncertainty)

        self.train()
        return (
            torch.stack(all_states, dim=0),     # (H, B, state_dim)
            torch.stack(all_rewards, dim=0),     # (H, B, 1)
            torch.stack(all_actions, dim=0),     # (H, B, action_dim)
            torch.stack(all_uncertainties, dim=0) # (H, B, 1)
        )

    @torch.no_grad()
    def compute_surprise(self, state, action, next_state, context_z=None):
        """
        Compute per-sample surprise: KL(posterior || prior) for each transition.
        High KL = the model was surprised = novel/unexplored state.

        Args:
            state: (B, state_dim)
            action: (B, action_dim)
            next_state: (B, state_dim)
            context_z: (B, context_z_dim) — PEARL context vector (optional)
        Returns:
            surprise: (B,) — per-sample KL divergence (non-negative)
        """
        B = state.shape[0]

        # Encode current state
        _, _, z_t = self.encoder(state)

        # Prior: what the model predicts (conditioned on PEARL context)
        h_t = self.prior.get_initial_hidden(B, state.device)
        h_next, mu_prior, log_var_prior = self.prior(h_t, z_t, action, context_z)

        # Posterior: what actually happened
        mu_post, log_var_post, _ = self.posterior(h_next, next_state)

        # Per-sample KL (sum over latent dims, keep batch dim)
        var_post = log_var_post.exp()
        var_prior = log_var_prior.exp()
        kl_per_sample = 0.5 * (
            log_var_prior - log_var_post
            + var_post / var_prior
            + (mu_post - mu_prior).pow(2) / var_prior
            - 1.0
        ).sum(dim=-1)  # (B,)

        return kl_per_sample.clamp(min=0.0)


# Legacy aliases for backward compatibility with old checkpoints
_DynamicsMLP = None  # Removed — old checkpoints will need RESET_TRAINING=true
LatentDynamicsEnsemble = None  # Removed — replaced by LatentWorldModel

