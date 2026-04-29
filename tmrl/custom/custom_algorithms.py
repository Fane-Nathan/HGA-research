# standard library imports
import itertools
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

# third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD

# local imports
import tmrl.custom.custom_models as core
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from tmrl.training import TrainingAgent
import tmrl.config.config_constants as cfg

import logging


def _add_tensor_stats(logs, prefix, x, dead_threshold=1e-3):
    if logs is None or x is None:
        return logs

    with torch.no_grad():
        x = x.detach()
        if not x.is_floating_point():
            x = x.float()

        if x.numel() == 0:
            logs[f"{prefix}_mean"] = 0.0
            logs[f"{prefix}_std"] = 0.0
            logs[f"{prefix}_abs_mean"] = 0.0
            logs[f"{prefix}_min"] = 0.0
            logs[f"{prefix}_max"] = 0.0
            return logs

        logs[f"{prefix}_mean"] = x.mean().item()
        logs[f"{prefix}_std"] = x.std(unbiased=False).item()
        logs[f"{prefix}_abs_mean"] = x.abs().mean().item()
        logs[f"{prefix}_min"] = x.min().item()
        logs[f"{prefix}_max"] = x.max().item()

        if x.dim() >= 2 and x.shape[-1] > 0:
            feature_view = x.reshape(-1, x.shape[-1])
            dim_std = feature_view.std(dim=0, unbiased=False)
            logs[f"{prefix}_dead_dim_ratio_1e3"] = (dim_std < dead_threshold).float().mean().item()
            logs[f"{prefix}_dim_std_min"] = dim_std.min().item()
            logs[f"{prefix}_dim_std_mean"] = dim_std.mean().item()
            logs[f"{prefix}_dim_std_max"] = dim_std.max().item()

    return logs


def _parameter_grad_norm(parameters):
    if parameters is None:
        return 0.0
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    total_sq_norm = 0.0
    found_grad = False
    for p in parameters:
        if p is None or p.grad is None:
            continue

        grad = p.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        total_sq_norm += grad.norm(2).item() ** 2
        found_grad = True

    return total_sq_norm ** 0.5 if found_grad else 0.0


def _module_grad_norm(module):
    if module is None:
        return 0.0
    return _parameter_grad_norm(module.parameters())


def _clear_parameter_grads(parameters):
    if parameters is None:
        return
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p is not None:
            p.grad = None


# Soft Actor-Critic ====================================================================================================


@dataclass(eq=0)
class SpinupSacAgent(TrainingAgent):  # Adapted from Spinup
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = core.MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning (SAC v2)
    learn_entropy_coef: bool = True  # if True, SAC v2 is used, else, SAC v1 is used
    target_entropy: float = None  # if None, the target entropy for SAC v2 is set automatically
    optimizer_actor: str = "adam"  # one of ["adam", "adamw", "sgd"]
    optimizer_critic: str = "adam"  # one of ["adam", "adamw", "sgd"]
    betas_actor: tuple = None  # for Adam and AdamW
    betas_critic: tuple = None  # for Adam and AdamW
    l2_actor: float = None  # weight decay
    l2_critic: float = None  # weight decay

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up optimizers for policy and q-function:

        self.optimizer_actor = self.optimizer_actor.lower()
        self.optimizer_critic = self.optimizer_critic.lower()
        if self.optimizer_actor not in ["adam", "adamw", "sgd"]:
            logging.warning(f"actor optimizer {self.optimizer_actor} is not valid, defaulting to sgd")
        if self.optimizer_critic not in ["adam", "adamw", "sgd"]:
            logging.warning(f"critic optimizer {self.optimizer_critic} is not valid, defaulting to sgd")
        if self.optimizer_actor == "adam":
            pi_optimizer_cls = Adam
        elif self.optimizer_actor == "adamw":
            pi_optimizer_cls = AdamW
        else:
            pi_optimizer_cls = SGD
        pi_optimizer_kwargs = {"lr": self.lr_actor}
        if self.optimizer_actor in ["adam, adamw"] and self.betas_actor is not None:
            pi_optimizer_kwargs["betas"] = tuple(self.betas_actor)
        if self.l2_actor is not None:
            pi_optimizer_kwargs["weight_decay"] = self.l2_actor

        if self.optimizer_critic == "adam":
            q_optimizer_cls = Adam
        elif self.optimizer_critic == "adamw":
            q_optimizer_cls = AdamW
        else:
            q_optimizer_cls = SGD
        q_optimizer_kwargs = {"lr": self.lr_critic}
        if self.optimizer_critic in ["adam, adamw"] and self.betas_critic is not None:
            q_optimizer_kwargs["betas"] = tuple(self.betas_critic)
        if self.l2_critic is not None:
            q_optimizer_kwargs["weight_decay"] = self.l2_critic

        self.pi_optimizer = pi_optimizer_cls(self.model.actor.parameters(), **pi_optimizer_kwargs)
        self.q_optimizer = q_optimizer_cls(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()), **q_optimizer_kwargs)

        # entropy coefficient:

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape)  # .astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        o, a, r, o2, d, _ = batch

        pi, logp_pi = self.model.actor(o)
        # FIXME? log_prob = log_prob.reshape(-1, 1)

        # loss_alpha:

        loss_alpha = None
        if self.learn_entropy_coef:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            limit = getattr(self, "alpha_floor", 0.05) 
            self.log_alpha.clamp_(min=np.log(limit))
        # Run one gradient descent step for Q1 and Q2

        # loss_q:

        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.model.actor(o2)

            # Target Q-values
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = (loss_q1 + loss_q2) / 2  # averaged for homogeneity with REDQ

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)

        # Next run one gradient descent step for actor.

        # loss_pi:

        # pi, logp_pi = self.model.actor(o)
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha_t * logp_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # FIXME: remove debug info
        with torch.no_grad():

            if not cfg.DEBUG_MODE:
                ret_dict = dict(
                    loss_actor=loss_pi.detach().item(),
                    loss_critic=loss_q.detach().item(),
                )
            else:
                q1_o2_a2 = self.model.q1(o2, a2)
                q2_o2_a2 = self.model.q2(o2, a2)
                q1_targ_pi = self.model_target.q1(o, pi)
                q2_targ_pi = self.model_target.q2(o, pi)
                q1_targ_a = self.model_target.q1(o, a)
                q2_targ_a = self.model_target.q2(o, a)

                diff_q1pt_qpt = (q1_pi_targ - q_pi_targ).detach()
                diff_q2pt_qpt = (q2_pi_targ - q_pi_targ).detach()
                diff_q1_q1t_a2 = (q1_o2_a2 - q1_pi_targ).detach()
                diff_q2_q2t_a2 = (q2_o2_a2 - q2_pi_targ).detach()
                diff_q1_q1t_pi = (q1_pi - q1_targ_pi).detach()
                diff_q2_q2t_pi = (q2_pi - q2_targ_pi).detach()
                diff_q1_q1t_a = (q1 - q1_targ_a).detach()
                diff_q2_q2t_a = (q2 - q2_targ_a).detach()
                diff_q1_backup = (q1 - backup).detach()
                diff_q2_backup = (q2 - backup).detach()
                diff_q1_backup_r = (q1 - backup + r).detach()
                diff_q2_backup_r = (q2 - backup + r).detach()

                ret_dict = dict(
                    loss_actor=loss_pi.detach().item(),
                    loss_critic=loss_q.detach().item(),
                    # debug:
                    debug_log_pi=logp_pi.detach().mean().item(),
                    debug_log_pi_std=logp_pi.detach().std().item(),
                    debug_logp_a2=logp_a2.detach().mean().item(),
                    debug_logp_a2_std=logp_a2.detach().std().item(),
                    debug_q_a1=q_pi.detach().mean().item(),
                    debug_q_a1_std=q_pi.detach().std().item(),
                    debug_q_a1_targ=q_pi_targ.detach().mean().item(),
                    debug_q_a1_targ_std=q_pi_targ.detach().std().item(),
                    debug_backup=backup.detach().mean().item(),
                    debug_backup_std=backup.detach().std().item(),
                    debug_q1=q1.detach().mean().item(),
                    debug_q1_std=q1.detach().std().item(),
                    debug_q2=q2.detach().mean().item(),
                    debug_q2_std=q2.detach().std().item(),
                    debug_diff_q1=diff_q1_backup.mean().item(),
                    debug_diff_q1_std=diff_q1_backup.std().item(),
                    debug_diff_q2=diff_q2_backup.mean().item(),
                    debug_diff_q2_std=diff_q2_backup.std().item(),
                    debug_diff_r_q1=diff_q1_backup_r.mean().item(),
                    debug_diff_r_q1_std=diff_q1_backup_r.std().item(),
                    debug_diff_r_q2=diff_q2_backup_r.mean().item(),
                    debug_diff_r_q2_std=diff_q2_backup_r.std().item(),
                    debug_diff_q1pt_qpt=diff_q1pt_qpt.mean().item(),
                    debug_diff_q2pt_qpt=diff_q2pt_qpt.mean().item(),
                    debug_diff_q1_q1t_a2=diff_q1_q1t_a2.mean().item(),
                    debug_diff_q2_q2t_a2=diff_q2_q2t_a2.mean().item(),
                    debug_diff_q1_q1t_pi=diff_q1_q1t_pi.mean().item(),
                    debug_diff_q2_q2t_pi=diff_q2_q2t_pi.mean().item(),
                    debug_diff_q1_q1t_a=diff_q1_q1t_a.mean().item(),
                    debug_diff_q2_q2t_a=diff_q2_q2t_a.mean().item(),
                    debug_diff_q1pt_qpt_std=diff_q1pt_qpt.std().item(),
                    debug_diff_q2pt_qpt_std=diff_q2pt_qpt.std().item(),
                    debug_diff_q1_q1t_a2_std=diff_q1_q1t_a2.std().item(),
                    debug_diff_q2_q2t_a2_std=diff_q2_q2t_a2.std().item(),
                    debug_diff_q1_q1t_pi_std=diff_q1_q1t_pi.std().item(),
                    debug_diff_q2_q2t_pi_std=diff_q2_q2t_pi.std().item(),
                    debug_diff_q1_q1t_a_std=diff_q1_q1t_a.std().item(),
                    debug_diff_q2_q2t_a_std=diff_q2_q2t_a.std().item(),
                    debug_r=r.detach().mean().item(),
                    debug_r_std=r.detach().std().item(),
                    debug_d=d.detach().mean().item(),
                    debug_d_std=d.detach().std().item(),
                    debug_a_0=a[:, 0].detach().mean().item(),
                    debug_a_0_std=a[:, 0].detach().std().item(),
                    debug_a_1=a[:, 1].detach().mean().item(),
                    debug_a_1_std=a[:, 1].detach().std().item(),
                    debug_a_2=a[:, 2].detach().mean().item(),
                    debug_a_2_std=a[:, 2].detach().std().item(),
                    debug_a1_0=pi[:, 0].detach().mean().item(),
                    debug_a1_0_std=pi[:, 0].detach().std().item(),
                    debug_a1_1=pi[:, 1].detach().mean().item(),
                    debug_a1_1_std=pi[:, 1].detach().std().item(),
                    debug_a1_2=pi[:, 2].detach().mean().item(),
                    debug_a1_2_std=pi[:, 2].detach().std().item(),
                    debug_a2_0=a2[:, 0].detach().mean().item(),
                    debug_a2_0_std=a2[:, 0].detach().std().item(),
                    debug_a2_1=a2[:, 1].detach().mean().item(),
                    debug_a2_1_std=a2[:, 1].detach().std().item(),
                    debug_a2_2=a2[:, 2].detach().mean().item(),
                    debug_a2_2_std=a2[:, 2].detach().std().item(),
                )

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# REDQ-SAC =============================================================================================================

@dataclass(eq=0)
class REDQSACAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = core.REDQMLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning
    learn_entropy_coef: bool = True
    target_entropy: float = None  # if None, the target entropy is set automatically
    n: int = 10  # number of REDQ parallel Q networks
    m: int = 2  # number of REDQ randomly sampled target networks
    q_updates_per_policy_update: int = 1  # in REDQ, this is the "UTD ratio" (20), this interplays with lr_actor

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device REDQ-SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer_list = [Adam(q.parameters(), lr=self.lr_critic) for q in self.model.qs]
        self.criterion = torch.nn.MSELoss()
        self.loss_pi = torch.zeros((1,), device=device)

        self.i_update = 0  # for UTD ratio

        if self.target_entropy is None:  # automatic entropy coefficient
            self.target_entropy = -np.prod(action_space.shape)  # .astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        self.i_update += 1
        update_policy = (self.i_update % self.q_updates_per_policy_update == 0)
        # DEBUG: Confirm new code is running
        # print(f"DEBUG: Train Step {self.i_update}, Update Policy: {update_policy}, Learn Ent: {self.learn_entropy_coef}")

        o, a, r, o2, d, _ = batch

        if update_policy:
            pi, logp_pi = self.model.actor(o)
        # FIXME? log_prob = log_prob.reshape(-1, 1)

        loss_alpha = None
        if self.learn_entropy_coef:
            alpha_t = torch.exp(self.log_alpha.detach())
            if update_policy:
                loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)

            sample_idxs = np.random.choice(self.n, self.m, replace=False)

            q_prediction_next_list = [self.model_target.qs[i](o2, a2) for i in sample_idxs]
            q_prediction_next_cat = torch.stack(q_prediction_next_list, -1)
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            backup = r.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (min_q - alpha_t * logp_a2.unsqueeze(dim=-1))

        q_prediction_list = [q(o, a) for q in self.model.qs]
        q_prediction_cat = torch.stack(q_prediction_list, -1)
        backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup

        loss_q = self.criterion(q_prediction_cat, backup)  # * self.n  # averaged for homogeneity with SAC

        for q in self.q_optimizer_list:
            q.zero_grad()
        loss_q.backward()

        if update_policy:
            for q in self.model.qs:
                q.requires_grad_(False)

            qs_pi = [q(o, pi) for q in self.model.qs]
            qs_pi_cat = torch.stack(qs_pi, -1)
            ave_q = torch.mean(qs_pi_cat, dim=1, keepdim=True)
            loss_pi = (alpha_t * logp_pi.unsqueeze(dim=-1) - ave_q).mean()
            self.pi_optimizer.zero_grad()
            loss_pi.backward()

            for q in self.model.qs:
                q.requires_grad_(True)

        for q_optimizer in self.q_optimizer_list:
            q_optimizer.step()

        if update_policy:
            self.pi_optimizer.step()

        if update_policy:
            with torch.no_grad():
                for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        if update_policy:
            self.loss_pi = loss_pi.detach()
        ret_dict = dict(
            loss_actor=self.loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
        )

        if self.learn_entropy_coef and loss_alpha is not None:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# ========== SHARED BACKBONE REDQ-SAC AGENT ==========
# Optimized for 4GB VRAM: Uses shared CNN backbone across actor + critics


@dataclass(eq=0)
class SharedBackboneREDQSACAgent(TrainingAgent):
    """
    REDQ-SAC Agent optimized for low VRAM GPUs.
    
    Uses SharedBackboneHybridActorCritic which shares ONE CNN across all networks.
    Key optimization: Features are extracted ONCE and reused for all Q-heads.
    """
    observation_space: type
    action_space: type
    device: str = None
    model_cls: type = core.SharedBackboneHybridActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    lr_entropy: float = 1e-3
    learn_entropy_coef: bool = True
    target_entropy: float = None
    n: int = 2  # number of Q networks (default 2 for VRAM)
    m: int = 2  # number of randomly sampled target networks
    q_updates_per_policy_update: int = 1

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space, n=self.n)
        logging.debug(f" device SharedBackbone-REDQ-SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        
        # Optimizer for actor's policy head only (features are fixed/detached for actor)
        self.pi_optimizer = Adam(
            list(self.model.actor.net.parameters()) +
            list(self.model.actor.mu_layer.parameters()) +
            list(self.model.actor.std_net.parameters()) +
            list(self.model.actor.log_std_layer.parameters()), 
            lr=self.lr_actor
        )
        
        # Optimizer for Q-heads AND Encoder (Critic drives representation)
        self._q_params = [p for q in self.model.qs for p in q.parameters()]
        self._encoder_params = list(self.model.actor.cnn.parameters()) + list(self.model.actor.float_mlp.parameters())
        if hasattr(self.model.actor, "fusion_norm"):
            self._encoder_params += list(self.model.actor.fusion_norm.parameters())
        if hasattr(self.model.actor, "fusion_gate"):
            self._encoder_params += list(self.model.actor.fusion_gate.parameters())
        if hasattr(self.model.actor, "context_encoder"):
            self._encoder_params += list(self.model.actor.context_encoder.parameters())
        if hasattr(self.model.actor, "film_generator"):
            self._encoder_params += list(self.model.actor.film_generator.parameters())

        # Keep shared representation updates slower than Q-head updates at high UTD.
        utd_ratio = max(1.0, float(self.q_updates_per_policy_update))
        encoder_lr = min(self.lr_critic / utd_ratio, 3e-5)
        self.q_optimizer = Adam([
            {'params': self._q_params, 'lr': self.lr_critic},
            {'params': self._encoder_params, 'lr': encoder_lr},
        ])
        
        self.criterion = torch.nn.MSELoss()
        self.loss_pi = torch.zeros((1,), device=device)
        self.loss_q = torch.zeros((1,), device=device)  # Initialize loss_q
        self.i_update = 0

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        self.i_update += 1
        update_policy = (self.i_update % self.q_updates_per_policy_update == 0)

        o, a, r, o2, d, _ = batch

        # Get current alpha
        if self.learn_entropy_coef:
            alpha_t = torch.exp(self.log_alpha.detach())
        else:
            alpha_t = self.alpha_t

        # === Target Q computation (with no_grad) ===
        with torch.no_grad():
            # Use current policy for next action, target critics for evaluation.
            features_o2_curr, _, _ = self.model.forward_features(o2)
            a2, logp_a2, _ = self.model.actor_from_features(features_o2_curr, None)

            # Extract target features for target Q-values
            features_o2_target, _, _ = self.model_target.forward_features(o2)

            sample_idxs = np.random.choice(self.n, self.m, replace=False)
            q_prediction_next_list = [self.model_target.qs[i](features_o2_target, a2) for i in sample_idxs]
            q_prediction_next_cat = torch.stack(q_prediction_next_list, -1)
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            backup = r.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (min_q - alpha_t * logp_a2.unsqueeze(dim=-1))

        # === Critic update ===
        # Extract features for current obs (WITH gradients for encoder from critics)
        features_o_critic, _, _ = self.model.forward_features(o)
        
        q_prediction_list = [q(features_o_critic, a) for q in self.model.qs]
        q_prediction_cat = torch.stack(q_prediction_list, -1)
        backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup

        loss_q = self.criterion(q_prediction_cat, backup)
        self.loss_q = loss_q.detach()

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # === Actor update (includes encoder gradients) ===
        loss_alpha = None
        if update_policy:
            for q in self.model.qs:
                q.requires_grad_(False)

            # Extract features DETACHED for actor (Actor doesn't update encoder)
            with torch.no_grad():
                features_o_actor, _, _ = self.model.forward_features(o)
            pi, logp_pi, _ = self.model.actor_from_features(features_o_actor, None)
            
            qs_pi = [q(features_o_actor, pi) for q in self.model.qs]
            qs_pi_cat = torch.stack(qs_pi, -1)
            ave_q = torch.mean(qs_pi_cat, dim=1, keepdim=True)
            loss_pi = (alpha_t * logp_pi.unsqueeze(dim=-1) - ave_q).mean()
            
            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()

            for q in self.model.qs:
                q.requires_grad_(True)

            # Entropy coefficient update (after actor update)
            if self.learn_entropy_coef:
                # Need fresh logp_pi for alpha gradient
                with torch.no_grad():
                    features_alpha, _, _ = self.model.forward_features(o)
                _, logp_pi_alpha, _ = self.model.actor_from_features(features_alpha, None)
                loss_alpha = -(self.log_alpha * (logp_pi_alpha + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                loss_alpha.backward()
                self.alpha_optimizer.step()

            self.loss_pi = loss_pi.detach()

        # === Polyak averaging ===
        if update_policy:
            with torch.no_grad():
                for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        ret_dict = dict(
            loss_actor=self.loss_pi.detach().item(),
            loss_critic=self.loss_q.detach().item(),
        )

        if self.learn_entropy_coef and loss_alpha is not None:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# ============== GRAC: Gradient-Rescaled Actor-Critic Components ==============

class GradientHealthTracker:
    """
    GHAE: Gradient-Health Adaptive Entropy (Novel — part of GRAC algorithm).

    Tracks per-dimension gradient health of the actor's mu_layer and provides
    entropy target boosts for dimensions with persistently vanishing gradients.

    Creates a self-healing feedback loop:
      vanishing gradients → more exploration → actions move from boundaries
      → gradients recover → exploration decreases → exploitation resumes
    """
    def __init__(self, dim_act, ema_decay=0.99, threshold=0.01,
                 boost=0.5, device='cpu'):
        self.dim_act = dim_act
        self.ema_decay = ema_decay
        self.threshold = threshold
        self.boost = boost
        self.grad_health = torch.ones(dim_act, device=device)

    def update(self, mu_layer_grad):
        """Update health scores from mu_layer weight gradient."""
        if mu_layer_grad is None:
            return
        # mu_layer.weight has shape (dim_act, hidden_dim)
        # Per-dimension gradient magnitude (mean across input dims)
        per_dim_mag = mu_layer_grad.abs().mean(dim=1)  # (dim_act,)
        self.grad_health = (self.ema_decay * self.grad_health +
                           (1 - self.ema_decay) * per_dim_mag.detach())

    def get_entropy_boost(self):
        """Compute per-dimension entropy target boost for unhealthy dims."""
        deficit = torch.relu(self.threshold - self.grad_health)
        return self.boost * deficit


# ============== DroQ SAC Agent ==========================

class DroQSACAgent(TrainingAgent):
    """
    DroQ (Dropout Q-functions) SAC Agent for maximum sample efficiency.
    
    Key features:
    - Uses only 2 Q-networks with Dropout+LayerNorm
    - Supports high UTD (Update-to-Data) ratios (20+)
    - Dropout provides ensemble-like diversity for uncertainty
    - Compatible with shared backbone architecture
    - EWC (Elastic Weight Consolidation) for continual learning across maps
    """
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space,
                 action_space,
                 device,
                 model_cls=core.DroQHybridActorCritic,
                 gamma=0.99,
                 polyak=0.995,
                 alpha=0.2,
                 lr_actor=1e-3,
                 lr_critic=1e-3,
                 lr_entropy=1e-3,
                 learn_entropy_coef=True,
                 target_entropy=None,
                 q_updates_per_policy_update=20,
                 model=None):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_entropy = lr_entropy
        self.learn_entropy_coef = learn_entropy_coef
        self.target_entropy = target_entropy
        self.n = 2  # DroQ always uses 2 Q-networks
        self.m = 2  # Use both for min-Q
        self.q_updates_per_policy_update = q_updates_per_policy_update
        self.alpha_floor = cfg.TMRL_CONFIG.get("ALG", {}).get("ALPHA_FLOOR", 0.08)

        # EWC (Elastic Weight Consolidation) for continual learning
        self.ewc_lambda = cfg.TMRL_CONFIG.get("ALG", {}).get("EWC_LAMBDA", 0.0)
        self._ewc_fisher = {}   # Fisher Information Matrix (diagonal)
        self._ewc_params = {}   # Optimal parameters from previous task
        self._ewc_active = False
        # Auto-load EWC state if it exists
        ewc_path = Path(cfg.WEIGHTS_FOLDER if hasattr(cfg, 'WEIGHTS_FOLDER') else r'C:\Users\felix\TmrlData\weights') / 'ewc_state.pkl'
        if ewc_path.exists() and self.ewc_lambda > 0:
            self.load_ewc_state(str(ewc_path))
            logging.info(f"EWC: Loaded consolidation state from {ewc_path}, lambda={self.ewc_lambda}")

        model = model if model is not None else model_cls(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        
        # Optimizer for actor's policy head only (features are fixed/detached for actor)
        self.pi_optimizer = Adam([
            {'params': list(self.model.actor.net.parameters()) +
                       list(self.model.actor.mu_layer.parameters()),
             'lr': self.lr_actor},
            {'params': list(self.model.actor.std_net.parameters()) +
                       (list(self.model.actor.log_std_layer.parameters())
                        if hasattr(self.model.actor, 'log_std_layer') else []),
             'lr': self.lr_actor},
        ], lr=self.lr_actor)
        
        # Optimizer for Q-heads AND Encoder (Critic drives representation)
        # Context encoder gets a lower LR because Q-loss gradients travel through
        # ~15 layers to reach it, becoming noisy. A lower LR prevents noise amplification.
        self._q_params = [p for q in self.model.qs for p in q.parameters()]
        self._encoder_params = list(self.model.actor.cnn.parameters()) + list(self.model.actor.float_mlp.parameters())
        if hasattr(self.model.actor, "fusion_norm"):
            self._encoder_params += list(self.model.actor.fusion_norm.parameters())
        if hasattr(self.model.actor, "fusion_gate"):
            self._encoder_params += list(self.model.actor.fusion_gate.parameters())
        
        self._context_params = []
        if hasattr(self.model.actor, "context_encoder"):
            self._context_params = list(self.model.actor.context_encoder.parameters())
        if hasattr(self.model.actor, "film_generator"):
            self._context_params += list(self.model.actor.film_generator.parameters())

        # Scale encoder-related learning rates by UTD to prevent representation churn.
        utd_ratio = max(1.0, float(self.q_updates_per_policy_update))
        encoder_lr = min(self.lr_critic / utd_ratio, 3e-5)
        context_lr = min(self.lr_critic / utd_ratio, 3e-5)

        optimizer_groups = [
            {'params': self._q_params, 'lr': self.lr_critic},
            {'params': self._encoder_params, 'lr': encoder_lr},
        ]
        if self._context_params:
            optimizer_groups.append({'params': self._context_params, 'lr': context_lr})
        self.q_optimizer = Adam(optimizer_groups)
        
        # Use MSELoss so critics can aggressively correct large Q-target errors.
        self.criterion = torch.nn.MSELoss()
        self.loss_pi = torch.zeros((1,), device=device)
        self.loss_q = torch.zeros((1,), device=device)
        self.i_update = 0

        dim_act = action_space.shape[0]
        if self.target_entropy is None:
            self.target_entropy = torch.full((dim_act,), -1.0, device=self.device)
        else:
            per_dim = float(self.target_entropy) / dim_act
            self.target_entropy = torch.full((dim_act,), per_dim, device=self.device)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(dim_act, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.full((dim_act,), float(self.alpha), device=self.device)

        # GRAC: Gradient-Health Adaptive Entropy tracker (configurable from config.json)
        ghae_threshold = cfg.TMRL_CONFIG.get("ALG", {}).get("GHAE_THRESHOLD", 0.02)
        ghae_boost = cfg.TMRL_CONFIG.get("ALG", {}).get("GHAE_BOOST", 2.0)
        self.grad_tracker = GradientHealthTracker(
            dim_act=dim_act, device=self.device,
            threshold=ghae_threshold, boost=ghae_boost
        )

        # ── World Model (RSSM) ─────────────────────────────────────────────
        wm_cfg = cfg.TMRL_CONFIG.get("WORLD_MODEL", {})
        self.wm_enabled = wm_cfg.get("ENABLED", False)
        if self.wm_enabled:
            from tmrl.custom.custom_models import LatentWorldModel
            wm_state_dim = 28  # speed + gear + rpm + lidar(19) + xyz(3) + progress(1) + crash(1) + progress_gain(1)
            wm_latent = wm_cfg.get("LATENT_DIM", 32)
            wm_gru = wm_cfg.get("GRU_DIM", 128)
            wm_hidden = wm_cfg.get("HIDDEN_DIM", 256)
            wm_kl_free = wm_cfg.get("KL_FREE_NATS", 1.0)
            self.dynamics = LatentWorldModel(
                state_dim=wm_state_dim, action_dim=dim_act,
                latent_dim=wm_latent, gru_dim=wm_gru,
                hidden_dim=wm_hidden, kl_free_nats=wm_kl_free
            ).to(device)
            self.dynamics_optimizer = Adam(self.dynamics.parameters(),
                                          lr=wm_cfg.get("MODEL_LR", 3e-4))
            self.wm_warmup = wm_cfg.get("WARMUP_STEPS", 3000)
            self.wm_horizon = wm_cfg.get("ROLLOUT_HORIZON", 15)
            self.wm_batch_size = wm_cfg.get("IMAGINED_BATCH_SIZE", 256)
            self.wm_train_steps = 0
            logging.info(f"World Model RSSM: latent={wm_latent}, gru={wm_gru}, "
                         f"horizon={self.wm_horizon}, warmup={self.wm_warmup}")
            self.curiosity_scale = wm_cfg.get("CURIOSITY_SCALE", 0.1)
            self._last_surprise_mean = 0.0  # tracks latest surprise for dynamic alpha floor
            self._last_verifier_trust = 1.0 # conviction score for dynamic alpha floor

            # Imagination Actor: learned policy for realistic imagined rollouts
            from tmrl.custom.custom_models import ImaginationActor, RunningMeanStd
            self.imag_actor = ImaginationActor(
                state_dim=wm_state_dim, action_dim=dim_act,
            ).to(device)
            self.imag_actor_optimizer = Adam(self.imag_actor.parameters(), lr=1e-3)
            self.imag_noise_scale = wm_cfg.get("IMAGINATION_NOISE_SCALE", 0.3)
            self.curiosity_reward_clip = wm_cfg.get("CURIOSITY_REWARD_CLIP", 5.0)
            self.curiosity_rms = RunningMeanStd()

    def get_actor(self):
        # Avoid deepcopy-based export for DroQ models:
        # certain parametrized tensors are not deepcopy-compatible in recent torch.
        return self.model.actor

    def _alpha_floor_tensor(self):
        alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
        dim_act = self.action_space.shape[0]
        floor_all = float(alg_cfg.get("ALPHA_FLOOR", 0.0))
        floor_cfg = alg_cfg.get("ALPHA_FLOOR_PER_DIM", None)
        if floor_cfg is not None:
            floors = [float(v) for v in floor_cfg]
        else:
            floors = [
                float(alg_cfg.get("ALPHA_FLOOR_STEER", floor_all)),
                float(alg_cfg.get("ALPHA_FLOOR_GAS", floor_all)),
                float(alg_cfg.get("ALPHA_FLOOR_BRAKE", floor_all)),
            ]
        if len(floors) < dim_act:
            floors.extend([floor_all] * (dim_act - len(floors)))
        return torch.tensor(floors[:dim_act], device=self.device, dtype=torch.float32)

    def _apply_alpha_floor(self):
        floor = self._alpha_floor_tensor()
        if self.learn_entropy_coef:
            with torch.no_grad():
                log_floor = torch.log(floor.clamp_min(1e-12))
                self.log_alpha.data.copy_(torch.maximum(self.log_alpha.data, log_floor))
            return torch.exp(self.log_alpha.detach()), floor
        return torch.maximum(self.alpha_t, floor), floor

    # ── World Model helpers ──────────────────────────────────────────────

    def _extract_critic_state(self, obs):
        """
        Extract the 28-dim Critic state vector from a batched observation tuple.
        Returns: (B, 28) tensor on self.device
        """
        if len(obs) == 9:
            speed, gear, rpm, images, lidar, xyz, progress, act1, act2 = obs
            B = images.shape[0]
            crash = torch.zeros((B, 1), device=speed.device)
            progress_gain = torch.zeros((B, 1), device=speed.device)
        else:
            speed, gear, rpm, images, lidar, xyz, progress, crash, progress_gain, act1, act2 = obs
            B = images.shape[0]
            
        speed = speed.view(B, -1)
        gear = gear.view(B, -1)
        rpm = rpm.view(B, -1)
        lidar = lidar.view(B, -1)
        xyz = xyz.view(B, -1)
        progress = progress.view(B, -1)
        crash = crash.view(B, -1)
        progress_gain = progress_gain.view(B, -1)
        return torch.cat((speed, gear, rpm, lidar, xyz, progress, crash, progress_gain), dim=-1)  # (B, 28)

    def _train_dynamics(self, o, a, r, o2, context_z=None):
        """
        Train the RSSM world model on real transitions.
        Uses reconstruction + KL divergence loss.
        Args:
            o:  observation tuple (batched)
            a:  actions (B, 3)
            r:  rewards (B,)
            o2: next observation tuple (batched)
            context_z: (B, 64) — PEARL context vector (optional)
        Returns:
            metrics: dict with component losses for logging
        """
        state = self._extract_critic_state(o).detach()
        next_state = self._extract_critic_state(o2).detach()
        reward = r.unsqueeze(-1) if r.dim() == 1 else r  # (B, 1)

        wm_input_metrics = {}
        _add_tensor_stats(wm_input_metrics, "wm_input/state", state)
        _add_tensor_stats(wm_input_metrics, "wm_input/next_state", next_state)
        _add_tensor_stats(wm_input_metrics, "wm_input/action", a)
        _add_tensor_stats(wm_input_metrics, "wm_input/reward", reward)
        wm_input_metrics["bridge/context_z_to_wm_present"] = 1.0 if context_z is not None else 0.0
        wm_input_metrics["bridge/context_z_requires_grad"] = float(context_z.requires_grad) if context_z is not None else 0.0
        if context_z is not None:
            _add_tensor_stats(wm_input_metrics, "wm_input/context_z", context_z)
            _add_tensor_stats(wm_input_metrics, "bridge/context_z", context_z)

        loss, metrics = self.dynamics.train_step(state, a, next_state, reward, context_z=context_z)
        metrics = dict(metrics or {})
        metrics.update(wm_input_metrics)

        self.dynamics_optimizer.zero_grad()
        loss.backward()
        metrics["wm_grad/encoder"] = _module_grad_norm(getattr(self.dynamics, "encoder", None))
        metrics["wm_grad/prior"] = _module_grad_norm(getattr(self.dynamics, "prior", None))
        metrics["wm_grad/posterior"] = _module_grad_norm(getattr(self.dynamics, "posterior", None))
        decoder = getattr(self.dynamics, "decoder", None)
        metrics["wm_grad/decoder"] = _module_grad_norm(getattr(decoder, "state_net", None))
        metrics["wm_grad/reward_head"] = _module_grad_norm(getattr(decoder, "reward_net", None))
        metrics["wm_grad/total"] = _module_grad_norm(self.dynamics)
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 5.0)
        self.dynamics_optimizer.step()

        self.wm_train_steps += 1
        return metrics

    def _extract_prev_actions(self, obs, B=None):
        """
        Extract previous actions (act1, act2) from a batched observation tuple.
        Returns: act1 (B, 3), act2 (B, 3)
        """
        if len(obs) == 9:
            _, _, _, images, _, _, _, act1, act2 = obs
        else:
            _, _, _, images, _, _, _, _, _, act1, act2 = obs
        if B is None:
            B = images.shape[0]
        return act1[:B].view(B, -1).detach(), act2[:B].view(B, -1).detach()

    def _imagined_critic_update(self, o, ctx=None, context_z=None, real_action=None):
        """
        Generate imagined rollouts in latent space and perform Critic gradient updates.
        Uses the RSSM prior to roll forward, decodes to critic state, then
        computes proper 1-step TD targets using the actual Q-network pipeline.
        """
        state = self._extract_critic_state(o).detach()
        B = min(state.shape[0], self.wm_batch_size)
        state = state[:B]
        if context_z is not None:
            context_z = context_z[:B].detach()
        if real_action is not None:
            real_action = real_action[:B].detach()

        # Extract real previous actions from batch to seed action history
        real_act1, real_act2 = self._extract_prev_actions(o, B)  # (B, 3) each

        # Policy function for imagination: use learned ImaginationActor + noise
        noise_scale = getattr(self, 'imag_noise_scale', 0.3)
        def policy_fn(critic_state):
            with torch.no_grad():
                base_action = self.imag_actor(critic_state)
            noise = torch.randn_like(base_action) * noise_scale
            return (base_action + noise).clamp(-1.0, 1.0)

        # Imagine H steps into the future (conditioned on PEARL context)
        with torch.no_grad():
            imag_states, imag_rewards, imag_actions, imag_uncertainties = self.dynamics.imagine(
                state, policy_fn, self.wm_horizon, self.gamma, context_z=context_z
            )

        # Use the last decoded state + reward for a 1-step TD critic update
        # Take transitions from each horizon step as independent training data
        H = imag_states.shape[0]
        imag_metrics = {}
        for h in range(H):
            _add_tensor_stats(imag_metrics, f"wm_imag/h{h}_state", imag_states[h])
            _add_tensor_stats(imag_metrics, f"wm_imag/h{h}_action", imag_actions[h])
            _add_tensor_stats(imag_metrics, f"wm_imag/h{h}_reward", imag_rewards[h])
            if imag_uncertainties is not None:
                _add_tensor_stats(imag_metrics, f"wm_imag/h{h}_uncertainty", imag_uncertainties[h])

        imag_metrics["bridge/context_z_to_critic_present"] = 1.0 if context_z is not None else 0.0
        imag_metrics["bridge/context_z_requires_grad"] = float(context_z.requires_grad) if context_z is not None else 0.0
        if context_z is not None:
            _add_tensor_stats(imag_metrics, "bridge/context_z", context_z)

        if H < 2:
            imag_metrics["wm_imagined_steps"] = 0
            return imag_metrics

        # === Build proper action history for critic input ===
        # The real critic expects (state_28, act1=a_{t-1}, act2=a_{t-2}, z_64)
        # We must track the 2-step action history through the imagination rollout.
        #
        # At horizon step h, the "previous actions" are:
        #   act1_h = imag_actions[h-1]  (action taken at step h-1)
        #   act2_h = imag_actions[h-2]  (action taken at step h-2)
        #
        # For h=0: act1=real_act1, act2=real_act2 (from the batch)
        # For h=1: act1=imag_actions[0], act2=real_act1
        # For h>=2: act1=imag_actions[h-1], act2=imag_actions[h-2]

        # Build act1_history and act2_history for each horizon step (H, B, 3)
        act1_history = []
        act2_history = []
        for h in range(H):
            if h == 0:
                act1_history.append(real_act1)
                act2_history.append(real_act2)
            elif h == 1:
                act1_history.append(imag_actions[0])
                act2_history.append(real_act1)
            else:
                act1_history.append(imag_actions[h - 1])
                act2_history.append(imag_actions[h - 2])

        act1_stack = torch.stack(act1_history, dim=0)  # (H, B, 3)
        act2_stack = torch.stack(act2_history, dim=0)  # (H, B, 3)

        # Flatten horizon: treat each (s_t, a_t, r_t, s_{t+1}) as a transition
        s_flat = imag_states[:-1].reshape(-1, self.dynamics.state_dim)    # ((H-1)*B, state_dim)
        a_flat = imag_actions[:-1].reshape(-1, self.dynamics.action_dim)  # ((H-1)*B, 3)
        r_flat = imag_rewards[:-1].reshape(-1, 1)                         # ((H-1)*B, 1)
        ns_flat = imag_states[1:].reshape(-1, self.dynamics.state_dim)    # ((H-1)*B, state_dim)
        u_t_flat = imag_uncertainties[:-1].reshape(-1, 1)                 # ((H-1)*B, 1)

        # Previous actions for current states (s_flat) and next states (ns_flat)
        act1_s = act1_stack[:-1].reshape(-1, self.dynamics.action_dim)   # ((H-1)*B, 3)
        act2_s = act2_stack[:-1].reshape(-1, self.dynamics.action_dim)   # ((H-1)*B, 3)
        act1_ns = act1_stack[1:].reshape(-1, self.dynamics.action_dim)   # ((H-1)*B, 3)
        act2_ns = act2_stack[1:].reshape(-1, self.dynamics.action_dim)   # ((H-1)*B, 3)

        _add_tensor_stats(imag_metrics, "bridge/state_real", state)
        _add_tensor_stats(imag_metrics, "bridge/state_imag", s_flat)
        if state.shape[-1] == s_flat.shape[-1]:
            imag_metrics["bridge/state_distribution_gap"] = (
                state.detach().mean(dim=0) - s_flat.detach().mean(dim=0)
            ).abs().mean().item()
        if real_action is not None:
            _add_tensor_stats(imag_metrics, "bridge/action_real", real_action)
        _add_tensor_stats(imag_metrics, "bridge/action_imag", a_flat)
        _add_tensor_stats(imag_metrics, "bridge/act1_real", real_act1)
        _add_tensor_stats(imag_metrics, "bridge/act2_real", real_act2)
        imag_metrics["bridge/act1_vs_imag_action_abs_diff"] = (act1_s.detach() - a_flat.detach()).abs().mean().item()
        imag_metrics["bridge/act2_vs_act1_abs_diff"] = (act2_s.detach() - act1_s.detach()).abs().mean().item()

        # Compute target Q-values using imagined next states
        with torch.no_grad():
            a2 = self.imag_actor(ns_flat)
            a2_noise = torch.randn_like(a2) * noise_scale
            a2 = (a2 + a2_noise).clamp(-1.0, 1.0)
            # Build critic_floats: the Q-heads take 34-dim (28 state + 3 act1 + 3 act2)
            # Use proper action history for next-state critic input
            critic_ns = torch.cat([ns_flat, act1_ns, act2_ns], dim=-1)

            # Use real PEARL context for Q-network input during imagination
            if context_z is not None:
                z_imag = context_z.unsqueeze(0).expand(H-1, -1, -1).reshape(-1, 64)
                critic_ns = torch.cat([critic_ns, z_imag], dim=-1)
            else:
                z_neutral = torch.zeros(ns_flat.shape[0], 64, device=self.device)
                critic_ns = torch.cat([critic_ns, z_neutral], dim=-1)

            # Get FiLM params — use zeros (neutral modulation) for imagined data
            if hasattr(self.model, 'film_generator'):
                z_for_film = torch.zeros(ns_flat.shape[0], 64, device=self.device)
                film_params_neutral = self.model.film_generator(z_for_film)
            else:
                film_params_neutral = None

            # Target Q from target network
            q_next_list = [q(critic_ns, a2, film_params_neutral) for q in self.model_target.qs]
            q_next_cat = torch.stack(q_next_list, -1)
            min_q_next = torch.min(q_next_cat, dim=1)[0]  # (N,)

            target_q = r_flat.squeeze(-1) + self.gamma * min_q_next  # (N,)

        # Critic update on imagined transitions — use proper action history
        critic_s = torch.cat([s_flat, act1_s, act2_s], dim=-1)
        
        if context_z is not None:
            z_imag_grad = context_z.unsqueeze(0).expand(H-1, -1, -1).reshape(-1, 64)
            critic_s = torch.cat([critic_s, z_imag_grad], dim=-1)
        else:
            if hasattr(self.model, 'context_encoder'):
                z_neutral_grad = torch.zeros(s_flat.shape[0], 64, device=self.device)
                critic_s = torch.cat([critic_s, z_neutral_grad], dim=-1)
            
        if hasattr(self.model, 'film_generator'):
            z_for_film_grad = torch.zeros(s_flat.shape[0], 64, device=self.device)
            film_params_grad = self.model.film_generator(z_for_film_grad)
        else:
            film_params_grad = None

        q_pred_list = [q(critic_s, a_flat, film_params_grad) for q in self.model.qs]
        q_pred_cat = torch.stack(q_pred_list, -1)  # (N, 2)
        target_q_expanded = target_q.unsqueeze(-1).expand_as(q_pred_cat)
        with torch.no_grad():
            q_imag = q_pred_cat.detach()
            target_q_detached = target_q.detach()
            imag_td_error = q_imag - target_q_expanded.detach()
            imag_metrics["bridge/q_imag_mean"] = q_imag.mean().item()
            imag_metrics["bridge/q_imag_std"] = q_imag.std(unbiased=False).item()
            imag_metrics["bridge/target_q_imag_mean"] = target_q_detached.mean().item()
            imag_metrics["bridge/target_q_imag_std"] = target_q_detached.std(unbiased=False).item()
            imag_metrics["bridge/imag_td_error_mean"] = imag_td_error.mean().item()
            imag_metrics["bridge/imag_td_error_abs_mean"] = imag_td_error.abs().mean().item()

        loss_unreduced = F.mse_loss(q_pred_cat, target_q_expanded, reduction='none')
        
        # === Verifier-Gated Imagination ===
        # Compute trust metric m_t from uncertainty u_t
        lambda_trust = cfg.TMRL_CONFIG.get("ALG", {}).get("VERIFIER_LAMBDA", 10.0)
        m_t = torch.exp(-lambda_trust * u_t_flat)  # ((H-1)*B, 1)
        m_t_expanded = m_t.expand_as(loss_unreduced)
        _add_tensor_stats(imag_metrics, "wm/verifier_uncertainty", u_t_flat)
        _add_tensor_stats(imag_metrics, "wm/verifier_trust", m_t)
        imag_metrics["wm/verifier_trust_saturation_low"] = (m_t.detach() < 0.05).float().mean().item()
        imag_metrics["wm/verifier_trust_saturation_high"] = (m_t.detach() > 0.95).float().mean().item()
        _add_tensor_stats(imag_metrics, "bridge/trust", m_t)
        imag_metrics["bridge/trust_saturation_low"] = (m_t.detach() < 0.05).float().mean().item()
        imag_metrics["bridge/trust_saturation_high"] = (m_t.detach() > 0.95).float().mean().item()
        
        # Weight loss by trust metric
        loss_q_imagined = (loss_unreduced * m_t_expanded).mean()

        self.q_optimizer.zero_grad()
        loss_q_imagined.backward()
        # Clip actor gradients loosely to prevent extreme spikes from deterministic pull
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self._q_params, 1.0)
        self.q_optimizer.step()

        imag_metrics.update({
            "wm_imagined_steps": H * B,
            "wm_imagined_q_loss": loss_q_imagined.item(),
            "verifier_uncertainty_mean": u_t_flat.mean().item(),
            "verifier_trust_mt_mean": m_t.mean().item(),
            "grad_norm_critic": actor_grad_norm.item(),
        })
        return imag_metrics

    # ── EWC: Elastic Weight Consolidation ────────────────────────────────

    def consolidate_task(self, memory=None, n_samples=2000):
        """
        Compute Fisher Information Matrix for the current task.
        Call this BEFORE switching to a new map.
        
        The Fisher matrix captures which parameters are most important
        for the current task. During future training, deviations from
        these parameters are penalized proportionally to their importance.
        """
        logging.info("EWC: Computing Fisher Information Matrix...")
        self.model.eval()
        
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
        
        # Use replay buffer if available, else use random noise
        if memory is not None:
            n_batches = max(1, n_samples // 64)
            for _ in range(n_batches):
                try:
                    batch = memory.sample()
                    # Forward pass to get loss
                    img_ctx = None
                    if len(batch) == 8:
                        o, a, r, o2, d, _, ctx_full, img_ctx_full = batch
                        ctx = ctx_full[:, :-1, :]
                        img_ctx = img_ctx_full[:, :-1, :, :, :]
                    elif len(batch) == 7:
                        o, a, r, o2, d, _, ctx_full = batch
                        ctx = ctx_full[:, :-1, :]
                    else:
                        o, a, r, o2, d, _ = batch
                        ctx = None
                    
                    uses_context = hasattr(self.model, 'context_encoder') and self.model.context_encoder is not None
                    if uses_context and ctx is not None:
                        fused, critic, film, _ = self.model.forward_features(o, ctx, img_context=img_ctx)
                    else:
                        fused, critic, film, _ = self.model.forward_features(o)
                    
                    q_list = [q(fused, a, film) for q in self.model.qs]
                    # Use Q-values as proxy for task importance
                    for q_val in q_list:
                        self.model.zero_grad()
                        q_val.mean().backward(retain_graph=True)
                        for name, param in self.model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                fisher[name] += param.grad.data.pow(2) / n_batches
                except Exception as e:
                    logging.warning(f"EWC: Skipping batch due to: {e}")
                    continue
        
        # Normalize and store
        for name in fisher:
            fisher[name] = fisher[name].clamp(max=100.0)  # prevent extreme values
        
        self._ewc_fisher = fisher
        self._ewc_params = {name: param.data.clone() 
                           for name, param in self.model.named_parameters() 
                           if param.requires_grad}
        self._ewc_active = True
        
        n_params = sum(f.numel() for f in fisher.values())
        mean_fisher = sum(f.mean().item() for f in fisher.values()) / max(len(fisher), 1)
        logging.info(f"EWC: Consolidated {n_params:,} parameters, mean Fisher={mean_fisher:.4f}")
        self.model.train()

    def _ewc_loss(self):
        """Compute EWC penalty: Σ F_i * (θ_i - θ*_i)²"""
        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self._ewc_fisher and name in self._ewc_params:
                fisher = self._ewc_fisher[name]
                optimal = self._ewc_params[name]
                loss += (fisher * (param - optimal).pow(2)).sum()
        return loss

    def save_ewc_state(self, path=None):
        """Save Fisher matrix + optimal params to disk."""
        if path is None:
            path = str(Path(r'C:\Users\felix\TmrlData\weights') / 'ewc_state.pkl')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            'fisher': {k: v.cpu() for k, v in self._ewc_fisher.items()},
            'params': {k: v.cpu() for k, v in self._ewc_params.items()},
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logging.info(f"EWC: Saved consolidation state to {path}")

    def load_ewc_state(self, path=None):
        """Load Fisher matrix + optimal params from disk."""
        if path is None:
            path = str(Path(r'C:\Users\felix\TmrlData\weights') / 'ewc_state.pkl')
        if not Path(path).exists():
            logging.warning(f"EWC: No state file at {path}")
            return
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self._ewc_fisher = {k: v.to(self.device) for k, v in state['fisher'].items()}
        self._ewc_params = {k: v.to(self.device) for k, v in state['params'].items()}
        self._ewc_active = True
        logging.info(f"EWC: Loaded consolidation state ({len(self._ewc_fisher)} params)")


    def train(self, batch):
        """
        DroQ training step.
        Same as SharedBackboneREDQSAC but with dropout enabled during training.
        Supports context-augmented batches for context-based meta-learning.
        """
        # Backward compatibility for checkpoints (init logic bypassed)
        if not hasattr(self, "q_updates_per_policy_update"):
             self.q_updates_per_policy_update = cfg.TMRL_CONFIG["ALG"]["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]
        if not hasattr(self, "det_reg_lambda"):
             self.det_reg_lambda = cfg.TMRL_CONFIG.get("ALG", {}).get("DET_REG_LAMBDA", 0.01)
        # Entropy is now fully managed by the Target Entropy controller (Auto-Annealer).
        # We removed alpha_floor here to perfectly align the SAC optimizer.

        # Lazy-init World Model for checkpoints loaded via pickle (bypasses __init__)
        if not hasattr(self, "wm_enabled"):
            wm_cfg = cfg.TMRL_CONFIG.get("WORLD_MODEL", {})
            self.wm_enabled = wm_cfg.get("ENABLED", False)
            if self.wm_enabled:
                from tmrl.custom.custom_models import LatentWorldModel
                dim_act = self.action_space.shape[0]
                wm_state_dim = 28  # speed + gear + rpm + lidar(19) + xyz(3) + progress(1) + crash(1) + progress_gain(1)
                wm_latent = wm_cfg.get("LATENT_DIM", 32)
                wm_gru = wm_cfg.get("GRU_DIM", 128)
                wm_hidden = wm_cfg.get("HIDDEN_DIM", 256)
                wm_kl_free = wm_cfg.get("KL_FREE_NATS", 1.0)
                self.dynamics = LatentWorldModel(
                    state_dim=wm_state_dim, action_dim=dim_act,
                    latent_dim=wm_latent, gru_dim=wm_gru,
                    hidden_dim=wm_hidden, kl_free_nats=wm_kl_free
                ).to(self.device)
                self.dynamics_optimizer = Adam(self.dynamics.parameters(),
                                              lr=wm_cfg.get("MODEL_LR", 3e-4))
                self.wm_warmup = wm_cfg.get("WARMUP_STEPS", 3000)
                self.wm_horizon = wm_cfg.get("ROLLOUT_HORIZON", 15)
                self.wm_batch_size = wm_cfg.get("IMAGINED_BATCH_SIZE", 256)
                self.wm_train_steps = 0
                logging.info(f"World Model RSSM (lazy init): latent={wm_latent}, gru={wm_gru}")
                self.curiosity_scale = wm_cfg.get("CURIOSITY_SCALE", 0.1)
                self._last_surprise_mean = 0.0
                self._last_verifier_trust = 1.0

        # Lazy-init ImaginationActor for checkpoints that predate this feature
        if self.wm_enabled and not hasattr(self, 'imag_actor'):
            from tmrl.custom.custom_models import ImaginationActor, RunningMeanStd
            wm_cfg = cfg.TMRL_CONFIG.get("WORLD_MODEL", {})
            dim_act = self.action_space.shape[0]
            self.imag_actor = ImaginationActor(
                state_dim=28, action_dim=dim_act,
            ).to(self.device)
            self.imag_actor_optimizer = Adam(self.imag_actor.parameters(), lr=1e-3)
            self.imag_noise_scale = wm_cfg.get("IMAGINATION_NOISE_SCALE", 0.3)
            self.curiosity_reward_clip = wm_cfg.get("CURIOSITY_REWARD_CLIP", 5.0)
            self.curiosity_rms = RunningMeanStd()
            logging.info("ImaginationActor lazy-initialized for existing checkpoint")

        # Always refresh curiosity config from live config (so config.json changes
        # take effect on trainer restart without needing RESET_TRAINING=true)
        if self.wm_enabled:
            wm_cfg_live = cfg.TMRL_CONFIG.get("WORLD_MODEL", {})
            self.curiosity_scale = wm_cfg_live.get("CURIOSITY_SCALE", 0.1)
            self.curiosity_reward_clip = wm_cfg_live.get("CURIOSITY_REWARD_CLIP", 5.0)


        self.i_update += 1
        # DroQ uses high UTD ratio
        update_policy = (self.i_update % self.q_updates_per_policy_update == 0)

        # Unpack batch - support context-augmented (7 or 8 elements) and standard (6 elements)
        img_ctx = None
        img_ctx_next = None
        if len(batch) == 8:
            o, a, r, o2, d, _, ctx_full, img_ctx_full = batch
            ctx = ctx_full[:, :-1, :]
            ctx_next = ctx_full[:, 1:, :]
            img_ctx = img_ctx_full[:, :-1, :, :, :]
            img_ctx_next = img_ctx_full[:, 1:, :, :, :]
        elif len(batch) == 7:
            o, a, r, o2, d, _, ctx_full = batch
            ctx = ctx_full[:, :-1, :]
            ctx_next = ctx_full[:, 1:, :]
        else:
            o, a, r, o2, d, _ = batch
            ctx = None
            ctx_next = None

        # Force update learning rates and target entropy from live config on every trainer restart.
        # Uses a versioned flag so bumping the version forces re-application on existing checkpoints.
        _OVERRIDE_VERSION = 3  # Bump this to force re-apply on next restart
        if getattr(self, "_override_version", 0) < _OVERRIDE_VERSION:
            import logging
            new_lr_actor = 1e-4
            new_lr_critic = 1e-4
            for param_group in self.pi_optimizer.param_groups:
                param_group['lr'] = new_lr_actor
            for param_group in self.q_optimizer.param_groups:
                param_group['lr'] = new_lr_critic
            # Fix GAMMA baked in checkpoint
            self.gamma = 0.99
            # Refresh target entropy from live config.json (prevents entropy death spiral)
            cfg_target = cfg.TMRL_CONFIG.get("ALG", {}).get("TARGET_ENTROPY", None)
            if cfg_target is not None:
                dim_act = self.action_space.shape[0]
                per_dim = float(cfg_target) / dim_act
                self.target_entropy = torch.full((dim_act,), per_dim, device=self.device)
                logging.info(f" === OVERRIDE: TARGET_ENTROPY={per_dim:.2f} per dim (total={cfg_target}) ===")
            # Refresh GHAE params from live config
            ghae_threshold = cfg.TMRL_CONFIG.get("ALG", {}).get("GHAE_THRESHOLD", 0.02)
            ghae_boost = cfg.TMRL_CONFIG.get("ALG", {}).get("GHAE_BOOST", 2.0)
            if hasattr(self, 'grad_tracker'):
                self.grad_tracker.threshold = ghae_threshold
                self.grad_tracker.boost = ghae_boost
                logging.info(f" === OVERRIDE: GHAE(threshold={ghae_threshold}, boost={ghae_boost}) ===")
            alpha_floor_t = self._alpha_floor_tensor()
            if alpha_floor_t.numel() >= 3:
                logging.info(
                    f" === OVERRIDE: ALPHA_FLOOR(steer={alpha_floor_t[0].item():.4f}, "
                    f"gas={alpha_floor_t[1].item():.4f}, brake={alpha_floor_t[2].item():.4f}) ==="
                )
            logging.info(f" === OVERRIDE: LR(Actor={new_lr_actor}, Critic={new_lr_critic}), GAMMA=0.99 ===")
            self._override_version = _OVERRIDE_VERSION

        # Ensure Q-networks are in training mode (dropout active)
        self.model.train()

        # Get current alpha
        alpha_t, alpha_floor_t = self._apply_alpha_floor()

        # Check if model supports FiLM context (None = Vanilla baseline)
        uses_context = hasattr(self.model, 'context_encoder') and self.model.context_encoder is not None

        # === Target Q computation (with no_grad, dropout active) ===
        with torch.no_grad():
            self.model_target.train()
            self.model.train()        
            
            # BUG 1 FIX: Use CURRENT policy to get next action a2
            if uses_context and ctx_next is not None:
                fused_o2_curr, _, film_o2_curr, z_o2 = self.model.forward_features(o2, ctx_next, img_context=img_ctx_next)
            else:
                fused_o2_curr, _, film_o2_curr, z_o2 = self.model.forward_features(o2)
            a2, logp_a2, _ = self.model.actor_from_features(fused_o2_curr, film_o2_curr, z=z_o2)
            self.model.train()        # Re-enable dropout

            # Now evaluate the Q-value of that action using the TARGET network
            if uses_context and ctx_next is not None:
                _, critic_o2_tgt, film_o2_tgt, _ = self.model_target.forward_features(o2, ctx_next, img_context=img_ctx_next)
            else:
                _, critic_o2_tgt, film_o2_tgt, _ = self.model_target.forward_features(o2)

            # Use both Q-networks for min-Q
            q_prediction_next_list = [q(critic_o2_tgt, a2, film_o2_tgt) for q in self.model_target.qs]
            q_prediction_next_cat = torch.stack(q_prediction_next_list, -1)
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            alpha_scalar = alpha_t.mean()

            # === Curiosity bonus: reward novel states the WM hasn't seen ===
            r_augmented = r
            if self.wm_enabled and self.wm_train_steps > self.wm_warmup:
                state_cur = self._extract_critic_state(o).detach()
                state_nxt = self._extract_critic_state(o2).detach()
                surprise = self.dynamics.compute_surprise(state_cur, a, state_nxt)  # (B,)
                # Normalize surprise with running stats for scale consistency
                self.curiosity_rms.update(surprise)
                surprise_norm = self.curiosity_rms.normalize(surprise)
                clip_val = getattr(self, 'curiosity_reward_clip', 5.0)
                surprise_norm = surprise_norm.clamp(-clip_val, clip_val)
                curiosity_bonus = self.curiosity_scale * surprise_norm
                r_augmented = r + curiosity_bonus

            backup = r_augmented.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (min_q - alpha_scalar * logp_a2.unsqueeze(dim=-1))

        # === Critic update (with dropout) ===
        self.model.train()  # Ensure dropout is active
        if uses_context and ctx is not None:
            _, critic_o_curr, film_o_critic, z_critic = self.model.forward_features(o, ctx, img_context=img_ctx)
        else:
            _, critic_o_curr, film_o_critic, z_critic = self.model.forward_features(o)
        
        q_prediction_list = [q(critic_o_curr, a, film_o_critic) for q in self.model.qs]
        q_prediction_cat = torch.stack(q_prediction_list, -1)
        backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup

        loss_q = self.criterion(q_prediction_cat, backup)

        # EWC: penalize deviation from previous task's optimal params
        loss_ewc = torch.zeros(1, device=loss_q.device)
        if self._ewc_active and self.ewc_lambda > 0:
            loss_ewc = self._ewc_loss()
            loss_q = loss_q + self.ewc_lambda * loss_ewc

        # === Variational KL divergence loss (PEARL) ===
        loss_kl = torch.zeros(1, device=loss_q.device)
        if uses_context and ctx is not None and hasattr(self.model.context_encoder, 'last_kl_div'):
            loss_kl = self.model.context_encoder.last_kl_div
            # β-VAE weighting: small enough not to overwhelm RL signal
            kl_beta = cfg.TMRL_CONFIG.get("ALG", {}).get("KL_BETA", 0.05)
            # Free nats: allow a minimum KL without penalty to prevent posterior collapse.
            # Without this, the KL term pushes q(z|c) → N(0,I), making z uninformative.
            kl_free_nats = cfg.TMRL_CONFIG.get("ALG", {}).get("KL_FREE_NATS", 1.0)
            loss_kl_clipped = torch.clamp(loss_kl - kl_free_nats, min=0.0)
            loss_q = loss_q + kl_beta * loss_kl_clipped

        self.loss_q = loss_q.detach()

        self.q_optimizer.zero_grad()
        loss_q.backward()
        # Per-module gradient clipping: context encoder gets tighter clip
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self._q_params, 1.0)
        torch.nn.utils.clip_grad_norm_(self._encoder_params, 1.0)
        if self._context_params:
            torch.nn.utils.clip_grad_norm_(self._context_params, 0.5)
        self.q_optimizer.step()

        # === Actor update ===
        loss_alpha = None
        effective_target_entropy_diag = None
        actor_bridge_metrics = {}
        if update_policy:
            for q in self.model.qs:
                q.requires_grad_(False)

            with torch.no_grad():
                if uses_context and ctx is not None:
                    fused_o_actor, critic_o_actor, film_o_actor, z_actor = self.model.forward_features(o, ctx, img_context=img_ctx)
                else:
                    fused_o_actor, critic_o_actor, film_o_actor, z_actor = self.model.forward_features(o)
            pi, logp_pi, logp_per_dim = self.model.actor_from_features(fused_o_actor, film_o_actor, z=z_actor)
            
            qs_pi = [q(critic_o_actor, pi, film_o_actor) for q in self.model.qs]
            qs_pi_cat = torch.stack(qs_pi, -1)
            min_q_pi = torch.min(qs_pi_cat, dim=1, keepdim=True)[0]
            std_q_pi = torch.std(qs_pi_cat, dim=1, keepdim=True)
            
            # Small SAC variant: uncertainty bonus and KL brake (ensemble disagreement)
            unc_bonus = cfg.TMRL_CONFIG.get("ALG", {}).get("UNCERTAINTY_BONUS", 0.0)
            kl_brake = cfg.TMRL_CONFIG.get("ALG", {}).get("KL_BRAKE", 0.0)
            q_target_pi = min_q_pi + (unc_bonus - kl_brake) * std_q_pi
            actor_bridge_metrics["bridge/actor_action_requires_grad"] = float(pi.requires_grad)
            actor_bridge_metrics["bridge/critic_input_requires_grad"] = float(critic_o_actor.requires_grad)
            try:
                dqda = torch.autograd.grad(
                    q_target_pi.mean(), pi, retain_graph=True, allow_unused=True
                )[0]
            except RuntimeError:
                dqda = None
            if dqda is not None:
                actor_bridge_metrics["bridge/dqda_norm"] = dqda.detach().norm(dim=-1).mean().item()
                actor_bridge_metrics["bridge/dqda_abs_mean"] = dqda.detach().abs().mean().item()
            else:
                actor_bridge_metrics["bridge/dqda_norm"] = 0.0
                actor_bridge_metrics["bridge/dqda_abs_mean"] = 0.0

            entropy_cost = (alpha_t * logp_per_dim).sum(dim=-1)
            loss_pi = (entropy_cost.unsqueeze(dim=-1) - q_target_pi).mean()

            # === SBR: Steering-Only Boundary Repulsion (GRAC Component 3) ===
            # Adapted "inverting gradients" concept: mild L2 penalty on steering
            # to discourage gratuitous saturation that causes gradient death.
            # Applied to steering ONLY (index 0) — gas/brake are unpenalized.
            # Coefficient is ~200x smaller than reward signal to avoid sluggishness.
            sbr_lambda = cfg.TMRL_CONFIG.get("ALG", {}).get("SBR_LAMBDA", 0.005)
            steering_action = pi[:, 0]  # index 0 = steer
            loss_sbr = sbr_lambda * (steering_action ** 2).mean()
            loss_pi = loss_pi + loss_sbr

            # EWC: lighter penalty on actor (0.1x) to allow policy adaptation
            if self._ewc_active and self.ewc_lambda > 0:
                loss_pi = loss_pi + self.ewc_lambda * 0.1 * self._ewc_loss()

            # === Advantage-Weighted Deterministic Bridge (AWDB) ===
            # Replaces volatile DPG mathematically with a normalized supervised teacher bridge.
            # Pull dynamically from self so tracking_offline.py (Auto-Annealer) can control it
            det_lambda = getattr(self, "det_reg_lambda", 0.05)
            loss_awdb = torch.zeros(1, device=self.device)
            if det_lambda > 0:
                # To prevent vanishing gradients on track corners (where tanh squashes the derivative to 0),
                # we must evaluate the MSE bridge mathematically BEFORE the tanh activation.
                
                # Extract the post-tanh squashed action (for Q-evaluation) and the pre-tanh 'mu' (for learning)
                pi_det_squashed, _, _, mu_pre = self.model.actor_from_features(
                    fused_o_actor, film_o_actor, test=True, with_logprob=False, z=z_actor, return_pretanh_mu=True)
                
                qs_det = [q(critic_o_actor, pi_det_squashed, film_o_actor) for q in self.model.qs]
                min_q_det = torch.min(torch.stack(qs_det, -1), dim=1)[0]
                
                # Advantage: How much better is the pure stochastic action (pi) vs deterministic action (pi_det)?
                advantage = (q_target_pi - min_q_det).detach()
                
                # Only pull deterministic towards stochastic IF stochastic is actually better (adv > 0)
                advantage_mask = torch.relu(advantage)
                
                # Reverse the successful stochastic 'pi' backwards into its pre-tanh infinite-space counterpart.
                # We detach() it permanently so Atanh instability can never traverse back into the network gradients.
                act_limit = getattr(self.model.actor, "act_limit", 1.0)
                pi_scaled = (pi.detach() / act_limit).clamp(-0.99999, 0.99999)
                pi_pre_target = torch.atanh(pi_scaled)
                
                # Direct pre-tanh Supervised Bridge (Absolutely zero vanishing gradients!)
                mse_bridge = torch.nn.functional.mse_loss(mu_pre, pi_pre_target, reduction='none').mean(dim=-1, keepdim=True)
                
                # Normalize advantage to solve explosion gradients: map to [0, 1] range softly
                adv_weight = advantage_mask / (advantage_mask.max() + 1e-8)
                
                loss_awdb = (mse_bridge * adv_weight).mean()
                loss_pi = loss_pi + det_lambda * loss_awdb

            self.pi_optimizer.zero_grad()
            loss_pi.backward()

            # GRAC: Update gradient health tracker AFTER backward, BEFORE optimizer step
            if hasattr(self, 'grad_tracker') and self.model.actor.mu_layer.weight.grad is not None:
                self.grad_tracker.update(self.model.actor.mu_layer.weight.grad)

            alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
            guard_enabled = alg_cfg.get("ACTOR_STABILITY_GUARD_ENABLED", True)
            guard_dqda_min = float(alg_cfg.get("ACTOR_GUARD_DQDA_MIN_NORM", 1e-4))
            guard_health_min = float(alg_cfg.get("ACTOR_GUARD_GRAD_HEALTH_MIN", 0.01))
            if hasattr(self, 'grad_tracker'):
                grad_health_mean = self.grad_tracker.grad_health.detach().mean().item()
            else:
                grad_health_mean = 1.0
            dqda_norm_value = actor_bridge_metrics.get("bridge/dqda_norm", 0.0)
            actor_stability_guard_active = (
                guard_enabled
                and dqda_norm_value < guard_dqda_min
                and grad_health_mean < guard_health_min
            )
            actor_bridge_metrics["guard/actor_stability_active"] = float(actor_stability_guard_active)
            actor_bridge_metrics["guard/actor_mu_step_blocked"] = float(actor_stability_guard_active)
            actor_bridge_metrics["guard/actor_std_step_allowed"] = 1.0
            actor_bridge_metrics["guard/dqda_min_norm"] = guard_dqda_min
            actor_bridge_metrics["guard/grad_health_min"] = guard_health_min
            actor_bridge_metrics["guard/grad_health_mean"] = grad_health_mean
            if actor_stability_guard_active:
                _clear_parameter_grads(
                    list(self.model.actor.net.parameters()) +
                    list(self.model.actor.mu_layer.parameters())
                )

            # Per-module clipping for actor: std_net gets tighter clip
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.actor.net.parameters()) +
                list(self.model.actor.mu_layer.parameters()), 1.0)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.actor.std_net.parameters()), 0.5)
            self.pi_optimizer.step()

            for q in self.model.qs:
                q.requires_grad_(True)

            # Entropy coefficient update
            if self.learn_entropy_coef:
                with torch.no_grad():
                    if uses_context and ctx is not None:
                        fused_alpha, _, film_alpha, z_alpha = self.model.forward_features(o, ctx)
                    else:
                        fused_alpha, _, film_alpha, z_alpha = self.model.forward_features(o)
                _, _, logp_per_dim_alpha = self.model.actor_from_features(fused_alpha, film_alpha, z=z_alpha)
                # Apply stall recovery entropy bump dynamically
                stall_bump = getattr(self, "stall_entropy_bump", 0.0)
                # GRAC: Apply GHAE entropy boost for dimensions with vanishing gradients
                ghae_boost = torch.zeros_like(self.target_entropy)
                if hasattr(self, 'grad_tracker'):
                    ghae_boost = self.grad_tracker.get_entropy_boost()
                # SAC target entropy is negative in this codepath. A positive
                # exploration boost must therefore make the target more negative.
                effective_target_entropy = self.target_entropy - stall_bump - ghae_boost
                effective_target_entropy_diag = effective_target_entropy.detach()
                
                loss_alpha = -(self.log_alpha * (logp_per_dim_alpha.detach().mean(dim=0) + effective_target_entropy)).sum()
                self.alpha_optimizer.zero_grad()
                loss_alpha.backward()
                torch.nn.utils.clip_grad_norm_(self.log_alpha, 1.0)
                self.alpha_optimizer.step()
                alpha_t, alpha_floor_t = self._apply_alpha_floor()

            self.loss_pi = loss_pi.detach()

        # === Polyak averaging ===
        if update_policy:
            with torch.no_grad():
                for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        # Diagnostics: expose alpha and log_std trends to catch entropy collapse early.
        with torch.no_grad():
            if uses_context and ctx is not None:
                fused_diag, critic_diag, film_diag, z_diag = self.model.forward_features(o, ctx)
            else:
                fused_diag, critic_diag, film_diag, z_diag = self.model.forward_features(o)
            if z_diag is None:
                z_diag = torch.zeros(fused_diag.shape[0], 64, device=fused_diag.device)
            actor_input_diag = torch.cat([fused_diag, z_diag], dim=-1)
            log_std_raw_diag = self.model.actor.std_net(actor_input_diag)
            log_std_diag = core._compute_log_std_smooth(log_std_raw_diag)

        ret_dict = dict(
            loss_actor=self.loss_pi.detach().item(),
            loss_critic=self.loss_q.detach().item(),
        )
        if update_policy and 'loss_awdb' in dir():
            ret_dict["loss_awdb"] = loss_awdb.detach().item() if isinstance(loss_awdb, torch.Tensor) else 0.0
        if update_policy and 'loss_sbr' in dir():
            ret_dict["loss_sbr"] = loss_sbr.detach().item() if isinstance(loss_sbr, torch.Tensor) else 0.0
        if self._ewc_active:
            ret_dict["ewc_loss"] = loss_ewc.detach().item() if 'loss_ewc' in dir() else 0.0
        if uses_context:
            ret_dict["kl_div_loss"] = loss_kl.detach().item() if isinstance(loss_kl, torch.Tensor) else 0.0
        ret_dict["debug_alpha_steer"] = alpha_t[0].item()
        ret_dict["debug_alpha_gas"] = alpha_t[1].item()
        ret_dict["debug_alpha_brake"] = alpha_t[2].item()
        ret_dict["debug_alpha_floor_steer"] = alpha_floor_t[0].item()
        ret_dict["debug_alpha_floor_gas"] = alpha_floor_t[1].item()
        ret_dict["debug_alpha_floor_brake"] = alpha_floor_t[2].item()
        with torch.no_grad():
            q_real = q_prediction_cat.detach()
            ret_dict["bridge/q_real_mean"] = q_real.mean().item()
            ret_dict["bridge/q_real_std"] = q_real.std(unbiased=False).item()
        ret_dict.update(actor_bridge_metrics)

        # Expose gradient movements for live tracking
        if 'q_grad_norm' in locals() and isinstance(q_grad_norm, torch.Tensor):
            ret_dict["grad_norm_critic"] = q_grad_norm.detach().item()
        if 'actor_grad_norm' in locals() and isinstance(actor_grad_norm, torch.Tensor):
            ret_dict["grad_norm_actor"] = actor_grad_norm.detach().item()

        ret_dict["debug_log_std_mean"] = log_std_diag.detach().mean().item()
        ret_dict["debug_log_std_min"] = log_std_diag.detach().min().item()

        # GRAC diagnostics: gradient health and GHAE boost per dimension
        if hasattr(self, 'grad_tracker'):
            ret_dict["grad_health_steer"] = self.grad_tracker.grad_health[0].item()
            ret_dict["grad_health_gas"] = self.grad_tracker.grad_health[1].item()
            ret_dict["grad_health_brake"] = self.grad_tracker.grad_health[2].item()
            ghae = self.grad_tracker.get_entropy_boost()
            ret_dict["ghae_boost_steer"] = ghae[0].item()
            ret_dict["ghae_boost_gas"] = ghae[1].item()
            ret_dict["ghae_boost_brake"] = ghae[2].item()
        if effective_target_entropy_diag is not None:
            ret_dict["debug_target_entropy_steer"] = effective_target_entropy_diag[0].item()
            ret_dict["debug_target_entropy_gas"] = effective_target_entropy_diag[1].item()
            ret_dict["debug_target_entropy_brake"] = effective_target_entropy_diag[2].item()

        if self.learn_entropy_coef and loss_alpha is not None:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.mean().item()

        # ── World Model training (RSSM) ─────────────────────────────────
        if self.wm_enabled:
            # Use z_critic (always computed) as context for the world model
            # z_actor is only available during policy updates, but WM trains every step
            wm_context_z = z_critic.detach() if z_critic is not None else None
            wm_metrics = self._train_dynamics(o, a, r, o2, context_z=wm_context_z)
            ret_dict.update(wm_metrics)
            ret_dict["dynamics_loss"] = wm_metrics.get("wm_total_loss", 0.0)
            ret_dict["wm_train_steps"] = self.wm_train_steps
            ret_dict["wm_kl"] = wm_metrics.get("wm_kl", 0.0)
            ret_dict["wm_recon_state"] = wm_metrics.get("wm_recon_state", 0.0)

            # ── Train ImaginationActor to mimic real policy ─────────────
            critic_state_for_imag = self._extract_critic_state(o).detach()
            real_action = a.detach()
            pred_action = self.imag_actor(critic_state_for_imag)
            imag_actor_loss = F.mse_loss(pred_action, real_action)
            self.imag_actor_optimizer.zero_grad()
            imag_actor_loss.backward()
            self.imag_actor_optimizer.step()
            ret_dict["imag_actor_loss"] = imag_actor_loss.item()

            if self.wm_train_steps > self.wm_warmup:
                imag_metrics = self._imagined_critic_update(
                    o, ctx=ctx if uses_context else None, context_z=wm_context_z, real_action=a.detach()
                )
                ret_dict.update(imag_metrics)
                if "bridge/q_imag_mean" in imag_metrics and "bridge/q_real_mean" in ret_dict:
                    ret_dict["bridge/q_imag_minus_real"] = (
                        imag_metrics["bridge/q_imag_mean"] - ret_dict["bridge/q_real_mean"]
                    )
                self._last_verifier_trust = imag_metrics.get("verifier_trust_mt_mean", 1.0)

                # Log curiosity bonus stats (uses normalized surprise)
                state_cur = self._extract_critic_state(o).detach()
                state_nxt = self._extract_critic_state(o2).detach()
                with torch.no_grad():
                    surprise = self.dynamics.compute_surprise(state_cur, a, state_nxt, context_z=wm_context_z)
                surprise_norm = self.curiosity_rms.normalize(surprise)
                clip_val = getattr(self, 'curiosity_reward_clip', 5.0)
                surprise_norm_clipped = surprise_norm.clamp(-clip_val, clip_val)
                ret_dict["curiosity_surprise_raw_mean"] = surprise.mean().item()
                self._last_surprise_mean = surprise.mean().item()  # track for diagnostics
                ret_dict["curiosity_surprise_norm_mean"] = surprise_norm_clipped.mean().item()
                ret_dict["curiosity_bonus_mean"] = (self.curiosity_scale * surprise_norm_clipped).mean().item()
                ret_dict["curiosity_rms_mean"] = self.curiosity_rms.mean
                ret_dict["curiosity_rms_std"] = self.curiosity_rms.var ** 0.5
                # Log alpha floors (decoupled from curiosity)
                floor_mults = torch.tensor([1.5, 0.8, 0.5])
                trust_mod = 1.0 + (0.5 - getattr(self, '_last_verifier_trust', 1.0))
                trust_mod = max(0.2, min(2.0, trust_mod))
                # Removed dynamic_alpha_floor from logs as the metric is now organic alpha.

        return ret_dict
