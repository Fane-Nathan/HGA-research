import unittest

import numpy as np
import torch
from gymnasium.spaces import Box

from tmrl.custom.custom_models import (
    CONTEXT_Z_DIM,
    EGO_CRITIC_FLOAT_DIM,
    ContextualDroQHybridActorCritic,
    DroQHybridActorCritic,
    SharedBackboneHybridActorCritic,
)


def _make_spaces():
    observation_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return observation_space, action_space


def _make_obs(batch_size=2):
    return (
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 4, 64, 64),
        torch.zeros(batch_size, 19),
        torch.zeros(batch_size, 3),
        torch.zeros(batch_size, 3),
    )


def _make_contextual_obs(batch_size=2, xyz_value=0.0, progress_value=0.0):
    return (
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 4, 64, 64),
        torch.zeros(batch_size, 19),
        torch.full((batch_size, 3), xyz_value),
        torch.full((batch_size, 1), progress_value),
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 3),
        torch.zeros(batch_size, 3),
    )


class TestNonContextInterfaceContract(unittest.TestCase):
    def _assert_contract(self, model):
        obs = _make_obs()
        features, film, z = model.forward_features(obs)
        self.assertIsNone(film)
        self.assertIsNone(z)
        self.assertEqual(features.ndim, 2)

        action, logp_total, logp_per_dim = model.actor_from_features(
            features, film, with_logprob=True
        )
        self.assertEqual(action.shape[-1], 3)
        self.assertEqual(logp_total.ndim, 1)
        self.assertEqual(logp_per_dim.shape[-1], 3)

    def test_shared_backbone_hybrid_actor_critic_contract(self):
        observation_space, action_space = _make_spaces()
        model = SharedBackboneHybridActorCritic(observation_space, action_space, n=2)
        self._assert_contract(model)

    def test_droq_hybrid_actor_critic_contract(self):
        observation_space, action_space = _make_spaces()
        model = DroQHybridActorCritic(observation_space, action_space, dropout_rate=0.01)
        self._assert_contract(model)


class TestContextualPrivilegedStateScrub(unittest.TestCase):
    def test_critic_features_ignore_xyz_and_absolute_progress(self):
        observation_space, action_space = _make_spaces()
        model = ContextualDroQHybridActorCritic(observation_space, action_space, dropout_rate=0.01)
        model.eval()

        obs_a = _make_contextual_obs(xyz_value=0.0, progress_value=0.0)
        obs_b = _make_contextual_obs(xyz_value=999.0, progress_value=1.0)

        with torch.no_grad():
            _, critic_a, _, _ = model.forward_features(obs_a)
            _, critic_b, _, _ = model.forward_features(obs_b)

        self.assertEqual(critic_a.shape[-1], EGO_CRITIC_FLOAT_DIM + CONTEXT_Z_DIM)
        self.assertTrue(torch.allclose(critic_a, critic_b))

    def test_contextual_q_heads_accept_scrubbed_critic_features(self):
        observation_space, action_space = _make_spaces()
        model = ContextualDroQHybridActorCritic(observation_space, action_space, dropout_rate=0.01)
        obs = _make_contextual_obs()

        critic_batch = obs[0].shape[0]
        action = torch.zeros(critic_batch, action_space.shape[0])
        _, critic_features, film, _ = model.forward_features(obs)
        q_values = model.q_from_features(critic_features, action, film)

        self.assertEqual(critic_features.shape[-1], EGO_CRITIC_FLOAT_DIM + CONTEXT_Z_DIM)
        self.assertEqual(len(q_values), 2)
        for q in q_values:
            self.assertEqual(q.shape, (critic_batch,))
