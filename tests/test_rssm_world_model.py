"""
Tests for the RSSM Latent World Model + ImaginationActor + RunningMeanStd.

Verifies:
- RSSMEncoder produces correct latent dimensions
- RSSMPrior transition produces correct shapes
- RSSMPosterior refinement produces correct shapes
- RSSMDecoder reconstructs correct dimensions
- LatentWorldModel.train_step produces valid loss and metrics
- LatentWorldModel.imagine produces correct rollout shapes
- KL divergence is non-negative
- ImaginationActor outputs correct shape and bounded actions
- RunningMeanStd normalizes correctly
- Imagination with learned policy produces correct shapes
"""
import unittest
import torch
import torch.nn as nn

from tmrl.custom.custom_models import (
    EGO_WM_STATE_DIM,
    RSSMEncoder,
    RSSMPrior,
    RSSMPosterior,
    RSSMDecoder,
    LatentWorldModel,
    ImaginationActor,
    RunningMeanStd,
)


BATCH = 8
STATE_DIM = EGO_WM_STATE_DIM
ACTION_DIM = 3
LATENT_DIM = 32
GRU_DIM = 128
HIDDEN_DIM = 256


class TestRSSMEncoder(unittest.TestCase):
    def setUp(self):
        self.enc = RSSMEncoder(STATE_DIM, LATENT_DIM, HIDDEN_DIM)
        self.enc.train()

    def test_output_shapes(self):
        state = torch.randn(BATCH, STATE_DIM)
        mu, log_var, z = self.enc(state)
        self.assertEqual(mu.shape, (BATCH, LATENT_DIM))
        self.assertEqual(log_var.shape, (BATCH, LATENT_DIM))
        self.assertEqual(z.shape, (BATCH, LATENT_DIM))

    def test_eval_deterministic(self):
        self.enc.eval()
        state = torch.randn(BATCH, STATE_DIM)
        mu, _, z = self.enc(state)
        # In eval mode, z should equal mu (no sampling)
        self.assertTrue(torch.allclose(z, mu))


class TestRSSMPrior(unittest.TestCase):
    def setUp(self):
        self.prior = RSSMPrior(LATENT_DIM, ACTION_DIM, GRU_DIM, HIDDEN_DIM)

    def test_transition_shapes(self):
        h = self.prior.get_initial_hidden(BATCH, 'cpu')
        z = torch.randn(BATCH, LATENT_DIM)
        action = torch.randn(BATCH, ACTION_DIM)
        h_next, mu_prior, log_var_prior = self.prior(h, z, action)
        self.assertEqual(h_next.shape, (BATCH, GRU_DIM))
        self.assertEqual(mu_prior.shape, (BATCH, LATENT_DIM))
        self.assertEqual(log_var_prior.shape, (BATCH, LATENT_DIM))

    def test_initial_hidden(self):
        h = self.prior.get_initial_hidden(BATCH, 'cpu')
        self.assertEqual(h.shape, (BATCH, GRU_DIM))
        self.assertTrue(torch.all(h == 0))


class TestRSSMPosterior(unittest.TestCase):
    def setUp(self):
        self.post = RSSMPosterior(GRU_DIM, STATE_DIM, LATENT_DIM, HIDDEN_DIM)
        self.post.train()

    def test_output_shapes(self):
        h = torch.randn(BATCH, GRU_DIM)
        obs = torch.randn(BATCH, STATE_DIM)
        mu, log_var, z = self.post(h, obs)
        self.assertEqual(mu.shape, (BATCH, LATENT_DIM))
        self.assertEqual(log_var.shape, (BATCH, LATENT_DIM))
        self.assertEqual(z.shape, (BATCH, LATENT_DIM))


class TestRSSMDecoder(unittest.TestCase):
    def setUp(self):
        self.dec = RSSMDecoder(LATENT_DIM, GRU_DIM, STATE_DIM, HIDDEN_DIM)

    def test_decode_shapes(self):
        z = torch.randn(BATCH, LATENT_DIM)
        h = torch.randn(BATCH, GRU_DIM)
        state_hat, reward_hat = self.dec(z, h)
        self.assertEqual(state_hat.shape, (BATCH, STATE_DIM))
        # reward_hat is (num_heads, B, 1) from ensemble
        self.assertEqual(reward_hat.shape[1], BATCH)
        self.assertEqual(reward_hat.shape[2], 1)


class TestLatentWorldModel(unittest.TestCase):
    def setUp(self):
        self.wm = LatentWorldModel(
            state_dim=STATE_DIM, action_dim=ACTION_DIM,
            latent_dim=LATENT_DIM, gru_dim=GRU_DIM,
            hidden_dim=HIDDEN_DIM, kl_free_nats=1.0
        )
        self.wm.train()

    def test_train_step_loss(self):
        state = torch.randn(BATCH, STATE_DIM)
        action = torch.randn(BATCH, ACTION_DIM)
        next_state = torch.randn(BATCH, STATE_DIM)
        reward = torch.randn(BATCH, 1)

        loss, metrics = self.wm.train_step(state, action, next_state, reward)

        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreater(loss.item(), 0)
        self.assertIn("wm_recon_state", metrics)
        self.assertIn("wm_recon_reward", metrics)
        self.assertIn("wm_kl", metrics)
        self.assertIn("wm_dyn_loss", metrics)
        self.assertIn("wm_rep_loss", metrics)
        self.assertIn("wm_latent_probe_loss", metrics)
        self.assertIn("wm_decoder_latent_use_loss", metrics)
        self.assertIn("wm/post_prior_mu_abs_diff", metrics)
        self.assertIn("wm_val/recon_zero_z", metrics)
        self.assertIn("wm_val/recon_shuffle_z", metrics)
        self.assertIn("wm_val/post_advantage_z_ablate_min", metrics)
        self.assertIn("wm_val/post_advantage_ratio", metrics)
        self.assertIn("wm_total_loss", metrics)

    def test_train_step_gradients(self):
        """Verify gradients flow through all RSSM components."""
        state = torch.randn(BATCH, STATE_DIM)
        action = torch.randn(BATCH, ACTION_DIM)
        next_state = torch.randn(BATCH, STATE_DIM)
        reward = torch.randn(BATCH, 1)

        loss, _ = self.wm.train_step(state, action, next_state, reward)
        loss.backward()

        # Check all modules received gradients
        for name, param in self.wm.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")

    def test_kl_non_negative(self):
        """KL divergence should always be >= 0."""
        mu_post = torch.randn(BATCH, LATENT_DIM)
        lv_post = torch.randn(BATCH, LATENT_DIM)
        mu_prior = torch.randn(BATCH, LATENT_DIM)
        lv_prior = torch.randn(BATCH, LATENT_DIM)
        kl = LatentWorldModel._kl_divergence(mu_post, lv_post, mu_prior, lv_prior)
        self.assertGreaterEqual(kl.item(), 0.0)

    def test_imagine_shapes(self):
        """Verify imagination rollout produces correct tensor shapes."""
        state = torch.randn(BATCH, STATE_DIM)
        horizon = 10

        def dummy_policy(s):
            return torch.randn(s.shape[0], ACTION_DIM)

        imag_states, imag_rewards, imag_actions, imag_uncertainties = self.wm.imagine(
            state, dummy_policy, horizon
        )
        self.assertEqual(imag_states.shape, (horizon, BATCH, STATE_DIM))
        self.assertEqual(imag_rewards.shape, (horizon, BATCH, 1))
        self.assertEqual(imag_actions.shape, (horizon, BATCH, ACTION_DIM))
        self.assertEqual(imag_uncertainties.shape, (horizon, BATCH, 1))

    def test_imagine_different_horizons(self):
        """Imagination should work at various horizon lengths."""
        state = torch.randn(BATCH, STATE_DIM)
        def dummy_policy(s):
            return torch.randn(s.shape[0], ACTION_DIM)

        for h in [1, 5, 15, 30]:
            imag_s, imag_r, imag_a, imag_u = self.wm.imagine(state, dummy_policy, h)
            self.assertEqual(imag_s.shape[0], h)
            self.assertEqual(imag_u.shape[0], h)

    def test_roundtrip_encode_decode_dims(self):
        """Encode -> decode should preserve dimensionality (not exact values)."""
        state = torch.randn(BATCH, STATE_DIM)
        _, _, z = self.wm.encoder(state)
        h = self.wm.prior.get_initial_hidden(BATCH, 'cpu')
        state_hat, reward_hat = self.wm.decoder(z, h)
        self.assertEqual(state_hat.shape, (BATCH, STATE_DIM))

    def test_compute_surprise(self):
        """Surprise should be non-negative per-sample KL."""
        state = torch.randn(BATCH, STATE_DIM)
        action = torch.randn(BATCH, ACTION_DIM)
        next_state = torch.randn(BATCH, STATE_DIM)
        surprise = self.wm.compute_surprise(state, action, next_state)
        self.assertEqual(surprise.shape, (BATCH,))
        self.assertTrue(torch.all(surprise >= 0))


class TestImaginationActor(unittest.TestCase):
    def setUp(self):
        self.actor = ImaginationActor(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    def test_output_shape(self):
        """Output should be (B, action_dim)."""
        state = torch.randn(BATCH, STATE_DIM)
        action = self.actor(state)
        self.assertEqual(action.shape, (BATCH, ACTION_DIM))

    def test_output_bounded(self):
        """Actions should be bounded in [-1, 1] due to tanh."""
        state = torch.randn(BATCH, STATE_DIM) * 10  # large inputs
        action = self.actor(state)
        self.assertTrue(torch.all(action >= -1.0))
        self.assertTrue(torch.all(action <= 1.0))

    def test_gradients_flow(self):
        """Loss should produce gradients for all parameters."""
        state = torch.randn(BATCH, STATE_DIM)
        target = torch.randn(BATCH, ACTION_DIM).clamp(-1, 1)
        pred = self.actor(state)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")

    def test_imagine_with_learned_policy(self):
        """Verify imagination works with ImaginationActor as policy_fn."""
        wm = LatentWorldModel(
            state_dim=STATE_DIM, action_dim=ACTION_DIM,
            latent_dim=LATENT_DIM, gru_dim=GRU_DIM,
            hidden_dim=HIDDEN_DIM, kl_free_nats=1.0
        )
        noise_scale = 0.3

        def policy_fn(critic_state):
            with torch.no_grad():
                base = self.actor(critic_state)
            noise = torch.randn_like(base) * noise_scale
            return (base + noise).clamp(-1.0, 1.0)

        state = torch.randn(BATCH, STATE_DIM)
        imag_s, imag_r, imag_a, imag_u = wm.imagine(state, policy_fn, horizon=5)
        self.assertEqual(imag_s.shape, (5, BATCH, STATE_DIM))
        self.assertEqual(imag_a.shape, (5, BATCH, ACTION_DIM))
        # Actions should be bounded
        self.assertTrue(torch.all(imag_a >= -1.0))
        self.assertTrue(torch.all(imag_a <= 1.0))


class TestRunningMeanStd(unittest.TestCase):
    def test_initial_state(self):
        rms = RunningMeanStd()
        self.assertAlmostEqual(rms.mean, 0.0)
        self.assertAlmostEqual(rms.var, 1.0)

    def test_normalize_with_defaults(self):
        """With default var=1.0, normalize should be approximately identity."""
        rms = RunningMeanStd()
        x = torch.tensor([1.0, 2.0, 3.0])
        normed = rms.normalize(x)
        self.assertTrue(torch.allclose(normed, x, atol=1e-6))

    def test_update_shifts_mean(self):
        """After feeding constant values, mean should converge."""
        rms = RunningMeanStd()
        for _ in range(100):
            rms.update(torch.full((32,), 5.0))
        self.assertAlmostEqual(rms.mean, 5.0, places=1)

    def test_normalize_scales_correctly(self):
        """After learning stats, normalize should produce ~unit variance output."""
        rms = RunningMeanStd()
        # Feed many samples with known std
        for _ in range(200):
            rms.update(torch.randn(64) * 10.0)  # std=10
        x = torch.randn(32) * 10.0
        normed = rms.normalize(x)
        # Normalized values should be roughly 10x smaller than input
        self.assertLess(normed.abs().mean().item(), x.abs().mean().item())

    def test_pickle_compatible(self):
        """RunningMeanStd should survive pickle round-trip (checkpoint compat)."""
        import pickle
        rms = RunningMeanStd()
        rms.update(torch.randn(100))
        data = pickle.dumps(rms)
        rms2 = pickle.loads(data)
        self.assertAlmostEqual(rms.mean, rms2.mean)
        self.assertAlmostEqual(rms.var, rms2.var)
        self.assertAlmostEqual(rms.count, rms2.count)

class TestRSSMDecoderBootstrap(unittest.TestCase):
    """Test bootstrap masking prevents reward head collapse."""

    def test_bootstrap_masks_shape_and_ratio(self):
        """Bootstrap masks should have correct shape and ~80% coverage."""
        dec = RSSMDecoder(LATENT_DIM, GRU_DIM, STATE_DIM, HIDDEN_DIM)
        masks = dec.generate_bootstrap_masks(BATCH, 'cpu')
        self.assertEqual(masks.shape, (dec.num_reward_heads, BATCH))
        # Each head should see ~80% of the batch
        for i in range(dec.num_reward_heads):
            coverage = masks[i].sum().item() / BATCH
            self.assertGreaterEqual(coverage, 0.5)
            self.assertLessEqual(coverage, 1.0)

    def test_bootstrap_masks_differ_across_heads(self):
        """Different heads should see different subsets."""
        dec = RSSMDecoder(LATENT_DIM, GRU_DIM, STATE_DIM, HIDDEN_DIM)
        masks = dec.generate_bootstrap_masks(64, 'cpu')  # larger batch for statistical significance
        # At least some masks should differ
        all_same = True
        for i in range(1, dec.num_reward_heads):
            if not torch.equal(masks[0], masks[i]):
                all_same = False
                break
        self.assertFalse(all_same, "All bootstrap masks are identical — no diversity")

    def test_reward_diversity_after_training(self):
        """After training with bootstrap, reward heads should produce diverse outputs."""
        wm = LatentWorldModel(
            state_dim=STATE_DIM, action_dim=ACTION_DIM,
            latent_dim=LATENT_DIM, gru_dim=GRU_DIM,
            hidden_dim=HIDDEN_DIM, kl_free_nats=1.0
        )
        optimizer = torch.optim.Adam(wm.parameters(), lr=1e-3)

        # Train for 20 steps to let heads diverge
        for _ in range(20):
            state = torch.randn(32, STATE_DIM)
            action = torch.randn(32, ACTION_DIM)
            next_state = torch.randn(32, STATE_DIM)
            reward = torch.randn(32, 1)
            loss, _ = wm.train_step(state, action, next_state, reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check eval-mode diversity
        wm.eval()
        with torch.no_grad():
            z = torch.randn(32, LATENT_DIM)
            h = torch.randn(32, GRU_DIM)
            _, reward_hats = wm.decoder(z, h)
            # reward_hats: (num_heads, 32, 1)
            variance = reward_hats.var(dim=0).mean().item()
            self.assertGreater(variance, 1e-4,
                f"Reward head variance too low ({variance:.2e}), ensemble may have collapsed")

    def test_distinct_init_seeds_diversity(self):
        """Fresh decoder should have diverse reward head outputs from distinct init."""
        dec = RSSMDecoder(LATENT_DIM, GRU_DIM, STATE_DIM, HIDDEN_DIM)
        dec.eval()
        z = torch.randn(16, LATENT_DIM)
        h = torch.randn(16, GRU_DIM)
        with torch.no_grad():
            _, reward_hats = dec(z, h)
            variance = reward_hats.var(dim=0).mean().item()
            self.assertGreater(variance, 1e-5,
                f"Initial reward head diversity too low ({variance:.2e})")


if __name__ == "__main__":
    unittest.main()
