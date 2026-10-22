import importlib.util
import pathlib
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


additive = load_module(
    "diffusion_kd_additive", "pointcept/models/diffusion_kd/additive.py"
)
denoiser = load_module(
    "diffusion_kd_denoiser", "pointcept/models/diffusion_kd/denoiser.py"
)


class AdditiveDiffusionScheduleTest(unittest.TestCase):
    def setUp(self):
        self.schedule = additive.AdditiveDiffusionSchedule(
            num_timesteps=20, gamma_max=1.0
        )

    def test_gamma_is_monotonic_and_clean_state_is_zero(self):
        self.assertEqual(self.schedule.gammas.shape[0], 21)
        self.assertEqual(self.schedule.gammas[0].item(), 0.0)
        self.assertTrue(
            torch.all(self.schedule.gammas[1:] >= self.schedule.gammas[:-1])
        )

    def test_forward_process_does_not_attenuate_signal(self):
        feature = torch.randn(5, 8)
        zero_noise = torch.zeros_like(feature)
        timestep = torch.tensor([0, 1, 5, 10, 20])
        noisy, _ = self.schedule.q_sample(feature, timestep, zero_noise)
        self.assertTrue(torch.equal(noisy, feature))

    def test_oracle_noise_recovers_clean_feature(self):
        clean = torch.randn(6, 4)
        noise = torch.randn_like(clean)
        timestep = torch.tensor([0, 1, 4, 8, 12, 20])
        noisy, _ = self.schedule.q_sample(clean, timestep, noise)
        recovered = self.schedule.reverse_step(
            noisy,
            timestep,
            torch.zeros_like(timestep),
            noise,
            deterministic=True,
        )
        self.assertTrue(torch.allclose(recovered, clean, atol=1e-6))

    def test_degradation_score_and_timestep_endpoints(self):
        teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        student = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        score = additive.pointwise_degradation_score(student, teacher)
        self.assertTrue(torch.allclose(score, torch.tensor([0.0, 1.0])))
        timestep = self.schedule.score_to_timestep(score)
        self.assertEqual(timestep.tolist(), [0, 20])


class PointwiseDenoiserTest(unittest.TestCase):
    def test_pointwise_time_and_sample_offsets(self):
        model = denoiser.PointwiseDiffusionDenoiser(
            feat_dim=4,
            time_dim=8,
            hidden_dim=8,
            num_heads=2,
            num_blocks=1,
            window_size=4,
            drop_path=0.0,
        ).eval()
        feature = torch.randn(7, 4, requires_grad=True)
        timestep = torch.arange(7, dtype=torch.float32)
        offset = torch.tensor([3, 7])

        batched = model(feature, timestep, offset)
        separate = torch.cat(
            [
                model(feature[:3], timestep[:3], torch.tensor([3])),
                model(feature[3:], timestep[3:], torch.tensor([4])),
            ],
            dim=0,
        )
        self.assertEqual(batched.shape, feature.shape)
        self.assertTrue(torch.allclose(batched, separate, atol=1e-6))
        batched.sum().backward()
        self.assertIsNotNone(feature.grad)

    def test_frozen_denoiser_still_backpropagates_to_student_feature(self):
        model = denoiser.PointwiseDiffusionDenoiser(
            feat_dim=4,
            time_dim=8,
            hidden_dim=8,
            num_heads=2,
            num_blocks=1,
            window_size=4,
            drop_path=0.0,
        )
        for parameter in model.parameters():
            parameter.requires_grad = False
        student_feature = torch.randn(5, 4, requires_grad=True)
        prediction = model(
            student_feature,
            torch.arange(5, dtype=torch.float32),
            torch.tensor([5]),
        )
        prediction.sum().backward()
        self.assertIsNotNone(student_feature.grad)


if __name__ == "__main__":
    unittest.main()
