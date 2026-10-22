from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pointwise_degradation_score(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a bounded point-wise feature discrepancy in [0, 1].

    The score is a distillation-time degradation proxy, not an estimate of a
    physical Gaussian variance. Inputs are detached so the student cannot
    reduce the objective by manipulating its assigned diffusion timestep.
    """
    if student_feat.shape != teacher_feat.shape:
        raise ValueError(
            "Student and teacher features must have the same shape, got "
            f"{tuple(student_feat.shape)} and {tuple(teacher_feat.shape)}."
        )
    if student_feat.dim() != 2:
        raise ValueError("Expected flattened point features with shape [N, C].")

    student_unit = F.normalize(student_feat.detach().float(), dim=-1, eps=eps)
    teacher_unit = F.normalize(teacher_feat.detach().float(), dim=-1, eps=eps)
    # Squared distance between unit vectors is in [0, 4].
    return ((student_unit - teacher_unit) ** 2).sum(dim=-1).div(4.0).clamp_(0.0, 1.0)


class AdditiveDiffusionSchedule(nn.Module):
    """Noise schedule for a signal-preserving additive diffusion process.

    The forward marginal is x_t = x_0 + sqrt(gamma_t) * epsilon.  Index zero
    represents the clean state and therefore gamma[0] is exactly zero.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        gamma_max: float = 1.0,
        cosine_s: float = 0.008,
    ) -> None:
        super().__init__()
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be positive.")
        if gamma_max <= 0:
            raise ValueError("gamma_max must be positive.")

        self.num_timesteps = int(num_timesteps)
        self.beta_schedule = str(beta_schedule)
        self.gamma_max = float(gamma_max)

        if beta_schedule == "linear":
            betas = torch.linspace(
                float(beta_start), float(beta_end), self.num_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "cosine":
            steps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1)
            alpha_bar = torch.cos(
                ((steps / self.num_timesteps + cosine_s) / (1.0 + cosine_s))
                * math.pi
                / 2.0
            ).pow(2)
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
            betas = betas.clamp(1e-8, 0.999)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alpha_bars = torch.cat(
            [torch.ones(1, dtype=torch.float32), torch.cumprod(alphas, dim=0)], dim=0
        )
        gammas = self.gamma_max * (1.0 - alpha_bars)
        gammas[0] = 0.0

        self.register_buffer("betas", betas, persistent=True)
        self.register_buffer("alpha_bars", alpha_bars, persistent=True)
        self.register_buffer("gammas", gammas, persistent=True)

    def _prepare_timesteps(self, t: torch.Tensor, num_points: int) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=self.gammas.device)
        t = t.to(device=self.gammas.device, dtype=torch.long).reshape(-1)
        if t.numel() == 1 and num_points != 1:
            t = t.expand(num_points)
        if t.numel() != num_points:
            raise ValueError(
                f"Expected one timestep or {num_points} point-wise timesteps, got {t.numel()}."
            )
        return t.clamp(0, self.num_timesteps)

    def gamma_at(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        t = self._prepare_timesteps(t, like.shape[0])
        gamma = self.gammas.index_select(0, t).to(dtype=like.dtype)
        return gamma.view(gamma.shape[0], *([1] * (like.dim() - 1)))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        if noise.shape != x0.shape:
            raise ValueError("noise and x0 must have the same shape.")
        gamma_t = self.gamma_at(t, x0)
        return x0 + torch.sqrt(gamma_t) * noise, noise

    def predict_x0(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        return xt - torch.sqrt(self.gamma_at(t, xt)) * predicted_noise

    @torch.no_grad()
    def score_to_timestep(self, score: torch.Tensor) -> torch.Tensor:
        """Map a bounded degradation score to the nearest gamma timestep."""
        score = score.detach().float().clamp(0.0, 1.0)
        target = score * self.gammas[-1]
        upper = torch.searchsorted(self.gammas, target).clamp(0, self.num_timesteps)
        lower = (upper - 1).clamp(0, self.num_timesteps)
        upper_distance = (self.gammas[upper] - target).abs()
        lower_distance = (target - self.gammas[lower]).abs()
        return torch.where(lower_distance <= upper_distance, lower, upper).long()

    def reverse_step(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        predicted_noise: torch.Tensor,
        deterministic: bool = True,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Move from arbitrary point-wise timesteps t to smaller timesteps."""
        t = self._prepare_timesteps(t, xt.shape[0])
        t_next = self._prepare_timesteps(t_next, xt.shape[0])
        if torch.any(t_next > t):
            raise ValueError("Reverse diffusion requires t_next <= t for every point.")

        gamma_t = self.gamma_at(t, xt)
        gamma_next = self.gamma_at(t_next, xt)
        active = gamma_t > 0
        safe_gamma_t = gamma_t.clamp_min(torch.finfo(xt.dtype).eps)
        x0_hat = xt - torch.sqrt(gamma_t) * predicted_noise
        ratio = gamma_next / safe_gamma_t
        mean = ratio * xt + (1.0 - ratio) * x0_hat

        if not deterministic:
            if noise is None:
                noise = torch.randn_like(xt)
            variance = gamma_next * (gamma_t - gamma_next).clamp_min(0.0) / safe_gamma_t
            mean = mean + torch.sqrt(variance.clamp_min(0.0)) * noise
        return torch.where(active, mean, xt)

    def reverse_time_pairs(
        self, start_t: torch.Tensor, num_steps: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build an accelerated, point-wise linear path from start_t to zero."""
        if num_steps < 1:
            raise ValueError("num_steps must be positive.")
        start_t = start_t.long().clamp(0, self.num_timesteps)
        path = [
            torch.round(start_t.float() * (1.0 - step / num_steps)).long()
            for step in range(num_steps + 1)
        ]
        path[-1] = torch.zeros_like(start_t)
        return list(zip(path[:-1], path[1:]))
