"""Explicit experimental timestep policies shared by training and diagnostics."""
import torch


@torch.no_grad()
def assign_timesteps(schedule, score, teacher_logits, offset, policy='discrepancy',
                     fixed_timestep=100, generator=None):
    score = score.detach().float()
    if policy == 'fixed':
        if not 0 <= int(fixed_timestep) <= schedule.num_timesteps:
            raise ValueError('fixed_timestep must be inside the diffusion schedule.')
        return torch.full_like(score, int(fixed_timestep), dtype=torch.long)
    if policy == 'discrepancy':
        return schedule.score_to_timestep(score)
    if policy not in ('confidence', 'shuffle'):
        raise ValueError('Unknown timestep_policy: ' + str(policy))
    if teacher_logits is None or teacher_logits.shape[0] != score.numel():
        raise ValueError('Confidence policies require corresponding teacher logits.')
    classes = teacher_logits.shape[-1]
    if classes < 2:
        raise ValueError('At least two classes are required.')
    confidence = teacher_logits.detach().float().softmax(-1).amax(-1)
    reliability = ((confidence - 1/classes) / (1-1/classes)).clamp(0, 1)
    times = schedule.score_to_timestep(score * reliability)
    if policy == 'shuffle':
        ends = [len(times)] if offset is None else offset.detach().cpu().tolist()
        start = 0
        result = times.clone()
        for end in ends:
            end = int(end)
            if end <= start or end > len(times):
                raise ValueError('Invalid sample offsets.')
            permutation = torch.randperm(end-start, device=times.device, generator=generator)
            result[start:end] = times[start:end][permutation]
            start = end
        if start != len(times):
            raise ValueError('Offsets do not cover all points.')
        return result
    return times
