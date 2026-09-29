"""Differentiable reverse cycle and reliability-aware per-example losses.

Copy this file into the archived CycleDiffusion project root.  It deliberately
does not alter inference methods: the first A->B pass stays under no_grad, while
the B->A pass calls the same decoder estimator through an unrolled,
differentiable sampler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.utils import fix_len_compatibility, sequence_mask


@dataclass(frozen=True)
class ReliabilityConfig:
    variant: str = "joint"
    content_temperature: float = 0.50
    acoustic_temperature: float = 1.00
    minimum_weight: float = 0.10
    hard_gate_threshold: float = 0.50


def reverse_diffusion_with_grad(decoder, z, mask, mean, ref, ref_mask,
                                mean_ref, c, n_timesteps, mode):
    """Equivalent to archived Diffusion.reverse_diffusion, without no_grad."""
    if mode not in {"pf", "em", "ml"}:
        raise ValueError("mode must be one of: pf, em, ml")
    h = 1.0 / n_timesteps
    xt = z * mask
    for index in range(n_timesteps):
        t = 1.0 - index * h
        time = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
        beta_t = decoder.get_beta(t)
        xt_ref = torch.stack(
            [decoder.compute_diffused_mean(ref, ref_mask, mean_ref, t)], dim=1
        )
        if mode == "pf":
            score = decoder.estimator(xt, mask, mean, xt_ref, ref_mask, c, time)
            dxt = 0.5 * (mean - xt - score) * (beta_t * h)
        else:
            if mode == "ml":
                kappa = decoder.get_gamma(0, t - h)
                kappa *= 1.0 - decoder.get_gamma(t - h, t, p=2.0)
                kappa /= decoder.get_gamma(0, t) * beta_t * h
                kappa -= 1.0
                omega = decoder.get_nu(t - h, t) / decoder.get_gamma(0, t)
                omega += decoder.get_mu(t - h, t)
                omega -= 0.5 * beta_t * h + 1.0
                sigma = decoder.get_sigma(t - h, t)
            else:
                kappa, omega = 0.0, 0.0
                sigma = math.sqrt(beta_t * h)
            score = decoder.estimator(xt, mask, mean, xt_ref, ref_mask, c, time)
            dxt = (mean - xt) * (0.5 * beta_t * h + omega)
            dxt -= score * (1.0 + kappa) * (beta_t * h)
            dxt += torch.randn_like(z, requires_grad=False) * sigma
        xt = (xt - dxt) * mask
    return xt


def differentiable_convert(model, x, x_lengths, x_ref, x_ref_lengths, c,
                           n_timesteps=6, mode="ml"):
    """Run conversion with gradients only through decoder sampling operations."""
    x, x_lengths = model.relocate_input([x, x_lengths])
    x_ref, x_ref_lengths, c = model.relocate_input([x_ref, x_ref_lengths, c])
    # The collate function right-pads every mel to ``train_frames``.  A batch
    # can nevertheless contain only shorter utterances, so deriving the mask
    # width from ``max(lengths)`` makes it narrower than the padded tensor.
    x_mask = sequence_mask(x_lengths, x.shape[-1]).unsqueeze(1).to(x.dtype)
    ref_mask = sequence_mask(
        x_ref_lengths, x_ref.shape[-1]
    ).unsqueeze(1).to(x_ref.dtype)
    with torch.no_grad():
        mean = model.encoder(x, x_mask)
        mean_x = model.decoder.compute_diffused_mean(x, x_mask, mean, 1.0)
        mean_ref = model.encoder(x_ref, ref_mask)

    batch = x.shape[0]
    # Keep the differentiable reconstruction aligned with the collated source
    # width even when every utterance in a batch is shorter than that width.
    source_width = x.shape[-1]
    padded_length = fix_len_compatibility(source_width)
    padded_mask = sequence_mask(x_lengths, padded_length).unsqueeze(1).to(x.dtype)
    padded_mean = x.new_zeros((batch, model.n_feats, padded_length))
    padded_z = x.new_zeros((batch, model.n_feats, padded_length))
    for item in range(batch):
        length = int(x_lengths[item])
        padded_mean[item, :, :length] = mean[item, :, :length]
        padded_z[item, :, :length] = mean_x[item, :, :length]
    padded_z = padded_z + torch.randn_like(padded_z, requires_grad=False)
    decoded = reverse_diffusion_with_grad(
        model.decoder, padded_z, padded_mask, padded_mean, x_ref, ref_mask,
        mean_ref, c, n_timesteps, mode,
    )
    return mean_x, decoded[:, :, :source_width]


def _masked_mean(values, lengths):
    mask = sequence_mask(lengths, values.shape[-1]).unsqueeze(1).to(values.dtype)
    denom = mask.sum(dim=(1, 2)).clamp_min(1.0) * values.shape[1]
    return (values * mask).sum(dim=(1, 2)) / denom


@torch.no_grad()
def reliability_components(model, source, lengths, pseudo, target_reference,
                           target_reference_lengths, target_ref_scores,
                           config):
    """Return detached speaker/content/acoustic reliability in [0, 1]."""
    mask = sequence_mask(lengths, source.shape[-1]).unsqueeze(1).to(source.dtype)
    source_content = model.encoder(source, mask)
    pseudo_content = model.encoder(pseudo, mask)
    content_error = _masked_mean((pseudo_content - source_content).abs(), lengths)
    content = torch.exp(-content_error / config.content_temperature)

    ref_mask = sequence_mask(
        target_reference_lengths, target_reference.shape[-1]
    ).unsqueeze(1).to(target_reference.dtype)
    source_mean = _masked_mean(source, lengths)
    source_var = _masked_mean((source - source_mean[:, None, None]) ** 2, lengths)
    ref_mean = _masked_mean(target_reference, target_reference_lengths)
    ref_var = _masked_mean(
        (target_reference - ref_mean[:, None, None]) ** 2,
        target_reference_lengths,
    )
    center = 0.5 * (source_mean + ref_mean)
    scale = torch.sqrt(0.5 * (source_var + ref_var)).clamp_min(1e-3)
    z = (pseudo - center[:, None, None]).abs() / scale[:, None, None]
    acoustic_excess = _masked_mean(F.relu(z - 4.0), lengths)
    finite = torch.isfinite(pseudo).all(dim=(1, 2)).to(pseudo.dtype)
    acoustic = finite * torch.exp(-acoustic_excess / config.acoustic_temperature)
    speaker = target_ref_scores.to(source.dtype).clamp(0.0, 1.0)
    return {"speaker": speaker, "content": content, "acoustic": acoustic}


def combine_reliability(parts, config):
    if config.variant == "uniform":
        weight = torch.ones_like(parts["content"])
    elif config.variant == "speaker_only":
        weight = parts["speaker"]
    elif config.variant == "content_only":
        weight = parts["content"]
    elif config.variant in {"joint", "hard_gate"}:
        weight = (parts["speaker"] * parts["content"] * parts["acoustic"]).pow(1 / 3)
        if config.variant == "hard_gate":
            weight = (weight >= config.hard_gate_threshold).to(weight.dtype)
    else:
        raise ValueError(f"unknown reliability variant: {config.variant}")
    if config.variant != "hard_gate":
        weight = weight.clamp(config.minimum_weight, 1.0)
    return weight.detach()


def weighted_cycle_l1(reconstruction, source, lengths, weights):
    per_example = _masked_mean((reconstruction - source).abs(), lengths)
    return (weights * per_example).sum() / weights.sum().clamp_min(1e-6), per_example
