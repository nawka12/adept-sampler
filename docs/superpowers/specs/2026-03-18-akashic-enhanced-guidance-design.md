# AkashicSolver Enhanced Guidance — Design Spec

**Date:** 2026-03-18
**Feature:** APG with Momentum + Guidance Interval Limiting
**Target:** AkashicSolver (new parameters, no changes to stepping math)

---

## Overview

Add two complementary CFG-level improvements to AkashicSolver as new optional parameters:

1. **Adaptive Projected Guidance (APG) with Momentum** — decomposes the guidance vector into parallel (oversaturation) and orthogonal (quality) components relative to the conditional prediction, scales the parallel component via `eta`, and optionally smooths the guidance direction with an accumulator across steps.
2. **Guidance Interval Limiting** — restricts CFG application to a configurable progress window, skipping it entirely at high-noise (harmful) and low-noise (unnecessary) steps.

Neither feature touches the Adams-Bashforth stepping math, tau blending, or noise injection. Both are upstream CFG-layer modifications.

---

## Architecture

### Hook type: `pre_cfg_function`

APG and guidance interval limiting are registered via `unet.set_model_sampler_pre_cfg_function()`, **not** `set_model_sampler_cfg_function`. The pre-cfg API appends to a list (see `model_patcher.py` line 82: `model_options["sampler_pre_cfg_function"] = model_options.get(..., []) + [fn]`), so it chains with the existing spectral modulation hook without conflict.

The existing spectral modulation hook uses `set_model_sampler_cfg_function` (post-CFG). The new enhanced guidance hook uses `set_model_sampler_pre_cfg_function` (pre-CFG). They operate at different points in the pipeline and do not interfere.

### New function: `create_enhanced_guidance_pre_cfg_hook()`

Lives in `scripts/custom_euler_ancestral_reforge.py` alongside `create_spectral_modulation_cfg_hook`.

Parameters captured at creation:
- `total_steps: int` — `len(sigmas) - 1`, for computing step progress
- All user-facing parameters (see below)

Closure state (stored as mutable list reference cells to allow mutation through closure):
- `step_counter: list[int]` — `[0]`, incremented each call after progress is computed
- `running_avg: list` — `[0]`, the guidance accumulator for momentum
- `prev_sigma: list` — `[None]`, last seen sigma for detecting inpainting restarts

### Registration in AkashicSolver setup

```python
if REFORGE_AVAILABLE and (apg_enabled or guidance_interval_enabled):
    try:
        unet = p.sd_model.forge_objects.unet.clone()  # clone before mutating
        enhanced_hook = create_enhanced_guidance_pre_cfg_hook(
            total_steps=len(final_sigmas) - 1,
            apg_enabled=apg_enabled,
            apg_eta=apg_eta,
            apg_norm_threshold=apg_norm_threshold,
            apg_momentum=apg_momentum,
            guidance_interval_enabled=guidance_interval_enabled,
            guidance_start=guidance_start,
            guidance_end=guidance_end,
        )
        unet.set_model_sampler_pre_cfg_function(
            enhanced_hook,
            disable_cfg1_optimization=True  # ensure uncond is always evaluated
        )
        p.sd_model.forge_objects.unet = unet
    except Exception as e:
        print(f"⚠️ Enhanced Guidance hook failed to register: {e}")
```

---

## Hook Body — Canonical Execution Order

The hook function follows the canonical event order from `nodes_apg.py`:

```
1. Guard: return early if only one condition (CFG disabled / uncond=None)
2. Extract cond, uncond, sigma, cond_scale
3. Inpainting reset check: if sigma increased vs prev_sigma, reset counter and running_avg
4. Compute progress from counter, then increment counter
5. Save prev_sigma
6. Guidance interval check: if outside window, return true-skip (conds_out[0] = uncond)
7. APG body (if apg_enabled)
8. Return modified conds_out
```

---

## Guidance Interval Limiting

```python
# Step 6: Guidance interval check
if guidance_interval_enabled:
    if progress < guidance_start or progress > guidance_end:
        # True skip: replace cond with uncond so framework computes
        # uncond + scale*(uncond-uncond) = uncond (no guidance)
        no_guidance = [uncond] + args["conds_out"][1:]
        return no_guidance
```

Returning `[uncond, uncond, ...]` causes the framework to produce a net-zero guidance delta. This is a true guidance skip, not a pass-through. Returning `args["conds_out"]` unmodified would still apply full standard CFG — the wrong behavior.

### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `guidance_interval_enabled` | bool | False | Enable guidance interval limiting |
| `guidance_start` | 0.0–1.0 | 0.1 | Skip guidance before this progress fraction (high-noise, harmful) |
| `guidance_end` | 0.0–1.0 | 0.9 | Skip guidance after this progress fraction (low-noise, unnecessary) |

Validation: if `guidance_start >= guidance_end`, log a warning and skip interval limiting.

---

## APG with Momentum

Full hook body pseudocode with correct ordering:

```python
def pre_cfg_hook(args):
    # 1. Guard: if only one condition output, CFG is disabled — return as-is
    if len(args["conds_out"]) < 2 or args["conds_out"][1] is None:
        return args["conds_out"]

    # 2. Extract values
    cond   = args["conds_out"][0]
    uncond = args["conds_out"][1]
    sigma  = args["sigma"][0]
    cond_scale = args["cond_scale"]

    # 3. Inpainting restart detection: sigma increasing means a new pass
    if prev_sigma[0] is not None and sigma > prev_sigma[0]:
        step_counter[0] = 0
        running_avg[0] = 0

    # 4. Compute progress then increment counter
    progress = step_counter[0] / max(total_steps, 1)  # range [0, (N-1)/N]
    step_counter[0] += 1

    # 5. Save sigma for next call
    prev_sigma[0] = sigma

    # 6. Guidance interval check (true skip)
    if guidance_interval_enabled:
        if progress < guidance_start or progress > guidance_end:
            return [uncond] + args["conds_out"][1:]  # zero-delta guidance

    if not apg_enabled:
        return args["conds_out"]

    # 7. APG body
    guidance = cond - uncond

    # Momentum: running accumulator (canonical form from nodes_apg.py)
    # Higher momentum = smoother but more lagged guidance direction
    if apg_momentum != 0:
        if not torch.is_tensor(running_avg[0]):
            running_avg[0] = guidance
        else:
            running_avg[0] = apg_momentum * running_avg[0] + guidance
        guidance = running_avg[0]
    # Note: norm_threshold is applied to the accumulated guidance vector,
    # not per-step guidance. This is intentional: constrains the smoothed
    # direction, consistent with the canonical implementation.

    # Norm threshold: clamp magnitude to prevent runaway early-step guidance
    if apg_norm_threshold > 0:
        guidance_norm = guidance.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale = torch.minimum(
            torch.ones_like(guidance_norm),
            apg_norm_threshold / guidance_norm
        )
        guidance = guidance * scale

    # Project guidance onto cond (not uncond)
    def project(v0, v1):
        v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
        v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
        v0_orthogonal = v0 - v0_parallel
        return v0_parallel, v0_orthogonal

    guidance_parallel, guidance_orthogonal = project(guidance, cond)

    # Reconstruct: orthogonal fully preserved, parallel scaled by eta.
    # eta=1.0 → standard CFG, eta=0.0 → pure orthogonal (full APG effect)
    modified_guidance = guidance_orthogonal + apg_eta * guidance_parallel

    # Output formulation (canonical from nodes_apg.py line 62):
    # This pre-scales modified_guidance so that after the framework multiplies
    # by cond_scale, the net effect is: uncond + cond_scale * modified_guidance.
    # Do NOT simplify this formula — removing the division breaks output magnitude.
    modified_cond = (uncond + modified_guidance) + (cond - uncond) / cond_scale

    # 8. Return
    return [modified_cond, uncond] + args["conds_out"][2:]
```

### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `apg_enabled` | bool | False | Enable APG with momentum |
| `apg_eta` | -1.0–2.0 | 1.0 | Scale of parallel guidance component. 1.0 = standard CFG, 0.0 = full APG |
| `apg_norm_threshold` | 0.0–50.0 | 5.0 | Clamp guidance vector norm. 0.0 = disabled |
| `apg_momentum` | -5.0–1.0 | 0.0 | Guidance accumulator coefficient. 0.0 = disabled. Negative = anti-momentum |

`apg_eta=1.0` default means APG is fully opt-in — the feature has zero effect until the user lowers eta below 1.0. This matches the canonical implementation's defaults.

---

## UI Placement

New collapsible subsection **"Enhanced Guidance"** in AkashicSolver's UI, below the existing "Spectral Modulation" and "Combat CFG Drift" sections.

Sub-groups:
- **APG**: `apg_enabled`, `apg_eta`, `apg_norm_threshold`, `apg_momentum`
- **Guidance Interval**: `guidance_interval_enabled`, `guidance_start`, `guidance_end`

---

## What Is Not Changed

- `sa_solver_step()` Adams-Bashforth stepping logic
- Tau blending or noise injection
- Phase-aware tau computation
- Existing spectral modulation or combat CFG drift logic
- Any other sampler (Adept, MCE, etc.)
- Scheduler code (AkashicAOS, AkashicEQFlow)

---

## References

- APG: *Adaptive Projected Guidance* (arXiv 2410.02416)
- Canonical implementation: `ldm_patched/contrib/nodes_apg.py` (ComfyUI)
- Guidance Interval: *Guiding a Diffusion Model with a Bad Version of Itself* (arXiv 2404.07724, Karras et al., NeurIPS 2024)
