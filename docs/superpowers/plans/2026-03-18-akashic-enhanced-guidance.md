# AkashicSolver Enhanced Guidance Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add APG with momentum and Guidance Interval Limiting as new optional parameters in AkashicSolver, using a `pre_cfg` hook that chains cleanly with the existing spectral modulation hook.

**Architecture:** A single new function `create_enhanced_guidance_pre_cfg_hook()` registered via `set_model_sampler_pre_cfg_function` (list-based, non-destructive). All edits are in one file. No stepping math is touched.

**Tech Stack:** PyTorch, Gradio (WebUI UI), reForge hook API (`ldm_patched/modules/model_patcher.py`)

**Spec:** `docs/superpowers/specs/2026-03-18-akashic-enhanced-guidance-design.md`

---

## Chunk 1: Hook Function + Defaults

### Task 1: Add defaults to `current_sampler_settings`

**Files:**
- Modify: `scripts/custom_euler_ancestral_reforge.py:190-194`

The `current_sampler_settings` dict at line 146 holds all sampler defaults. The "Additional CFG fixes" block ends at line 194. Add 7 new keys after line 194.

- [ ] **Step 1: Read lines 188-195 to get exact context**

```python
# Expected content around line 190:
    'akashic_spectral_mod': False,
    'akashic_spectral_percentile': 5.0,
    'akashic_combat_cfg_drift': False,
    'akashic_combat_drift_intensity': 0.5,
}
```

- [ ] **Step 2: Add new defaults**

In `scripts/custom_euler_ancestral_reforge.py`, replace:
```python
    'akashic_combat_cfg_drift': False,  # Combat CFG mean drift
    'akashic_combat_drift_intensity': 0.5,  # Combat drift intensity (0-1)
}
```
With:
```python
    'akashic_combat_cfg_drift': False,  # Combat CFG mean drift
    'akashic_combat_drift_intensity': 0.5,  # Combat drift intensity (0-1)
    # Enhanced Guidance settings (APG + Guidance Interval)
    'akashic_apg_enabled': False,
    'akashic_apg_eta': 1.0,
    'akashic_apg_norm_threshold': 5.0,
    'akashic_apg_momentum': 0.0,
    'akashic_guidance_interval_enabled': False,
    'akashic_guidance_start': 0.1,
    'akashic_guidance_end': 0.9,
}
```

- [ ] **Step 3: Verify by running a Python syntax check**

```bash
cd /mnt/a/stable-diffusion-webui-reForge/extensions/cussam-sampler
python -c "
import ast, sys
with open('scripts/custom_euler_ancestral_reforge.py') as f:
    src = f.read()
try:
    ast.parse(src)
    print('OK: syntax valid')
except SyntaxError as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"
```
Expected: `OK: syntax valid`

- [ ] **Step 4: Commit**

```bash
git add scripts/custom_euler_ancestral_reforge.py
git commit -m "feat(akashic): add enhanced guidance defaults to current_sampler_settings"
```

---

### Task 2: Add `create_enhanced_guidance_pre_cfg_hook()` function

**Files:**
- Modify: `scripts/custom_euler_ancestral_reforge.py` — insert after `create_spectral_modulation_cfg_hook` (ends around line 1340)

The new function goes right after the spectral modulation hook factory, which ends with a closing `return spectral_cfg_hook` and `return` statement. Find the exact end by searching for `def create_spectral_modulation_cfg_hook` — it's the only such function.

- [ ] **Step 1: Write a standalone test for the hook logic**

Create `tests/test_enhanced_guidance_hook.py`:

```python
"""
Standalone test for create_enhanced_guidance_pre_cfg_hook logic.
Run with: python tests/test_enhanced_guidance_hook.py
No WebUI needed.
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── minimal shims so the import doesn't fail without WebUI ───────────────────
import types

# Core WebUI modules
for mod in ['modules', 'modules.shared', 'modules.scripts', 'gradio']:
    sys.modules.setdefault(mod, types.ModuleType(mod))
sys.modules['modules'].scripts = sys.modules['modules.scripts']
sys.modules['gradio'].Checkbox = lambda **kw: None
sys.modules['gradio'].Slider = lambda **kw: None
sys.modules['gradio'].Dropdown = lambda **kw: None
sys.modules['gradio'].Row = type('Row', (), {'__enter__': lambda s,*a: s, '__exit__': lambda s,*a: None})
sys.modules['gradio'].Group = sys.modules['gradio'].Row
sys.modules['gradio'].Accordion = sys.modules['gradio'].Row
sys.modules['gradio'].Markdown = lambda *a, **kw: None
sys.modules['gradio'].update = lambda **kw: kw
sys.modules['modules.shared'] = types.SimpleNamespace(opts=types.SimpleNamespace(sd_model_checkpoint=''))

# k_diffusion / k_diff (used by the sampling module)
for mod in ['k_diff', 'k_diff.k_diffusion', 'k_diff.k_diffusion.sampling',
            'k_diffusion', 'k_diffusion.sampling']:
    m = types.ModuleType(mod)
    # stub out anything the source file calls at import time
    m.sampling = m
    sys.modules.setdefault(mod, m)
# ─────────────────────────────────────────────────────────────────────────────

from scripts.custom_euler_ancestral_reforge import create_enhanced_guidance_pre_cfg_hook

B, C, H, W = 1, 4, 8, 8  # tiny latent

def make_args(cond_val=2.0, uncond_val=1.0, sigma=5.0, cond_scale=7.5, extra_conds=None):
    cond   = torch.full((B, C, H, W), cond_val)
    uncond = torch.full((B, C, H, W), uncond_val)
    conds_out = [cond, uncond]
    if extra_conds:
        conds_out += extra_conds
    return {
        "conds_out": conds_out,
        "sigma": torch.tensor([sigma]),
        "cond_scale": cond_scale,
    }

def test_passthrough_when_disabled():
    """With apg_enabled=False and guidance_interval_enabled=False, hook is a no-op."""
    hook = create_enhanced_guidance_pre_cfg_hook(
        total_steps=20,
        apg_enabled=False, apg_eta=1.0, apg_norm_threshold=5.0, apg_momentum=0.0,
        guidance_interval_enabled=False, guidance_start=0.1, guidance_end=0.9,
    )
    args = make_args()
    original_cond = args["conds_out"][0].clone()
    result = hook(args)
    assert torch.allclose(result[0], original_cond), "cond should be unchanged"
    assert torch.allclose(result[1], args["conds_out"][1]), "uncond should be unchanged"
    print("PASS: passthrough_when_disabled")

def test_guidance_interval_skip():
    """Progress=0 (< guidance_start=0.1) should return [uncond, uncond, ...]."""
    hook = create_enhanced_guidance_pre_cfg_hook(
        total_steps=20,
        apg_enabled=False, apg_eta=1.0, apg_norm_threshold=0.0, apg_momentum=0.0,
        guidance_interval_enabled=True, guidance_start=0.1, guidance_end=0.9,
    )
    args = make_args()
    uncond = args["conds_out"][1].clone()
    result = hook(args)
    # First step (progress=0) is below guidance_start=0.1 → true skip
    assert torch.allclose(result[0], uncond), "cond[0] should be replaced with uncond for true skip"
    print("PASS: guidance_interval_skip")

def test_guidance_interval_pass():
    """Progress in the middle of the window should NOT skip."""
    hook = create_enhanced_guidance_pre_cfg_hook(
        total_steps=20,
        apg_enabled=False, apg_eta=1.0, apg_norm_threshold=0.0, apg_momentum=0.0,
        guidance_interval_enabled=True, guidance_start=0.1, guidance_end=0.9,
    )
    # advance counter to step 5 (progress = 5/20 = 0.25, inside [0.1, 0.9])
    for i in range(5):
        hook(make_args(sigma=10.0 - i))  # monotonically decreasing sigma
    args = make_args(sigma=4.0)
    result = hook(args)
    # cond != uncond (cond_val=2.0, uncond_val=1.0), so if skip didn't fire
    # result[0] should NOT equal uncond
    assert not torch.allclose(result[0], args["conds_out"][1]), "cond should NOT be replaced mid-window"
    print("PASS: guidance_interval_pass")

def test_apg_eta_one_matches_canonical_formula():
    """apg_eta=1.0, no momentum: output must match the canonical APG formula exactly."""
    hook = create_enhanced_guidance_pre_cfg_hook(
        total_steps=20,
        apg_enabled=True, apg_eta=1.0, apg_norm_threshold=0.0, apg_momentum=0.0,
        guidance_interval_enabled=False, guidance_start=0.1, guidance_end=0.9,
    )
    cond_scale = 7.5
    args = make_args(cond_val=2.0, uncond_val=1.0, cond_scale=cond_scale)
    cond   = args["conds_out"][0]
    uncond = args["conds_out"][1]

    result = hook(args)

    # With eta=1.0: modified_guidance = ortho + 1.0*parallel = full guidance (cond - uncond)
    # Canonical output formula: (uncond + modified_guidance) + (cond - uncond) / cond_scale
    guidance = cond - uncond
    expected = (uncond + guidance) + guidance / cond_scale
    assert torch.allclose(result[0], expected, atol=1e-5), \
        f"eta=1.0 output does not match canonical formula"
    assert not torch.isnan(result[0]).any(), "result should not contain NaN"
    print("PASS: apg_eta_one_matches_canonical_formula")

def test_apg_eta_zero_reduces_saturation():
    """apg_eta=0 removes the parallel component — output magnitude should be <= eta=1 case."""
    cond_scale = 7.5
    args_eta1 = make_args(cond_val=3.0, uncond_val=1.0, cond_scale=cond_scale)
    args_eta0 = make_args(cond_val=3.0, uncond_val=1.0, cond_scale=cond_scale)

    hook_eta1 = create_enhanced_guidance_pre_cfg_hook(
        total_steps=20, apg_enabled=True, apg_eta=1.0,
        apg_norm_threshold=0.0, apg_momentum=0.0,
        guidance_interval_enabled=False, guidance_start=0.0, guidance_end=1.0,
    )
    hook_eta0 = create_enhanced_guidance_pre_cfg_hook(
        total_steps=20, apg_enabled=True, apg_eta=0.0,
        apg_norm_threshold=0.0, apg_momentum=0.0,
        guidance_interval_enabled=False, guidance_start=0.0, guidance_end=1.0,
    )

    cond_eta1 = hook_eta1(args_eta1)[0]
    cond_eta0 = hook_eta0(args_eta0)[0]

    # With eta=0, parallel component removed → modified_guidance norm should be smaller
    guidance_eta1 = (cond_eta1 - args_eta1["conds_out"][1]).norm()
    guidance_eta0 = (cond_eta0 - args_eta0["conds_out"][1]).norm()
    assert guidance_eta0 <= guidance_eta1 + 1e-5, \
        f"eta=0 guidance norm ({guidance_eta0:.4f}) should be <= eta=1 ({guidance_eta1:.4f})"
    print("PASS: apg_eta_zero_reduces_saturation")

def test_single_cond_guard():
    """If only one condition (CFG disabled), hook returns as-is without error."""
    hook = create_enhanced_guidance_pre_cfg_hook(
        total_steps=20, apg_enabled=True, apg_eta=0.0,
        apg_norm_threshold=5.0, apg_momentum=0.5,
        guidance_interval_enabled=True, guidance_start=0.1, guidance_end=0.9,
    )
    cond = torch.ones(B, C, H, W)
    args = {"conds_out": [cond], "sigma": torch.tensor([5.0]), "cond_scale": 7.5}
    result = hook(args)
    assert len(result) == 1, "should return list of length 1"
    assert torch.allclose(result[0], cond), "single cond returned unchanged"
    print("PASS: single_cond_guard")

def test_extra_conds_preserved():
    """Extra conditioning tensors beyond index 1 are preserved unchanged."""
    hook = create_enhanced_guidance_pre_cfg_hook(
        total_steps=20, apg_enabled=True, apg_eta=0.5,
        apg_norm_threshold=0.0, apg_momentum=0.0,
        guidance_interval_enabled=False, guidance_start=0.1, guidance_end=0.9,
    )
    extra = torch.full((B, C, H, W), 99.0)
    args = make_args(extra_conds=[extra])
    result = hook(args)
    assert len(result) == 3, "should preserve all 3 conds_out entries"
    assert torch.allclose(result[2], extra), "extra cond at index 2 should be unchanged"
    print("PASS: extra_conds_preserved")

def test_inpainting_reset():
    """Sigma increasing (inpainting restart) resets the step counter and running_avg."""
    hook = create_enhanced_guidance_pre_cfg_hook(
        total_steps=20, apg_enabled=True, apg_eta=0.5,
        apg_norm_threshold=0.0, apg_momentum=0.8,
        guidance_interval_enabled=True, guidance_start=0.5, guidance_end=0.9,
    )
    # Run 15 steps (progress = 0.75, inside window)
    for i in range(15):
        hook(make_args(sigma=10.0 - i * 0.5))
    # Now simulate inpainting restart: sigma jumps up
    args_restart = make_args(sigma=14.0)
    result = hook(args_restart)
    # After reset, step_counter=0, progress=0/20=0.0 < guidance_start=0.5 → skip
    uncond = args_restart["conds_out"][1]
    assert torch.allclose(result[0], uncond), "after restart, first step should be a guidance skip"
    print("PASS: inpainting_reset")

if __name__ == "__main__":
    tests = [
        test_passthrough_when_disabled,
        test_guidance_interval_skip,
        test_guidance_interval_pass,
        test_apg_eta_one_matches_canonical_formula,
        test_apg_eta_zero_reduces_saturation,
        test_single_cond_guard,
        test_extra_conds_preserved,
        test_inpainting_reset,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)} test(s) FAILED: {failed}")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
```

- [ ] **Step 2: Run the test and confirm it fails (function not yet defined)**

```bash
cd /mnt/a/stable-diffusion-webui-reForge/extensions/cussam-sampler
mkdir -p tests
python tests/test_enhanced_guidance_hook.py 2>&1 | head -20
```
Expected: ImportError or AttributeError — `create_enhanced_guidance_pre_cfg_hook` does not exist yet.

- [ ] **Step 3: Implement `create_enhanced_guidance_pre_cfg_hook()`**

Find `create_spectral_modulation_cfg_hook` in `scripts/custom_euler_ancestral_reforge.py` (around line 1296). Read to the end of the function (look for the second `return` that closes the factory). Insert the new function immediately after it.

The new function to insert (after the closing line of `create_spectral_modulation_cfg_hook`):

```python

def create_enhanced_guidance_pre_cfg_hook(
    total_steps,
    apg_enabled=True,
    apg_eta=1.0,
    apg_norm_threshold=5.0,
    apg_momentum=0.0,
    guidance_interval_enabled=False,
    guidance_start=0.1,
    guidance_end=0.9,
):
    """
    Pre-CFG hook implementing Adaptive Projected Guidance (APG) with momentum
    and Guidance Interval Limiting for AkashicSolver.

    Registered via set_model_sampler_pre_cfg_function (list-based, non-destructive).
    Operates on args["conds_out"] before CFG multiplication.

    References:
      - APG: arXiv 2410.02416
      - Guidance Interval: arXiv 2404.07724 (Karras et al., NeurIPS 2024)
      - Canonical APG impl: ldm_patched/contrib/nodes_apg.py
    """
    # Mutable list cells — the only way to mutate closed-over state in Python 3
    step_counter = [0]      # incremented each call after progress is computed
    running_avg = [0]       # guidance accumulator for momentum (scalar 0 = uninitialised)
    prev_sigma = [None]     # last seen sigma, used to detect inpainting restarts

    # Validate guidance interval parameters
    _interval_active = guidance_interval_enabled and (guidance_start < guidance_end)
    if guidance_interval_enabled and not _interval_active:
        print(f"⚠️ Enhanced Guidance: guidance_start ({guidance_start}) >= guidance_end ({guidance_end}), interval limiting disabled.")

    def project(v0, v1):
        """Project v0 onto v1, return (parallel, orthogonal) components."""
        v1_norm = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
        v0_parallel = (v0 * v1_norm).sum(dim=[-1, -2, -3], keepdim=True) * v1_norm
        v0_orthogonal = v0 - v0_parallel
        return v0_parallel, v0_orthogonal

    def pre_cfg_hook(args):
        # 1. Guard: single-condition path (CFG disabled — uncond not evaluated)
        if len(args["conds_out"]) < 2 or args["conds_out"][1] is None:
            return args["conds_out"]

        # 2. Extract values
        cond       = args["conds_out"][0]
        uncond     = args["conds_out"][1]
        sigma      = args["sigma"][0]
        cond_scale = args["cond_scale"]

        # 3. Inpainting restart detection: sigma increasing means a new pass started
        if prev_sigma[0] is not None and sigma > prev_sigma[0]:
            step_counter[0] = 0
            running_avg[0] = 0

        # 4. Compute progress then increment counter
        #    Range: [0, (N-1)/N] — never reaches 1.0, safe for 1-step edge case
        progress = step_counter[0] / max(total_steps, 1)
        step_counter[0] += 1

        # 5. Save sigma for next call
        prev_sigma[0] = sigma

        # 6. Guidance interval check — true skip (zero-delta guidance)
        if _interval_active:
            if progress < guidance_start or progress > guidance_end:
                # Replace cond with uncond so framework computes:
                # uncond + scale*(uncond-uncond) = uncond  (no guidance)
                return [uncond] + args["conds_out"][1:]

        # 7. APG body
        if not apg_enabled:
            return args["conds_out"]

        guidance = cond - uncond

        # Momentum: running accumulator (canonical form from nodes_apg.py)
        # norm_threshold is applied to the accumulated vector (intentional —
        # constrains the smoothed direction, not per-step guidance)
        if apg_momentum != 0:
            if not torch.is_tensor(running_avg[0]):
                running_avg[0] = guidance
            else:
                running_avg[0] = apg_momentum * running_avg[0] + guidance
            guidance = running_avg[0]

        # Norm threshold: soft clamp to prevent runaway guidance magnitude
        if apg_norm_threshold > 0:
            guidance_norm = guidance.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale = torch.minimum(
                torch.ones_like(guidance_norm),
                apg_norm_threshold / guidance_norm
            )
            guidance = guidance * scale

        # Project guidance onto cond (not uncond — this is the correct APG reference)
        guidance_parallel, guidance_orthogonal = project(guidance, cond)

        # Reconstruct: orthogonal fully preserved, parallel scaled by eta
        # eta=1.0 → standard CFG behaviour; eta=0.0 → pure orthogonal (full APG)
        modified_guidance = guidance_orthogonal + apg_eta * guidance_parallel

        # Output formulation (canonical from nodes_apg.py line 62).
        # Pre-scales modified_guidance so that after the framework multiplies by
        # cond_scale the net result is: uncond + cond_scale * modified_guidance.
        # Do NOT simplify — removing the division breaks output magnitude.
        modified_cond = (uncond + modified_guidance) + (cond - uncond) / cond_scale

        # 8. Return — preserve any extra conds_out entries beyond index 1
        return [modified_cond, uncond] + args["conds_out"][2:]

    return pre_cfg_hook
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
cd /mnt/a/stable-diffusion-webui-reForge/extensions/cussam-sampler
python tests/test_enhanced_guidance_hook.py
```
Expected:
```
PASS: passthrough_when_disabled
PASS: guidance_interval_skip
PASS: guidance_interval_pass
PASS: apg_eta_one_matches_canonical_formula
PASS: apg_eta_zero_reduces_saturation
PASS: single_cond_guard
PASS: extra_conds_preserved
PASS: inpainting_reset

All 8 tests passed.
```

- [ ] **Step 5: Syntax check**

```bash
python -c "import ast; ast.parse(open('scripts/custom_euler_ancestral_reforge.py').read()); print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/custom_euler_ancestral_reforge.py tests/test_enhanced_guidance_hook.py
git commit -m "feat(akashic): add create_enhanced_guidance_pre_cfg_hook with APG + guidance interval"
```

---

## Chunk 2: UI + Wiring

### Task 3: Add UI widgets for Enhanced Guidance

**Files:**
- Modify: `scripts/custom_euler_ancestral_reforge.py` — AkashicSolver UI section (~line 1918-1958)

The "Additional CFG Fixes" section ends at line 1958 (after the `combat_drift_options` visibility handler). Insert a new "Enhanced Guidance" section immediately after line 1958 (before the `with gr.Group(visible=False) as mirror_correction_euler_options:` block).

- [ ] **Step 1: Read lines 1940-1962 to confirm insertion point**

Confirm the structure ends with:
```python
                            self.akashic_combat_cfg_drift.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[self.akashic_combat_cfg_drift],
                                outputs=[combat_drift_options]
                            )

                        with gr.Group(visible=False) as mirror_correction_euler_options:
```

- [ ] **Step 2: Insert Enhanced Guidance UI block**

After the `self.akashic_combat_cfg_drift.change(...)` block and before `with gr.Group(visible=False) as mirror_correction_euler_options:`, insert:

```python

                            # Enhanced Guidance Section
                            gr.Markdown("---")
                            gr.Markdown("**Enhanced Guidance** (APG + Guidance Interval)")
                            with gr.Row():
                                self.akashic_apg_enabled = gr.Checkbox(
                                    label='APG',
                                    value=False,
                                    info="Adaptive Projected Guidance: reduces oversaturation"
                                )
                                self.akashic_guidance_interval_enabled = gr.Checkbox(
                                    label='Guidance Interval',
                                    value=False,
                                    info="Restrict CFG to a progress window (skip harmful early/late steps)"
                                )

                            with gr.Group(visible=False) as apg_options:
                                with gr.Row():
                                    self.akashic_apg_eta = gr.Slider(
                                        label='APG Eta',
                                        minimum=-1.0, maximum=2.0, value=1.0, step=0.05,
                                        info="Parallel guidance scale. 1.0=standard CFG, 0.0=full APG"
                                    )
                                    self.akashic_apg_norm_threshold = gr.Slider(
                                        label='Norm Threshold',
                                        minimum=0.0, maximum=50.0, value=5.0, step=0.5,
                                        info="Clamp guidance vector magnitude (0=disabled)"
                                    )
                                self.akashic_apg_momentum = gr.Slider(
                                    label='APG Momentum',
                                    minimum=-5.0, maximum=1.0, value=0.0, step=0.05,
                                    info="Guidance accumulator coefficient (0=disabled)"
                                )

                            with gr.Group(visible=False) as guidance_interval_options:
                                with gr.Row():
                                    self.akashic_guidance_start = gr.Slider(
                                        label='Guidance Start',
                                        minimum=0.0, maximum=1.0, value=0.1, step=0.05,
                                        info="Skip guidance before this progress fraction"
                                    )
                                    self.akashic_guidance_end = gr.Slider(
                                        label='Guidance End',
                                        minimum=0.0, maximum=1.0, value=0.9, step=0.05,
                                        info="Skip guidance after this progress fraction"
                                    )

                            # Visibility handlers for Enhanced Guidance
                            self.akashic_apg_enabled.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[self.akashic_apg_enabled],
                                outputs=[apg_options]
                            )
                            self.akashic_guidance_interval_enabled.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[self.akashic_guidance_interval_enabled],
                                outputs=[guidance_interval_options]
                            )
```

- [ ] **Step 3: Syntax check**

```bash
python -c "import ast; ast.parse(open('scripts/custom_euler_ancestral_reforge.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/custom_euler_ancestral_reforge.py
git commit -m "feat(akashic): add Enhanced Guidance UI widgets (APG + guidance interval)"
```

---

### Task 4: Wire all plumbing — `ui()` return, unpacking, settings, hook registration, generation params, infotext

**Files:**
- Modify: `scripts/custom_euler_ancestral_reforge.py` — 6 separate locations

This task has 6 distinct edit sites. Read each location before editing.

#### 4a: `ui()` return list (around line 2265-2273)

- [ ] **Step 1: Read lines 2263-2274 to confirm context**

Find:
```python
            self.akashic_ndb_strength,
            self.akashic_eqvae_mode,
            # Additional CFG Fixes settings
            self.akashic_spectral_mod,
            self.akashic_spectral_percentile,
            self.akashic_combat_cfg_drift,
            self.akashic_combat_drift_intensity,
            self.vae_reflection,
        ]
```

- [ ] **Step 2: Add new widgets to return list**

Replace the end of that list:
```python
            self.akashic_combat_cfg_drift,
            self.akashic_combat_drift_intensity,
            self.vae_reflection,
        ]
```
With:
```python
            self.akashic_combat_cfg_drift,
            self.akashic_combat_drift_intensity,
            self.vae_reflection,
            # Enhanced Guidance
            self.akashic_apg_enabled,
            self.akashic_apg_eta,
            self.akashic_apg_norm_threshold,
            self.akashic_apg_momentum,
            self.akashic_guidance_interval_enabled,
            self.akashic_guidance_start,
            self.akashic_guidance_end,
        ]
```

#### 4b: `process_before_every_sampling` parameter unpacking (around line 2297-2301)

- [ ] **Step 3: Read lines 2297-2302 to confirm context**

Find:
```python
            # Additional CFG Fixes settings
            akashic_spectral_mod, akashic_spectral_percentile,
            akashic_combat_cfg_drift, akashic_combat_drift_intensity,
            vae_reflection,
        ) = script_args
```

- [ ] **Step 4: Extend unpacking**

Replace:
```python
            akashic_combat_cfg_drift, akashic_combat_drift_intensity,
            vae_reflection,
        ) = script_args
```
With:
```python
            akashic_combat_cfg_drift, akashic_combat_drift_intensity,
            vae_reflection,
            # Enhanced Guidance
            akashic_apg_enabled, akashic_apg_eta, akashic_apg_norm_threshold, akashic_apg_momentum,
            akashic_guidance_interval_enabled, akashic_guidance_start, akashic_guidance_end,
        ) = script_args
```

#### 4c: `current_sampler_settings.update()` (around line 2541-2547)

- [ ] **Step 5: Read lines 2541-2548 to confirm context**

Find:
```python
            # CFG Enhancement settings
            'akashic_spectral_mod': akashic_spectral_mod,
            'akashic_spectral_percentile': akashic_spectral_percentile,
            'akashic_combat_cfg_drift': akashic_combat_cfg_drift,
            'akashic_combat_drift_intensity': akashic_combat_drift_intensity,
            'vae_reflection': vae_reflection,
        })
```

- [ ] **Step 6: Add new settings to update dict**

Replace:
```python
            'akashic_combat_drift_intensity': akashic_combat_drift_intensity,
            'vae_reflection': vae_reflection,
        })
```
With:
```python
            'akashic_combat_drift_intensity': akashic_combat_drift_intensity,
            'vae_reflection': vae_reflection,
            # Enhanced Guidance
            'akashic_apg_enabled': akashic_apg_enabled,
            'akashic_apg_eta': akashic_apg_eta,
            'akashic_apg_norm_threshold': akashic_apg_norm_threshold,
            'akashic_apg_momentum': akashic_apg_momentum,
            'akashic_guidance_interval_enabled': akashic_guidance_interval_enabled,
            'akashic_guidance_start': akashic_guidance_start,
            'akashic_guidance_end': akashic_guidance_end,
        })
```

#### 4d: Hook registration (after the spectral modulation hook block, around line 2573)

- [ ] **Step 7: Read lines 2559-2575 to confirm context**

Find:
```python
        # --- Spectral Modulation CFG Hook ---
        ...
            except Exception as e:
                print(f"⚠️ Failed to apply Spectral Modulation CFG hook: {e}")

        if enable_custom:
```

- [ ] **Step 8: Insert enhanced guidance hook registration block**

After the spectral modulation hook's `except` block and before `if enable_custom:`, insert:

```python
        # --- Enhanced Guidance Pre-CFG Hook (APG + Guidance Interval) ---
        if REFORGE_AVAILABLE and (akashic_apg_enabled or akashic_guidance_interval_enabled):
            try:
                if hasattr(p, 'sd_model') and hasattr(p.sd_model, 'forge_objects'):
                    unet = p.sd_model.forge_objects.unet.clone()
                    enhanced_hook = create_enhanced_guidance_pre_cfg_hook(
                        total_steps=max(p.steps, 1),
                        apg_enabled=akashic_apg_enabled,
                        apg_eta=akashic_apg_eta,
                        apg_norm_threshold=akashic_apg_norm_threshold,
                        apg_momentum=akashic_apg_momentum,
                        guidance_interval_enabled=akashic_guidance_interval_enabled,
                        guidance_start=akashic_guidance_start,
                        guidance_end=akashic_guidance_end,
                    )
                    unet.set_model_sampler_pre_cfg_function(
                        enhanced_hook,
                        disable_cfg1_optimization=True,
                    )
                    p.sd_model.forge_objects.unet = unet
                    parts = []
                    if akashic_apg_enabled:
                        parts.append(f"APG(η={akashic_apg_eta}, mom={akashic_apg_momentum})")
                    if akashic_guidance_interval_enabled:
                        parts.append(f"Interval[{akashic_guidance_start:.2f}–{akashic_guidance_end:.2f}]")
                    print(f"✨ Enhanced Guidance active: {', '.join(parts)}")
            except Exception as e:
                print(f"⚠️ Failed to apply Enhanced Guidance hook: {e}")

```

**Note on `_total_steps`:** The final sigma schedule is computed later inside the sampler function, not at hook registration time. `p.steps` is a safe approximation — the hook only needs `total_steps` for progress computation and off-by-one here is immaterial.

#### 4e: `p.extra_generation_params.update()` (around line 2632-2636)

- [ ] **Step 9: Read lines 2632-2637 to confirm context**

Find:
```python
                # Additional CFG Fixes parameters
                'akashic_spectral_mod': akashic_spectral_mod if use_akashic_solver else False,
                'akashic_combat_cfg_drift': akashic_combat_cfg_drift if use_akashic_solver else False,
                'akashic_combat_drift_intensity': akashic_combat_drift_intensity if use_akashic_solver and akashic_combat_cfg_drift else 'N/A',
                'vae_reflection': vae_reflection,
            })
```

- [ ] **Step 10: Add enhanced guidance to generation params**

Replace:
```python
                'vae_reflection': vae_reflection,
            })
```
With:
```python
                'vae_reflection': vae_reflection,
                # Enhanced Guidance
                'akashic_apg_enabled': akashic_apg_enabled if use_akashic_solver else False,
                'akashic_apg_eta': akashic_apg_eta if use_akashic_solver and akashic_apg_enabled else 'N/A',
                'akashic_apg_norm_threshold': akashic_apg_norm_threshold if use_akashic_solver and akashic_apg_enabled else 'N/A',
                'akashic_apg_momentum': akashic_apg_momentum if use_akashic_solver and akashic_apg_enabled else 'N/A',
                'akashic_guidance_interval_enabled': akashic_guidance_interval_enabled if use_akashic_solver else False,
                'akashic_guidance_start': akashic_guidance_start if use_akashic_solver and akashic_guidance_interval_enabled else 'N/A',
                'akashic_guidance_end': akashic_guidance_end if use_akashic_solver and akashic_guidance_interval_enabled else 'N/A',
            })
```

#### 4f: `infotext_fields` (around line 2218-2223)

- [ ] **Step 11: Read lines 2218-2224 to confirm context**

Find:
```python
            (self.akashic_combat_cfg_drift, lambda p: str(p.get('akashic_combat_cfg_drift', 'false')).lower() == 'true' if 'akashic_combat_cfg_drift' in p else gr.update()),
            (self.akashic_combat_drift_intensity, lambda p: gr.update() if p.get('akashic_combat_drift_intensity') in (None, 'N/A') else float(p['akashic_combat_drift_intensity'])),
            (self.vae_reflection, lambda p: str(p.get('vae_reflection', 'false')).lower() == 'true' if 'vae_reflection' in p else gr.update()),
        ]
```

- [ ] **Step 12: Add infotext entries**

Replace:
```python
            (self.vae_reflection, lambda p: str(p.get('vae_reflection', 'false')).lower() == 'true' if 'vae_reflection' in p else gr.update()),
        ]
```
With:
```python
            (self.vae_reflection, lambda p: str(p.get('vae_reflection', 'false')).lower() == 'true' if 'vae_reflection' in p else gr.update()),
            # Enhanced Guidance infotext
            (self.akashic_apg_enabled, lambda p: str(p.get('akashic_apg_enabled', 'false')).lower() == 'true' if 'akashic_apg_enabled' in p else gr.update()),
            (self.akashic_apg_eta, lambda p: gr.update() if p.get('akashic_apg_eta') in (None, 'N/A') else float(p['akashic_apg_eta'])),
            (self.akashic_apg_norm_threshold, lambda p: gr.update() if p.get('akashic_apg_norm_threshold') in (None, 'N/A') else float(p['akashic_apg_norm_threshold'])),
            (self.akashic_apg_momentum, lambda p: gr.update() if p.get('akashic_apg_momentum') in (None, 'N/A') else float(p['akashic_apg_momentum'])),
            (self.akashic_guidance_interval_enabled, lambda p: str(p.get('akashic_guidance_interval_enabled', 'false')).lower() == 'true' if 'akashic_guidance_interval_enabled' in p else gr.update()),
            (self.akashic_guidance_start, lambda p: gr.update() if p.get('akashic_guidance_start') in (None, 'N/A') else float(p['akashic_guidance_start'])),
            (self.akashic_guidance_end, lambda p: gr.update() if p.get('akashic_guidance_end') in (None, 'N/A') else float(p['akashic_guidance_end'])),
        ]
```

- [ ] **Step 13: Final syntax check**

```bash
python -c "import ast; ast.parse(open('scripts/custom_euler_ancestral_reforge.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 14: Run the full test suite to verify nothing broke**

```bash
python tests/test_enhanced_guidance_hook.py
```
Expected: `All 8 tests passed.`

- [ ] **Step 15: Commit**

```bash
git add scripts/custom_euler_ancestral_reforge.py
git commit -m "feat(akashic): wire Enhanced Guidance through UI, settings, hook registration, and generation params"
```

---

## Chunk 3: Verification

### Task 5: Integration verification

- [ ] **Step 1: Confirm no regressions in existing parameters**

```bash
python -c "
import ast
src = open('scripts/custom_euler_ancestral_reforge.py').read()
ast.parse(src)

# Verify all expected symbols exist
assert 'create_enhanced_guidance_pre_cfg_hook' in src, 'hook factory missing'
assert 'akashic_apg_enabled' in src, 'apg_enabled missing'
assert 'akashic_guidance_interval_enabled' in src, 'interval_enabled missing'
assert 'set_model_sampler_pre_cfg_function' in src, 'pre_cfg registration missing'
assert 'disable_cfg1_optimization=True' in src, 'disable_cfg1_optimization missing'
assert 'enhanced_hook = create_enhanced_guidance_pre_cfg_hook' in src, 'hook call site missing'
# Verify old settings still present
assert 'akashic_spectral_mod' in src, 'spectral_mod broken'
assert 'akashic_combat_cfg_drift' in src, 'combat_cfg_drift broken'
print('All assertions passed.')
"
```

- [ ] **Step 2: Verify test file runs cleanly**

```bash
python tests/test_enhanced_guidance_hook.py
```
Expected: `All 8 tests passed.`

- [ ] **Step 3: Check that the `ui()` return list and unpacking have matching length**

```bash
python -c "
src = open('scripts/custom_euler_ancestral_reforge.py').read()
# Count self.akashic_apg_enabled occurrences — should be in UI return, unpacking, settings, params, infotext
count = src.count('akashic_apg_enabled')
print(f'akashic_apg_enabled occurrences: {count}')
assert count >= 5, f'Expected at least 5 occurrences (defaults, ui return, unpack, settings, params, infotext), got {count}'
print('OK')
"
```

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "test: add verification for Enhanced Guidance wiring"
```

---

## Recommended Test Settings (Manual)

When loading the WebUI, verify with AkashicSolver selected:

| Feature | Setting | Expected outcome |
|---------|---------|-----------------|
| APG only | `apg_enabled=True, apg_eta=0.0, apg_momentum=0.0, guidance_interval=False` | Less saturated colors at high CFG vs stock |
| Guidance interval only | `guidance_interval_enabled=True, start=0.1, end=0.9, apg=False` | Subtly cleaner early composition |
| Both combined | `apg_enabled=True, apg_eta=0.0, guidance_interval_enabled=True` | Combined effect |
| Both disabled | All off | Identical output to before this change |
| CFG=1.0 (no neg prompt) | Any combo | No crash, single-cond guard fires |
