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
sys.modules['modules.scripts'].Script = type('Script', (), {
    'title': lambda s: '', 'show': lambda s, *a: False,
    'ui': lambda s, *a: [], 'run': lambda s, *a: None,
})
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
_k_diff_sampling = types.ModuleType('k_diff.k_diffusion.sampling')
_k_diff_kdiff = types.ModuleType('k_diff.k_diffusion')
_k_diff_kdiff.sampling = _k_diff_sampling
_k_diff_root = types.ModuleType('k_diff')
_k_diff_root.k_diffusion = _k_diff_kdiff
_k_diff_root.sampling = _k_diff_sampling

_kdiff_sampling = types.ModuleType('k_diffusion.sampling')
_kdiff_root = types.ModuleType('k_diffusion')
_kdiff_root.sampling = _kdiff_sampling

for mod, obj in [
    ('k_diff', _k_diff_root),
    ('k_diff.k_diffusion', _k_diff_kdiff),
    ('k_diff.k_diffusion.sampling', _k_diff_sampling),
    ('k_diffusion', _kdiff_root),
    ('k_diffusion.sampling', _kdiff_sampling),
]:
    sys.modules.setdefault(mod, obj)
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
            import traceback
            print(f"FAIL: {t.__name__}: {e}")
            traceback.print_exc()
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)} test(s) FAILED: {failed}")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
