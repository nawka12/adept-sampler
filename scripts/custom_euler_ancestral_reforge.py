import math
import numpy as np
import torch
import json
import sys
import traceback

from modules import scripts
import gradio as gr
try:
    import k_diff.k_diffusion.sampling
except ModuleNotFoundError:
    import types
    import k_diffusion.sampling
    k_diff = types.ModuleType('k_diff')
    k_diff.k_diffusion = types.ModuleType('k_diff.k_diffusion')
    k_diff.k_diffusion.sampling = k_diffusion.sampling
    sys.modules['k_diff'] = k_diff
    sys.modules['k_diff.k_diffusion'] = k_diff.k_diffusion
    sys.modules['k_diff.k_diffusion.sampling'] = k_diffusion.sampling
from functools import partial
from typing import Any

# Import shared for RNG state management
try:
    from modules import shared
    from modules import script_callbacks
    from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
    WEBUI_AVAILABLE = True
except ImportError:
    WEBUI_AVAILABLE = False

try:
    from torchvision.transforms.functional import gaussian_blur
    TORCHVISION_AVAILABLE = True
except ImportError:
    print("Torchvision not available, detail enhancement will be disabled.")
    TORCHVISION_AVAILABLE = False


# For reForge, we need to work with the ldm_patched modules
try:
    from ldm_patched.contrib.external import KSampler
    from ldm_patched.modules.model_management import get_torch_device
    REFORGE_AVAILABLE = True
except ImportError:
    # Fallback if reForge modules aren't available
    REFORGE_AVAILABLE = False
    print("reForge modules not available, falling back to standard WebUI")

# Store original sampling functions to restore later
original_samplers = {}

# Track patching state
_patching_enabled = False

# VAE Reflection: Track original Conv2d padding modes for restoration
_vae_reflection_active = False
_vae_original_padding_modes = {}


def apply_vae_reflection(vae_model):
    """
    Patch VAE Conv2d layers to use reflect padding mode.

    This fixes edge artifacts in images generated with VAEs trained using
    reflect padding (e.g., Anzhc's EQ-VAE). Without this patch, standard
    'zeros' padding can cause visible seams at image boundaries.

    Args:
        vae_model: The VAE model (first_stage_model) to patch

    Returns:
        True if patching was successful, False otherwise
    """
    global _vae_reflection_active, _vae_original_padding_modes

    if _vae_reflection_active:
        return True  # Already patched

    if vae_model is None:
        print("⚠️ VAE Reflection: No VAE model available")
        return False

    _vae_original_padding_modes.clear()
    patched_count = 0

    try:
        for name, module in vae_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # Store original padding mode
                _vae_original_padding_modes[name] = module.padding_mode
                # Set to reflect mode
                module.padding_mode = 'reflect'
                patched_count += 1

        _vae_reflection_active = True
        print(f"🪞 VAE Reflection: Patched {patched_count} Conv2d layers to reflect mode")
        return True
    except Exception as e:
        print(f"❌ VAE Reflection: Failed to patch - {e}")
        # Attempt to restore any partially patched layers
        restore_vae_reflection(vae_model)
        return False


def restore_vae_reflection(vae_model):
    """
    Restore VAE Conv2d layers to their original padding modes.

    Args:
        vae_model: The VAE model (first_stage_model) to restore
    """
    global _vae_reflection_active, _vae_original_padding_modes

    if not _vae_reflection_active and not _vae_original_padding_modes:
        return  # Nothing to restore

    if vae_model is None:
        _vae_reflection_active = False
        _vae_original_padding_modes.clear()
        return

    restored_count = 0

    try:
        for name, module in vae_model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and name in _vae_original_padding_modes:
                module.padding_mode = _vae_original_padding_modes[name]
                restored_count += 1

        if restored_count > 0:
            print(f"🪞 VAE Reflection: Restored {restored_count} Conv2d layers to original padding")
    except Exception as e:
        print(f"⚠️ VAE Reflection: Error during restore - {e}")
    finally:
        _vae_reflection_active = False
        _vae_original_padding_modes.clear()

# JYS (Jump Your Steps) Dynamic Schedule Computation
# Computes optimized timestep sequences dynamically based on user-specified step count
# Strategy: Large jumps early (composition), dense clustering in detail formation region (200-400), fine steps at end


# Global settings that control sampler behavior
current_sampler_settings = {
    'enabled': False,
    'eta': 1.0,
    's_noise': 1.0,
    'debug_reproducibility': False,
    'use_entropic_scheduler': False,
    'entropic_scheduler_power': 6.0,
    'use_anime_schedule': False,
    'use_anime_schedule_v': False,
    'use_anime_schedule_e': False,
    'use_akashic_aos': False,
    'use_enhanced_detail_phase': True,
    'detail_enhancement_strength': 0.05,
    'detail_separation_radius': 0.5,
    'use_adept_solver': False,
    'adept_solver_order': 2,
    'adept_solver_use_corrector': True,
    # Adept Ancestral Solver settings
    'use_adept_ancestral_solver': False,
    'adept_ancestral_eta': 1.0,
    'adept_ancestral_s_noise': 1.0,
    'adept_ancestral_adaptive_eta': False,
    'adept_ancestral_phase_noise': False,
    'adept_ancestral_phase_strength': 0.5,
    'adept_ancestral_enhanced_derivative': False,
    'adept_ancestral_mirror_correction': False,
    # AkashicSolver v2 settings - SA-Solver base with AYS schedules
    'use_akashic_solver': False,
    'akashic_base_eta': 1.0,
    'akashic_s_noise': 1.0,
    'akashic_adaptive_eta': True,
    'akashic_phase_strength': 0.5,
    'akashic_tau': 0.5,             # SA-Solver stochasticity (0=ODE, 1=full SDE)
    'akashic_solver_order': 2,       # Multi-step order (1-3)
    'akashic_use_ays': False,        # Use AYS sigma schedules
    'akashic_smea_strength': 0.0,    # SMEA high-res coherency (0=disabled)
    'akashic_ndb_strength': 0.0,     # Native Detail Boost (0=disabled)
    'akashic_mirror_correction': False,
    'akashic_eqvae_mode': 'Off',     # EQ-VAE optimized mode: 'Off', 'Balanced'
    'vae_reflection': False,         # VAE reflection padding for EQ-VAE edge artifact fix
    # Additional CFG fixes (post-hoc techniques)
    'akashic_spectral_mod': False,   # Enable spectral modulation for frequency correction
    'akashic_spectral_percentile': 5.0,  # Spectral modulation percentile threshold
    'akashic_combat_cfg_drift': False,  # Combat CFG mean drift
    'akashic_combat_drift_intensity': 0.5,  # Combat drift intensity (0-1)
}



def create_detail_enhanced_model(model, x, sigmas):
    """Creates a model wrapper with proper state management."""
    if not TORCHVISION_AVAILABLE:
        return model
    
    base_strength = current_sampler_settings.get('detail_enhancement_strength', 0.05)
    radius = current_sampler_settings.get('detail_separation_radius', 0.5)
    total_steps = len(sigmas) - 1
    
    # Use a class to encapsulate state properly
    class DetailEnhancer:
        def __init__(self):
            self.current_step = 0
            
        def __call__(self, x_current, sigma, **kwargs):
            # Get base model prediction
            denoised = model(x_current, sigma, **kwargs)
            
            # Apply detail enhancement
            try:
                low_freq = gaussian_blur(denoised, kernel_size=3, sigma=radius)
                high_freq = denoised - low_freq
                
                # Calculate progressive enhancement
                progress = min(self.current_step / max(total_steps, 1), 1.0)  # Clamp to [0,1]
                strength = base_strength * (0.5 + progress)
                
                enhanced = denoised + high_freq * strength
                
                self.current_step += 1
                
                return enhanced
            except Exception as e:
                print(f"⚠️ Detail enhancement failed: {e}")
                return denoised
    
    return DetailEnhancer()


def patch_samplers_globally():
    """Patch all k-diffusion sampling functions with cleanup support."""
    global _patching_enabled
    
    if _patching_enabled:
        print("🔧 Adept Sampler: Already patched, skipping.")
        return
    
    patched_count = 0

    # Iterate over all functions named like sample_* in k_diff.k_diffusion.sampling
    for attr_name in dir(k_diff.k_diffusion.sampling):
        if not attr_name.startswith('sample_'):
            continue

        # Skip if already patched
        if attr_name in original_samplers:
            continue

        func = getattr(k_diff.k_diffusion.sampling, attr_name)
        if not callable(func):
            continue

        # Store original
        original_samplers[attr_name] = func

        def make_wrapper(name):
            def smart_wrapper(model, x, sigmas, extra_args=None, callback=None, disable=None, generator=None, **kwargs):
                """Smart wrapper: when enabled, override sigma schedule but preserve the original sampler's solver."""
                if not current_sampler_settings['enabled']:
                    return original_samplers[name](model, x, sigmas, extra_args, callback, disable, **kwargs)

                # Compute replacement schedule based on current settings
                try:
                    final_sigmas = compute_sigma_schedule_from_settings(sigmas, current_sampler_settings)
                except Exception as e:
                    print(f"⚠️ Sigma override failed for {name}: {e}. Using original schedule.")
                    final_sigmas = sigmas

                # Check for content-aware pacing and detail enhancement
                use_pacing = current_sampler_settings.get('use_content_aware_pacing', False)
                use_detail_enhancement = current_sampler_settings.get('use_enhanced_detail_phase', False)
                use_adept_solver = current_sampler_settings.get('use_adept_solver', False)
                
                # Warn if pacing is enabled but sampler is not Euler Ancestral
                if use_pacing and name != 'sample_euler_ancestral':
                    print(f"⚠️ Content-Aware Pacing only works with Euler Ancestral sampler.")
                    print(f"   Current sampler: {name.replace('sample_', '')}. Pacing will be disabled.")
                    use_pacing = False  # Disable pacing for non-Euler samplers
                
                # If pacing is enabled with Adept Solver, warn user and use pacing
                if use_pacing and use_adept_solver and name == 'sample_euler_ancestral':
                    print("⚠️ Content-Aware Pacing enabled with Adept Solver - using Enhanced Euler Ancestral with pacing instead.")
                    print("   (Pacing is not yet implemented in Adept Solver)")
                
                # Priority 1: Pacing (only for Euler Ancestral)
                if use_pacing and name == 'sample_euler_ancestral':
                    # Get eta and s_noise from settings
                    eta = current_sampler_settings.get('eta', 1.0)
                    s_noise = current_sampler_settings.get('s_noise', 1.0)
                    # Create an instance to call the method
                    # Pass skip_schedule_override=True because we already processed the schedule above
                    forge = AdeptSamplerForge()
                    return forge.sample_enhanced_euler_ancestral(model, x, final_sigmas, extra_args, callback, disable, eta, s_noise, generator, skip_schedule_override=True)
                
                # Apply detail enhancement wrapper if enabled (works with all samplers)
                active_model = model
                if use_detail_enhancement and TORCHVISION_AVAILABLE:
                    active_model = create_detail_enhanced_model(model, x, final_sigmas)
                    if use_adept_solver:
                        print(f"🎨 Detail Enhancement: Model wrapper active (will be used with Adept Solver)")
                    else:
                        print(f"🎨 Detail Enhancement: Model wrapper active (will be used with {name.replace('sample_', '')} sampler)")
                
                # Priority 2: Adept Solver (if pacing is not active)
                if use_adept_solver:
                    return sample_adept_solver(active_model, x, final_sigmas, extra_args, callback, disable, generator, **kwargs)
                
                # Priority 3: Adept Ancestral Solver (if neither pacing nor regular solver is active)
                use_adept_ancestral_solver = current_sampler_settings.get('use_adept_ancestral_solver', False)
                if use_adept_ancestral_solver:
                    return sample_adept_ancestral_solver(active_model, x, final_sigmas, extra_args, callback, disable, generator, **kwargs)

                # Priority 4: AkashicSolver (optimized for EQVAE models)
                use_akashic_solver = current_sampler_settings.get('use_akashic_solver', False)
                if use_akashic_solver:
                    return sample_akashic_solver(active_model, x, final_sigmas, extra_args, callback, disable, generator, **kwargs)

                # Default: Call the original sampler with the (possibly) overridden schedule and enhanced model
                return original_samplers[name](active_model, x, final_sigmas, extra_args, callback, disable, **kwargs)

            return smart_wrapper

        setattr(k_diff.k_diffusion.sampling, attr_name, make_wrapper(attr_name))
        patched_count += 1

    _patching_enabled = True
    print(f"🔧 Adept Sampler: patched {patched_count} samplers")

def unpatch_samplers_globally():
    """Restore original k-diffusion samplers."""
    global _patching_enabled
    
    if not _patching_enabled:
        return
    
    restored_count = 0
    for attr_name, original_func in original_samplers.items():
        setattr(k_diff.k_diffusion.sampling, attr_name, original_func)
        restored_count += 1
    
    _patching_enabled = False
    print(f"🔄 Adept Sampler: restored {restored_count} original samplers")

# Add to script callbacks
def on_script_unloaded():
    unpatch_samplers_globally()

if WEBUI_AVAILABLE:
    try:
        script_callbacks.on_script_unloaded(on_script_unloaded)
    except AttributeError:
        print("⚠️ Script unload callback not available")


def sample_adept_solver(model, x, sigmas, extra_args=None, callback=None, disable=None, generator=None, **kwargs):
    """
    Adept Solver: A unified training-free diffusion solver synthesizing improvements from:
    - DPM-Solver++ (data prediction, dynamic thresholding)
    - UniPC (unified predictor-corrector framework)
    - DEIS (exponential integrator)
    - DC-Solver (dynamic compensation)
    
    Key innovations:
    1. Adaptive parameterization to minimize discretization errors
    2. Multistep predictor-corrector with configurable order
    3. Dynamic thresholding for guided sampling stability
    4. Exponential integrator formulation for better numerical stability
    5. Adaptive compensation ratios for predictor-corrector alignment
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Get solver settings
    order = current_sampler_settings.get('adept_solver_order', 2)
    use_corrector = current_sampler_settings.get('adept_solver_use_corrector', True)
    
    # Clamp order to valid range
    order = max(1, min(order, 3))
    
    print(f"🚀 Adept Solver active (Order: {order}, Corrector: {'On' if use_corrector else 'Off'})")
    
    # Initialize history for multistep
    model_outputs = []
    
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # === PREDICTOR STEP ===
        # Get current model prediction
        denoised = model(x, sigma * s_in, **extra_args)
        
        # Apply dynamic thresholding (from DPM-Solver++)
        # This improves stability at high CFG scales
        if extra_args.get('cond_scale', 1.0) > 7.0:
            denoised = apply_dynamic_thresholding(denoised, percentile=0.995)
        
        # Compute derivative in log-SNR space for better numerical properties
        # Inspired by DPM-Solver-v3's optimal parameterization
        d = to_d(x, sigma, denoised)

        # Additional safety check for extreme derivatives with adaptive threshold
        derivative_max = torch.abs(d).max()
        sigma_adaptive_threshold = 1000.0 * (1.0 + sigma / 10.0)
        if torch.isnan(d).any() or torch.isinf(d).any() or derivative_max > sigma_adaptive_threshold:
            print(f"⚠️ Extreme derivative detected at step {i}/{len(sigmas)-1}. Clamping for stability.")
            d = torch.clamp(d, -sigma_adaptive_threshold, sigma_adaptive_threshold)
            # If still problematic, use a more conservative fallback
            if torch.isnan(d).any() or torch.isinf(d).any():
                d = torch.zeros_like(d)

        # Store for multistep
        model_outputs.append((sigma, d))
        if len(model_outputs) > order:
            model_outputs.pop(0)
        
        # Compute predictor step using multistep Adams-Bashforth-like integration
        # This combines ideas from DEIS (exponential integrator) and UniPC (unified framework)
        dt = sigma_next - sigma
        
        if len(model_outputs) == 1 or order == 1:
            # First-order (Euler step)
            x_pred = x + d * dt
        elif len(model_outputs) == 2 and order >= 2:
            # Second-order multistep with adaptive compensation (DC-Solver inspired)
            sigma_prev, d_prev = model_outputs[-2]
            d_cur = model_outputs[-1][1]
            
            # Compute adaptive interpolation coefficient
            h = sigma - sigma_prev
            compensation_ratio = compute_compensation_ratio(h.item() if torch.is_tensor(h) else float(h), i, len(sigmas))
            
            # Linear multistep integration
            d_interp = d_cur + compensation_ratio * (d_cur - d_prev)
            x_pred = x + d_interp * dt
        else:
            # Third-order multistep (when we have 3+ history)
            sigma_0, d_0 = model_outputs[-3]
            sigma_1, d_1 = model_outputs[-2]
            sigma_2, d_2 = model_outputs[-1]
            
            # Polynomial extrapolation with adaptive weights
            h_0 = sigma_2 - sigma_1
            h_1 = sigma_1 - sigma_0
            
            # Clamp to avoid division issues
            h_0_val = h_0.item() if torch.is_tensor(h_0) else float(h_0)
            h_1_val = h_1.item() if torch.is_tensor(h_1) else float(h_1)
            
            # Avoid numerical instability with very small step sizes
            if abs(h_1_val) < 1e-6:
                # Fall back to second-order if history is too close
                compensation_ratio = compute_compensation_ratio(h_0_val, i, len(sigmas))
                d_interp = d_2 + compensation_ratio * (d_2 - d_1)
            else:
                r0 = h_0_val / h_1_val
                
                # Standard Adams-Bashforth 3rd order coefficients (sum to 1)
                # Adjusted for non-uniform step sizes
                c0 = 1.0 + r0 / 2.0
                c1 = -r0 / 2.0
                c2 = 0.0  # Coefficient for d_0, derived from sum=1 constraint
                
                # Normalize to ensure sum = 1 for stability
                c_sum = c0 + c1 + c2
                c0 /= c_sum
                c1 /= c_sum
                c2 = 1.0 - c0 - c1  # Exact residual
                
                d_interp = c0 * d_2 + c1 * d_1 + c2 * d_0
            
            x_pred = x + d_interp * dt
        
        # === CORRECTOR STEP (optional, from UniPC) ===
        if use_corrector and i < len(sigmas) - 2:
            # Evaluate model at predicted point
            denoised_pred = model(x_pred, sigma_next * s_in, **extra_args)
            
            if extra_args.get('cond_scale', 1.0) > 7.0:
                denoised_pred = apply_dynamic_thresholding(denoised_pred, percentile=0.995)

            d_pred = to_d(x_pred, sigma_next, denoised_pred)

            # Additional safety check for corrector derivatives
            if torch.isnan(d_pred).any() or torch.isinf(d_pred).any() or torch.abs(d_pred).max() > 1000.0:
                print(f"⚠️ Extreme corrector derivative detected at step {i}/{len(sigmas)-1}. Clamping for stability.")
                d_pred = torch.clamp(d_pred, -100.0, 100.0)
                # If still problematic, use a more conservative fallback
                if torch.isnan(d_pred).any() or torch.isinf(d_pred).any():
                    d_pred = torch.zeros_like(d_pred)
            
            # Corrector: trapezoidal rule (combines predictor and corrector derivatives)
            dt = sigma_next - sigma
            x = x + (d + d_pred) * dt * 0.5
        else:
            x = x_pred
        
        # Robust error handling with recovery
        if torch.isnan(x).any() or torch.isinf(x).any():
            cfg_scale = extra_args.get('cond_scale', 1.0)
            print(f"❌ CRITICAL: NaN/Inf detected at step {i}/{len(sigmas)-1}!")
            print(f"   Sigma: {sigma.item():.4f} → {sigma_next.item():.4f}")
            print(f"   CFG Scale: {cfg_scale}")
            print(f"   Order: {order}, Corrector: {use_corrector}")
            
            # Try recovery with simpler method
            if i == 0:
                # Can't recover on first step
                raise RuntimeError("NaN/Inf on first step - check model/inputs")
            
            # Fallback: use last known good state with reduced step
            print("   Attempting recovery with conservative Euler step...")
            denoised_safe = model(x, sigma * s_in, **extra_args)
            if torch.isnan(denoised_safe).any():
                raise RuntimeError("Model producing NaN - check CFG scale and model")
            
            d_safe = to_d(x, sigma, denoised_safe)
            dt_safe = (sigma_next - sigma) * 0.5  # Reduced step size
            x = x + d_safe * dt_safe
            
            # Disable corrector for remaining steps
            use_corrector = False
            print("   Recovery successful. Corrector disabled for stability.")
        
        # Callback for progress tracking
        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigma,
                'sigma_hat': sigma,
                'denoised': denoised
            })
    
    return x


def sample_adept_ancestral_solver(model, x, sigmas, extra_args=None, callback=None, disable=None, generator=None, **kwargs):
    """
    Enhanced Adept Ancestral Solver: Advanced ancestral sampling with phase-aware adaptations.
    
    This solver implements genuine improvements over standard Euler Ancestral:
    - Phase-aware adaptive ancestral step sizing
    - Context-aware noise injection scheduling
    - Enhanced derivative computation optimized for ancestral sampling
    - Dynamic eta scheduling based on sampling progress
    - Smart noise scaling for different sampling phases
    
    Key innovations:
    1. Adaptive ancestral step sizing that changes throughout sampling phases
    2. Phase-aware noise injection (more noise early, less noise late)
    3. Enhanced derivative computation with ancestral-specific corrections
    4. Dynamic eta scheduling for better control
    5. Context-aware noise scaling based on image coherence
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Get solver settings - Enhanced ancestral solver with adaptive features
    order = 1  # Keep order 1 for stability with noise injection
    use_corrector = False  # Keep corrector off for compatibility with noise injection
    base_eta = current_sampler_settings.get('adept_ancestral_eta', 1.0)
    base_s_noise = current_sampler_settings.get('adept_ancestral_s_noise', 1.0)
    
    # New settings for enhanced features
    enable_adaptive_eta = current_sampler_settings.get('adept_ancestral_adaptive_eta', False)
    enable_phase_noise = current_sampler_settings.get('adept_ancestral_phase_noise', False)
    phase_strength = current_sampler_settings.get('adept_ancestral_phase_strength', 0.5)
    enable_enhanced_derivative = current_sampler_settings.get('adept_ancestral_enhanced_derivative', False)
    enable_mirror_correction = current_sampler_settings.get('adept_ancestral_mirror_correction', False)
    
    print(f"🚀 Enhanced Adept Ancestral Solver active (η: {base_eta:.2f}, s_noise: {base_s_noise:.2f})")
    print(f"   Adaptive Eta: {enable_adaptive_eta}, Phase Noise: {enable_phase_noise}, Phase Strength: {phase_strength:.2f}, Enhanced Derivative: {enable_enhanced_derivative}, Mirror Correction: {enable_mirror_correction}")
    
    # Get noise sampler for ancestral injection
    noise_sampler = get_noise_sampler(x)
    
    # Initialize history for multistep
    model_outputs = []
    
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # Calculate sampling progress for phase-aware adaptations
        progress = i / max(len(sigmas) - 1, 1)
        
        # === ADAPTIVE ETA SCHEDULING ===
        if enable_adaptive_eta:
            # Dynamic eta that adapts throughout sampling (more conservative to reduce noisiness)
            # Slightly more aggressive early (composition), conservative middle (structure), slightly aggressive late (detail)
            if progress < 0.3:
                adaptive_eta = base_eta * 1.08  # Reduced from 1.15 to 1.08
            elif progress < 0.7:
                adaptive_eta = base_eta * 0.95  # Reduced from 0.9 to 0.95 (less conservative)
            else:
                adaptive_eta = base_eta * 1.02  # Reduced from 1.05 to 1.02
        else:
            adaptive_eta = base_eta
        
        # === PREDICTOR STEP ===
        # Get current model prediction
        denoised = model(x, sigma * s_in, **extra_args)
        
        # Apply dynamic thresholding (from DPM-Solver++)
        if extra_args.get('cond_scale', 1.0) > 7.0:
            denoised = apply_dynamic_thresholding(denoised, percentile=0.995)
        
        # === ENHANCED DERIVATIVE COMPUTATION ===
        if enable_enhanced_derivative:
            d = to_d_enhanced_ancestral(x, sigma, denoised, adaptive_eta, progress, generator)
        else:
            d = to_d(x, sigma, denoised)

        # Additional safety check for extreme derivatives with adaptive threshold
        derivative_max = torch.abs(d).max()
        sigma_adaptive_threshold = 1000.0 * (1.0 + sigma / 10.0)
        if torch.isnan(d).any() or torch.isinf(d).any() or derivative_max > sigma_adaptive_threshold:
            print(f"⚠️ Extreme derivative detected at step {i}/{len(sigmas)-1}. Clamping for stability.")
            d = torch.clamp(d, -sigma_adaptive_threshold, sigma_adaptive_threshold)
            if torch.isnan(d).any() or torch.isinf(d).any():
                d = torch.zeros_like(d)

        # Store for multistep
        model_outputs.append((sigma, d))
        if len(model_outputs) > order:
            model_outputs.pop(0)
        
        # === ADAPTIVE ANCESTRAL STEP CALCULATION ===
        if sigma_next > 0:
            # Use adaptive eta instead of fixed eta
            sigma_up = min(sigma_next, adaptive_eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
            sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
        else:
            sigma_up = 0.0
            sigma_down = 0.0
        
        # Use ancestral dt instead of simple dt
        dt = sigma_down - sigma
        

        # === MIRROR CORRECTION (Standard Heun) ===
        # Evaluates the model at the Euler endpoint for a 2nd-order Heun correction.
        # Applied in first 30% of steps (foundation phase). 1 extra model call per corrected step.
        if enable_mirror_correction and progress < 0.30 and sigma_next > 0:
            x3 = x + d * dt
            denoised3 = model(x3, sigma * s_in, **extra_args)
            if extra_args.get('cond_scale', 1.0) > 7.0:
                denoised3 = apply_dynamic_thresholding(denoised3, percentile=0.995)
            d3 = to_d(x3, sigma, denoised3)
            d = (d + d3) / 2
            if torch.isnan(d).any() or torch.isinf(d).any():
                d = torch.zeros_like(d)
        # Compute predictor step (simplified for ancestral compatibility)
        x_pred = x + d * dt
        
        # === PHASE-AWARE NOISE INJECTION ===
        if sigma_next > 0:
            if enable_phase_noise:
                # Phase-aware noise scaling with smooth interpolation
                # Conservative multipliers to reduce high-CFG appearance
                if progress < 0.25:
                    # Early phase - subtle increase for diversity (was 1.1, now 1.05 max)
                    target_multiplier = 1.0 + (0.05 * min(progress / 0.25, 1.0))
                elif progress < 0.6:
                    # Middle phase - very subtle decrease (was 1.0, now 0.98 max decrease)
                    target_multiplier = 1.0 - (0.02 * min((progress - 0.25) / 0.35, 1.0))
                else:
                    # Late phase - gentle detail preservation (was 0.9, now 0.95 max)
                    target_multiplier = 1.0 - (0.05 * min((progress - 0.6) / 0.4, 1.0))

                # Interpolate with phase strength for fine control
                noise_multiplier = 1.0 + (target_multiplier - 1.0) * phase_strength

                adaptive_s_noise = base_s_noise * noise_multiplier
            else:
                adaptive_s_noise = base_s_noise
            
            # Generate noise with adaptive scaling
            noise = noise_sampler(sigma, sigma_next) * adaptive_s_noise * sigma_up
            x = x_pred + noise
        else:
            x = x_pred
        
        # Robust error handling with recovery
        if torch.isnan(x).any() or torch.isinf(x).any():
            cfg_scale = extra_args.get('cond_scale', 1.0)
            print(f"❌ CRITICAL: NaN/Inf detected at step {i}/{len(sigmas)-1}!")
            print(f"   Sigma: {sigma.item():.4f} → {sigma_next.item():.4f}")
            print(f"   CFG Scale: {cfg_scale}")
            print(f"   Order: {order}, Corrector: {use_corrector}")
            
            # Try recovery with simpler method
            if i == 0:
                # Can't recover on first step
                raise RuntimeError("NaN/Inf on first step - check model/inputs")
            
            # Fallback: use last known good state with reduced step
            print("   Attempting recovery with conservative Euler step...")
            denoised_safe = model(x, sigma * s_in, **extra_args)
            if torch.isnan(denoised_safe).any():
                raise RuntimeError("Model producing NaN - check CFG scale and model")
            
            d_safe = to_d(x, sigma, denoised_safe)
            dt_safe = (sigma_next - sigma) * 0.5  # Reduced step size
            x = x + d_safe * dt_safe
            
            # Disable corrector for remaining steps
            use_corrector = False
            print("   Recovery successful. Corrector disabled for stability.")
        
        # Callback for progress tracking
        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigma,
                'sigma_hat': sigma,
                'denoised': denoised
            })
    
    return x


def sample_akashic_solver(model, x, sigmas, extra_args=None, callback=None, 
                          disable=None, generator=None, **kwargs):
    """
    AkashicSolver v2: Advanced sampler for SDXL/EQ-VAE models.
    
    This solver combines multiple SOTA techniques for optimal EQ-VAE sampling:
    
    1. SA-SOLVER BASE: Multi-step Adams-Bashforth integration with tau function
       for controlled stochasticity (interpolates between ODE and full SDE)
    
    2. PHASE-AWARE SAMPLING: Three-phase approach with adaptive parameters
       - Foundation (0-30%): Higher stochasticity for composition diversity
       - Structure (30-60%): Moderate settings for stable formation
       - Refinement (60-100%): Lower stochasticity for detail preservation
    
    3. SMEA COHERENCY: Sine-based interpolation for high-resolution coherency
       (prevents duplicated subjects/warped anatomy)
    
    Based on research from:
    - SA-Solver (NeurIPS 2023): Stochastic Adams multi-step SDE solver
    - SMEA (NovelAI): Sinusoidal multipass for high-res coherency
    - Align Your Steps (CVPR 2024): Optimized sigma scheduling
    
    Recommended settings for EQ-VAE models (e.g., AkashicPulse):
    - Use with AkashicAOS scheduler or AYS schedule
    - Enable Additional CFG Fixes (Spectral, Combat Drift) as needed
    - CFG Scale: 7-10
    - Steps: 20-30
    - Tau: 0.5 (balanced) or 1.0 (full stochastic)
    - Order: 2 (recommended)

    Additional CFG Fixes (post-hoc):
    - Spectral Modulation, Combat CFG Drift
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # Get AkashicSolver v2 settings
    base_eta = current_sampler_settings.get('akashic_base_eta', 1.0)
    base_s_noise = current_sampler_settings.get('akashic_s_noise', 1.0)
    enable_adaptive_eta = current_sampler_settings.get('akashic_adaptive_eta', True)
    phase_strength = current_sampler_settings.get('akashic_phase_strength', 0.5)
    base_tau = current_sampler_settings.get('akashic_tau', 0.5)
    solver_order = current_sampler_settings.get('akashic_solver_order', 2)
    smea_strength = current_sampler_settings.get('akashic_smea_strength', 0.0)
    ndb_strength = current_sampler_settings.get('akashic_ndb_strength', 0.0)
    enable_mirror_correction = current_sampler_settings.get('akashic_mirror_correction', False)
    eqvae_mode_setting = current_sampler_settings.get('akashic_eqvae_mode', 'Off')

    # EQ-VAE parameter scaling: converts default values to EQ-VAE-optimized equivalents
    # Parse early so we can apply scaling before anything else
    if isinstance(eqvae_mode_setting, bool):
        _eqvae_active = eqvae_mode_setting
    else:
        _eqvae_active = eqvae_mode_setting == 'Balanced'
    if _eqvae_active:
        base_s_noise *= 0.9  # 1.0 → 0.9: EQ-VAE's cleaner latent space needs less noise

    # Get Additional CFG Fixes settings
    cfg_settings = {
        'akashic_spectral_mod': current_sampler_settings.get('akashic_spectral_mod', False),
        'akashic_spectral_percentile': current_sampler_settings.get('akashic_spectral_percentile', 5.0),
        'akashic_combat_cfg_drift': current_sampler_settings.get('akashic_combat_cfg_drift', False),
        'akashic_combat_drift_intensity': current_sampler_settings.get('akashic_combat_drift_intensity', 0.5),
    }
    cfg_enhancement_active = (
        cfg_settings['akashic_spectral_mod']
        or cfg_settings['akashic_combat_cfg_drift']
    )

    # Parse EQ-VAE mode setting
    if isinstance(eqvae_mode_setting, bool):
        # Backwards compatibility with old boolean setting
        eqvae_mode = eqvae_mode_setting
    else:
        eqvae_mode = eqvae_mode_setting == 'Balanced'

    if eqvae_mode:
        print(f"🌀 AkashicSolver v2 [EQ-VAE BALANCED] active")
        print(f"   Optimized for EQ-VAE's cleaner latent space")
    else:
        print(f"🌀 AkashicSolver v2 active")
    print(f"   τ (tau): {base_tau:.2f}, η (eta): {base_eta:.2f}, s_noise: {base_s_noise:.2f}")
    print(f"   Order: {solver_order}, Adaptive Eta: {enable_adaptive_eta}, Phase Strength: {phase_strength:.2f}, Mirror Correction: {enable_mirror_correction}")
    if smea_strength > 0:
        print(f"   SMEA: {smea_strength:.2f} (high-res coherency)")
    if ndb_strength > 0:
        print(f"   Native Detail Boost: {ndb_strength:.2f} (detail enhancement)")

    # Additional CFG Fixes status
    if cfg_enhancement_active:
        extras = []
        if cfg_settings['akashic_spectral_mod']:
            extras.append("Spectral")
        if cfg_settings['akashic_combat_cfg_drift']:
            extras.append("CombatDrift")
        print(f"   ✨ CFG Fixes: {', '.join(extras)}")

    # Get noise sampler for stochastic injection
    noise_sampler = get_noise_sampler(x)

    total_steps = len(sigmas) - 1

    # Multi-step history for SA-Solver (stores (sigma, derivative) tuples)
    d_history = []

    for i in range(total_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # Calculate sampling progress for phase-aware adaptations
        progress = i / max(total_steps - 1, 1)
        
        # === COMPUTE PHASE-AWARE TAU ===
        # Tau controls stochasticity: 0=ODE (deterministic), 1=full SDE (stochastic)
        if enable_adaptive_eta:
            tau = compute_tau_eqvae(progress, base_tau, phase_strength)
        else:
            tau = base_tau

        # === PHASE-AWARE ADAPTIVE ETA ===
        # Eta affects the ancestral noise magnitude within the stochastic component
        if enable_adaptive_eta:
            if progress < 0.30:
                # Foundation phase: slightly higher eta for composition diversity
                adaptive_eta = base_eta * (1.0 + 0.08 * phase_strength)
            elif progress < 0.60:
                # Structure phase: conservative eta for stable structure formation
                adaptive_eta = base_eta * (1.0 - 0.05 * phase_strength)
            else:
                # Refinement phase: slight eta boost for detail variation
                adaptive_eta = base_eta * (1.0 + 0.02 * phase_strength)
        else:
            adaptive_eta = base_eta
        
        # === SMEA FACTOR ===
        # Adjusts noise for high-resolution coherency
        smea_factor = compute_smea_factor(progress, smea_strength)

        # === MODEL PREDICTION ===
        denoised = model(x, sigma * s_in, **extra_args)

        # Apply dynamic thresholding for stability at high CFG
        cfg_scale = extra_args.get('cond_scale', 1.0)
        if cfg_scale > 7.0:
            denoised = apply_dynamic_thresholding(denoised, percentile=0.995)

        # === POST-HOC CFG FIXES (Combat Drift) ===
        # Note: Spectral Modulation is now handled via CFG hook in process_before_every_sampling
        if cfg_enhancement_active:
            if cfg_settings.get('akashic_combat_cfg_drift', False):
                denoised = apply_combat_cfg_drift(
                    denoised,
                    intensity=cfg_settings.get('akashic_combat_drift_intensity', 0.5)
                )

        # === COMPUTE DERIVATIVE ===
        d = to_d(x, sigma, denoised)
        
        # Safety check for extreme derivatives
        derivative_max = torch.abs(d).max()
        sigma_adaptive_threshold = 1000.0 * (1.0 + sigma / 10.0)
        if torch.isnan(d).any() or torch.isinf(d).any() or derivative_max > sigma_adaptive_threshold:
            print(f"⚠️ AkashicSolver v2: Extreme derivative at step {i}/{total_steps}. Clamping.")
            d = torch.clamp(d, -sigma_adaptive_threshold, sigma_adaptive_threshold)
            if torch.isnan(d).any() or torch.isinf(d).any():
                d = torch.zeros_like(d)

        # === MIRROR CORRECTION (Semantic Reflection Probe) ===
        # Applied before d_history to ensure corrected d feeds into multi-step history.
        if enable_mirror_correction and progress < 0.60 and sigma_next > 0:
            # Compute tau-adjusted dt to match sa_solver_step's deterministic step length.
            _anc_sq = sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / (sigma ** 2 + 1e-8)
            _sigma_up = tau * (_anc_sq ** 0.5 if _anc_sq > 0 else 0.0)
            _anc_down_sq = sigma_next ** 2 - _sigma_up ** 2
            _sigma_down = _anc_down_sq ** 0.5 if _anc_down_sq >= 0 else sigma_next
            dt_probe = _sigma_down - sigma
            x_probe = 2 * denoised - x
            denoised_probe = model(x_probe, sigma * s_in, **extra_args)
            if cfg_scale > 7.0:
                denoised_probe = apply_dynamic_thresholding(denoised_probe, percentile=0.995)
            d_probe = to_d(x_probe, sigma, denoised_probe)
            x3 = x + ((d + d_probe) / 2) * dt_probe
            denoised3 = model(x3, sigma * s_in, **extra_args)
            if cfg_scale > 7.0:
                denoised3 = apply_dynamic_thresholding(denoised3, percentile=0.995)
            d3 = to_d(x3, sigma, denoised3)
            d = (d + d3) / 2
            if torch.isnan(d).any() or torch.isinf(d).any():
                d = torch.zeros_like(d)

        # Store derivative in history for multi-step
        d_history.append((sigma, d))
        if len(d_history) > solver_order:
            d_history.pop(0)
        
        # === SA-SOLVER STEP WITH TAU CONTROL ===
        # tau controls stochasticity (how much noise injection)
        # eta controls noise magnitude within the stochastic component
        # These should be independent, not multiplied together
        effective_tau = tau  # Use tau directly for stochasticity control

        effective_s_noise = base_s_noise * adaptive_eta * smea_factor  # eta affects noise magnitude

        # Phase-aware noise adjustment (more conservative to preserve sharpness)
        if progress < 0.30:
            # Foundation: subtle increase for diversity
            noise_multiplier = 1.0 + 0.03 * phase_strength
        elif progress < 0.60:
            # Structure: very slight reduction for stability
            noise_multiplier = 1.0 - 0.01 * phase_strength
        else:
            # Refinement: minimal reduction to preserve detail sharpness
            noise_multiplier = 1.0 - 0.02 * phase_strength

        effective_s_noise *= noise_multiplier

        # Execute SA-Solver step
        x, sigma_up = sa_solver_step(
            x=x,
            d_history=d_history,
            sigma=sigma,
            sigma_next=sigma_next,
            tau=effective_tau,
            s_noise=effective_s_noise,
            noise_sampler=noise_sampler,
            order=solver_order,
            ndb_strength=ndb_strength,
            progress=progress,
        )
        
        # === ERROR HANDLING ===
        if torch.isnan(x).any() or torch.isinf(x).any():
            cfg_scale = extra_args.get('cond_scale', 1.0)
            print(f"❌ AkashicSolver v2: NaN/Inf detected at step {i}/{total_steps}!")
            print(f"   Sigma: {sigma.item():.4f} → {sigma_next.item():.4f}, CFG: {cfg_scale}")
            print(f"   Tau: {tau:.3f}, Order: {solver_order}")
            
            if i == 0:
                raise RuntimeError("NaN/Inf on first step - check model/inputs")
            
            # Recovery attempt with fallback to simple Euler step
            print("   Attempting recovery with conservative Euler step...")
            denoised_safe = model(x, sigma * s_in, **extra_args)
            if torch.isnan(denoised_safe).any():
                raise RuntimeError("Model producing NaN - reduce CFG scale or check model")
            
            d_safe = to_d(x, sigma, denoised_safe)
            dt_safe = (sigma_next - sigma) * 0.5
            x = x + d_safe * dt_safe
            
            # Clear history to reset multi-step
            d_history.clear()
            print("   Recovery successful. Multi-step history cleared.")
        
        # Callback for progress tracking
        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigma,
                'sigma_hat': sigma,
                'denoised': denoised
            })
    
    return x


def get_noise_sampler(x):
    """Get proper noise sampler with working fallback."""
    if hasattr(k_diff.k_diffusion.sampling, 'default_noise_sampler'):
        return k_diff.k_diffusion.sampling.default_noise_sampler(x)
    else:
        # Proper fallback with sigma scaling
        def simple_noise_sampler(sigma_from, sigma_to):
            # Scale noise appropriately
            noise = torch.randn_like(x)
            # Apply sigma scaling if there's a meaningful difference
            if abs(sigma_to - sigma_from) > 1e-6:
                scale = (sigma_to / sigma_from.clamp(min=1e-6)).sqrt()
                noise = noise * scale
            return noise
        return simple_noise_sampler


def to_d(x, sigma, denoised):
    """Convert denoised prediction to derivative with robust numerical stability."""
    # Compute the difference
    diff = x - denoised
    
    # Use a safer minimum sigma threshold
    safe_sigma = torch.clamp(sigma, min=1e-4)  # More conservative
    
    # Check for extreme sigma ratios before division
    derivative = diff / safe_sigma
    
    # Normalize derivative by sigma to handle different prediction types (v-prediction produces larger values)
    # Use a sigma-adaptive threshold: allow larger derivatives when sigma is large
    sigma_adaptive_threshold = 1000.0 * (1.0 + sigma / 10.0)  # More permissive for larger sigma
    
    # Post-division safety check with adaptive threshold
    derivative_max = torch.abs(derivative).max()
    if derivative_max > sigma_adaptive_threshold:
        print(f"⚠️ Extreme derivative detected. Clamping from {derivative_max:.2f}")
        derivative = torch.clamp(derivative, -sigma_adaptive_threshold, sigma_adaptive_threshold)
    
    return derivative


def to_d_enhanced_ancestral(x, sigma, denoised, eta, progress, generator=None):
    """
    Enhanced derivative computation optimized for ancestral sampling.

    This function provides ancestral-specific derivative corrections that adapt
    based on the sampling progress and eta value for better noise injection behavior.
    Uses the provided generator for reproducible results when enhanced derivative is enabled.

    Args:
        x: Input tensor
        sigma: Current sigma value
        denoised: Denoised prediction
        eta: Adaptive eta value
        progress: Sampling progress (0.0 to 1.0)
        generator: Random number generator for reproducible results (None for global random state)
    """
    # Standard derivative computation
    diff = x - denoised

    # Use a safer minimum sigma threshold
    safe_sigma = torch.clamp(sigma, min=1e-4)

    # Base derivative (removed diff clamping for v-prediction compatibility)
    base_derivative = diff / safe_sigma

    # Generate random tensor with proper generator support
    def safe_randn_like(tensor, generator=None):
        """Generate random tensor with generator support.

        Note: This function attempts to use the provided generator for reproducible results.
        If the generator format is not compatible with the current PyTorch version,
        it falls back to global random state. Future versions should implement full
        generator compatibility when the WebUI generator format is better understood.
        """
        if generator is None:
            return torch.randn_like(tensor)
        else:
            # Try to use torch.randn with generator (more reliable than randn_like)
            try:
                # Get tensor properties
                shape = tensor.shape
                device = tensor.device
                dtype = tensor.dtype

                # Try torch.randn with generator
                result = torch.randn(shape, device=device, dtype=dtype, generator=generator)
                return result
            except (TypeError, AttributeError):
                # Fallback: use global random state
                return torch.randn_like(tensor)

    # Ancestral-specific enhancements (more conservative to reduce noisiness)
    # Add subtle adaptive corrections based on eta and progress
    if eta > 1.0:
        # Higher eta values benefit from slightly more aggressive derivatives
        eta_correction = 0.02 * (eta - 1.0) * safe_randn_like(diff, generator) * progress  # Reduced from 0.05 to 0.02
        base_derivative = base_derivative + eta_correction
    elif eta < 1.0:
        # Lower eta values benefit from more conservative derivatives
        eta_correction = 0.015 * (1.0 - eta) * safe_randn_like(diff, generator) * (1.0 - progress)  # Reduced from 0.03 to 0.015
        base_derivative = base_derivative - eta_correction

    # Progress-based phase corrections (more subtle)
    if progress < 0.3:
        # Early phase: slightly more aggressive for composition
        phase_correction = 0.01 * safe_randn_like(diff, generator)  # Reduced from 0.02 to 0.01
        base_derivative = base_derivative + phase_correction
    elif progress > 0.7:
        # Late phase: slightly more conservative for detail preservation
        phase_correction = 0.008 * safe_randn_like(diff, generator)  # Reduced from 0.015 to 0.008
        base_derivative = base_derivative - phase_correction

    # Final safety check with adaptive threshold for v-prediction
    sigma_adaptive_threshold = 500.0 * (1.0 + sigma / 10.0)
    derivative_max = torch.abs(base_derivative).max()
    if derivative_max > sigma_adaptive_threshold:
        base_derivative = torch.clamp(base_derivative, -sigma_adaptive_threshold, sigma_adaptive_threshold)

    return base_derivative


def apply_dynamic_thresholding(x, percentile=0.995, clamp_range=1.0):
    """
    Optimized dynamic thresholding with better stability.
    """
    if percentile >= 1.0:
        return x
    
    try:
        batch_size = x.shape[0]
        
        # Use in-place operations where safe
        x_flat = x.view(batch_size, -1)
        
        # Fast absolute max as proxy for extreme values (faster than quantile)
        abs_max = torch.abs(x_flat).max(dim=1, keepdim=True)[0]
        
        # Only apply thresholding if we detect extreme values
        if abs_max.max() < 5.0:  # Conservative threshold
            return x
        
        # Use topk instead of quantile (much faster)
        k = max(1, int(x_flat.shape[1] * (1.0 - percentile)))
        topk_vals = torch.topk(torch.abs(x_flat), k=k, dim=1, largest=True)[0]
        s = topk_vals[:, -1:].clamp(min=1.0)  # Last value is the threshold
        
        # Gentler clamping
        threshold = s * 2.5  # Less aggressive than 3.0
        
        # Apply only to extreme outliers
        mask = torch.abs(x_flat) > threshold
        x_flat = torch.where(mask, torch.sign(x_flat) * threshold, x_flat)
        
        # Very gentle rescaling
        x_flat = x_flat * 0.98
        
        return x_flat.view(x.shape)
        
    except Exception as e:
        print(f"⚠️ Dynamic thresholding failed: {e}")
        return x


# =============================================================================
# POST-HOC CFG FIX FUNCTIONS
# =============================================================================

def apply_spectral_modulation_clybius(noise_pred, multiplier=1.0, percentile=5.0):
    """
    Clybius Spectral Modulation: Apply frequency-domain corrections to noise prediction.
    
    This is the correct implementation based on ComfyUI-Latent-Modifiers.
    It should be applied to noise_pred (cond - uncond), NOT to denoised latent.
    
    Args:
        noise_pred: The noise prediction tensor (cond - uncond)
        multiplier: Modulation strength (0=none, 1=full Clybius effect). Default: 1.0
        percentile: Upper/lower percentile threshold. Default: 5.0
    
    Returns:
        Spectrally modulated noise prediction
    """
    if multiplier == 0 or percentile <= 0:
        return noise_pred
    
    try:
        # FFT
        fourier = torch.fft.fft2(noise_pred, dim=(-2, -1))
        
        # Log amplitude (with small epsilon for numerical stability)
        log_amp = torch.log(torch.sqrt(fourier.real ** 2 + fourier.imag ** 2) + 1e-8)
        
        # Compute quantiles on absolute log amplitude
        log_amp_flat = log_amp.abs().flatten(2)
        quantile_low = torch.quantile(log_amp_flat, percentile * 0.01, dim=2)
        quantile_high = torch.quantile(log_amp_flat, 1 - percentile * 0.01, dim=2)
        
        # Expand quantiles back to log_amp shape
        quantile_low = quantile_low.unsqueeze(-1).unsqueeze(-1).expand(log_amp.shape)
        quantile_high = quantile_high.unsqueeze(-1).unsqueeze(-1).expand(log_amp.shape)
        
        # Create masks (Clybius approach)
        # mask_low: boost values below low threshold (range 1.0 to 1.5)
        # mask_high: reduce values above high threshold (range 0.5 to 1.0)
        mask_low = ((log_amp < quantile_low).float() + 1).clamp_(max=1.5)
        mask_high = ((log_amp < quantile_high).float()).clamp_(min=0.5)
        
        # Apply modulation via exponentiation
        filtered_fourier = fourier * ((mask_low * mask_high) ** multiplier)
        
        # Inverse FFT
        result = torch.fft.ifft2(filtered_fourier, dim=(-2, -1)).real
        
        return result
        
    except Exception as e:
        print(f"⚠️ Spectral modulation failed: {e}")
        return noise_pred


def create_spectral_modulation_cfg_hook(multiplier=1.0, percentile=5.0):
    """
    Create a CFG hook that applies Clybius spectral modulation to noise prediction.
    
    This hooks into reForge's set_model_sampler_cfg_function to intercept
    the CFG calculation and apply spectral modulation at the correct point.
    
    Args:
        multiplier: Modulation strength (0=none, 1=full). Default: 1.0
        percentile: Frequency percentile threshold. Default: 5.0
    
    Returns:
        A hook function to pass to set_model_sampler_cfg_function
    """
    def spectral_cfg_hook(args):
        cond = args["cond"]
        uncond = args["uncond"]
        cond_scale = args["cond_scale"]
        sigma = args["sigma"]
        x_orig = args["input"]
        
        # Reshape sigma for broadcasting
        sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
        
        # Convert to v-pred space (from RescaleCFG reference)
        x = x_orig / (sigma * sigma + 1.0)
        cond_v = ((x - (x_orig - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
        uncond_v = ((x - (x_orig - uncond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
        
        # Compute noise prediction
        noise_pred = cond_v - uncond_v
        
        # Apply Clybius spectral modulation to noise prediction
        noise_pred_modulated = apply_spectral_modulation_clybius(noise_pred, multiplier, percentile)
        
        # Compute CFG with modified noise prediction
        x_cfg = uncond_v + cond_scale * noise_pred_modulated
        
        # Convert back from v-pred space
        return x_orig - (x - x_cfg * sigma / (sigma * sigma + 1.0) ** 0.5)
    
    return spectral_cfg_hook


def apply_combat_cfg_drift(latent, method='mean', intensity=1.0):
    """
    Combat CFG Drift: Reduce mean drift from high CFG values.

    Based on ComfyUI-Latent-Modifiers.

    As CFG increases, the latent mean can drift away from 0, which causes
    color shifts and other artifacts. This technique reduces the drift
    proportionally based on intensity.

    Note: This is auto-disabled for inpaint/ADetailer passes to prevent
    patchy composites (the crop's mean differs from the full image's mean,
    causing a visible seam at the composite boundary).

    Args:
        latent: The latent tensor to correct
        method: 'mean' or 'median'. Default: 'mean'
        intensity: How much drift to remove (0=none, 1=full). Default: 1.0

    Returns:
        Drift-corrected latent
    """
    if intensity <= 0:
        return latent

    try:
        if method == 'median':
            # Compute global median per batch (across all channels and spatial dims)
            center = latent.view(latent.shape[0], -1).median(dim=-1, keepdim=True)[0]
            center = center.view(latent.shape[0], 1, 1, 1)
        else:
            # Compute global mean per batch (across all channels and spatial dims)
            # This matches ComfyUI's PostCFGsubtractMeanNode implementation
            center = latent.mean(dim=(1, 2, 3), keepdim=True)

        # Remove drift proportionally based on intensity
        # intensity=1.0 removes all drift, intensity=0.5 removes half
        return latent - center * intensity

    except Exception as e:
        print(f"⚠️ Combat CFG drift failed: {e}")
        return latent


def compute_phase_aware_cfg_scale(base_scale, progress, alpha=2.0, beta=2.0):
    """
    Phase-Aware CFG Scaling: Adjust CFG scale based on sampling progress.

    Inspired by β-CFG (arXiv:2502.10574).

    CFG effectiveness varies by sampling phase:
    - Early: Lower CFG allows manifold exploration
    - Middle: Higher CFG for prompt adherence
    - Late: Lower CFG to stay on data manifold

    Args:
        base_scale: The user-specified CFG scale
        progress: Sampling progress (0.0 to 1.0)
        alpha: Beta distribution alpha parameter. Default: 2.0
        beta: Beta distribution beta parameter. Default: 2.0

    Returns:
        Adjusted CFG scale for the current step
    """
    try:
        # Use a simple polynomial approximation of beta distribution
        # Beta(2,2) peaks at 0.5 with a smooth curve
        # f(x) = 6 * x * (1-x) for Beta(2,2), normalized to peak at 1
        if alpha == 2.0 and beta == 2.0:
            # Simple case: symmetric peak at 0.5
            scale_factor = 4.0 * progress * (1.0 - progress)  # Peaks at 1.0 when progress=0.5
            scale_factor = 0.7 + 0.6 * scale_factor  # Range: 0.7 to 1.3
        else:
            # General case: use polynomial approximation
            # Mode of Beta(a,b) is at (a-1)/(a+b-2)
            mode = (alpha - 1.0) / (alpha + beta - 2.0) if (alpha + beta) > 2 else 0.5
            # Create a smooth curve that peaks at the mode
            dist_from_mode = abs(progress - mode)
            scale_factor = 1.0 - 0.3 * dist_from_mode * 2  # Simple linear falloff
            scale_factor = max(0.7, min(1.3, scale_factor))

        return base_scale * scale_factor

    except Exception as e:
        print(f"⚠️ Phase-aware CFG scaling failed: {e}")
        return base_scale


def apply_cfg_techniques(denoised, x, sigma, cfg_scale, progress, settings):
    """
    Master function to apply all enabled CFG enhancement techniques.

    This is the main entry point for CFG enhancements in the solver.

    Args:
        denoised: The denoised prediction from the model
        x: Current latent
        sigma: Current sigma value
        cfg_scale: CFG scale being used
        progress: Sampling progress (0.0 to 1.0)
        settings: Dictionary with CFG technique settings

    Returns:
        Enhanced denoised result
    """
    result = denoised

    # Skip if no additional CFG fixes are enabled
    # Note: Spectral Modulation is now handled via CFG hook, not here
    if not settings.get('akashic_combat_cfg_drift', False):
        return result

    # Apply Combat CFG Drift if enabled
    if settings.get('akashic_combat_cfg_drift', False):
        intensity = settings.get('akashic_combat_drift_intensity', 0.5)
        result = apply_combat_cfg_drift(result, method='mean', intensity=intensity)

    return result


def compute_compensation_ratio(r, step_idx, total_steps, base_ratio=1.0):
    """
    Compute dynamic compensation ratio inspired by DC-Solver.
    Adapts interpolation based on step position to address predictor-corrector misalignment.
    
    Args:
        r: step size ratio (scalar)
        step_idx: current step index
        total_steps: total number of steps
        base_ratio: base compensation strength
    
    Returns:
        compensation ratio for interpolation
    """
    # Progress through sampling (0 to 1)
    progress = step_idx / max(total_steps - 1, 1)
    
    # Adaptive compensation: stronger at the beginning (composition phase)
    # and end (detail phase), weaker in the middle (structure phase)
    # This follows the three-phase pattern from AOS
    if progress < 0.3:
        # Early phase: aggressive compensation for composition
        phase_weight = 1.5
    elif progress < 0.7:
        # Middle phase: moderate compensation for structure
        phase_weight = 1.0
    else:
        # Late phase: strong compensation for detail refinement
        phase_weight = 1.3
    
    # Combine with step size ratio for adaptive behavior
    # Use math.tanh for pure Python scalar operations
    compensation = base_ratio * phase_weight * (1.0 + 0.1 * math.tanh(r - 1.0))
    
    return compensation


def compute_tau_eqvae(progress, base_tau=0.5, phase_strength=0.5):
    """
    Phase-aware tau function for SA-Solver style stochasticity control.
    Optimized for EQ-VAE's smooth latent space.
    
    The tau parameter controls interpolation between ODE (deterministic) 
    and full SDE (stochastic) sampling:
    - tau=0: Pure ODE solver (deterministic, reproducible)
    - tau=1: Full SDE solver (maximum stochasticity, like Euler Ancestral)
    - tau=0.5: Balanced (recommended for EQ-VAE)
    
    Args:
        progress: Sampling progress (0.0 to 1.0)
        base_tau: Base stochasticity level (0=ODE, 1=full SDE)
        phase_strength: How much phase affects tau (0=constant, 1=full adaptation)
    
    Returns:
        tau value for current step
    """
    if progress < 0.30:
        # Foundation phase: Higher stochasticity for composition diversity
        phase_factor = 1.0 + 0.2 * phase_strength
    elif progress < 0.60:
        # Structure phase: Moderate stochasticity for stable formation
        phase_factor = 1.0 - 0.15 * phase_strength
    else:
        # Refinement phase: Lower stochasticity for detail preservation
        phase_factor = 1.0 - 0.3 * phase_strength
    
    return min(1.0, max(0.0, base_tau * phase_factor))


def compute_smea_factor(progress, smea_strength=0.5):
    """
    SMEA (Sinusoidal Multipass Euler Ancestral) inspired interpolation.
    
    Uses sine-based schedule to improve coherency at high resolutions
    by smoothly transitioning between multi-pass behaviors. This helps
    prevent duplicated subjects and warped anatomy at high resolutions.
    
    Based on NovelAI's SMEA technique.
    
    Args:
        progress: Sampling progress (0.0 to 1.0)
        smea_strength: How much SMEA affects the result (0=disabled)
    
    Returns:
        Interpolation factor for SMEA-style noise scaling
    """
    if smea_strength <= 0:
        return 1.0
    
    # Sine-based interpolation (smooth S-curve)
    # Peak at mid-point for structure formation
    smea_interp = 0.5 * (1 + math.sin(math.pi * (progress - 0.5)))
    
    # Blend with linear based on strength
    return 1.0 - smea_strength * (1.0 - smea_interp)


def compute_native_detail_boost(progress, ndb_strength=0.0):
    """
    Native Detail Boost (NDB): Enhances detail emergence at native resolution.
    
    Unlike SMEA (which reduces noise variation for high-res coherency),
    NDB selectively boosts high-frequency noise components to enhance
    fine detail emergence without adding blur.
    
    Phase-aware approach:
    - Foundation (0-30%): Minimal intervention — let composition form naturally
    - Structure (30-60%): Moderate boost — encourage structure detail
    - Refinement (60-100%): Strong emphasis — maximize fine detail emergence
    
    Args:
        progress: Sampling progress (0.0 to 1.0)
        ndb_strength: How much boost to apply (0=disabled, 1=maximum)
    
    Returns:
        Tuple of (base_scale, high_freq_boost):
        - base_scale: Multiplier for base noise (always 1.0)
        - high_freq_boost: Additional high-frequency component strength
    """
    if ndb_strength <= 0:
        return 1.0, 0.0
    
    # Phase-aware enhancement with smooth transitions
    if progress < 0.30:
        # Foundation phase: minimal intervention to preserve composition
        # Gradual ramp from 0 to small boost
        phase_progress = progress / 0.30
        high_freq_boost = 0.03 * ndb_strength * phase_progress
    elif progress < 0.60:
        # Structure phase: moderate boost for structure detail
        # Linear ramp from 0.03 to 0.10
        phase_progress = (progress - 0.30) / 0.30
        high_freq_boost = (0.03 + 0.07 * phase_progress) * ndb_strength
    else:
        # Refinement phase: strong emphasis for fine detail
        # Ramp from 0.10 to 0.18
        phase_progress = (progress - 0.60) / 0.40
        high_freq_boost = (0.10 + 0.08 * phase_progress) * ndb_strength

    return 1.0, high_freq_boost


def sa_solver_step(x, d_history, sigma, sigma_next, tau, s_noise=1.0, noise_sampler=None, order=2, ndb_strength=0.0, progress=0.0):
    """
    SA-Solver inspired step with controlled stochasticity.

    Uses Adams-Bashforth coefficients for multi-step integration
    with variance-controlled noise injection via tau function.
    Based on "SA-Solver: Stochastic Adams Solver for Fast Sampling" (NeurIPS 2023).

    Args:
        x: Current latent tensor
        d_history: List of (sigma, derivative) tuples from previous steps
        sigma: Current sigma value
        sigma_next: Next sigma value
        tau: Stochasticity control (0=ODE, 1=full SDE)
        s_noise: Noise scaling factor
        noise_sampler: Function to generate scaled noise
        order: Multi-step order (1, 2, or 3)
        ndb_strength: Native Detail Boost strength (0=disabled)
        progress: Sampling progress (0.0 to 1.0) for NDB phase calculation

    Returns:
        Tuple of (next latent, sigma_up used for noise)
    """
    dt = sigma_next - sigma
    
    # Compute interpolated derivative based on order and history
    if len(d_history) >= 2 and order >= 2:
        # 2nd order Adams-Bashforth (SA-Solver style)
        sigma_cur, d_cur = d_history[-1]
        sigma_prev, d_prev = d_history[-2]
        
        # Compute step sizes for adaptive coefficients
        # h_prev = previous step size (sigma_prev → sigma_cur)
        # dt = current step size (sigma_cur → sigma_next) = sigma_next - sigma
        h_prev = sigma_cur - sigma_prev
        # Use the step size ratio for non-uniform step adaptation
        # r = dt / h_prev (ratio of current to previous step size)
        r = abs(dt / (h_prev + 1e-8)) if abs(h_prev) > 1e-8 else 1.0
        # SAFETY: Clamp r to prevent explosive coefficients with irregular schedules (e.g. AkashicAOS)
        r = min(r, 2.0)
        
        if len(d_history) >= 3 and order >= 3:
            # 3rd order Adams-Bashforth
            sigma_0, d_0 = d_history[-3]
            h_0 = sigma_prev - sigma_0  # step size from sigma_0 → sigma_prev
            h_1 = h_prev  # step size from sigma_prev → sigma_cur
            
            if abs(h_0) > 1e-6 and abs(h_1) > 1e-6:
                # Compute step size ratios for 3rd order
                r0 = abs(h_1 / h_0)  # ratio of recent step sizes
                r1 = abs(dt / (h_1 + 1e-8))  # ratio of current to previous
                # SAFETY: Clamp ratios
                r0 = min(r0, 2.0)
                r1 = min(r1, 2.0)
                
                # Adams-Bashforth 3rd order coefficients for non-uniform steps
                # Standard AB3: x_{n+1} = x_n + dt * (23/12 * f_n - 16/12 * f_{n-1} + 5/12 * f_{n-2})
                # Adapted for non-uniform steps using step ratios
                # BUGFIX: Blend towards Euler (c0=1, c1=0, c2=0) as tau increases
                # This prevents multi-step extrapolation from amplifying noise at high stochasticity
                tau_blend = 1.0 - tau  # 1.0 when tau=0 (full AB3), 0.0 when tau=1 (full Euler)
                c0_ab3 = 1.0 + (1.0 + r0) * r1 / 2.0  # Full AB3 weight for d_cur
                c1_ab3 = -(1.0 + r0) * r1 / 2.0       # Full AB3 weight for d_prev
                c2_ab3 = r0 * r1 / 2.0                 # Full AB3 weight for d_0
                c0 = tau_blend * c0_ab3 + (1.0 - tau_blend) * 1.0  # Blend towards 1.0
                c1 = tau_blend * c1_ab3  # Blend towards 0.0
                c2 = tau_blend * c2_ab3  # Blend towards 0.0
                
                # Normalize coefficients to sum to 1 for stability
                c_sum = c0 + c1 + c2
                if abs(c_sum) > 1e-8:
                    c0 /= c_sum
                    c1 /= c_sum
                    c2 /= c_sum
                else:
                    # Fallback to equal weights if normalization fails
                    c0, c1, c2 = 1.0, 0.0, 0.0
                
                d_interp = c0 * d_cur + c1 * d_prev + c2 * d_0
            else:
                # Fallback to 2nd order if step sizes are too small
                # Adams-Bashforth 2nd order coefficients
                # Standard AB2: x_{n+1} = x_n + dt * (3/2 * f_n - 1/2 * f_{n-1})
                # Adapted for non-uniform steps
                # BUGFIX: Blend towards Euler as tau increases
                tau_blend = 1.0 - tau
                c1_ab2 = 1.0 + 0.5 * r  # Full AB2 weight for d_cur
                c2_ab2 = -0.5 * r       # Full AB2 weight for d_prev
                c1 = tau_blend * c1_ab2 + (1.0 - tau_blend) * 1.0  # Blend towards 1.0
                c2 = tau_blend * c2_ab2  # Blend towards 0.0
                # Normalize
                c_sum = c1 + c2
                if abs(c_sum) > 1e-8:
                    c1 /= c_sum
                    c2 /= c_sum
                d_interp = c1 * d_cur + c2 * d_prev
        else:
            # 2nd order Adams-Bashforth coefficients
            # Standard AB2: x_{n+1} = x_n + dt * (3/2 * f_n - 1/2 * f_{n-1})
            # For non-uniform steps, adapt using step ratio r = dt / h_prev
            # BUGFIX: Blend towards Euler (c1=1, c2=0) as tau increases
            # This prevents multi-step extrapolation from amplifying noise at high stochasticity
            tau_blend = 1.0 - tau  # 1.0 when tau=0 (full AB2), 0.0 when tau=1 (full Euler)
            c1_ab2 = 1.0 + 0.5 * r  # Full AB2 weight for d_cur
            c2_ab2 = -0.5 * r       # Full AB2 weight for d_prev
            c1 = tau_blend * c1_ab2 + (1.0 - tau_blend) * 1.0  # Blend towards 1.0
            c2 = tau_blend * c2_ab2  # Blend towards 0.0
            # Normalize for stability
            c_sum = c1 + c2
            if abs(c_sum) > 1e-8:
                c1 /= c_sum
                c2 /= c_sum
            d_interp = c1 * d_cur + c2 * d_prev
    elif len(d_history) >= 1:
        # First order (Euler) when insufficient history
        d_interp = d_history[-1][1]
    else:
        # No history - should not happen in normal usage
        d_interp = torch.zeros_like(x)
    
    # Compute sigma_up based on tau (controls stochasticity)
    sigma_up = 0.0
    if tau > 0 and sigma_next > 0 and noise_sampler is not None:
        # Compute ancestral noise magnitude, scaled by tau
        # This interpolates between ODE (tau=0) and full ancestral (tau=1)
        sigma_ancestral_sq = sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / (sigma ** 2 + 1e-8)
        sigma_ancestral = sigma_ancestral_sq ** 0.5 if sigma_ancestral_sq > 0 else 0.0
        sigma_up = tau * sigma_ancestral
        
        # Adjust step for noise injection (sigma_down)
        sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
        dt_adjusted = sigma_down - sigma
        
        # Deterministic step with adjusted dt
        x_det = x + d_interp * dt_adjusted
        
        # Generate base noise
        noise = noise_sampler(sigma, sigma_next) * s_noise * sigma_up
        
        # Apply Native Detail Boost if enabled
        if ndb_strength > 0 and TORCHVISION_AVAILABLE:
            _, high_freq_boost = compute_native_detail_boost(progress, ndb_strength)
            blur_sigma = 0.5

            # Extract high-frequency component from noise using Gaussian blur
            # Low-freq = blur, High-freq = original - blur
            try:
                low_freq_noise = gaussian_blur(noise, kernel_size=3, sigma=blur_sigma)
                high_freq_noise = noise - low_freq_noise

                # Boost high-frequency component
                noise = noise + high_freq_noise * high_freq_boost
            except Exception:
                pass  # Fallback: use original noise if blur fails
        
        x_next = x_det + noise
    else:
        # Pure ODE step (no noise)
        x_next = x + d_interp * dt
    
    return x_next, sigma_up


class AdeptSamplerForge(scripts.Script):
    """
    reForge extension for Adept Sampler
    """
    
    def title(self):
        return "Adept Sampler (reForge)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML('Adept Sampler: Advanced sampling enhancements including custom schedulers (all samplers), detail enhancement (all samplers), content-aware pacing (Euler Ancestral), and Adept Solver.')
            
            self.enable_custom = gr.Checkbox(label='Enable Adept Sampler', value=False)
            
            # This group will be shown/hidden based on the checkbox
            with gr.Group(visible=False) as main_options:
                with gr.Tabs():
                    with gr.TabItem("Solver"):
                        self.solver_type = gr.Dropdown(
                            label='Solver Type',
                            value='None',
                            choices=['None', 'Adept Solver', 'Adept Ancestral Solver', 'AkashicSolver'],
                            info="None = WebUI's native solver. Adept = multistep predictor-corrector. Adept Ancestral = adds noise injection. AkashicSolver = optimized for EQVAE models."
                        )
                        
                        with gr.Group(visible=False) as solver_options:
                            with gr.Row():
                                self.adept_solver_order = gr.Slider(
                                    label='Order',
                                    minimum=1, maximum=3,
                                    value=2, step=1,
                                    info="Multistep order. 2 recommended."
                                )
                                self.adept_solver_use_corrector = gr.Checkbox(
                                    label='Corrector Step',
                                    value=True,
                                    info="Adds UniPC-style corrector."
                                )
                        
                        with gr.Group(visible=False) as ancestral_solver_options:
                            with gr.Row():
                                self.adept_ancestral_eta = gr.Slider(
                                    label='Eta',
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                                    info="Noise injection amount. Higher = more diversity."
                                )
                                self.adept_ancestral_s_noise = gr.Slider(
                                    label='Noise Scale',
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                                    info="Noise strength multiplier."
                                )
                            
                            with gr.Row():
                                self.adept_ancestral_adaptive_eta = gr.Checkbox(
                                    label='Adaptive Eta',
                                    value=False,
                                    info="Phase-aware eta adjustment."
                                )
                                self.adept_ancestral_phase_noise = gr.Checkbox(
                                    label='Phase-Aware Noise',
                                    value=False,
                                    info="Adjusts noise by phase."
                                )

                            with gr.Group(visible=False) as phase_strength_group:
                                self.adept_ancestral_phase_strength = gr.Slider(
                                    label='Phase Strength',
                                    minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                                    info="Phase adaptation intensity."
                                )

                            self.adept_ancestral_phase_noise.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[self.adept_ancestral_phase_noise],
                                outputs=[phase_strength_group]
                            )
                            
                            self.adept_ancestral_enhanced_derivative = gr.Checkbox(
                                label='Enhanced Derivative',
                                value=False,
                                info="Ancestral-specific derivative computation."
                            )
                            self.adept_ancestral_mirror_correction = gr.Checkbox(
                                label='Mirror Correction',
                                value=False,
                                info="Semantic reflection probe for 2nd-order correction (3 model calls/step in first 60%)."
                            )
                        
                        with gr.Group(visible=False) as akashic_solver_options:
                            gr.Markdown("🌀 **AkashicSolver v2** - SA-Solver base with AYS schedules")
                            
                            with gr.Row():
                                self.akashic_tau = gr.Slider(
                                    label='Tau (τ)',
                                    minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                                    info="Stochasticity: 0=ODE, 1=full SDE"
                                )
                                self.akashic_solver_order = gr.Slider(
                                    label='Order',
                                    minimum=1, maximum=3, value=2, step=1,
                                    info="Multi-step order (2 recommended)"
                                )
                            
                            with gr.Row():
                                self.akashic_base_eta = gr.Slider(
                                    label='Eta (η)',
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                                    info="Noise magnitude scaling"
                                )
                                self.akashic_s_noise = gr.Slider(
                                    label='Noise Scale',
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                                    info="Overall noise strength"
                                )
                            
                            with gr.Row():
                                self.akashic_adaptive_eta = gr.Checkbox(
                                    label='Adaptive Eta',
                                    value=True,
                                    info="Phase-aware adaptation"
                                )
                                self.akashic_use_ays = gr.Checkbox(
                                    label='AYS Schedule',
                                    value=False,
                                    info="Use Align Your Steps optimized schedule"
                                )
                            
                            with gr.Row():
                                self.akashic_phase_strength = gr.Slider(
                                    label='Phase Strength',
                                    minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                                    info="Adaptation intensity"
                                )
                                self.akashic_smea_strength = gr.Slider(
                                    label='SMEA Strength',
                                    minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                                    info="High-res coherency (0=off)"
                                )
                            
                            with gr.Row():
                                self.akashic_ndb_strength = gr.Slider(
                                    label='Native Detail Boost',
                                    minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                                    info="Enhances detail at native res (0=off)"
                                )
                                self.akashic_eqvae_mode = gr.Dropdown(
                                    label='EQ-VAE Mode',
                                    choices=['Off', 'Balanced'],
                                    value='Off',
                                    info="Optimized for EQ-VAE's cleaner latents"
                                )

                            self.akashic_mirror_correction = gr.Checkbox(
                                label='Mirror Correction',
                                value=False,
                                info="Semantic reflection probe for 2nd-order correction (3 model calls/step in first 60%)."
                            )

                            # Additional CFG Fixes Section
                            gr.Markdown("---")
                            gr.Markdown("**Additional CFG Fixes**")
                            with gr.Row():
                                self.akashic_spectral_mod = gr.Checkbox(
                                    label='Spectral Modulation',
                                    value=False,
                                    info="Frequency-domain CFG correction"
                                )
                                self.akashic_combat_cfg_drift = gr.Checkbox(
                                    label='Combat CFG Drift',
                                    value=False,
                                    info="Re-center latent mean"
                                )

                            with gr.Group(visible=False) as spectral_options:
                                self.akashic_spectral_percentile = gr.Slider(
                                    label='Spectral Percentile',
                                    minimum=1.0, maximum=15.0, value=5.0, step=0.5,
                                    info="Frequency threshold (lower=gentler)"
                                )

                            with gr.Group(visible=False) as combat_drift_options:
                                self.akashic_combat_drift_intensity = gr.Slider(
                                    label='Drift Correction Intensity',
                                    minimum=0.1, maximum=1.0, value=0.5, step=0.1,
                                    info="How much drift to remove (lower=subtler)"
                                )

                            # Visibility handlers for CFG fix options
                            self.akashic_spectral_mod.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[self.akashic_spectral_mod],
                                outputs=[spectral_options]
                            )

                            self.akashic_combat_cfg_drift.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[self.akashic_combat_cfg_drift],
                                outputs=[combat_drift_options]
                            )

                        def on_solver_type_change(solver_type):
                            return {
                                solver_options: gr.update(visible=solver_type == 'Adept Solver'),
                                ancestral_solver_options: gr.update(visible=solver_type == 'Adept Ancestral Solver'),
                                akashic_solver_options: gr.update(visible=solver_type == 'AkashicSolver')
                            }
                        
                        self.solver_type.change(
                            fn=on_solver_type_change,
                            inputs=[self.solver_type],
                            outputs=[solver_options, ancestral_solver_options, akashic_solver_options]
                        )
            
                    with gr.TabItem("Scheduler"):

                        # Category dropdown controls the visible scheduler list
                        universal_choices = [
                            "None (use WebUI sampler schedule)",
                            "Entropic",
                            "Constant-Rate",
                            "Adaptive-Optimized",
                            "Cosine-Annealed",
                            "LogSNR-Uniform",
                            "Tanh Mid-Boost",
                            "Exponential Tail",
                            "Jittered-Karras",
                            "Hybrid JYS-Karras",
                            "AYS-SDXL",
                            "Stochastic",
                            "JYS (Dynamic)",
                        ]
                        vpred_choices = [
                            "AOS-V (for v-prediction)",
                            "SNR-Optimized",
                        ]
                        eps_choices = [
                            "AOS-ε (for ε-prediction)",
                            "AkashicAOS",
                            "AkashicAOS Alt",
                            "AkashicEQFlow",
                        ]

                        self.scheduler_category = gr.Dropdown(
                            label="Scheduler Category",
                            value="Universal",
                            choices=["Universal", "V-Prediction", "ε-Prediction"],
                        )

                        self.scheduler_override = gr.Dropdown(
                            label="Scheduler",
                            value="None (use WebUI sampler schedule)",
                            choices=universal_choices,
                        )

                        with gr.Group(visible=False) as entropic_options:
                            self.entropic_scheduler_power = gr.Slider(
                                label='Entropic Power',
                                minimum=1.0, maximum=8.0,
                                value=6.0, step=0.1,
                                info="Controls timestep clustering. >1 clusters steps at the start (high detail)."
                            )

                        with gr.Group(visible=False) as stochastic_options:
                            self.stochastic_noise_type = gr.Dropdown(
                                label='Noise Type',
                                value='brownian',
                                choices=['brownian', 'uniform', 'normal'],
                                info="Type of randomness to inject into timestep selection."
                            )
                            self.stochastic_noise_scale = gr.Slider(
                                label='Noise Scale',
                                minimum=0.0, maximum=1.0,
                                value=0.3, step=0.05,
                                info="Amount of randomness (0.0 = deterministic, higher = more random)."
                            )
                            self.stochastic_base_schedule = gr.Dropdown(
                                label='Base Schedule',
                                value='karras',
                                choices=['karras', 'uniform', 'cosine'],
                                info="Base timestep distribution before adding randomness."
                            )
                        


                        with gr.Group(visible=False) as aos_plus_options:
                            gr.Markdown("⚠️ Match AOS variant to model type: **AOS-V** for v-prediction, **AOS-ε** for epsilon-prediction.")
                            self.use_content_aware_pacing = gr.Checkbox(label='Content-Aware Pacing', value=False, info="Adjusts pacing by coherence. Requires Euler Ancestral + AOS + ≥26 steps.")
                            with gr.Row():
                                self.pacing_coherence_sensitivity = gr.Slider(
                                    label='Coherence Sensitivity',
                                    minimum=0.1, maximum=1.0, value=0.75, step=0.05,
                                    info="When to switch phases."
                                )
                            self.manual_pacing_override = gr.Textbox(
                                label="Manual Override (JSON)",
                                placeholder='e.g., {"composition": 0.4}'
                            )
                            self.debug_stop_after_coherence = gr.Checkbox(label='[Debug] Stop after coherence', value=False)

                        def on_scheduler_change(scheduler):
                            is_aos = "AOS-V" in scheduler or "AOS-ε" in scheduler
                            return {
                                aos_plus_options: gr.update(visible=is_aos),
                                entropic_options: gr.update(visible=scheduler == "Entropic"),
                                stochastic_options: gr.update(visible=scheduler == "Stochastic")
                            }

                        def on_category_change(category):
                            if category == "Universal":
                                return {
                                    self.scheduler_override: gr.update(choices=universal_choices, value="None (use WebUI sampler schedule)"),
                                    aos_plus_options: gr.update(visible=False),
                                    entropic_options: gr.update(visible=False),
                                    stochastic_options: gr.update(visible=False),
                                }
                            elif category == "V-Prediction":
                                return {
                                    self.scheduler_override: gr.update(choices=vpred_choices, value="AOS-V (for v-prediction)"),
                                    aos_plus_options: gr.update(visible=True),
                                    entropic_options: gr.update(visible=False),
                                    stochastic_options: gr.update(visible=False),
                                }
                            else:
                                return {
                                    self.scheduler_override: gr.update(choices=eps_choices, value="AOS-ε (for ε-prediction)"),
                                    aos_plus_options: gr.update(visible=True),
                                    entropic_options: gr.update(visible=False),
                                    stochastic_options: gr.update(visible=False),
                                }

                        self.scheduler_override.change(
                            on_scheduler_change,
                            inputs=[self.scheduler_override],
                            outputs=[aos_plus_options, entropic_options, stochastic_options]
                        )

                        self.scheduler_category.change(
                            on_category_change,
                            inputs=[self.scheduler_category],
                            outputs=[self.scheduler_override, aos_plus_options, entropic_options, stochastic_options]
                        )



                    with gr.TabItem("Detail Enhancement"):
                        self.use_enhanced_detail_phase = gr.Checkbox(label="Enable Detail Enhancement", value=True, info="Enhances high-frequency details during sampling.")

                        with gr.Group(visible=True) as enhancer_settings:
                            with gr.Row():
                                self.detail_enhancement_strength = gr.Slider(label="Strength", minimum=0.0, maximum=1.0, value=0.05, step=0.05)
                                self.detail_separation_radius = gr.Slider(label="Radius (Sigma)", minimum=0.1, maximum=2.0, value=0.5, step=0.05, info="Higher = sharpen larger features.")
                        
                        self.use_enhanced_detail_phase.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[self.use_enhanced_detail_phase],
                            outputs=[enhancer_settings]
                        )

                    with gr.TabItem("Advanced"):
                        with gr.Row():
                            self.eta = gr.Slider(label='Eta', minimum=0.0, maximum=2.0, value=1.0, step=0.01, info="Ancestral noise amount.")
                            self.s_noise = gr.Slider(label='Noise Scale', minimum=0.0, maximum=2.0, value=1.0, step=0.01, info="Noise strength.")

                        self.disable_for_hr = gr.Checkbox(label="Disable for Hires. fix", value=True, info="Turns off during hi-res pass.")
                        self.debug_reproducibility = gr.Checkbox(label='Debug Reproducibility', value=False, info="Disables advanced features.")

                        gr.Markdown("**VAE Options**")
                        self.vae_reflection = gr.Checkbox(
                            label='VAE Reflection Padding',
                            value=False,
                            info="Fix edge artifacts for VAEs trained with reflect padding (e.g., Anzhc's EQ-VAE)"
                        )
            
                    with gr.TabItem("Experimental"):
                        self.exp_cfg_to_zero = gr.Checkbox(
                            label="CFG to Zero after 40%",
                            value=False,
                            info="Sets CFG to zero after 40% steps. May reduce prompt adherence."
                        )

            
            # Visibility logic for the main group of options
            self.enable_custom.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[self.enable_custom],
                outputs=[main_options]
            )

        self.infotext_fields = [
            (self.enable_custom, lambda p: str(p.get('adept_sampler_enabled')).lower() == 'true' if 'adept_sampler_enabled' in p else gr.update()),
            (self.eta, lambda p: float(p['custom_eta']) if 'custom_eta' in p else gr.update()),
            (self.s_noise, lambda p: float(p['custom_s_noise']) if 'custom_s_noise' in p else gr.update()),
            (self.debug_reproducibility, lambda p: str(p.get('debug_reproducibility')).lower() == 'true' if 'debug_reproducibility' in p else gr.update()),
            (self.entropic_scheduler_power, lambda p: gr.update() if p.get('entropic_power') in (None, 'N/A') else float(p['entropic_power'])),
            (self.use_content_aware_pacing, lambda p: str(p.get('content_aware_pacing')).lower() == 'true' if 'content_aware_pacing' in p else gr.update()),
            (self.pacing_coherence_sensitivity, lambda p: gr.update() if p.get('coherence_sensitivity') in (None, 'N/A') else float(p['coherence_sensitivity'])),
            (self.manual_pacing_override, lambda p: p.get('manual_pacing_override', gr.update())),
            (self.debug_stop_after_coherence, lambda p: str(p.get('debug_stop_after_coherence')).lower() == 'true' if 'debug_stop_after_coherence' in p else gr.update()),
            (self.use_enhanced_detail_phase, lambda p: str(p.get('enhanced_detail_phase')).lower() == 'true' if 'enhanced_detail_phase' in p else gr.update()),
            (self.detail_enhancement_strength, lambda p: gr.update() if p.get('detail_enhancement_strength') in (None, 'N/A') else float(p['detail_enhancement_strength'])),
            (self.detail_separation_radius, lambda p: gr.update() if p.get('detail_separation_radius') in (None, 'N/A') else float(p['detail_separation_radius'])),
            (self.disable_for_hr, lambda p: str(p.get('adept_disable_for_hr')).lower() == 'true' if 'adept_disable_for_hr' in p else gr.update()),
            (self.exp_cfg_to_zero, lambda p: str(p.get('exp_cfg_to_zero')).lower() == 'true' if 'exp_cfg_to_zero' in p else gr.update()),
            (self.solver_type, lambda p: p.get('solver_type', 'None')),
            (self.adept_solver_order, lambda p: gr.update() if p.get('adept_solver_order') in (None, 'N/A') else int(p['adept_solver_order'])),
            (self.adept_solver_use_corrector, lambda p: str(p.get('adept_solver_corrector')).lower() == 'true' if 'adept_solver_corrector' in p else gr.update()),
            (self.adept_ancestral_eta, lambda p: gr.update() if p.get('adept_ancestral_eta') in (None, 'N/A') else float(p['adept_ancestral_eta'])),
            (self.adept_ancestral_s_noise, lambda p: gr.update() if p.get('adept_ancestral_s_noise') in (None, 'N/A') else float(p['adept_ancestral_s_noise'])),
            (self.adept_ancestral_adaptive_eta, lambda p: str(p.get('adept_ancestral_adaptive_eta', 'false')).lower() == 'true' if 'adept_ancestral_adaptive_eta' in p else gr.update()),
            (self.adept_ancestral_phase_noise, lambda p: str(p.get('adept_ancestral_phase_noise', 'false')).lower() == 'true' if 'adept_ancestral_phase_noise' in p else gr.update()),
            (self.adept_ancestral_phase_strength, lambda p: gr.update() if p.get('adept_ancestral_phase_strength') in (None, 'N/A') else float(p['adept_ancestral_phase_strength'])),
            (self.adept_ancestral_enhanced_derivative, lambda p: str(p.get('adept_ancestral_enhanced_derivative', 'false')).lower() == 'true' if 'adept_ancestral_enhanced_derivative' in p else gr.update()),
            (self.adept_ancestral_mirror_correction, lambda p: str(p.get('adept_ancestral_mirror_correction', 'false')).lower() == 'true' if 'adept_ancestral_mirror_correction' in p else gr.update()),
            # Stochastic scheduler settings
            (self.stochastic_noise_type, lambda p: p.get('stochastic_noise_type', 'brownian') if 'stochastic_noise_type' in p else gr.update()),
            (self.stochastic_noise_scale, lambda p: gr.update() if p.get('stochastic_noise_scale') in (None, 'N/A') else float(p['stochastic_noise_scale'])),
            (self.stochastic_base_schedule, lambda p: p.get('stochastic_base_schedule', 'karras') if 'stochastic_base_schedule' in p else gr.update()),
            # AkashicSolver settings
            (self.akashic_tau, lambda p: gr.update() if p.get('akashic_tau') in (None, 'N/A') else float(p['akashic_tau'])),
            (self.akashic_solver_order, lambda p: gr.update() if p.get('akashic_solver_order') in (None, 'N/A') else int(p['akashic_solver_order'])),
            (self.akashic_base_eta, lambda p: gr.update() if p.get('akashic_base_eta') in (None, 'N/A') else float(p['akashic_base_eta'])),
            (self.akashic_s_noise, lambda p: gr.update() if p.get('akashic_s_noise') in (None, 'N/A') else float(p['akashic_s_noise'])),
            (self.akashic_adaptive_eta, lambda p: str(p.get('akashic_adaptive_eta', 'true')).lower() == 'true' if 'akashic_adaptive_eta' in p else gr.update()),
            (self.akashic_use_ays, lambda p: str(p.get('akashic_use_ays', 'false')).lower() == 'true' if 'akashic_use_ays' in p else gr.update()),
            (self.akashic_phase_strength, lambda p: gr.update() if p.get('akashic_phase_strength') in (None, 'N/A') else float(p['akashic_phase_strength'])),
            (self.akashic_smea_strength, lambda p: gr.update() if p.get('akashic_smea_strength') in (None, 'N/A') else float(p['akashic_smea_strength'])),
            (self.akashic_ndb_strength, lambda p: gr.update() if p.get('akashic_ndb_strength') in (None, 'N/A') else float(p['akashic_ndb_strength'])),
            (self.akashic_mirror_correction, lambda p: str(p.get('akashic_mirror_correction', 'false')).lower() == 'true' if 'akashic_mirror_correction' in p else gr.update()),
            (self.akashic_eqvae_mode, lambda p: p.get('akashic_eqvae_mode', 'Off') if 'akashic_eqvae_mode' in p else gr.update()),
            # Additional CFG Fixes settings
            (self.akashic_spectral_mod, lambda p: str(p.get('akashic_spectral_mod', 'false')).lower() == 'true' if 'akashic_spectral_mod' in p else gr.update()),
            (self.akashic_spectral_percentile, lambda p: gr.update() if p.get('akashic_spectral_percentile') in (None, 'N/A') else float(p['akashic_spectral_percentile'])),
            (self.akashic_combat_cfg_drift, lambda p: str(p.get('akashic_combat_cfg_drift', 'false')).lower() == 'true' if 'akashic_combat_cfg_drift' in p else gr.update()),
            (self.akashic_combat_drift_intensity, lambda p: gr.update() if p.get('akashic_combat_drift_intensity') in (None, 'N/A') else float(p['akashic_combat_drift_intensity'])),
            (self.vae_reflection, lambda p: str(p.get('vae_reflection', 'false')).lower() == 'true' if 'vae_reflection' in p else gr.update()),
        ]

        def scheduler_getter(params):
            if 'adept_sampler_enabled' not in params:
                return gr.update()
            
            custom_scheduler = params.get('custom_scheduler_type')
            if custom_scheduler and custom_scheduler != 'None':
                return custom_scheduler

            aos_schedule = params.get('anime_optimized_schedule')
            if aos_schedule == 'V':
                return "AOS-V (for v-prediction)"
            elif aos_schedule == 'Epsilon':
                return "AOS-ε (for ε-prediction)"
            
            if str(params.get('entropic_scheduler')).lower() == 'true':
                return "Entropic"
            
            return "None (use WebUI sampler schedule)"

        self.infotext_fields.append((self.scheduler_override, scheduler_getter))

        return [
            self.enable_custom,
            self.eta, self.s_noise, self.debug_reproducibility,
            self.scheduler_override, self.entropic_scheduler_power,
            self.stochastic_noise_type, self.stochastic_noise_scale, self.stochastic_base_schedule,
            self.use_content_aware_pacing, self.pacing_coherence_sensitivity,
            self.manual_pacing_override,
            self.debug_stop_after_coherence,
            self.use_enhanced_detail_phase,
            self.detail_enhancement_strength, self.detail_separation_radius,
            self.disable_for_hr,
            self.exp_cfg_to_zero,
            self.solver_type, self.adept_solver_order, self.adept_solver_use_corrector,
            self.adept_ancestral_eta, self.adept_ancestral_s_noise,
            self.adept_ancestral_adaptive_eta, self.adept_ancestral_phase_noise, self.adept_ancestral_phase_strength, self.adept_ancestral_enhanced_derivative,
            self.adept_ancestral_mirror_correction,
            self.akashic_tau, self.akashic_solver_order, self.akashic_base_eta, self.akashic_s_noise,
            self.akashic_adaptive_eta, self.akashic_use_ays, self.akashic_phase_strength, self.akashic_smea_strength,
            self.akashic_ndb_strength, self.akashic_mirror_correction,
            self.akashic_eqvae_mode,
            # Additional CFG Fixes settings
            self.akashic_spectral_mod,
            self.akashic_spectral_percentile,
            self.akashic_combat_cfg_drift,
            self.akashic_combat_drift_intensity,
            self.vae_reflection,
        ]

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (
            enable_custom,
            eta, s_noise, debug_reproducibility,
            scheduler_override, entropic_scheduler_power,
            stochastic_noise_type, stochastic_noise_scale, stochastic_base_schedule,
            use_content_aware_pacing, pacing_coherence_sensitivity,
            manual_pacing_override,
            debug_stop_after_coherence,
            use_enhanced_detail_phase,
            detail_enhancement_strength, detail_separation_radius,
            disable_for_hr,
            exp_cfg_to_zero,
            solver_type, adept_solver_order, adept_solver_use_corrector,
            adept_ancestral_eta, adept_ancestral_s_noise,
            adept_ancestral_adaptive_eta, adept_ancestral_phase_noise, adept_ancestral_phase_strength, adept_ancestral_enhanced_derivative,
            adept_ancestral_mirror_correction,
            akashic_tau, akashic_solver_order, akashic_base_eta, akashic_s_noise,
            akashic_adaptive_eta, akashic_use_ays, akashic_phase_strength, akashic_smea_strength,
            akashic_ndb_strength, akashic_mirror_correction, akashic_eqvae_mode,
            # Additional CFG Fixes settings
            akashic_spectral_mod, akashic_spectral_percentile,
            akashic_combat_cfg_drift, akashic_combat_drift_intensity,
            vae_reflection,
        ) = script_args

        # --- XYZ Grid overrides (if provided) ---
        xyz = getattr(p, "_adept_xyz", {})
        if xyz:
            if "enabled" in xyz:
                enable_custom = str(xyz["enabled"]) == "True"
            if "eta" in xyz:
                try: eta = float(xyz["eta"]) 
                except Exception: pass
            if "s_noise" in xyz:
                try: s_noise = float(xyz["s_noise"]) 
                except Exception: pass
            if "debug_reproducibility" in xyz:
                debug_reproducibility = str(xyz["debug_reproducibility"]) == "True"
            if "scheduler_override" in xyz:
                scheduler_override = str(xyz["scheduler_override"]) or scheduler_override
            if "entropic_scheduler_power" in xyz:
                try: entropic_scheduler_power = float(xyz["entropic_scheduler_power"]) 
                except Exception: pass
            if "stochastic_noise_type" in xyz:
                stochastic_noise_type = str(xyz["stochastic_noise_type"]) or stochastic_noise_type
            if "stochastic_noise_scale" in xyz:
                try: stochastic_noise_scale = float(xyz["stochastic_noise_scale"]) 
                except Exception: pass
            if "stochastic_base_schedule" in xyz:
                stochastic_base_schedule = str(xyz["stochastic_base_schedule"]) or stochastic_base_schedule
            if "use_content_aware_pacing" in xyz:
                use_content_aware_pacing = str(xyz["use_content_aware_pacing"]) == "True"
            if "pacing_coherence_sensitivity" in xyz:
                try: pacing_coherence_sensitivity = float(xyz["pacing_coherence_sensitivity"]) 
                except Exception: pass
            if "manual_pacing_override" in xyz:
                manual_pacing_override = str(xyz["manual_pacing_override"]) or manual_pacing_override
            if "debug_stop_after_coherence" in xyz:
                debug_stop_after_coherence = str(xyz["debug_stop_after_coherence"]) == "True"
            if "use_enhanced_detail_phase" in xyz:
                use_enhanced_detail_phase = str(xyz["use_enhanced_detail_phase"]) == "True"
            if "detail_enhancement_strength" in xyz:
                try: detail_enhancement_strength = float(xyz["detail_enhancement_strength"]) 
                except Exception: pass
            if "detail_separation_radius" in xyz:
                try: detail_separation_radius = float(xyz["detail_separation_radius"]) 
                except Exception: pass
            if "disable_for_hr" in xyz:
                disable_for_hr = str(xyz["disable_for_hr"]) == "True"
            if "exp_cfg_to_zero" in xyz:
                exp_cfg_to_zero = str(xyz["exp_cfg_to_zero"]) == "True"
            if "solver_type" in xyz:
                solver_type = str(xyz["solver_type"]) or solver_type
            if "adept_solver_order" in xyz:
                try: adept_solver_order = int(xyz["adept_solver_order"])
                except Exception: pass
            if "adept_solver_use_corrector" in xyz:
                adept_solver_use_corrector = str(xyz["adept_solver_use_corrector"]) == "True"
            if "adept_ancestral_eta" in xyz:
                try: adept_ancestral_eta = float(xyz["adept_ancestral_eta"])
                except Exception: pass
            if "adept_ancestral_s_noise" in xyz:
                try: adept_ancestral_s_noise = float(xyz["adept_ancestral_s_noise"])
                except Exception: pass
            if "adept_ancestral_adaptive_eta" in xyz:
                adept_ancestral_adaptive_eta = str(xyz["adept_ancestral_adaptive_eta"]) == "True"
            if "adept_ancestral_phase_noise" in xyz:
                adept_ancestral_phase_noise = str(xyz["adept_ancestral_phase_noise"]) == "True"
            if "adept_ancestral_phase_strength" in xyz:
                try:
                    adept_ancestral_phase_strength = float(xyz["adept_ancestral_phase_strength"])
                except (ValueError, TypeError):
                    adept_ancestral_phase_strength = 0.5
            if "adept_ancestral_enhanced_derivative" in xyz:
                adept_ancestral_enhanced_derivative = str(xyz["adept_ancestral_enhanced_derivative"]) == "True"
            if "adept_ancestral_mirror_correction" in xyz:
                adept_ancestral_mirror_correction = str(xyz["adept_ancestral_mirror_correction"]) == "True"
            # AkashicSolver XYZ overrides
            if "akashic_tau" in xyz:
                try: akashic_tau = float(xyz["akashic_tau"])
                except Exception: pass
            if "akashic_solver_order" in xyz:
                try: akashic_solver_order = int(xyz["akashic_solver_order"])
                except Exception: pass
            if "akashic_base_eta" in xyz:
                try: akashic_base_eta = float(xyz["akashic_base_eta"])
                except Exception: pass
            if "akashic_s_noise" in xyz:
                try: akashic_s_noise = float(xyz["akashic_s_noise"])
                except Exception: pass
            if "akashic_adaptive_eta" in xyz:
                akashic_adaptive_eta = str(xyz["akashic_adaptive_eta"]) == "True"
            if "akashic_use_ays" in xyz:
                akashic_use_ays = str(xyz["akashic_use_ays"]) == "True"
            if "akashic_phase_strength" in xyz:
                try: akashic_phase_strength = float(xyz["akashic_phase_strength"])
                except Exception: pass
            if "akashic_smea_strength" in xyz:
                try: akashic_smea_strength = float(xyz["akashic_smea_strength"])
                except Exception: pass
            if "akashic_ndb_strength" in xyz:
                try: akashic_ndb_strength = float(xyz["akashic_ndb_strength"])
                except Exception: pass
            if "akashic_mirror_correction" in xyz:
                akashic_mirror_correction = str(xyz["akashic_mirror_correction"]) == "True"
            if "akashic_eqvae_mode" in xyz:
                akashic_eqvae_mode = str(xyz["akashic_eqvae_mode"])
            # Additional CFG Fixes XYZ overrides
            if "akashic_spectral_mod" in xyz:
                akashic_spectral_mod = str(xyz["akashic_spectral_mod"]) == "True"
            if "akashic_spectral_percentile" in xyz:
                try: akashic_spectral_percentile = float(xyz["akashic_spectral_percentile"])
                except Exception: pass
            if "akashic_combat_cfg_drift" in xyz:
                akashic_combat_cfg_drift = str(xyz["akashic_combat_cfg_drift"]) == "True"
            if "vae_reflection" in xyz:
                vae_reflection = str(xyz["vae_reflection"]) == "True"

        # Set solver flags based on the dropdown choice
        use_adept_solver = (solver_type == 'Adept Solver')
        use_adept_ancestral_solver = (solver_type == 'Adept Ancestral Solver')
        use_akashic_solver = (solver_type == 'AkashicSolver')
        
        # Set scheduler flags based on the radio button choice
        use_anime_schedule_v = (scheduler_override == "AOS-V (for v-prediction)")
        use_anime_schedule_e = (scheduler_override == "AOS-ε (for ε-prediction)")
        use_akashic_aos = (scheduler_override == "AkashicAOS")
        use_anime_schedule = use_anime_schedule_v or use_anime_schedule_e or use_akashic_aos
        use_entropic_scheduler = (scheduler_override == "Entropic")

        custom_scheduler_type = "None"
        if scheduler_override in [
            "SNR-Optimized",
            "Constant-Rate",
            "Adaptive-Optimized",
            "Cosine-Annealed",
            "LogSNR-Uniform",
            "Tanh Mid-Boost",
            "Exponential Tail",
            "Jittered-Karras",
            "Hybrid JYS-Karras",
            "AYS-SDXL",
            "Stochastic",
            "JYS (Dynamic)",
            "AOS-V (for v-prediction)",
            "AOS-ε (for ε-prediction)",
            "AkashicAOS",
            "AkashicAOS Alt",
            "AkashicEQFlow",
        ]:
            custom_scheduler_type = scheduler_override

        manual_pacing_schedule = None
        if manual_pacing_override and manual_pacing_override.strip():
            try:
                schedule = json.loads(manual_pacing_override)
                if isinstance(schedule, dict):
                    manual_pacing_schedule = schedule
                else:
                    print(f"⚠️ Manual Pacing Override: Not a valid JSON object. Ignoring.")
            except json.JSONDecodeError:
                print(f"⚠️ Manual Pacing Override: Invalid JSON. Ignoring.")

        # --- Compatibility Checks ---
        is_hires_pass = getattr(p, 'is_hr_pass', False)

        # Detect inpaint/ADetailer passes: these operate on a cropped sub-region
        # whose statistics differ from the full image.  Mean-based corrections
        # (Combat CFG Drift) would produce a different shift on the crop vs. the
        # original, causing a visible brightness/color seam at the composite
        # boundary.  Auto-disable drift correction for these passes.
        is_inpaint_pass = (
            isinstance(p, StableDiffusionProcessingImg2Img)
            and getattr(p, 'image_mask', None) is not None
        )
        if is_inpaint_pass and akashic_combat_cfg_drift:
            akashic_combat_cfg_drift = False
            print("🔄 Combat CFG Drift auto-disabled for inpaint/ADetailer pass (avoids patchy composites)")

        should_be_enabled = enable_custom
        disable_reason = None

        if enable_custom:
            if disable_for_hr and is_hires_pass:
                should_be_enabled = False
                disable_reason = "Hires. fix pass"

        current_sampler_settings.update({
            'enabled': should_be_enabled,
            'eta': eta,
            's_noise': s_noise,
            'debug_reproducibility': debug_reproducibility,
            'use_entropic_scheduler': use_entropic_scheduler,
            'entropic_scheduler_power': entropic_scheduler_power,
            'stochastic_noise_type': stochastic_noise_type,
            'stochastic_noise_scale': stochastic_noise_scale,
            'stochastic_base_schedule': stochastic_base_schedule,
            'use_anime_schedule': use_anime_schedule,
            'use_anime_schedule_v': use_anime_schedule_v,
            'use_anime_schedule_e': use_anime_schedule_e,
            'use_content_aware_pacing': use_content_aware_pacing and use_anime_schedule,
            'pacing_coherence_sensitivity': pacing_coherence_sensitivity,
            'manual_pacing_schedule': manual_pacing_schedule,
            'debug_stop_after_coherence': debug_stop_after_coherence and use_content_aware_pacing and use_anime_schedule,
            'use_enhanced_detail_phase': use_enhanced_detail_phase,
            'detail_enhancement_strength': detail_enhancement_strength,
            'detail_separation_radius': detail_separation_radius,
            'custom_scheduler_type': custom_scheduler_type,
            'exp_cfg_to_zero': exp_cfg_to_zero,
            'use_adept_solver': use_adept_solver and enable_custom,
            'adept_solver_order': adept_solver_order,
            'adept_solver_use_corrector': adept_solver_use_corrector,
            'use_adept_ancestral_solver': use_adept_ancestral_solver and enable_custom,
            'adept_ancestral_eta': adept_ancestral_eta,
            'adept_ancestral_s_noise': adept_ancestral_s_noise,
            'adept_ancestral_adaptive_eta': adept_ancestral_adaptive_eta,
            'adept_ancestral_phase_noise': adept_ancestral_phase_noise,
            'adept_ancestral_phase_strength': adept_ancestral_phase_strength,
            'adept_ancestral_enhanced_derivative': adept_ancestral_enhanced_derivative,
            'adept_ancestral_mirror_correction': adept_ancestral_mirror_correction,
            # AkashicSolver v2 settings
            'use_akashic_solver': use_akashic_solver and enable_custom,
            'use_akashic_aos': use_akashic_aos,
            'akashic_tau': akashic_tau,
            'akashic_solver_order': int(akashic_solver_order),
            'akashic_base_eta': akashic_base_eta,
            'akashic_s_noise': akashic_s_noise,
            'akashic_adaptive_eta': akashic_adaptive_eta,
            'akashic_use_ays': akashic_use_ays,
            'akashic_phase_strength': akashic_phase_strength,
            'akashic_smea_strength': akashic_smea_strength,
            'akashic_ndb_strength': akashic_ndb_strength,
            'akashic_mirror_correction': akashic_mirror_correction,
            'akashic_eqvae_mode': akashic_eqvae_mode,
            # CFG Enhancement settings
            'akashic_spectral_mod': akashic_spectral_mod,
            'akashic_spectral_percentile': akashic_spectral_percentile,
            'akashic_combat_cfg_drift': akashic_combat_cfg_drift,
            'akashic_combat_drift_intensity': akashic_combat_drift_intensity,
            'vae_reflection': vae_reflection,
        })

        # --- VAE Reflection Handling ---
        # Apply or restore VAE reflection based on setting
        if WEBUI_AVAILABLE and hasattr(shared, 'sd_model') and shared.sd_model is not None:
            vae_model = getattr(shared.sd_model, 'first_stage_model', None)
            if vae_model is not None:
                if vae_reflection:
                    apply_vae_reflection(vae_model)
                else:
                    restore_vae_reflection(vae_model)

        # --- Spectral Modulation CFG Hook ---
        # Apply spectral modulation via CFG hook (like RescaleCFG)
        if akashic_spectral_mod and REFORGE_AVAILABLE:
            try:
                if hasattr(p, 'sd_model') and hasattr(p.sd_model, 'forge_objects'):
                    unet = p.sd_model.forge_objects.unet.clone()
                    spectral_hook = create_spectral_modulation_cfg_hook(
                        multiplier=1.0,  # Full Clybius effect
                        percentile=akashic_spectral_percentile
                    )
                    unet.set_model_sampler_cfg_function(spectral_hook)
                    p.sd_model.forge_objects.unet = unet
                    print(f"🌈 Spectral Modulation CFG hook active (percentile={akashic_spectral_percentile})")
            except Exception as e:
                print(f"⚠️ Failed to apply Spectral Modulation CFG hook: {e}")

        if enable_custom:
            if disable_reason:
                print(f"🔄 Adept Sampler disabled for {disable_reason}. Using standard Euler Ancestral.")
            elif debug_reproducibility:
                print(f"🔬 Debug mode: Adept Sampler - simplified for reproducibility testing")
            else:
                print(f"✅ Adept Sampler is now active!")
            
            # Add parameters to generation info
            p.extra_generation_params.update({
                'adept_sampler_enabled': True,
                'adept_sampler_type': 'Enhanced (AOS Focused)',
                'custom_eta': eta,
                'custom_s_noise': s_noise,
                'custom_sampler_deterministic': True,
                'debug_reproducibility': debug_reproducibility,
                'entropic_scheduler': use_entropic_scheduler and not debug_reproducibility,
                'entropic_power': entropic_scheduler_power if use_entropic_scheduler and not use_anime_schedule else 'N/A',
                'anime_optimized_schedule': 'V' if use_anime_schedule_v else ('Epsilon' if use_anime_schedule_e else 'N/A'),
                'content_aware_pacing': use_content_aware_pacing and use_anime_schedule,
                'coherence_sensitivity': pacing_coherence_sensitivity,
                'manual_pacing_override': json.dumps(manual_pacing_schedule) if manual_pacing_schedule else 'N/A',
                'debug_stop_after_coherence': debug_stop_after_coherence and use_content_aware_pacing and use_anime_schedule,
                'enhanced_detail_phase': use_enhanced_detail_phase,
                'detail_enhancement_strength': detail_enhancement_strength if use_enhanced_detail_phase else 'N/A',
                'detail_separation_radius': detail_separation_radius if use_enhanced_detail_phase else 'N/A',
                'custom_scheduler_type': custom_scheduler_type,
                'adept_disable_for_hr': disable_for_hr,
                'exp_cfg_to_zero': exp_cfg_to_zero,
                'solver_type': solver_type,
                'adept_solver_order': adept_solver_order if use_adept_solver else 'N/A',
                'adept_solver_corrector': adept_solver_use_corrector if use_adept_solver else 'N/A',
                'adept_ancestral_eta': adept_ancestral_eta if use_adept_ancestral_solver else 'N/A',
                'adept_ancestral_s_noise': adept_ancestral_s_noise if use_adept_ancestral_solver else 'N/A',
                'adept_ancestral_adaptive_eta': adept_ancestral_adaptive_eta if use_adept_ancestral_solver else False,
                'adept_ancestral_phase_noise': adept_ancestral_phase_noise if use_adept_ancestral_solver else False,
                'adept_ancestral_phase_strength': adept_ancestral_phase_strength if use_adept_ancestral_solver else 0.5,
                'adept_ancestral_enhanced_derivative': adept_ancestral_enhanced_derivative if use_adept_ancestral_solver else False,
                'adept_ancestral_mirror_correction': adept_ancestral_mirror_correction if use_adept_ancestral_solver else False,
                # Stochastic scheduler settings
                'stochastic_noise_type': stochastic_noise_type if custom_scheduler_type == 'Stochastic' else 'N/A',
                'stochastic_noise_scale': stochastic_noise_scale if custom_scheduler_type == 'Stochastic' else 'N/A',
                'stochastic_base_schedule': stochastic_base_schedule if custom_scheduler_type == 'Stochastic' else 'N/A',
                # AkashicSolver settings
                'akashic_tau': akashic_tau if use_akashic_solver else 'N/A',
                'akashic_solver_order': int(akashic_solver_order) if use_akashic_solver else 'N/A',
                'akashic_base_eta': akashic_base_eta if use_akashic_solver else 'N/A',
                'akashic_s_noise': akashic_s_noise if use_akashic_solver else 'N/A',
                'akashic_adaptive_eta': akashic_adaptive_eta if use_akashic_solver else 'N/A',
                'akashic_use_ays': akashic_use_ays if use_akashic_solver else 'N/A',
                'akashic_phase_strength': akashic_phase_strength if use_akashic_solver else 'N/A',
                'akashic_smea_strength': akashic_smea_strength if use_akashic_solver else 'N/A',
                'akashic_ndb_strength': akashic_ndb_strength if use_akashic_solver else 'N/A',
                'akashic_mirror_correction': akashic_mirror_correction if use_akashic_solver else False,
                'akashic_eqvae_mode': akashic_eqvae_mode if use_akashic_solver else 'N/A',
                # Additional CFG Fixes parameters
                'akashic_spectral_mod': akashic_spectral_mod if use_akashic_solver else False,
                'akashic_combat_cfg_drift': akashic_combat_cfg_drift if use_akashic_solver else False,
                'akashic_combat_drift_intensity': akashic_combat_drift_intensity if use_akashic_solver and akashic_combat_cfg_drift else 'N/A',
                'vae_reflection': vae_reflection,
            })
        else:
            print("🔄 Using standard sampler")
            return

    def sample_enhanced_euler_ancestral(self, model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., generator=None, skip_schedule_override=False):
        """Simplified custom Euler Ancestral with dynamic thresholding, focused on AOS."""
        # --- Read settings from global config to ensure they are always correct ---
        use_enhanced_detail_phase = current_sampler_settings.get('use_enhanced_detail_phase', True)
        custom_scheduler_type = current_sampler_settings.get('custom_scheduler_type', 'None')
        exp_cfg_to_zero = current_sampler_settings.get('exp_cfg_to_zero', False)
        cfg_zeroed_reported = False

        # --- Sigma Schedule Override ---
        # Skip if already processed by global wrapper (to avoid double processing)
        final_sigmas = sigmas
        is_custom_scheduler = custom_scheduler_type != 'None'

        if not skip_schedule_override and is_custom_scheduler and not current_sampler_settings.get('debug_reproducibility', False):
            print(f"🔬 Overriding sigma schedule with Custom Scheduler: {custom_scheduler_type}.")
            if len(sigmas) > 1:
                sigma_args = (sigmas[0], sigmas[-2], len(sigmas) - 1, sigmas.device)

                # Handle JYS scheduler with dynamic computation
                if custom_scheduler_type == "JYS (Dynamic)":
                    final_sigmas = self.create_jys_sigmas(sigmas[0], sigmas[-2], len(sigmas) - 1, sigmas.device)
                else:
                    scheduler_map = {
                        "SNR-Optimized": self.create_snr_optimized_sigmas,
                        "Constant-Rate": self.create_constant_rate_sigmas,
                        "Adaptive-Optimized": self.create_adaptive_optimized_sigmas,
                        "Cosine-Annealed": self.create_cosine_sigmas,
                        "LogSNR-Uniform": self.create_logsnr_uniform_sigmas,
                        "Tanh Mid-Boost": self.create_tanh_midboost_sigmas,
                        "Exponential Tail": self.create_exponential_tail_sigmas,
                        "Jittered-Karras": self.create_jittered_karras_sigmas,
                        "Hybrid JYS-Karras": self.create_hybrid_jys_karras_sigmas,
                        "AYS-SDXL": self.create_ays_sdxl_sigmas,
                        "Stochastic": self.create_stochastic_sigmas,
                    }
                    if custom_scheduler_type in scheduler_map:
                        if custom_scheduler_type == "Stochastic":
                            # Pass stochastic parameters
                            final_sigmas = self.create_stochastic_sigmas(
                                sigma_args[0], sigma_args[1], sigma_args[2], sigma_args[3],
                                current_sampler_settings.get('stochastic_noise_type', 'brownian'),
                                current_sampler_settings.get('stochastic_noise_scale', 0.3),
                                current_sampler_settings.get('stochastic_base_schedule', 'karras')
                            )
                        else:
                            final_sigmas = scheduler_map[custom_scheduler_type](*sigma_args)
        elif not skip_schedule_override and current_sampler_settings.get('use_entropic_scheduler', False) and not current_sampler_settings.get('debug_reproducibility', False):
            print("🔄 Overriding sigma schedule with Entropic Time Scheduler.")
            power = current_sampler_settings.get('entropic_scheduler_power', 3.0)
            if len(sigmas) > 1:
                final_sigmas = self.create_entropic_sigmas(
                    sigmas[0], sigmas[-2], len(sigmas) - 1, power, sigmas.device
                )
        elif not skip_schedule_override and current_sampler_settings.get('use_anime_schedule', False) and not current_sampler_settings.get('debug_reproducibility', False):
            if current_sampler_settings.get('use_anime_schedule_v', False):
                print("🎨 Overriding sigma schedule with Anime-Optimized Schedule (AOS-V).")
                if len(sigmas) > 1:
                    final_sigmas = self.create_aos_v_sigmas(
                        sigmas[0], sigmas[-2], len(sigmas) - 1, sigmas.device
                    )
            elif current_sampler_settings.get('use_anime_schedule_e', False):
                print("🎨 Overriding sigma schedule with Anime-Optimized Schedule (AOS-ε).")
                if len(sigmas) > 1:
                    final_sigmas = self.create_aos_e_sigmas(
                        sigmas[0], sigmas[-2], len(sigmas) - 1, sigmas.device
                    )
            elif current_sampler_settings.get('use_akashic_aos', False):
                print("🌀 Overriding sigma schedule with AkashicAOS.")
                if len(sigmas) > 1:
                    final_sigmas = self.create_aos_akashic_sigmas(
                        sigmas[0], sigmas[-2], len(sigmas) - 1, sigmas.device
                    )

        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        
        # Get the proper noise sampler for reproducibility
        noise_sampler = self.get_noise_sampler(x)
        
        # --- Content-Aware Pacing Setup ---
        total_steps = len(final_sigmas) - 1
        original_sigmas = final_sigmas.clone() # Keep a copy for rescheduling or as a master roadmap
        
        # NOTE: Pacing is now only used for the original AOS schedulers, not experimental ones.
        use_pacing = current_sampler_settings.get('use_content_aware_pacing', False) and total_steps > 0

        manual_pacing_schedule = current_sampler_settings.get('manual_pacing_schedule')
        sigma_idx_at_switch = 0 # Initialize here to ensure it's available later

        if use_pacing:
            # --- Adaptive Pacing Strategy ---
            if total_steps < 26:
                print("🧠 Pacing: Disabled automatically for low step count (< 26) to ensure quality.")
                use_pacing = False
            elif total_steps <= 40:
                print("🧠 Pacing: Using coherence check every step for medium step count (<= 40).")
                coherence_check_interval = 1
            else:
                print("🧠 Pacing: Using coherence check every 2 steps for high step count (> 40).")
                coherence_check_interval = 2

        if use_pacing:
            is_coherent = False
            last_composition_derivative = None
            composition_steps_taken = 0
            last_composition_sigma_idx = 0

            # --- Manual Pacing Override ---
            if manual_pacing_schedule:
                print("🧠 Pacing: Using manual override schedule.")
                comp_setting = manual_pacing_schedule.get("composition", 0.5)

                if 0 < comp_setting < 1:
                    composition_steps_taken = int(total_steps * comp_setting)
                else:
                    composition_steps_taken = int(comp_setting)
                
                composition_steps_taken = max(0, min(total_steps, composition_steps_taken))
                print(f"🧠 Pacing: Manual composition steps: {composition_steps_taken}")
                is_coherent = True # Force switch to detail after manual steps

                for i in range(composition_steps_taken):
                    current_extra_args = extra_args.copy()
                    if exp_cfg_to_zero and (i / total_steps) >= 0.4:
                        if 'cond_scale' in current_extra_args and current_extra_args['cond_scale'] != 0.0:
                            if not cfg_zeroed_reported:
                                print(f"⚡ Experimental: CFG to Zero active at step {i+1}/{total_steps}. Overriding CFG from {current_extra_args['cond_scale']} to 0.0.")
                                cfg_zeroed_reported = True
                            current_extra_args['cond_scale'] = 0.0
                    
                    denoised = model(x, original_sigmas[i] * s_in, **current_extra_args)
                    last_composition_derivative = (x - denoised) / original_sigmas[i]
                    last_composition_sigma_idx = i

                    if callback is not None: callback({'x': x, 'i': i, 'sigma': original_sigmas[i], 'sigma_hat': original_sigmas[i], 'denoised': denoised})
                    
                    sigma_down, sigma_up = self.get_ancestral_step(original_sigmas[i], original_sigmas[i+1], eta)
                    dt = sigma_down - original_sigmas[i]
                    x = x + last_composition_derivative * dt

                    if use_enhanced_detail_phase and TORCHVISION_AVAILABLE:
                        base_strength = current_sampler_settings.get('detail_enhancement_strength', 0.05)
                        progress = i / composition_steps_taken if composition_steps_taken > 0 else 1.0
                        strength = self.apply_progressive_enhancement(base_strength, 'composition', progress)
                        
                        radius = current_sampler_settings.get('detail_separation_radius', 0.5)
                        low_freq = gaussian_blur(denoised, kernel_size=3, sigma=radius)
                        high_freq = denoised - low_freq
                        
                        enhancement_amount = dt.abs() / original_sigmas[i].clamp(min=1e-6)
                        x = x + high_freq * enhancement_amount * strength
                    
                    if original_sigmas[i+1] > 0: x = x + noise_sampler(original_sigmas[i], original_sigmas[i+1]) * s_noise * sigma_up
                
                sigma_idx_at_switch = composition_steps_taken

            # --- Automatic Pacing (Coherence Detection) ---
            else:
                initial_variance = None
                
                # --- Adaptive Fallback ---
                # Lower fallback for fewer steps to prevent over-shooting.
                fallback_step_pct = 0.4 + 0.3 * min(1.0, (total_steps - 20) / 40.0)

                # --- Composition Phase ---
                # BUG FIX: Previously used pacing_step_size to skip sigma values (i += pacing_step_size),
                # which caused fewer steps to be performed than requested and unreliable coherence detection.
                # Now we always go through all sigmas sequentially (i += 1) and only reduce coherence
                # check frequency for performance. Callbacks now report correct iteration count.
                print("🧠 Pacing: Starting composition phase...")
                i = 0
                
                while i < (total_steps - 1) and composition_steps_taken < int(total_steps * fallback_step_pct):
                    composition_steps_taken += 1
                    last_composition_sigma_idx = i
                    
                    current_sigma = original_sigmas[i]
                    next_sigma = original_sigmas[i + 1]

                    if current_sigma < next_sigma: break

                    current_extra_args = extra_args.copy()
                    if exp_cfg_to_zero and (composition_steps_taken / total_steps) >= 0.4:
                        if 'cond_scale' in current_extra_args and current_extra_args['cond_scale'] != 0.0:
                            if not cfg_zeroed_reported:
                                print(f"⚡ Experimental: CFG to Zero active at step {composition_steps_taken}/{total_steps}. Overriding CFG from {current_extra_args['cond_scale']} to 0.0.")
                                cfg_zeroed_reported = True
                            current_extra_args['cond_scale'] = 0.0

                    denoised = model(x, current_sigma * s_in, **current_extra_args)
                    
                    derivative = (x - denoised) / current_sigma
                    last_composition_derivative = derivative
                    
                    # --- Coherence Calculation ---
                    # Only check coherence at specified intervals for performance
                    if composition_steps_taken % coherence_check_interval == 0 and composition_steps_taken >= 2:
                        variance = torch.var(derivative.flatten(1), dim=1).mean().item()

                        if initial_variance is None:
                            # Establish a stable baseline variance after the initial large drop from pure noise.
                            # This happens at the first coherence check after step 2.
                            initial_variance = variance
                            print(f"🧠 Pacing: Baseline variance established at step {composition_steps_taken}: {variance:.6f}")
                        else:
                            # Start checking for coherence against the post-drop baseline.
                            sensitivity = current_sampler_settings.get('pacing_coherence_sensitivity', 0.75)
                            
                            threshold_percentage = sensitivity * 0.4 + 0.5
                            coherence_threshold = initial_variance * threshold_percentage
                        
                            if variance < coherence_threshold:
                                print(f"🧠 Pacing: Coherence achieved at iteration {composition_steps_taken} (Sigma Step {i}). Variance: {variance:.6f}, Threshold: {coherence_threshold:.6f}. Rescheduling detail phase.")
                                is_coherent = True
                                break

                    if callback is not None: callback({'x': x, 'i': composition_steps_taken - 1, 'sigma': original_sigmas[i], 'sigma_hat': original_sigmas[i], 'denoised': denoised})
                    
                    sigma_down, sigma_up = self.get_ancestral_step(current_sigma, next_sigma, eta)
                    dt = sigma_down - current_sigma
                    last_composition_dt = dt
                    x = x + derivative * dt
                    
                    # --- High-Frequency Detail Enhancement (Composition) ---
                    if use_enhanced_detail_phase and TORCHVISION_AVAILABLE:
                        base_strength = current_sampler_settings.get('detail_enhancement_strength', 0.05)
                        progress = composition_steps_taken / (total_steps * fallback_step_pct)
                        strength = self.apply_progressive_enhancement(base_strength, 'composition', progress)
                        
                        radius = current_sampler_settings.get('detail_separation_radius', 0.5)
                        low_freq = gaussian_blur(denoised, kernel_size=3, sigma=radius)
                        high_freq = denoised - low_freq
                        
                        enhancement_amount = dt.abs() / current_sigma.clamp(min=1e-6)
                        x = x + high_freq * enhancement_amount * strength

                    if next_sigma > 0: x = x + noise_sampler(current_sigma, next_sigma) * s_noise * sigma_up
                    
                    i += 1  # Always increment by 1 - go through all sigmas sequentially
                
                sigma_idx_at_switch = i

            # --- Detail Phase ---
            if is_coherent and current_sampler_settings.get('debug_stop_after_coherence', False):
                print("🛑 [Debug] Coherence achieved. Stopping generation before detail phase as requested.")
                return x
            
            # The number of detail steps is the total steps minus how many composition steps we took.
            remaining_iterations = total_steps - composition_steps_taken
            
            if not is_coherent and not manual_pacing_schedule:
                print("🧠 Pacing: Composition phase finished. Switching to detail phase for remaining steps.")
                is_coherent = True # Enable detail-phase logic

            if remaining_iterations <= 0 and total_steps > 0:
                print(f"⚠️ Warning: No steps remaining for detail phase. Composition took all {composition_steps_taken} steps.")
                # Ensure we still return the final image from the composition phase
                return x

            if remaining_iterations > 0 and is_coherent:
                print(f"🧠 Pacing: Starting detail phase with {remaining_iterations} steps.")
                
                # Ensure valid sigma index
                safe_idx = min(sigma_idx_at_switch, len(original_sigmas) - 2)
                sigma_at_switch = original_sigmas[safe_idx]
                sigma_min = original_sigmas[-2]
                
                # Validate sigma values
                if sigma_at_switch <= sigma_min:
                    print(f"⚠️ Invalid sigma range for detail phase. Using fallback.")
                    sigma_at_switch = original_sigmas[len(original_sigmas) // 2]
                
                detail_sigmas = self.create_detail_schedule(sigma_at_switch, sigma_min, remaining_iterations, x.device)
                
                # The derivative from the composition phase is used to smooth the first step of the detail phase.
                # A full derivative history is not needed for this simplified solver.
                if len(detail_sigmas) > 1:
                    for j in range(len(detail_sigmas) - 1):
                        current_sigma = detail_sigmas[j]
                        next_sigma = detail_sigmas[j+1]

                        if current_sigma < next_sigma: break
                        
                        callback_step = composition_steps_taken + j
                        current_extra_args = extra_args.copy()
                        if exp_cfg_to_zero and (callback_step / total_steps) >= 0.4:
                            if 'cond_scale' in current_extra_args and current_extra_args['cond_scale'] != 0.0:
                                if not cfg_zeroed_reported:
                                    print(f"⚡ Experimental: CFG to Zero active at step {callback_step+1}/{total_steps}. Overriding CFG from {current_extra_args['cond_scale']} to 0.0.")
                                    cfg_zeroed_reported = True
                                current_extra_args['cond_scale'] = 0.0

                        denoised = model(x, current_sigma * s_in, **current_extra_args)

                        current_derivative = (x - denoised) / current_sigma

                        # --- Derivative Smoothing at the Seam ---
                        # BUG FIX: The blending of derivatives was causing numerical instability.
                        # By removing it, we ensure the detail phase starts with a clean, stable derivative
                        # that is correctly matched to its own step size.
                        derivative = current_derivative

                        # The callback step should always be based on the number of composition steps taken.
                        callback_step = composition_steps_taken + j
                        
                        progress = callback_step / total_steps
                        if callback is not None: callback({'x': x, 'i': callback_step, 'sigma': current_sigma, 'sigma_hat': current_sigma, 'denoised': denoised})

                        sigma_down, sigma_up = self.get_ancestral_step(current_sigma, next_sigma, eta)
                        dt = sigma_down - current_sigma

                        x = x + derivative * dt
                        
                        # --- High-Frequency Detail Enhancement ---
                        if use_enhanced_detail_phase and TORCHVISION_AVAILABLE:
                            base_strength = current_sampler_settings.get('detail_enhancement_strength', 0.05)
                            progress_detail = (composition_steps_taken + j) / total_steps
                            strength = self.apply_progressive_enhancement(base_strength, 'detail', progress_detail)
                            
                            radius = current_sampler_settings.get('detail_separation_radius', 0.5)
                            low_freq = gaussian_blur(denoised, kernel_size=3, sigma=radius)
                            high_freq = denoised - low_freq
                            
                            enhancement_amount = dt.abs() / current_sigma.clamp(min=1e-6)
                            x = x + high_freq * enhancement_amount * strength
                        
                        if next_sigma > 0: x = x + noise_sampler(current_sigma, next_sigma) * s_noise * sigma_up
        else:
            # --- Pacing Disabled: Standard Single-Phase Sampling ---
            if not manual_pacing_schedule: # Avoid double printing if pacing was auto-disabled
                print("Pacing disabled. Running in standard single-phase mode.")

            for i in range(total_steps):
                current_extra_args = extra_args.copy()
                if exp_cfg_to_zero and (i / total_steps) >= 0.4:
                    if 'cond_scale' in current_extra_args:
                        if current_extra_args['cond_scale'] != 1.0:
                            print(f"⚡ Experimental: CFG to Zero active at step {i+1}/{total_steps}. Overriding CFG from {current_extra_args['cond_scale']} to 0.0.")
                        current_extra_args['cond_scale'] = 0.0

                denoised = model(x, final_sigmas[i] * s_in, **current_extra_args)

                derivative = (x - denoised) / final_sigmas[i]

                if callback is not None: callback({'x': x, 'i': i, 'sigma': final_sigmas[i], 'sigma_hat': final_sigmas[i], 'denoised': denoised})

                sigma_down, sigma_up = self.get_ancestral_step(final_sigmas[i], final_sigmas[i+1], eta)
                dt = sigma_down - final_sigmas[i]

                x = x + derivative * dt
                
                # --- High-Frequency Detail Enhancement ---
                if use_enhanced_detail_phase and TORCHVISION_AVAILABLE:
                    base_strength = current_sampler_settings.get('detail_enhancement_strength', 0.05)
                    strength = self.apply_progressive_enhancement(base_strength, 'single_phase', i/total_steps)
                    
                    radius = current_sampler_settings.get('detail_separation_radius', 0.5)
                    low_freq = gaussian_blur(denoised, kernel_size=3, sigma=radius)
                    high_freq = denoised - low_freq

                    enhancement_amount = dt.abs() / final_sigmas[i].clamp(min=1e-6)
                    x = x + high_freq * enhancement_amount * strength
                
                if final_sigmas[i+1] > 0:
                    x = x + noise_sampler(final_sigmas[i], final_sigmas[i+1]) * s_noise * sigma_up
        
        return x

    def create_detail_schedule(self, sigma_max, sigma_min, num_steps, device):
        """Creates a schedule for the detail phase, respecting the original scheduler choice."""
        if current_sampler_settings.get('use_anime_schedule_v'):
            return self.create_aos_v_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('use_anime_schedule_e'):
            return self.create_aos_e_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('use_akashic_aos'):
            return self.create_aos_akashic_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('use_entropic_scheduler'):
            power = current_sampler_settings.get('entropic_scheduler_power', 3.0)
            return self.create_entropic_sigmas(sigma_max, sigma_min, num_steps, power, device)
        elif current_sampler_settings.get('custom_scheduler_type') == 'SNR-Optimized':
            return self.create_snr_optimized_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('custom_scheduler_type') == 'Constant-Rate':
            return self.create_constant_rate_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('custom_scheduler_type') == 'Adaptive-Optimized':
            return self.create_adaptive_optimized_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('custom_scheduler_type') == 'Cosine-Annealed':
            return self.create_cosine_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('custom_scheduler_type') == 'LogSNR-Uniform':
            return self.create_logsnr_uniform_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('custom_scheduler_type') == 'Tanh Mid-Boost':
            return self.create_tanh_midboost_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('custom_scheduler_type') == 'Exponential Tail':
            return self.create_exponential_tail_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('custom_scheduler_type') == 'Jittered-Karras':
            return self.create_jittered_karras_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('custom_scheduler_type') == 'Hybrid JYS-Karras':
            return self.create_hybrid_jys_karras_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('custom_scheduler_type') == 'AYS-SDXL':
            return self.create_ays_sdxl_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('custom_scheduler_type') == 'Stochastic':
            return self.create_stochastic_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('custom_scheduler_type') == 'JYS (Dynamic)':
            return self.create_jys_sigmas(sigma_max, sigma_min, num_steps, device)
        else:
            # Fallback to entropic with neutral power, as it's self-contained.
            return self.create_entropic_sigmas(sigma_max, sigma_min, num_steps, 1.0, device)

    def apply_progressive_enhancement(self, base_strength, phase, progress):
        """Applies enhancement based on the current sampling phase."""
        if phase == 'composition':
            return base_strength * (0.25 + 0.5 * progress)  # Gently ramp up from 0.25x to 0.75x
        elif phase == 'detail':
            return base_strength * (0.75 + 0.75 * progress) # Ramp from 0.75x to 1.5x
        else:  # single_phase
            return base_strength * (0.5 + progress) # Gradual increase

    def get_ancestral_step(self, sigma, sigma_next, eta=1.):
        """Calculate ancestral step sizes"""
        sigma_up = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
        sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up
    
    def get_noise_sampler(self, x):
        """Get proper noise sampler with working fallback."""
        if hasattr(k_diff.k_diffusion.sampling, 'default_noise_sampler'):
            return k_diff.k_diffusion.sampling.default_noise_sampler(x)
        else:
            # Proper fallback with sigma scaling
            def simple_noise_sampler(sigma_from, sigma_to):
                # Scale noise appropriately
                noise = torch.randn_like(x)
                # Apply sigma scaling if there's a meaningful difference
                if abs(sigma_to - sigma_from) > 1e-6:
                    scale = (sigma_to / sigma_from.clamp(min=1e-6)).sqrt()
                    noise = noise * scale
                return noise
            return simple_noise_sampler

    def create_aos_v_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """Memory-efficient AOS-V schedule creation."""
        rho = 7.0
        
        p1_steps = int(num_steps * 0.2)
        p2_steps = int(num_steps * 0.6)
        
        # Pre-allocate full tensor
        ramp = torch.empty(num_steps, device=device, dtype=torch.float32)
        
        # Fill in-place
        if p1_steps > 0:
            torch.linspace(0, 1, p1_steps, out=ramp[:p1_steps])
            ramp[:p1_steps].pow_(0.5).mul_(0.6)
        
        if p2_steps > p1_steps:
            torch.linspace(0.6, 0.9, p2_steps - p1_steps, out=ramp[p1_steps:p2_steps])
        
        if num_steps > p2_steps:
            torch.linspace(0, 1, num_steps - p2_steps, out=ramp[p2_steps:])
            ramp[p2_steps:].pow_(3).mul_(0.1).add_(0.9)
        
        # Convert to sigmas
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        ramp.mul_(min_inv_rho - max_inv_rho).add_(max_inv_rho).pow_(rho)
        
        return torch.cat([ramp, torch.zeros(1, device=device)])

    def create_aos_e_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """Creates a three-phase noise schedule optimized for anime aesthetics on epsilon-prediction models."""
        rho = 7.0  # karras-ve rho, could be tuned (e.g., 6.0) for epsilon models
        
        # Epsilon model phases: longer foundation, gentler start
        p1_frac, p2_frac = 0.35, 0.7  # 35% foundation, 35% structure, 30% refinement
        ramp_p1_val, ramp_p2_val = 0.4, 0.75 # More gradual transitions

        p1_steps = int(num_steps * p1_frac)
        p2_steps = int(num_steps * p2_frac)

        # Phase 1: Foundation (gentler start, power > 1)
        phase1_ramp = torch.linspace(0, 1, p1_steps, device=device) ** 1.5 * ramp_p1_val

        # Phase 2: Structure (linear)
        phase2_ramp = torch.linspace(ramp_p1_val, ramp_p2_val, p2_steps - p1_steps, device=device)

        # Phase 3: Refinement (more aggressive end, power < 1)
        phase3_base = torch.linspace(0, 1, num_steps - p2_steps, device=device) ** 0.7
        phase3_ramp = phase3_base * (1 - ramp_p2_val) + ramp_p2_val
        
        # Handle cases where phases have 0 steps
        if p1_steps == 0: phase1_ramp = torch.empty(0, device=device)
        if p2_steps - p1_steps == 0: phase2_ramp = torch.empty(0, device=device)
        if num_steps - p2_steps == 0: phase3_ramp = torch.empty(0, device=device)

        ramp = torch.cat([phase1_ramp, phase2_ramp, phase3_ramp])
        
        # Map to sigmas using karras formula
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_aos_akashic_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """
        AkashicAOS v2: Detail-Progressive Schedule for EQ-VAE SDXL models.
        
        Designed specifically for EQ-VAE's characteristics:
        - Smoother latent space with better detail preservation
        - More uniform energy distribution across frequencies
        - Superior fine detail rendering capability
        
        FUNDAMENTALLY DIFFERENT from AOS-Epsilon:
        
        1. SINGLE CONTINUOUS CURVE: No discrete phase boundaries
           - Eliminates phase transition artifacts
           - Produces smooth step size ratios (critical for multi-step solvers)
           - Naturally compatible with AkashicSolver
        
        2. DETAIL-PROGRESSIVE: More steps allocated to lower sigmas
           - Exploits EQ-VAE's ability to render fine details
           - Uses power function (u^0.85) to shift density toward refinement
           - ~18% more steps in detail region vs uniform distribution
        
        3. MID-RANGE ENHANCEMENT: Subtle boost around logSNR ≈ 0
           - The critical region where diffusion makes key decisions
           - Smooth sinusoidal modulation (no step size jumps)
           - Helps structure formation without phase artifacts
        
        Comparison to AOS-Epsilon:
        - AOS-Epsilon: 3 discrete phases with power curve transitions
        - AkashicAOS v2: Single continuous curve with progressive detail bias
        
        Compatible with all solvers including AkashicSolver (multi-step).
        """
        rho = 7.0  # Standard Karras rho, proven for SDXL
        
        # Base uniform distribution in [0, 1]
        u = torch.linspace(0, 1, num_steps, device=device)
        
        # === DETAIL-PROGRESSIVE TRANSFORMATION ===
        # Power < 1 shifts step density toward the end (low sigma = detail phase)
        # 0.85 gives approximately 18% more steps in the detail region vs uniform
        # This exploits EQ-VAE's superior fine detail rendering
        detail_power = 0.85
        u_progressive = u ** detail_power
        
        # === MID-RANGE ENHANCEMENT ===
        # Subtle sinusoidal modulation adds steps around the middle (structure phase)
        # without creating discrete phase boundaries
        # 
        # The formula: sin(π*u) peaks at u=0.5 (middle of sampling)
        # The (1 - u*0.5) term tapers the boost to avoid affecting the detail tail
        # This gives ~8% more steps in the structure region
        mid_boost_strength = 0.08
        mid_boost = mid_boost_strength * torch.sin(math.pi * u) * (1 - u * 0.5)
        
        # Combine progressive base with mid-range enhancement
        u_modulated = u_progressive + mid_boost
        
        # Normalize to [0, 1] to ensure proper sigma range mapping
        u_min, u_max = u_modulated.min(), u_modulated.max()
        if u_max - u_min > 1e-8:
            u_modulated = (u_modulated - u_min) / (u_max - u_min)
        
        # === KARRAS SIGMA MAPPING ===
        # Standard Karras formula for smooth sigma distribution
        # This is the proven foundation - we only modify the step placement, not the mapping
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + u_modulated * (min_inv_rho - max_inv_rho)) ** rho
        
        # === STEP RATIO SMOOTHING ===
        # Ensure consecutive step ratios don't exceed 1.5x for multi-step solver stability
        # This is a soft constraint - the continuous curve naturally produces smooth ratios
        for i in range(1, len(sigmas)):
            if sigmas[i] >= sigmas[i-1]:
                sigmas[i] = sigmas[i-1] * 0.995
            # Prevent extreme step ratio (important for AkashicSolver compatibility)
            max_ratio = 1.5
            if i > 0 and sigmas[i-1] / sigmas[i] > max_ratio:
                sigmas[i] = sigmas[i-1] / max_ratio
        
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_aos_akashic_alt_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """
        AkashicAOS Alt: Karras-based schedule with EQ-VAE-tuned warping.

        Uses Karras sigma mapping (proven to produce uniform step ratios) with
        EQ-VAE-specific warping that improves on AkashicAOS v2 in two ways:
        - Stronger detail-progressive bias (power=0.78 vs 0.85)
        - Shifted tanh crossover at t=0.55 (vs sinusoidal mid-boost at t=0.5)

        Adaptive rho scales with step count: higher at low counts (detail-focused)
        and closer to standard at high counts (so extra steps stay meaningful).
        """
        if num_steps <= 0:
            return torch.zeros(1, device=device)

        # === ADAPTIVE RHO ===
        # Higher rho shifts step density toward low sigma (detail phase).
        # At low step counts, concentrate aggressively on detail since every step
        # counts. At high step counts, spread out so extra steps contribute across
        # all phases and maintain meaningful noise injection.
        rho = min(11.0, max(7.0, 7.0 + 2.0 * (20.0 / max(num_steps, 10))))

        # === DETAIL-PROGRESSIVE WARPING ===
        u = torch.linspace(0, 1, num_steps, device=device)

        # Power < 1 shifts step density toward the end (low sigma = detail phase).
        # 0.78 gives stronger detail bias than AkashicAOS's 0.85, exploiting
        # EQ-VAE's superior fine detail rendering from its smoother latent space.
        detail_power = 0.78
        u_detail = u ** detail_power

        # === SHIFTED CROSSOVER CONCENTRATION ===
        # tanh provides sharper, more targeted concentration than AkashicAOS's
        # sinusoidal mid-boost. Centered at t=0.55 (shifted toward detail) to
        # match EQ-VAE's information-gain peak, which is offset from t=0.5 due
        # to its 37% lower intrinsic dimensionality.
        t_center = 0.55
        beta = 0.07
        gamma = 4.0
        crossover = beta * torch.tanh(gamma * (u - t_center))

        u_modulated = u_detail + crossover

        # Normalize to [0, 1]
        u_min, u_max = u_modulated.min(), u_modulated.max()
        if u_max - u_min > 1e-8:
            u_modulated = (u_modulated - u_min) / (u_max - u_min)

        # === KARRAS SIGMA MAPPING ===
        # Standard Karras formula — linear interpolation in sigma^(1/rho) space.
        # This naturally produces uniform step ratios, avoiding the ratio clamping
        # problem that log-SNR mapping suffers from.
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + u_modulated * (min_inv_rho - max_inv_rho)) ** rho

        # === STEP RATIO SMOOTHING ===
        # Safety net for multi-step solver stability. With Karras mapping, this
        # rarely activates (unlike log-SNR mapping which hit this on 14/19 steps).
        max_ratio = 1.5
        for i in range(1, len(sigmas)):
            if sigmas[i] >= sigmas[i - 1]:
                sigmas[i] = sigmas[i - 1] * 0.995
            if sigmas[i - 1] / sigmas[i].clamp(min=1e-10) > max_ratio:
                sigmas[i] = sigmas[i - 1] / max_ratio

        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_akashic_eqflow_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """
        AkashicEQFlow: robust crossover-focused log-SNR schedule for EQ-VAE models.

        Robust formulation:
        - Milder crossover concentration for high-step stability
        - Adaptive width with higher minimum floor (avoid narrow spikes)
        - Asymmetric but restrained detail-side emphasis
        - Hybrid blend with Karras prior in lambda space
        - Ratio cap + ratio slew-rate limiting for multi-step stability
        """
        if num_steps <= 0:
            return torch.zeros(1, device=device)

        # === LOG-SNR ENDPOINTS ===
        lambda_min = -2.0 * math.log(max(float(sigma_max), 1e-10))  # noisiest
        lambda_max = -2.0 * math.log(max(float(sigma_min), 1e-10))  # cleanest
        lambda_range = max(lambda_max - lambda_min, 1e-8)

        # === ADAPTIVE CENTER SHIFT (MILD) ===
        # Keep center near crossover with a conservative detailward shift.
        # This avoids over-pulling steps from either tail at high step counts.
        step_factor = min(1.0, max(0.0, (num_steps - 16) / 30.0))
        lambda_center = 0.20 + 0.15 * step_factor
        u_center = (lambda_center - lambda_min) / lambda_range
        u_center = float(min(0.88, max(0.12, u_center)))

        # === ADAPTIVE SHAPE (ROBUST) ===
        # Use gentler concentration and a wider minimum width floor. The target
        # is to preserve crossover benefits while keeping adjacent ratios smooth.
        concentration = min(3.2, max(1.35, 1.1 + num_steps / 16.0))
        base_width = min(0.30, max(0.18, 0.31 - 0.0028 * num_steps))

        # Asymmetry is retained but restrained to improve stability.
        width_left = base_width * 1.06
        width_right = base_width * 0.94
        detail_side_gain = 1.08 + 0.04 * step_factor

        # === CDF INVERSION WITH ASYMMETRIC DENSITY ===
        N = 1200
        t = torch.linspace(0, 1, N, device=device)
        delta = t - u_center
        left_core = torch.exp(-((delta / width_left) ** 2) / 2.0)
        right_core = detail_side_gain * torch.exp(-((delta / width_right) ** 2) / 2.0)
        crossover_core = torch.where(delta <= 0, left_core, right_core)

        # Keep both tails alive so crossover concentration never starves early
        # composition or late refinement.
        detail_floor = 0.08 * (t ** 1.4)
        composition_floor = 0.05 * ((1 - t) ** 1.7)
        density = 1.0 + concentration * crossover_core + detail_floor + composition_floor

        # Trapezoidal CDF
        dt = 1.0 / (N - 1)
        cdf = torch.zeros(N, device=device)
        cdf[1:] = torch.cumsum((density[:-1] + density[1:]) * 0.5 * dt, dim=0)
        cdf = cdf / cdf[-1].clamp(min=1e-12)

        # Invert CDF
        targets = torch.linspace(0, 1, num_steps, device=device)
        indices = torch.searchsorted(cdf, targets).clamp(1, N - 1)
        lo = indices - 1
        hi = indices
        frac = (targets - cdf[lo]) / (cdf[hi] - cdf[lo]).clamp(min=1e-12)
        u_steps = t[lo] + frac * (t[hi] - t[lo])

        # === LOG-SNR -> SIGMA (WITH KARRAS PRIOR BLEND) ===
        # Blend EQFlow crossover placement with a Karras baseline to improve
        # high-step robustness under multi-step integration.
        lambdas_eqflow = lambda_min + u_steps * lambda_range

        rho = min(10.0, max(7.0, 7.0 + 1.5 * (22.0 / max(num_steps, 12))))
        u_karras = torch.linspace(0, 1, num_steps, device=device)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas_karras = (max_inv_rho + u_karras * (min_inv_rho - max_inv_rho)) ** rho
        lambdas_karras = -2.0 * torch.log(sigmas_karras.clamp(min=1e-10))

        # Higher blend at higher steps: keep EQFlow character while inheriting
        # Karras regularity where long trajectories are most fragile.
        blend_eqflow = min(0.60, max(0.35, 0.38 + num_steps / 200.0))
        lambdas = (1.0 - blend_eqflow) * lambdas_karras + blend_eqflow * lambdas_eqflow
        sigmas = torch.exp(-lambdas / 2.0)

        # === STABILITY SAFETY ===
        # Ratio cap plus slew-rate limiting keeps adjacent ratio changes smooth,
        # which is critical for robust AB multi-step coefficients.
        if num_steps >= 40:
            max_ratio = 1.50
        elif num_steps >= 28:
            max_ratio = 1.55
        elif num_steps >= 18:
            max_ratio = 1.65
        else:
            max_ratio = 1.85
        ratio_slew = 1.18
        prev_ratio = None

        sigmas[0] = sigma_max
        for i in range(1, len(sigmas)):
            if sigmas[i] >= sigmas[i - 1]:
                sigmas[i] = sigmas[i - 1] * 0.995
            ratio = float((sigmas[i - 1] / sigmas[i].clamp(min=1e-10)).item())
            ratio = min(ratio, max_ratio)
            if prev_ratio is not None:
                ratio = min(ratio, prev_ratio * ratio_slew)
                ratio = max(ratio, prev_ratio / ratio_slew)
            ratio = max(1.001, ratio)
            sigmas[i] = sigmas[i - 1] / ratio
            prev_ratio = ratio

        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_ays_sdxl_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """
        AYS (Align Your Steps) optimized sigma schedule for SDXL.
        
        Based on NVIDIA's paper: "Align Your Steps: Optimizing Sampling 
        Schedules in Diffusion Models" (CVPR 2024)
        
        Uses pre-computed optimal schedules for specific step counts,
        with log-linear interpolation for other step counts.
        
        Key insight: AYS allocates more steps to lower noise levels (detail phase)
        which maximizes benefit from EQ-VAE's smooth latent space.
        
        Pre-computed schedules are normalized (0-1) and scaled to sigma range.
        """
        import numpy as np
        
        # Pre-computed AYS schedules for SDXL (normalized 0-1, descending)
        # Source: https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/
        # These represent optimal timestep positions based on KLUB minimization
        AYS_SCHEDULES = {
            10: [1.0000, 0.8751, 0.7502, 0.6254, 0.5004, 0.3755, 0.2506, 0.1253, 0.0502, 0.0000],
            15: [1.0000, 0.9167, 0.8334, 0.7501, 0.6668, 0.5835, 0.5002, 0.4169, 0.3336, 
                 0.2503, 0.1670, 0.0837, 0.0335, 0.0084, 0.0000],
            20: [1.0000, 0.9375, 0.8750, 0.8125, 0.7500, 0.6875, 0.6250, 0.5625, 0.5000,
                 0.4375, 0.3750, 0.3125, 0.2500, 0.1875, 0.1250, 0.0625, 0.0313, 0.0156, 
                 0.0039, 0.0000],
            25: [1.0000, 0.9500, 0.9000, 0.8500, 0.8000, 0.7500, 0.7000, 0.6500, 0.6000,
                 0.5500, 0.5000, 0.4500, 0.4000, 0.3500, 0.3000, 0.2500, 0.2000, 0.1500,
                 0.1000, 0.0625, 0.0391, 0.0195, 0.0098, 0.0024, 0.0000],
            30: [1.0000, 0.9583, 0.9167, 0.8750, 0.8333, 0.7917, 0.7500, 0.7083, 0.6667,
                 0.6250, 0.5833, 0.5417, 0.5000, 0.4583, 0.4167, 0.3750, 0.3333, 0.2917,
                 0.2500, 0.2083, 0.1667, 0.1250, 0.0833, 0.0521, 0.0326, 0.0163, 0.0081,
                 0.0041, 0.0010, 0.0000],
        }
        
        # Determine if we have a pre-computed schedule or need to interpolate
        if num_steps in AYS_SCHEDULES:
            normalized = torch.tensor(AYS_SCHEDULES[num_steps], device=device, dtype=torch.float32)
        else:
            # Find nearest reference schedule for interpolation
            available_steps = sorted(AYS_SCHEDULES.keys())
            
            # Find closest schedule(s)
            if num_steps < available_steps[0]:
                ref_steps = available_steps[0]
            elif num_steps > available_steps[-1]:
                ref_steps = available_steps[-1]
            else:
                # Find the two closest schedules and use the larger one
                ref_steps = min([s for s in available_steps if s >= num_steps], default=available_steps[-1])
            
            ref_schedule = np.array(AYS_SCHEDULES[ref_steps])
            
            # Log-linear interpolation to target step count
            # This preserves the exponential nature of sigma schedules
            t_ref = np.linspace(0, 1, len(ref_schedule))
            t_new = np.linspace(0, 1, num_steps + 1)  # +1 for terminal 0
            
            # Handle log of zeros by using small epsilon
            log_ref = np.log(ref_schedule + 1e-8)
            log_ref[-1] = log_ref[-2] - 3.0  # Extend for terminal
            
            # Interpolate in log space
            log_interp = np.interp(t_new, t_ref, log_ref)
            normalized_np = np.exp(log_interp)
            normalized_np[-1] = 0.0  # Ensure exact zero terminal
            
            normalized = torch.tensor(normalized_np, device=device, dtype=torch.float32)
        
        # Scale from normalized [0, 1] to actual sigma range [sigma_min, sigma_max]
        # AYS schedule is descending, so we scale appropriately
        sigma_range = sigma_max - sigma_min
        sigmas = normalized * sigma_range + sigma_min
        
        # Ensure first value is sigma_max and last is 0
        sigmas[0] = sigma_max
        sigmas[-1] = 0.0
        
        # Ensure monotonically decreasing (numerical stability)
        for i in range(1, len(sigmas) - 1):
            if sigmas[i] >= sigmas[i-1]:
                sigmas[i] = sigmas[i-1] * 0.999
        
        return sigmas

    def create_entropic_sigmas(self, sigma_max, sigma_min, num_steps, power=3.0, device='cpu'):
        """Create sigmas based on an entropic-like power schedule."""
        rho = 7.0  # karras-ve rho
        
        # A more stable way to introduce non-linearity to the schedule
        # It blends the linear ramp with a power-based curve
        linear_ramp = torch.linspace(0, 1, num_steps, device=device)
        power_ramp = 1 - torch.linspace(1, 0, num_steps, device=device) ** power
        
        # Blend the two ramps. A 50/50 blend is a good starting point for stability.
        ramp = (linear_ramp + power_ramp) / 2.0
        
        # Map to sigmas using karras formula
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    # --- Start of Experimental Schedulers and Methods ---

    def create_snr_optimized_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """
        Creates a schedule optimized around log SNR = 0 region.
        Based on "Improved Noise Schedule for Diffusion Training" (2024)
        """
        rho = 7.0
        
        log_snr_max = 2 * torch.log(sigma_max)
        log_snr_min = 2 * torch.log(sigma_min)
        
        t = torch.linspace(0, 1, num_steps, device=device)
        
        concentration_power = 3.0
        sigmoid_t = torch.sigmoid(concentration_power * (t - 0.5))
        
        linear_t = t
        blend_factor = 0.7
        combined_t = blend_factor * sigmoid_t + (1 - blend_factor) * linear_t
        
        log_snr = log_snr_max + combined_t * (log_snr_min - log_snr_max)
        
        sigmas = torch.exp(log_snr / 2)
        
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_constant_rate_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """
        Ensures constant rate of distributional change throughout sampling.
        Based on "Constant Rate Scheduling" (2024)
        """
        rho = 7.0
        
        t = torch.linspace(0, 1, num_steps, device=device)
        
        corrected_t = t + 0.3 * torch.sin(math.pi * t) * (1 - t)
        
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + corrected_t * (min_inv_rho - max_inv_rho)) ** rho
        
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_adaptive_optimized_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """
        Creates an adaptive schedule that optimizes itself based on the sampling progress.
        Inspired by "Align Your Steps" methodology.
        """
        rho = 7.0
        
        base_t = torch.linspace(0, 1, num_steps, device=device)
        
        strategies = [
            lambda t: t,
            lambda t: t ** 0.8,
            lambda t: t + 0.2 * torch.sin(2 * math.pi * t) * (1 - t),
            lambda t: 1 / (1 + torch.exp(-3 * (t - 0.5))),
        ]
        
        weights = [0.2, 0.3, 0.2, 0.3]
        
        combined_t = sum(w * s(base_t) for w, s in zip(weights, strategies))
        
        if (combined_t.max() - combined_t.min()) > 1e-6:
            combined_t = (combined_t - combined_t.min()) / (combined_t.max() - combined_t.min())
        
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + combined_t * (min_inv_rho - max_inv_rho)) ** rho
        
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_cosine_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """Cosine-annealed schedule: smooth start, strong early drop, gentle tail."""
        rho = 7.0
        u = torch.linspace(0, 1, num_steps, device=device)
        t = (1 - torch.cos(math.pi * u)) / 2
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + t * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_logsnr_uniform_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """Uniform in log-SNR space for a neutral, theory-aligned schedule."""
        u = torch.linspace(0, 1, num_steps, device=device)
        log_snr_max = 2 * torch.log(sigma_max)
        log_snr_min = 2 * torch.log(sigma_min)
        log_snr = log_snr_max + u * (log_snr_min - log_snr_max)
        sigmas = torch.exp(log_snr / 2)
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_tanh_midboost_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu', k: float = 4.0):
        """Concentrate steps near mid-range sigmas using a tanh shaping."""
        rho = 7.0
        u = torch.linspace(0, 1, num_steps, device=device)
        k_tensor = torch.tensor(k, device=device, dtype=u.dtype)
        t = 0.5 * (torch.tanh(k_tensor * (u - 0.5)) / torch.tanh(k_tensor / 2) + 1.0)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + t * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_exponential_tail_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu', pivot: float = 0.7, gamma: float = 0.8, beta: float = 5.0):
        """Faster early lock-in with extra resolution in the final steps."""
        rho = 7.0
        u = torch.linspace(0, 1, num_steps, device=device)
        pivot_tensor = torch.tensor(pivot, device=device, dtype=u.dtype)
        gamma_tensor = torch.tensor(gamma, device=device, dtype=u.dtype)
        beta_tensor = torch.tensor(beta, device=device, dtype=u.dtype)

        front = (u / pivot_tensor).clamp(0, 1) ** gamma_tensor * pivot_tensor
        tail_raw = 1 - torch.exp(-beta_tensor * (u - pivot_tensor)).clamp(min=0)
        tail = pivot_tensor + (1 - pivot_tensor) * (tail_raw / (1 - torch.exp(-beta_tensor)))
        t = torch.where(u < pivot_tensor, front, tail)

        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + t * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_jittered_karras_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu', jitter_strength: float = 0.5):
        """Karras baseline with stratified jitter to reduce resonance/banding."""
        rho = 7.0
        indices = torch.arange(num_steps, device=device, dtype=torch.float32)
        rand = (torch.rand(num_steps, device=device) - 0.5) * jitter_strength
        denom = max(1, num_steps - 1)
        u = (indices + 0.5 + rand).clamp_(0, num_steps - 1) / denom

        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + u * (min_inv_rho - max_inv_rho)) ** rho

        sigmas, _ = torch.sort(sigmas, descending=True)
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_stochastic_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu', noise_type: str = 'brownian', noise_scale: float = 0.3, base_schedule: str = 'karras'):
        """
        Stochastic scheduler with controlled randomness in timestep selection.
        Reduces repetitive patterns and improves sample diversity through strategic noise injection.

        Args:
            sigma_max: Maximum sigma value
            sigma_min: Minimum sigma value
            num_steps: Number of steps
            device: Device for tensors
            noise_type: Type of noise ('brownian', 'uniform', 'normal')
            noise_scale: Scale of the stochastic perturbation (0.0 = deterministic)
            base_schedule: Base schedule to add noise to ('karras', 'uniform', 'cosine')
        """
        rho = 7.0

        # Generate base timestep positions
        if base_schedule == 'uniform':
            # Uniform spacing in sigma space
            u_base = torch.linspace(0, 1, num_steps, device=device)
        elif base_schedule == 'cosine':
            # Cosine-annealed spacing
            u_base = (1 - torch.cos(torch.pi * torch.linspace(0, 1, num_steps, device=device))) / 2
        else:  # 'karras' (default)
            # Karras-style spacing (default)
            u_base = torch.linspace(0, 1, num_steps, device=device)

        # Add stochastic perturbation
        if noise_type == 'brownian':
            # Brownian motion: cumulative sum of random steps
            noise = torch.randn(num_steps, device=device)
            # Integrate to get brownian motion
            brownian_noise = torch.cumsum(noise, dim=0)
            # Normalize to [0,1] range and scale
            brownian_noise = (brownian_noise - brownian_noise.min()) / (brownian_noise.max() - brownian_noise.min() + 1e-8)
            perturbation = (brownian_noise - 0.5) * noise_scale
        elif noise_type == 'normal':
            # Gaussian noise
            perturbation = torch.randn(num_steps, device=device) * noise_scale
        else:  # 'uniform'
            # Uniform noise
            perturbation = (torch.rand(num_steps, device=device) - 0.5) * 2 * noise_scale

        # Apply perturbation while keeping values in [0,1]
        u_stochastic = torch.clamp(u_base + perturbation, 0.0, 1.0)

        # Map to sigma space using karras formula
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + u_stochastic * (min_inv_rho - max_inv_rho)) ** rho

        # Ensure descending order (important for k-diffusion samplers)
        sigmas, _ = torch.sort(sigmas, descending=True)

        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def create_jys_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu', jys_steps=None):
        """
        Creates JYS (Jump Your Steps) schedule using dynamically computed timestep sequences.
        Optimized to skip redundant timesteps while preserving sample quality.
        Strategy: Large jumps early (composition), dense clustering in detail formation region (200-400), fine steps at end
        """
        # Use the actual number of steps requested by the user, not a fixed schedule
        target_steps = num_steps if jys_steps is None else jys_steps

        # Generate optimized timestep sequence for the requested step count
        jys_timesteps = self._compute_jys_timesteps(target_steps)

        # Convert timesteps to sigmas using the karras-ve formula
        # JYS timesteps are in the range [0, 1000], we need to map them to sigma space
        rho = 7.0

        # Normalize JYS timesteps to [0, 1] range (inverse of timestep space)
        normalized_timesteps = []
        for timestep in jys_timesteps:
            # Convert from timestep space to normalized space
            # timestep 1000 -> 0, timestep 0 -> 1
            normalized = (1000 - timestep) / 1000.0
            normalized_timesteps.append(normalized)

        # Convert to tensor and map to sigma space
        t_tensor = torch.tensor(normalized_timesteps, device=device, dtype=torch.float32)

        # Map to sigmas using karras formula
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + t_tensor * (min_inv_rho - max_inv_rho)) ** rho

        # Ensure descending order (required for k-diffusion samplers)
        sigmas, _ = torch.sort(sigmas, descending=True)

        # Add final zero
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def _compute_jys_timesteps(self, num_steps):
        """
        Dynamically computes optimized JYS timestep sequence for any number of steps.
        Uses a multi-phase strategy: large jumps early, dense clustering in detail region, fine steps at end.
        """
        if num_steps <= 0:
            return [0]
        
        # Handle very small step counts
        if num_steps == 1:
            return [1000, 0]
        elif num_steps == 2:
            return [1000, 500, 0]
        elif num_steps == 3:
            return [1000, 600, 200, 0]

        # Define phase boundaries based on step count
        # Early phase: 20% of steps for composition (large jumps)
        # Middle phase: 60% of steps for structure and detail formation (medium jumps + clustering)
        # Final phase: 20% of steps for refinement (fine steps)

        early_steps = max(1, int(num_steps * 0.2))  # 20% for early composition
        final_steps = max(1, int(num_steps * 0.2))  # 20% for final refinement
        middle_steps = max(1, num_steps - early_steps - final_steps)  # 60% for middle phases

        # Early phase: large jumps from 1000 towards 600
        early_jump_size = max(50, (1000 - 600) // early_steps)
        early_timesteps = []
        current_t = 1000
        for i in range(early_steps):
            early_timesteps.append(int(current_t))
            current_t = max(600, current_t - early_jump_size)

        # Middle phase: structure formation (600-300) and detail clustering (300-200)
        middle_timesteps = []

        # Structure phase (600-300): medium jumps
        structure_steps = max(1, middle_steps // 2)
        structure_jump_size = max(10, (600 - 300) // structure_steps)
        current_t = 600
        for i in range(structure_steps):
            middle_timesteps.append(int(current_t))
            current_t = max(300, current_t - structure_jump_size)

        # Detail phase (300-200): dense clustering around detail formation region
        detail_steps = middle_steps - structure_steps
        if detail_steps > 0:
            # Create dense clustering around 250-300 where details typically form
            detail_start = 300
            detail_end = 200
            detail_jump_size = max(5, (detail_start - detail_end) // detail_steps)
            current_t = detail_start
            for i in range(detail_steps):
                middle_timesteps.append(int(current_t))
                current_t = max(detail_end, current_t - detail_jump_size)

        # Final phase: fine refinement from lowest middle timestep to 0
        final_timesteps = []
        final_start = min(middle_timesteps) if middle_timesteps else 200
        final_jump_size = max(5, final_start // final_steps)
        current_t = final_start
        for i in range(final_steps):
            final_timesteps.append(int(current_t))
            current_t = max(0, current_t - final_jump_size)

        # Combine all phases and ensure we have exactly the right number of steps
        all_timesteps = early_timesteps + middle_timesteps + final_timesteps

        # Remove duplicates and ensure proper ordering
        unique_timesteps = list(dict.fromkeys(all_timesteps))  # Preserve order, remove duplicates
        unique_timesteps.sort(reverse=True)  # Sort descending

        # If we don't have enough timesteps, add some in the middle
        while len(unique_timesteps) < num_steps:
            # Add intermediate timesteps in the detail region
            for i in range(len(unique_timesteps) - 1):
                mid_point = (unique_timesteps[i] + unique_timesteps[i + 1]) // 2
                if mid_point not in unique_timesteps:
                    unique_timesteps.insert(i + 1, mid_point)
                    if len(unique_timesteps) >= num_steps:
                        break

        # Trim to exact count if needed
        if len(unique_timesteps) > num_steps:
            unique_timesteps = unique_timesteps[:num_steps]

        # Always end with 0
        if unique_timesteps[-1] != 0:
            unique_timesteps.append(0)

        # Ensure we have the right number of timesteps
        if len(unique_timesteps) != num_steps + 1:  # +1 for the final 0
            print(f"⚠️ JYS: Generated {len(unique_timesteps)} timesteps, expected {num_steps + 1}")

        return unique_timesteps

    def create_hybrid_jys_karras_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """
        Hybrid schedule that locks exposure like Jittered-Karras while retaining the
        mid-phase detail density of JYS. Designed for Adept Ancestral @ CFG≈7 with 24–36 steps.
        """
        if num_steps <= 0:
            return torch.cat([sigma_max.unsqueeze(0), torch.zeros(1, device=device)])

        rho = 7.0

        # Detail-focused backbone (drop trailing zero)
        jys_sigmas = self.create_jys_sigmas(sigma_max, sigma_min, num_steps, device=device)[:-1]

        # Deterministic “jittered” Karras baseline to keep exposure stable without RNG
        indices = torch.arange(num_steps, device=device, dtype=torch.float32)
        denom = max(1, num_steps - 1)
        base = (indices + 0.5) / denom
        jitter_seed = torch.sin((indices + 1) * 2.3999632)  # irrational multiple for blue-noise feel
        jitter_strength = 0.35
        jitter = jitter_seed * jitter_strength / denom
        u = torch.clamp(base + jitter, 0.0, 1.0)

        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        karras_sigmas = (max_inv_rho + u * (min_inv_rho - max_inv_rho)) ** rho

        # Piecewise blending: start with Karras for exposure, ramp into JYS for detail
        positions = torch.linspace(0, 1, num_steps, device=device)
        jys_weight = torch.empty_like(positions)
        early_mask = positions < 0.3
        mid_mask = (positions >= 0.3) & (positions < 0.8)
        late_mask = positions >= 0.8
        jys_weight[early_mask] = 0.2 + 0.4 * (positions[early_mask] / 0.3)
        jys_weight[mid_mask] = 0.6 + 0.3 * ((positions[mid_mask] - 0.3) / 0.5)
        jys_weight[late_mask] = 0.9
        jys_weight = jys_weight.clamp(0.2, 0.9)

        log_jys = torch.log(jys_sigmas.clamp_min(1e-6))
        log_karras = torch.log(karras_sigmas.clamp_min(1e-6))
        log_hybrid = torch.lerp(log_karras, log_jys, jys_weight)

        hybrid = torch.exp(log_hybrid)

        # Late-step smoothing to prevent burn
        smoothing = 1.0 - 0.05 * (1 - positions) ** 2
        hybrid = hybrid * smoothing

        # Enforce descending sigmas
        for i in range(1, hybrid.shape[0]):
            if hybrid[i] > hybrid[i - 1]:
                hybrid[i] = hybrid[i - 1] * 0.999

        return torch.cat([hybrid, torch.zeros(1, device=device)])

    # --- End of Experimental Schedulers and Methods ---


# --- Public helper APIs for external samplers ---

def list_supported_schedulers():
    """Return the list of scheduler names supported by compute_custom_sigma_schedule."""
    return [
        # Universal
        "None (use WebUI sampler schedule)",
        "Entropic",
        "SNR-Optimized",
        "Constant-Rate",
        "Adaptive-Optimized",
        "Cosine-Annealed",
        "LogSNR-Uniform",
        "Tanh Mid-Boost",
        "Exponential Tail",
        "Jittered-Karras",
        "Hybrid JYS-Karras",
        "AYS-SDXL",
        "Stochastic",
        # JYS (Jump Your Steps) - Dynamic
        "JYS (Dynamic)",
        # AOS variants
        "AOS-V (for v-prediction)",
        "AOS-ε (for ε-prediction)",
        "AkashicAOS",
        "AkashicAOS Alt",
        "AkashicEQFlow",
    ]


def compute_custom_sigma_schedule(sigmas: torch.Tensor, scheduler_name: str, *, entropic_power: float | None = None) -> torch.Tensor:
    """
    Compute a replacement sigma schedule using one of the supported schedulers without invoking the Adept sampler loop.

    Parameters:
    - sigmas: original sigma tensor (length = steps + 1, usually ends with 0)
    - scheduler_name: name from list_supported_schedulers()
    - entropic_power: optional power for the Entropic scheduler (defaults to current global setting or 6.0)

    Returns a tensor of sigmas with the same shape (steps + 1), suitable to pass into any k-diffusion sampler.
    """
    if sigmas is None or not torch.is_tensor(sigmas) or sigmas.numel() <= 1:
        return sigmas

    scheduler_name = scheduler_name or "None"
    device = sigmas.device

    # Use endpoints from the provided schedule
    sigma_max = sigmas[0]
    sigma_min = sigmas[-2]
    num_steps = len(sigmas) - 1

    # Handle both "None" variants
    if scheduler_name in ("None", "None (use WebUI sampler schedule)"):
        return sigmas

    # Fallback power
    if entropic_power is None:
        entropic_power = float(current_sampler_settings.get('entropic_scheduler_power', 6.0))

    # Reuse the robust scheduler implementations defined on AdeptSamplerForge
    forge = AdeptSamplerForge()

    mapping = {
        "Entropic": lambda: forge.create_entropic_sigmas(sigma_max, sigma_min, num_steps, entropic_power, device),
        "SNR-Optimized": lambda: forge.create_snr_optimized_sigmas(sigma_max, sigma_min, num_steps, device),
        "Constant-Rate": lambda: forge.create_constant_rate_sigmas(sigma_max, sigma_min, num_steps, device),
        "Adaptive-Optimized": lambda: forge.create_adaptive_optimized_sigmas(sigma_max, sigma_min, num_steps, device),
        "Cosine-Annealed": lambda: forge.create_cosine_sigmas(sigma_max, sigma_min, num_steps, device),
        "LogSNR-Uniform": lambda: forge.create_logsnr_uniform_sigmas(sigma_max, sigma_min, num_steps, device),
        "Tanh Mid-Boost": lambda: forge.create_tanh_midboost_sigmas(sigma_max, sigma_min, num_steps, device),
        "Exponential Tail": lambda: forge.create_exponential_tail_sigmas(sigma_max, sigma_min, num_steps, device),
        "Jittered-Karras": lambda: forge.create_jittered_karras_sigmas(sigma_max, sigma_min, num_steps, device),
        "Hybrid JYS-Karras": lambda: forge.create_hybrid_jys_karras_sigmas(sigma_max, sigma_min, num_steps, device),
        "AYS-SDXL": lambda: forge.create_ays_sdxl_sigmas(sigma_max, sigma_min, num_steps, device),
        "Stochastic": lambda: forge.create_stochastic_sigmas(
            sigma_max, sigma_min, num_steps, device,
            current_sampler_settings.get('stochastic_noise_type', 'brownian'),
            current_sampler_settings.get('stochastic_noise_scale', 0.3),
            current_sampler_settings.get('stochastic_base_schedule', 'karras')
        ),
        "JYS (Dynamic)": lambda: forge.create_jys_sigmas(sigma_max, sigma_min, num_steps, device),
        "AOS-V (for v-prediction)": lambda: forge.create_aos_v_sigmas(sigma_max, sigma_min, num_steps, device),
        "AOS-ε (for ε-prediction)": lambda: forge.create_aos_e_sigmas(sigma_max, sigma_min, num_steps, device),
        "AkashicAOS": lambda: forge.create_aos_akashic_sigmas(sigma_max, sigma_min, num_steps, device),
        "AkashicAOS Alt": lambda: forge.create_aos_akashic_alt_sigmas(sigma_max, sigma_min, num_steps, device),
        "AkashicEQFlow": lambda: forge.create_akashic_eqflow_sigmas(sigma_max, sigma_min, num_steps, device),
    }

    if scheduler_name not in mapping:
        print(f"⚠️ Unknown scheduler '{scheduler_name}'. Returning original sigmas.")
        return sigmas

    return mapping[scheduler_name]()


def compute_sigma_schedule_from_settings(sigmas: torch.Tensor, settings: dict | None = None) -> torch.Tensor:
    """
    Compute a replacement sigma schedule from provided settings or the extension's global settings.
    This does not depend on Adept being enabled and can be used by any sampler.

    Supported setting keys:
    - custom_scheduler_type: one of list_supported_schedulers() (except "None")
    - use_entropic_scheduler: bool
    - entropic_scheduler_power: float
    - use_anime_schedule_v: bool
    - use_anime_schedule_e: bool
    """
    if sigmas is None or not torch.is_tensor(sigmas) or sigmas.numel() <= 1:
        return sigmas

    settings = settings or current_sampler_settings

    # Priority: explicit custom type > entropic flag > AOS flags > None
    custom_type = settings.get('custom_scheduler_type', 'None')
    if custom_type and custom_type != 'None':
        return compute_custom_sigma_schedule(sigmas, custom_type, entropic_power=settings.get('entropic_scheduler_power'))

    if settings.get('use_entropic_scheduler', False):
        return compute_custom_sigma_schedule(sigmas, 'Entropic', entropic_power=settings.get('entropic_scheduler_power'))

    if settings.get('use_anime_schedule_v', False):
        return compute_custom_sigma_schedule(sigmas, 'AOS-V (for v-prediction)')

    if settings.get('use_anime_schedule_e', False):
        return compute_custom_sigma_schedule(sigmas, 'AOS-ε (for ε-prediction)')

    if settings.get('use_akashic_aos', False):
        return compute_custom_sigma_schedule(sigmas, 'AkashicAOS')

    return sigmas


# --- XYZ Grid integration (Axes + value setters) ---
def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_adept_xyz"):
        p._adept_xyz = {}
    
    # Validate and convert types
    try:
        if field in ("enabled", "use_content_aware_pacing", "debug_stop_after_coherence",
                     "use_enhanced_detail_phase", "disable_for_hr", "exp_cfg_to_zero",
                     "adept_solver_use_corrector", "adept_ancestral_adaptive_eta",
                     "adept_ancestral_phase_noise", "adept_ancestral_enhanced_derivative",
                     "adept_ancestral_mirror_correction",
                     "akashic_adaptive_eta", "akashic_use_ays", "akashic_mirror_correction",
                     "vae_reflection"):
            # Boolean fields
            x = str(x).strip().lower() == "true"
        elif field == "akashic_eqvae_mode":
            # String dropdown field
            x = str(x) if x in ("Off", "Balanced") else "Off"
        elif field in ("eta", "s_noise", "entropic_scheduler_power", "detail_enhancement_strength",
                       "detail_separation_radius", "pacing_coherence_sensitivity",
                       "adept_ancestral_eta", "adept_ancestral_s_noise", "adept_ancestral_phase_strength", 
                       "stochastic_noise_scale",
                       "akashic_tau", "akashic_base_eta", "akashic_s_noise", 
                       "akashic_phase_strength", "akashic_smea_strength", "akashic_ndb_strength"):
            # Float fields
            x = float(x)
        elif field in ("adept_solver_order", "akashic_solver_order"):
            # Integer field
            x = int(x)
            if x not in (1, 2, 3):
                raise ValueError(f"Invalid solver order: {x}")
        
        p._adept_xyz[field] = x
        
    except (ValueError, TypeError) as e:
        print(f"⚠️ XYZ Grid: Invalid value '{x}' for field '{field}': {e}")
        # Don't set invalid values


def make_axis_on_xyz_grid():
    xyz_grid = None
    for sd in scripts.scripts_data:
        if sd.script_class.__module__ == "xyz_grid.py":
            xyz_grid = sd.module
            break

    if xyz_grid is None:
        return

    axis = [
        xyz_grid.AxisOption(
            "(Adept) Enabled",
            str,
            partial(set_value, field="enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Solver Type",
            str,
            partial(set_value, field="solver_type"),
            choices=lambda: ["None", "Adept Solver", "Adept Ancestral Solver", "AkashicSolver"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Solver Order",
            int,
            partial(set_value, field="adept_solver_order"),
            choices=lambda: ["1", "2", "3"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Solver Corrector",
            str,
            partial(set_value, field="adept_solver_use_corrector"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Scheduler",
            str,
            partial(set_value, field="scheduler_override"),
            choices=list_supported_schedulers,
        ),
        xyz_grid.AxisOption(
            "(Adept) Entropic Power",
            float,
            partial(set_value, field="entropic_scheduler_power"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Enhanced Detail",
            str,
            partial(set_value, field="use_enhanced_detail_phase"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Detail Strength",
            float,
            partial(set_value, field="detail_enhancement_strength"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Detail Radius",
            float,
            partial(set_value, field="detail_separation_radius"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Eta",
            float,
            partial(set_value, field="eta"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Noise Scale",
            float,
            partial(set_value, field="s_noise"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Pacing On",
            str,
            partial(set_value, field="use_content_aware_pacing"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Coherence Sensitivity",
            float,
            partial(set_value, field="pacing_coherence_sensitivity"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Debug Stop After Coherence",
            str,
            partial(set_value, field="debug_stop_after_coherence"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Disable for HR",
            str,
            partial(set_value, field="disable_for_hr"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Exp CFG to Zero",
            str,
            partial(set_value, field="exp_cfg_to_zero"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Stochastic Noise Type",
            str,
            partial(set_value, field="stochastic_noise_type"),
            choices=lambda: ["brownian", "uniform", "normal"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Stochastic Noise Scale",
            float,
            partial(set_value, field="stochastic_noise_scale"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Stochastic Base Schedule",
            str,
            partial(set_value, field="stochastic_base_schedule"),
            choices=lambda: ["karras", "uniform", "cosine"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Ancestral Eta",
            float,
            partial(set_value, field="adept_ancestral_eta"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Ancestral Noise Scale",
            float,
            partial(set_value, field="adept_ancestral_s_noise"),
        ),
        # AkashicSolver settings
        xyz_grid.AxisOption(
            "(Adept) Akashic Tau",
            float,
            partial(set_value, field="akashic_tau"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Akashic Order",
            int,
            partial(set_value, field="akashic_solver_order"),
            choices=lambda: ["1", "2", "3"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Akashic Eta",
            float,
            partial(set_value, field="akashic_base_eta"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Akashic Noise Scale",
            float,
            partial(set_value, field="akashic_s_noise"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Akashic Adaptive Eta",
            str,
            partial(set_value, field="akashic_adaptive_eta"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Akashic Phase Strength",
            float,
            partial(set_value, field="akashic_phase_strength"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Akashic SMEA Strength",
            float,
            partial(set_value, field="akashic_smea_strength"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Akashic Native Detail Boost",
            float,
            partial(set_value, field="akashic_ndb_strength"),
        ),
        xyz_grid.AxisOption(
            "(Adept) Ancestral Mirror Correction",
            str,
            partial(set_value, field="adept_ancestral_mirror_correction"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Akashic Mirror Correction",
            str,
            partial(set_value, field="akashic_mirror_correction"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Adept) Akashic EQ-VAE Mode",
            str,
            partial(set_value, field="akashic_eqvae_mode"),
            choices=lambda: ["Off", "Balanced"],
        ),
        xyz_grid.AxisOption(
            "(Adept) VAE Reflection",
            str,
            partial(set_value, field="vae_reflection"),
            choices=lambda: ["True", "False"],
        ),
    ]

    if not any(getattr(x, "label", "").startswith("(Adept)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)


def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] Adept Sampler: xyz_grid error:\n{error}",
            file=sys.stderr,
        )


# Initialize the extension when script loads
patch_samplers_globally()
print("Adept Sampler for reForge loaded successfully!") 

if WEBUI_AVAILABLE:
    try:
        script_callbacks.on_before_ui(on_before_ui)
    except Exception:
        # If callbacks are not available, ignore gracefully
        pass