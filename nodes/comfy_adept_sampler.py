"""
ComfyUI-compatible Adept Sampler implementation.
This integrates properly with ComfyUI's sampling system.
"""

import math
import json
import torch
from .schedulers import AdeptSchedulers

# Try to import torchvision for detail enhancement
try:
    from torchvision.transforms.functional import gaussian_blur
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def adept_euler_ancestral_sample(model, x, sigmas, extra_args=None, callback=None, disable=None, 
                                scheduler_type="karras", eta=1.0, s_noise=1.0, entropic_power=6.0,
                                enable_content_aware_pacing=False, coherence_sensitivity=0.75,
                                manual_pacing_override="", enable_detail_enhancement=True,
                                detail_enhancement_strength=0.05, detail_separation_radius=0.5,
                                debug_stop_after_coherence=False):
    """
    Enhanced Euler Ancestral sampler with content-aware pacing and detail enhancement.
    Compatible with ComfyUI's sampling interface.
    """
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    total_steps = len(sigmas) - 1
    original_sigmas = sigmas.clone()
    
    # Override sigmas with custom scheduler if specified
    if scheduler_type != "karras" and len(sigmas) > 1:
        custom_sigmas = _create_custom_sigmas(scheduler_type, sigmas[0], sigmas[-2], len(sigmas) - 1, 
                                            sigmas.device, entropic_power)
        if custom_sigmas is not None:
            sigmas = custom_sigmas
            original_sigmas = sigmas.clone()
            print(f"🔬 Using custom scheduler: {scheduler_type}")
    
    # Parse manual pacing override
    manual_pacing_schedule = None
    if manual_pacing_override and manual_pacing_override.strip():
        try:
            schedule = json.loads(manual_pacing_override)
            if isinstance(schedule, dict):
                manual_pacing_schedule = schedule
        except json.JSONDecodeError:
            print("⚠️ Manual Pacing Override: Invalid JSON. Ignoring.")
    
    # Check if we should use content-aware pacing (only for AOS schedules)
    use_pacing = (enable_content_aware_pacing and 
                 ("AOS" in scheduler_type) and 
                 total_steps > 0)
    
    if use_pacing:
        # Adaptive pacing strategy
        if total_steps < 26:
            print("🧠 Pacing: Disabled automatically for low step count (< 26) to ensure quality.")
            use_pacing = False
        elif total_steps <= 40:
            print("🧠 Pacing: Using conservative single-step pacing for medium step count (<= 40).")
            pacing_step_size = 1
        else:
            print("🧠 Pacing: Using aggressive double-step pacing for high step count (> 40).")
            pacing_step_size = 2
    
    if use_pacing:
        return _sample_with_pacing(
            model, x, original_sigmas, extra_args, callback, disable,
            eta, s_noise, manual_pacing_schedule, coherence_sensitivity, pacing_step_size,
            enable_detail_enhancement, detail_enhancement_strength,
            detail_separation_radius, debug_stop_after_coherence, scheduler_type
        )
    else:
        return _sample_single_phase(
            model, x, sigmas, extra_args, callback, disable,
            eta, s_noise, enable_detail_enhancement, detail_enhancement_strength,
            detail_separation_radius
        )


def _create_custom_sigmas(scheduler_type, sigma_max, sigma_min, num_steps, device, entropic_power=6.0):
    """Create custom sigma schedule."""
    scheduler_fn = AdeptSchedulers.get_scheduler_fn(scheduler_type)
    if scheduler_fn is None:
        return None
        
    if scheduler_type == "Entropic":
        return scheduler_fn(sigma_max, sigma_min, num_steps, entropic_power, device)
    else:
        return scheduler_fn(sigma_max, sigma_min, num_steps, device)


def _sample_with_pacing(model, x, sigmas, extra_args, callback, disable, eta, s_noise,
                       manual_pacing_schedule, coherence_sensitivity, pacing_step_size,
                       enable_detail_enhancement, detail_enhancement_strength,
                       detail_separation_radius, debug_stop_after_coherence, scheduler_type):
    """Sample with content-aware pacing (two-phase sampling)."""
    
    total_steps = len(sigmas) - 1
    is_coherent = False
    last_composition_derivative = None
    composition_steps_taken = 0
    sigma_idx_at_switch = 0
    
    s_in = x.new_ones([x.shape[0]])
    
    if manual_pacing_schedule:
        # Manual pacing override
        print("🧠 Pacing: Using manual override schedule.")
        comp_setting = manual_pacing_schedule.get("composition", 0.5)
        
        if 0 < comp_setting < 1:
            composition_steps_taken = int(total_steps * comp_setting)
        else:
            composition_steps_taken = int(comp_setting)
        
        composition_steps_taken = max(0, min(total_steps, composition_steps_taken))
        print(f"🧠 Pacing: Manual composition steps: {composition_steps_taken}")
        is_coherent = True
        
        for i in range(composition_steps_taken):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            last_composition_derivative = (x - denoised) / sigmas[i]
            
            if callback is not None and not disable:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            
            sigma_down, sigma_up = _get_ancestral_step(sigmas[i], sigmas[i+1], eta)
            dt = sigma_down - sigmas[i]
            x = x + last_composition_derivative * dt
            
            if enable_detail_enhancement and TORCHVISION_AVAILABLE:
                progress = i / composition_steps_taken if composition_steps_taken > 0 else 1.0
                strength = _apply_progressive_enhancement(detail_enhancement_strength, 'composition', progress)
                x = _apply_detail_enhancement(x, denoised, dt, sigmas[i], strength, detail_separation_radius)
            
            if sigmas[i+1] > 0:
                x = x + torch.randn_like(x) * s_noise * sigma_up
        
        sigma_idx_at_switch = composition_steps_taken
    else:
        # Automatic pacing (coherence detection)
        initial_variance = None
        fallback_step_pct = 0.4 + 0.3 * min(1.0, (total_steps - 20) / 40.0)
        
        print("🧠 Pacing: Starting composition phase...")
        i = 0
        
        while i < (total_steps - 1) and composition_steps_taken < int(total_steps * fallback_step_pct):
            composition_steps_taken += 1
            
            current_sigma = sigmas[i]
            next_sigma_idx = min(i + pacing_step_size, total_steps)
            next_sigma = sigmas[next_sigma_idx]
            
            if current_sigma < next_sigma:
                break
            
            denoised = model(x, current_sigma * s_in, **extra_args)
            derivative = (x - denoised) / current_sigma
            last_composition_derivative = derivative
            
            # Coherence calculation
            variance = torch.var(derivative.flatten(1), dim=1).mean().item()
            
            if composition_steps_taken == 2:
                initial_variance = variance
            elif composition_steps_taken > 2 and initial_variance is not None:
                threshold_percentage = coherence_sensitivity * 0.4 + 0.5
                coherence_threshold = initial_variance * threshold_percentage
                
                if variance < coherence_threshold:
                    print(f"🧠 Pacing: Coherence achieved at iteration {composition_steps_taken} (Sigma Step {i}). Rescheduling detail phase.")
                    is_coherent = True
                    break
            
            if callback is not None and not disable:
                callback({'x': x, 'i': i, 'sigma': current_sigma, 'sigma_hat': current_sigma, 'denoised': denoised})
            
            sigma_down, sigma_up = _get_ancestral_step(current_sigma, next_sigma, eta)
            dt = sigma_down - current_sigma
            x = x + derivative * dt
            
            if enable_detail_enhancement and TORCHVISION_AVAILABLE:
                progress = composition_steps_taken / (total_steps * fallback_step_pct)
                strength = _apply_progressive_enhancement(detail_enhancement_strength, 'composition', progress)
                x = _apply_detail_enhancement(x, denoised, dt, current_sigma, strength, detail_separation_radius)
            
            if next_sigma > 0:
                x = x + torch.randn_like(x) * s_noise * sigma_up
            
            i = next_sigma_idx
        
        sigma_idx_at_switch = i
    
    # Debug stop after coherence
    if is_coherent and debug_stop_after_coherence:
        print("🛑 [Debug] Coherence achieved. Stopping generation before detail phase as requested.")
        return x
    
    # Detail phase
    remaining_iterations = total_steps - composition_steps_taken
    
    if not is_coherent and not manual_pacing_schedule:
        print("🧠 Pacing: Composition phase finished. Switching to detail phase for remaining steps.")
        is_coherent = True
    
    if remaining_iterations <= 0 and total_steps > 0:
        print(f"⚠️ Warning: No steps remaining for detail phase. Composition took all {composition_steps_taken} steps.")
        return x
    
    if remaining_iterations > 0 and is_coherent:
        print(f"🧠 Pacing: Starting detail phase with {remaining_iterations} steps.")
        
        sigma_at_switch = sigmas[min(sigma_idx_at_switch, total_steps)]
        sigma_min = sigmas[-2]
        detail_sigmas = _create_detail_schedule(sigma_at_switch, sigma_min, remaining_iterations, x.device, scheduler_type)
        
        if len(detail_sigmas) > 1:
            for j in range(len(detail_sigmas) - 1):
                current_sigma = detail_sigmas[j]
                next_sigma = detail_sigmas[j+1]
                
                if current_sigma < next_sigma:
                    break
                
                denoised = model(x, current_sigma * s_in, **extra_args)
                derivative = (x - denoised) / current_sigma
                
                callback_step = composition_steps_taken + j
                if callback is not None and not disable:
                    callback({'x': x, 'i': callback_step, 'sigma': current_sigma, 'sigma_hat': current_sigma, 'denoised': denoised})
                
                sigma_down, sigma_up = _get_ancestral_step(current_sigma, next_sigma, eta)
                dt = sigma_down - current_sigma
                x = x + derivative * dt
                
                if enable_detail_enhancement and TORCHVISION_AVAILABLE:
                    progress_detail = callback_step / total_steps
                    strength = _apply_progressive_enhancement(detail_enhancement_strength, 'detail', progress_detail)
                    x = _apply_detail_enhancement(x, denoised, dt, current_sigma, strength, detail_separation_radius)
                
                if next_sigma > 0:
                    x = x + torch.randn_like(x) * s_noise * sigma_up
    
    return x


def _sample_single_phase(model, x, sigmas, extra_args, callback, disable, eta, s_noise,
                        enable_detail_enhancement, detail_enhancement_strength,
                        detail_separation_radius):
    """Standard single-phase sampling."""
    
    total_steps = len(sigmas) - 1
    s_in = x.new_ones([x.shape[0]])
    
    for i in range(total_steps):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        derivative = (x - denoised) / sigmas[i]
        
        if callback is not None and not disable:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        sigma_down, sigma_up = _get_ancestral_step(sigmas[i], sigmas[i+1], eta)
        dt = sigma_down - sigmas[i]
        x = x + derivative * dt
        
        if enable_detail_enhancement and TORCHVISION_AVAILABLE:
            strength = _apply_progressive_enhancement(detail_enhancement_strength, 'single_phase', i/total_steps)
            x = _apply_detail_enhancement(x, denoised, dt, sigmas[i], strength, detail_separation_radius)
        
        if sigmas[i+1] > 0:
            x = x + torch.randn_like(x) * s_noise * sigma_up
    
    return x


def _get_ancestral_step(sigma, sigma_next, eta=1.):
    """Calculate ancestral step sizes."""
    sigma_up = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
    sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def _apply_progressive_enhancement(base_strength, phase, progress):
    """Apply enhancement based on the current sampling phase."""
    if phase == 'composition':
        return base_strength * (0.25 + 0.5 * progress)  # Gently ramp up from 0.25x to 0.75x
    elif phase == 'detail':
        return base_strength * (0.75 + 0.75 * progress)  # Ramp from 0.75x to 1.5x
    else:  # single_phase
        return base_strength * (0.5 + progress)  # Gradual increase


def _apply_detail_enhancement(x, denoised, dt, sigma, strength, radius):
    """Apply high-frequency detail enhancement."""
    if not TORCHVISION_AVAILABLE:
        return x
        
    low_freq = gaussian_blur(denoised, kernel_size=3, sigma=radius)
    high_freq = denoised - low_freq
    enhancement_amount = dt.abs() / sigma.clamp(min=1e-6)
    
    return x + high_freq * enhancement_amount * strength


def _create_detail_schedule(sigma_max, sigma_min, num_steps, device, scheduler_type):
    """Create a schedule for the detail phase, respecting the original scheduler choice."""
    if "AOS-V" in scheduler_type:
        return AdeptSchedulers.create_aos_v_sigmas(sigma_max, sigma_min, num_steps, device)
    elif "AOS-ε" in scheduler_type:
        return AdeptSchedulers.create_aos_e_sigmas(sigma_max, sigma_min, num_steps, device)
    elif scheduler_type == "Entropic":
        return AdeptSchedulers.create_entropic_sigmas(sigma_max, sigma_min, num_steps, 6.0, device)
    else:
        # Fallback to entropic with neutral power
        return AdeptSchedulers.create_entropic_sigmas(sigma_max, sigma_min, num_steps, 1.0, device)


class AdeptSamplerComfy:
    """ComfyUI Custom Sampling node that uses the Adept sampler."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            },
            "optional": {
                "scheduler_type": (["karras", "AOS-V", "AOS-ε", "Entropic", "SNR-Optimized", "Constant-Rate", "Adaptive-Optimized"], {"default": "karras"}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "entropic_power": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "enable_content_aware_pacing": ("BOOLEAN", {"default": False}),
                "coherence_sensitivity": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 1.0, "step": 0.05}),
                "manual_pacing_override": ("STRING", {"default": ""}),
                "enable_detail_enhancement": ("BOOLEAN", {"default": True}),
                "detail_enhancement_strength": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05}),
                "detail_separation_radius": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}),
                "debug_stop_after_coherence": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"
    
    def sample(self, model, noise, guider, sampler, sigmas, latent_image, 
               scheduler_type="karras", eta=1.0, s_noise=1.0, entropic_power=6.0,
               enable_content_aware_pacing=False, coherence_sensitivity=0.75,
               manual_pacing_override="", enable_detail_enhancement=True,
               detail_enhancement_strength=0.05, detail_separation_radius=0.5,
               debug_stop_after_coherence=False):
        """Custom sampling with Adept Sampler."""
        
        # Replace the sampler function with our enhanced version
        def enhanced_sampler_function(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
            return adept_euler_ancestral_sample(
                model, x, sigmas, extra_args, callback, disable,
                scheduler_type, eta, s_noise, entropic_power,
                enable_content_aware_pacing, coherence_sensitivity,
                manual_pacing_override, enable_detail_enhancement,
                detail_enhancement_strength, detail_separation_radius,
                debug_stop_after_coherence
            )
        
        # Get latent
        latent = latent_image
        batch_size = latent["samples"].shape[0]
        
        # Generate noise
        noise_sampler = noise(model, batch_size)
        
        # Sample using ComfyUI's sampling interface with our custom sampler
        # This is a simplified version - in practice you'd need to properly integrate
        # with ComfyUI's sampling system
        samples = guider.sample(noise_sampler, sigmas, enhanced_sampler_function, model, latent)
        
        return ({"samples": samples},)