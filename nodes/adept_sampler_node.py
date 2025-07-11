"""
Adept Sampler Node for ComfyUI.
Enhanced Euler Ancestral sampler with advanced features.
"""

import math
import json
import torch
from .schedulers import AdeptSchedulers
import comfy.model_management
import comfy.samplers
import comfy.k_diffusion.sampling
from comfy.ldm.modules.diffusionmodules.openaimodel import timestep_embedding

# Try to import torchvision for detail enhancement
try:
    from torchvision.transforms.functional import gaussian_blur
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class AdeptSamplerNode:
    """Enhanced Euler Ancestral sampler with content-aware pacing and detail enhancement."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "scheduler": (["karras", "AOS-V", "AOS-ε", "Entropic", "SNR-Optimized", "Constant-Rate", "Adaptive-Optimized"],),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "sigmas": ("SIGMAS",),
                "entropic_power": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "enable_content_aware_pacing": ("BOOLEAN", {"default": False}),
                "coherence_sensitivity": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 1.0, "step": 0.05}),
                "manual_pacing_override": ("STRING", {"default": ""}),
                "enable_detail_enhancement": ("BOOLEAN", {"default": True}),
                "detail_enhancement_strength": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05}),
                "detail_separation_radius": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}),
                "debug_stop_after_coherence": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"
    
    def create_model_function(self, model, positive, negative, cfg):
        diffusion_model = model.model.diffusion_model
        device = comfy.model_management.get_torch_device()
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        diffusion_model = diffusion_model.to(device, dtype)
        model_sampling = model.model.model_sampling.to(device)
        dtype = diffusion_model.dtype  # Update dtype after casting

        def model_fn(x, sigma):
            device = x.device

            if not isinstance(sigma, torch.Tensor) or sigma.dim() == 0:
                if not isinstance(sigma, torch.Tensor):
                    sigma = torch.tensor(sigma, device=device)
                sigma = sigma.reshape((1,)).repeat(x.shape[0])

            timestep = model_sampling.timestep(sigma)
            if isinstance(timestep, float):
                timestep = torch.tensor([timestep] * x.shape[0], dtype=torch.float32, device=device)
            elif isinstance(timestep, torch.Tensor):
                timestep = timestep.to(device=device, dtype=torch.int64)
                if timestep.dim() == 0:
                    timestep = timestep.repeat(x.shape[0])
                elif timestep.dim() == 1 and timestep.shape[0] == 1:
                    timestep = timestep.repeat(x.shape[0])
            else:
                raise ValueError("Unexpected type for timestep")

            dtype = diffusion_model.dtype
            x = x.to(dtype)

            if dtype == torch.float16:
                timestep = timestep.to(torch.float32)
                original_time_embed_dtype = diffusion_model.time_embed[0].weight.dtype
                diffusion_model.time_embed.to(torch.float32)

                from types import MethodType
                original_time_embed_forward = diffusion_model.time_embed.forward
                def patched_time_embed_forward(self, input):
                    return original_time_embed_forward(input.to(torch.float32)).to(dtype)
                diffusion_model.time_embed.forward = MethodType(patched_time_embed_forward, diffusion_model.time_embed)
            else:
                timestep = timestep.to(dtype)

            # Handle multiple conditionings by concatenating contexts
            def compute_pred(cond_list, uncond_list):
                # Process positive conditionings
                if cond_list:
                    if len(cond_list) > 1:
                        pos_cond = torch.cat([cond[0] for cond in cond_list], dim=1).to(device, dtype)
                        pos_extra = cond_list[0][1]  # Use extras from first conditioning
                    else:
                        pos_cond = cond_list[0][0].to(device, dtype)
                        pos_extra = cond_list[0][1]
                    pos_kwargs = {'context': pos_cond}
                    if 'pooled_output' in pos_extra:
                        y = pos_extra['pooled_output'].to(device, dtype)
                        if hasattr(diffusion_model, 'label_emb') and diffusion_model.label_emb is not None:
                            expected_y_dim = diffusion_model.label_emb[0][0].in_features
                            provided_dim = y.shape[-1]
                            if provided_dim == expected_y_dim:
                                pos_kwargs['y'] = y
                            elif expected_y_dim == 2816 and provided_dim == 1280:
                                orig_size = pos_extra.get('original_size', (x.shape[3] * 8, x.shape[2] * 8))
                                orig_width = orig_size[0]
                                orig_height = orig_size[1]
                                crop_coords = pos_extra.get('crop_coords', (0, 0))
                                crop_w = crop_coords[1]  # left
                                crop_h = crop_coords[0]  # top
                                target_size = pos_extra.get('target_size', orig_size)
                                target_width = target_size[0]
                                target_height = target_size[1]
                                time_ids = torch.tensor([orig_width, orig_height, crop_w, crop_h, target_width, target_height], dtype=torch.float32, device=device)
                                add_emb = torch.cat([timestep_embedding(time_ids[i:i+1], 256) for i in range(6)], dim=1).to(dtype)
                                add_emb = add_emb.repeat(y.shape[0], 1)
                                pos_kwargs['y'] = torch.cat([y, add_emb], dim=1)
                            elif provided_dim < expected_y_dim:
                                print(f"Warning: Mismatch in y dimension: expected {expected_y_dim}, got {provided_dim}. Padding with zeros.")
                                padding = torch.zeros((y.shape[0], expected_y_dim - provided_dim), dtype=y.dtype, device=device)
                                pos_kwargs['y'] = torch.cat([y, padding], dim=-1)
                            else:
                                print(f"Warning: Mismatch in y dimension: expected {expected_y_dim}, got {provided_dim}. Using zeros instead.")
                                pos_kwargs['y'] = torch.zeros((y.shape[0], expected_y_dim), dtype=y.dtype, device=device)
                        else:
                            pos_kwargs['y'] = y
                    elif hasattr(diffusion_model, 'label_emb') and diffusion_model.label_emb is not None:
                        expected_y_dim = diffusion_model.label_emb[0][0].in_features
                        print("Warning: Model expects y conditioning but none provided. Using zeros.")
                        pos_kwargs['y'] = torch.zeros((x.shape[0], expected_y_dim), dtype=dtype, device=device)
                    avg_cond = diffusion_model(x, timestep, **pos_kwargs)
                else:
                    avg_cond = torch.zeros_like(x)

                # Process negative conditionings
                if uncond_list:
                    if len(uncond_list) > 1:
                        neg_cond = torch.cat([cond[0] for cond in uncond_list], dim=1).to(device, dtype)
                        neg_extra = uncond_list[0][1]  # Use extras from first conditioning
                    else:
                        neg_cond = uncond_list[0][0].to(device, dtype)
                        neg_extra = uncond_list[0][1]
                    neg_kwargs = {'context': neg_cond}
                    if 'pooled_output' in neg_extra:
                        y = neg_extra['pooled_output'].to(device, dtype)
                        if hasattr(diffusion_model, 'label_emb') and diffusion_model.label_emb is not None:
                            expected_y_dim = diffusion_model.label_emb[0][0].in_features
                            provided_dim = y.shape[-1]
                            if provided_dim == expected_y_dim:
                                neg_kwargs['y'] = y
                            elif expected_y_dim == 2816 and provided_dim == 1280:
                                orig_size = neg_extra.get('original_size', (x.shape[3] * 8, x.shape[2] * 8))
                                orig_width = orig_size[0]
                                orig_height = orig_size[1]
                                crop_coords = neg_extra.get('crop_coords', (0, 0))
                                crop_w = crop_coords[1]
                                crop_h = crop_coords[0]
                                target_size = neg_extra.get('target_size', orig_size)
                                target_width = target_size[0]
                                target_height = target_size[1]
                                time_ids = torch.tensor([orig_width, orig_height, crop_w, crop_h, target_width, target_height], dtype=torch.float32, device=device)
                                add_emb = torch.cat([timestep_embedding(time_ids[i:i+1], 256) for i in range(6)], dim=1).to(dtype)
                                add_emb = add_emb.repeat(y.shape[0], 1)
                                neg_kwargs['y'] = torch.cat([y, add_emb], dim=1)
                            elif provided_dim < expected_y_dim:
                                print(f"Warning: Mismatch in y dimension: expected {expected_y_dim}, got {provided_dim}. Padding with zeros.")
                                padding = torch.zeros((y.shape[0], expected_y_dim - provided_dim), dtype=y.dtype, device=device)
                                neg_kwargs['y'] = torch.cat([y, padding], dim=-1)
                            else:
                                print(f"Warning: Mismatch in y dimension: expected {expected_y_dim}, got {provided_dim}. Using zeros instead.")
                                neg_kwargs['y'] = torch.zeros((y.shape[0], expected_y_dim), dtype=y.dtype, device=device)
                        else:
                            neg_kwargs['y'] = y
                    elif hasattr(diffusion_model, 'label_emb') and diffusion_model.label_emb is not None:
                        expected_y_dim = diffusion_model.label_emb[0][0].in_features
                        print("Warning: Model expects y conditioning but none provided. Using zeros.")
                        neg_kwargs['y'] = torch.zeros((x.shape[0], expected_y_dim), dtype=dtype, device=device)
                    avg_uncond = diffusion_model(x, timestep, **neg_kwargs)
                else:
                    avg_uncond = torch.zeros_like(x)

                noise_pred = avg_uncond + cfg * (avg_cond - avg_uncond)
                return noise_pred

            noise_pred = compute_pred(positive, negative)

            if dtype == torch.float16:
                diffusion_model.time_embed.forward = original_time_embed_forward
                diffusion_model.time_embed.to(original_time_embed_dtype)

            denoised = model_sampling.calculate_denoised(sigma, noise_pred, x)

            return denoised

        return model_fn

    def sample(self, model, positive, negative, latent_image, steps, cfg, eta, s_noise, scheduler, denoise,
               sigmas=None, entropic_power=6.0, enable_content_aware_pacing=False, coherence_sensitivity=0.75,
               manual_pacing_override="", enable_detail_enhancement=True, detail_enhancement_strength=0.05,
               detail_separation_radius=0.5, debug_stop_after_coherence=False, seed=0):
        """Main sampling function."""
        
        device = comfy.model_management.get_torch_device()
        
        torch.manual_seed(seed)
        
        model_fn = self.create_model_function(model, positive, negative, cfg)
        model_sampling = model.model.model_sampling.to(device)
        
        if sigmas is None:
            sigmas = self._get_sigmas(model_sampling, scheduler, steps, denoise, entropic_power, device)
        
        x = latent_image["samples"].to(device)
        noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler(x)
        
        result = self._enhanced_euler_ancestral_sample(
            model_fn, x, sigmas, eta, s_noise,
            scheduler, enable_content_aware_pacing, coherence_sensitivity,
            manual_pacing_override, enable_detail_enhancement,
            detail_enhancement_strength, detail_separation_radius,
            debug_stop_after_coherence, noise_sampler
        )
        
        return ({"samples": result},)
    
    def _get_sigmas(self, model_sampling, scheduler, steps, denoise, entropic_power, device):
        """Generate sigma schedule using model's noise scale."""
        if denoise is None:
            denoise = 1.0
        full_steps = steps if denoise >= 1.0 else math.ceil(steps / denoise)
        sigma_max = model_sampling.sigma_max.item()
        sigma_min = model_sampling.sigma_min.item()
        
        if scheduler == "karras":
            sigmas = self._get_karras_sigmas(full_steps, device, sigma_max, sigma_min)
        else:
            scheduler_fn = AdeptSchedulers.get_scheduler_fn(scheduler)
            if scheduler_fn is None:
                sigmas = self._get_karras_sigmas(full_steps, device, sigma_max, sigma_min)
            else:
                if scheduler == "Entropic":
                    sigmas = scheduler_fn(sigma_max, sigma_min, full_steps, entropic_power, device)
                else:
                    sigmas = scheduler_fn(sigma_max, sigma_min, full_steps, device)
        
        if denoise < 1.0:
            start_idx = full_steps - steps
            sigmas = sigmas[start_idx:]
            if len(sigmas) == steps:
                sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
        
        return sigmas
    
    def _get_karras_sigmas(self, steps, device, sigma_max, sigma_min, rho=7.0):
        """Generate standard Karras schedule with provided sigma range."""
        step_indices = torch.arange(0, steps, dtype=torch.float32, device=device)
        
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        
        sigmas = (max_inv_rho + step_indices / (steps - 1) * (min_inv_rho - max_inv_rho)) ** rho
        
        return torch.cat([sigmas, torch.zeros(1, device=device)])
    
    def _enhanced_euler_ancestral_sample(self, model_fn, x, sigmas, eta, s_noise,
                                       scheduler, enable_content_aware_pacing, coherence_sensitivity,
                                       manual_pacing_override, enable_detail_enhancement,
                                       detail_enhancement_strength, detail_separation_radius,
                                       debug_stop_after_coherence, noise_sampler):
        """Enhanced Euler Ancestral sampling with content-aware pacing and detail enhancement."""
        
        total_steps = len(sigmas) - 1
        original_sigmas = sigmas.clone()
        
        manual_pacing_schedule = None
        if manual_pacing_override and manual_pacing_override.strip():
            try:
                schedule = json.loads(manual_pacing_override)
                if isinstance(schedule, dict):
                    manual_pacing_schedule = schedule
            except json.JSONDecodeError:
                print("⚠️ Manual Pacing Override: Invalid JSON. Ignoring.")
        
        use_pacing = (enable_content_aware_pacing and 
                      ("AOS" in scheduler) and 
                      total_steps > 0)
        
        if use_pacing:
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
            return self._sample_with_pacing(
                model_fn, x, original_sigmas, manual_pacing_schedule, coherence_sensitivity, pacing_step_size,
                enable_detail_enhancement, detail_enhancement_strength,
                detail_separation_radius, debug_stop_after_coherence, eta, s_noise, scheduler, noise_sampler
            )
        else:
            return self._sample_single_phase(
                model_fn, x, sigmas, eta, s_noise,
                enable_detail_enhancement, detail_enhancement_strength,
                detail_separation_radius, noise_sampler
            )
    
    def _sample_with_pacing(self, model_fn, x, sigmas, manual_pacing_schedule, coherence_sensitivity, pacing_step_size,
                           enable_detail_enhancement, detail_enhancement_strength,
                           detail_separation_radius, debug_stop_after_coherence, eta, s_noise, scheduler, noise_sampler):
        """Sample with content-aware pacing (two-phase sampling)."""
        
        total_steps = len(sigmas) - 1
        is_coherent = False
        last_composition_derivative = None
        composition_steps_taken = 0
        sigma_idx_at_switch = 0

        if manual_pacing_schedule:
            print("🧠 Pacing: Using manual override schedule.")
            comp_setting = manual_pacing_schedule.get("composition", 0.5)
            
            if 0 < comp_setting < 1:
                composition_steps_taken = int(total_steps * comp_setting)
            else:
                composition_steps_taken = int(comp_setting)
            
            composition_steps_taken = max(0, min(total_steps, composition_steps_taken))
            print(f"🧠 Pacing: Manual composition steps: {composition_steps_taken}")
            is_coherent = True

            # Run composition phase
            for i in range(composition_steps_taken):
                denoised = model_fn(x, sigmas[i].repeat(x.shape[0]))
                last_composition_derivative = (x - denoised) / sigmas[i]
                
                sigma_down, sigma_up = self._get_ancestral_step(sigmas[i], sigmas[i+1], eta)
                dt = sigma_down - sigmas[i]
                x = x + last_composition_derivative * dt
                
                if enable_detail_enhancement and TORCHVISION_AVAILABLE:
                    progress = i / composition_steps_taken if composition_steps_taken > 0 else 1.0
                    strength = self._apply_progressive_enhancement(detail_enhancement_strength, 'composition', progress)
                    x = self._apply_detail_enhancement(x, denoised, dt, sigmas[i], strength, detail_separation_radius)
                x = torch.clamp(x, min=-10.0, max=10.0)
                
                if sigmas[i+1] > 0:
                    x = x + noise_sampler(sigmas[i], sigmas[i+1]) * s_noise * sigma_up
                    x = torch.clamp(x, min=-10.0, max=10.0)
            
            sigma_idx_at_switch = composition_steps_taken
        else:
            # Automatic pacing
            initial_variance = None
            fallback_step_pct = 0.4 + 0.3 * torch.clamp(torch.tensor((total_steps - 20) / 40.0), 0.0, 1.0).item()
            
            print("🧠 Pacing: Starting composition phase...")
            i = 0
            
            while i < (total_steps - 1) and composition_steps_taken < int(total_steps * fallback_step_pct):
                composition_steps_taken += 1
                
                current_sigma = sigmas[i]
                next_sigma_idx = min(i + pacing_step_size, total_steps)
                next_sigma = sigmas[next_sigma_idx]
                
                if current_sigma <= next_sigma:
                    break
                
                denoised = model_fn(x, current_sigma.repeat(x.shape[0]))
                derivative = (x - denoised) / current_sigma
                last_composition_derivative = derivative
                
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
                
                sigma_down, sigma_up = self._get_ancestral_step(current_sigma, next_sigma, eta)
                dt = sigma_down - current_sigma
                x = x + derivative * dt
                
                if enable_detail_enhancement and TORCHVISION_AVAILABLE:
                    progress = composition_steps_taken / (total_steps * fallback_step_pct)
                    strength = self._apply_progressive_enhancement(detail_enhancement_strength, 'composition', progress)
                    x = self._apply_detail_enhancement(x, denoised, dt, current_sigma, strength, detail_separation_radius)
                x = torch.clamp(x, min=-10.0, max=10.0)
                
                if next_sigma > 0:
                    x = x + noise_sampler(current_sigma, next_sigma) * s_noise * sigma_up
                    x = torch.clamp(x, min=-10.0, max=10.0)
                
                i = next_sigma_idx
            
            sigma_idx_at_switch = i
            if not is_coherent:
                print("🧠 Pacing: Composition phase finished without reaching coherence. Proceeding to detail phase.")
                is_coherent = True
        
        if is_coherent and debug_stop_after_coherence and not manual_pacing_schedule:
            print("🛑 [Debug] Coherence achieved. Stopping generation before detail phase as requested.")
            return x
        
        remaining_iterations = total_steps - sigma_idx_at_switch
        
        if remaining_iterations <= 0 and total_steps > 0:
            if composition_steps_taken > 0:
                print(f"⚠️ Warning: No steps remaining for detail phase. Composition took all {sigma_idx_at_switch} sigma steps.")
            return x
        
        if remaining_iterations > 0 and is_coherent:
            print(f"🧠 Pacing: Starting detail phase with {remaining_iterations} steps.")
            
            sigma_at_switch = sigmas[min(sigma_idx_at_switch, total_steps)]
            sigma_min = sigmas[-2]
            detail_sigmas = self._create_detail_schedule(sigma_at_switch, sigma_min, remaining_iterations, x.device, scheduler)
            
            if len(detail_sigmas) > 1:
                for j in range(len(detail_sigmas) - 1):
                    current_sigma = detail_sigmas[j]
                    next_sigma = detail_sigmas[j+1]
                    
                    if current_sigma < next_sigma:
                        break
                    
                    denoised = model_fn(x, current_sigma.repeat(x.shape[0]))
                    derivative = (x - denoised) / current_sigma
                    
                    sigma_down, sigma_up = self._get_ancestral_step(current_sigma, next_sigma, eta)
                    dt = sigma_down - current_sigma
                    x = x + derivative * dt
                    
                    if enable_detail_enhancement and TORCHVISION_AVAILABLE:
                        progress_detail = (composition_steps_taken + j) / total_steps
                        strength = self._apply_progressive_enhancement(detail_enhancement_strength, 'detail', progress_detail)
                        x = self._apply_detail_enhancement(x, denoised, dt, current_sigma, strength, detail_separation_radius)
                    
                    if next_sigma > 0:
                        x = x + noise_sampler(current_sigma, next_sigma) * s_noise * sigma_up
        
        return x

    def _sample_single_phase(self, model_fn, x, sigmas, eta, s_noise,
                            enable_detail_enhancement, detail_enhancement_strength,
                            detail_separation_radius, noise_sampler):
        """Standard single-phase sampling."""
        
        total_steps = len(sigmas) - 1
        
        for i in range(total_steps):
            denoised = model_fn(x, sigmas[i].repeat(x.shape[0]))
            derivative = (x - denoised) / sigmas[i]
            
            sigma_down, sigma_up = self._get_ancestral_step(sigmas[i], sigmas[i+1], eta)
            dt = sigma_down - sigmas[i]
            x = x + derivative * dt
            
            if enable_detail_enhancement and TORCHVISION_AVAILABLE:
                strength = self._apply_progressive_enhancement(detail_enhancement_strength, 'single_phase', i/total_steps)
                x = self._apply_detail_enhancement(x, denoised, dt, sigmas[i], strength, detail_separation_radius)
            x = torch.clamp(x, min=-10.0, max=10.0)
            
            if sigmas[i+1] > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i+1]) * s_noise * sigma_up
                x = torch.clamp(x, min=-10.0, max=10.0)
        
        return x
    
    def _get_ancestral_step(self, sigma, sigma_next, eta=1.):
        """Calculate ancestral step sizes."""
        sigma_up = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
        sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up
    
    def _apply_progressive_enhancement(self, base_strength, phase, progress):
        """Apply enhancement based on the current sampling phase."""
        if phase == 'composition':
            return base_strength * (0.25 + 0.5 * progress)  # Gently ramp up from 0.25x to 0.75x
        elif phase == 'detail':
            return base_strength * (0.75 + 0.75 * progress)  # Ramp from 0.75x to 1.5x
        else:  # single_phase
            return base_strength * (0.5 + progress)  # Gradual increase
    
    def _apply_detail_enhancement(self, x, denoised, dt, sigma, strength, radius):
        """Apply high-frequency detail enhancement."""
        if not TORCHVISION_AVAILABLE:
            return x
            
        low_freq = gaussian_blur(denoised, kernel_size=3, sigma=radius)
        high_freq = denoised - low_freq
        enhancement_amount = dt.abs() / sigma.clamp(min=1e-6)
        
        return x + high_freq * enhancement_amount * strength
    
    def _create_detail_schedule(self, sigma_max, sigma_min, num_steps, device, scheduler):
        """Create a schedule for the detail phase, respecting the original scheduler choice."""
        if "AOS-V" in scheduler:
            return AdeptSchedulers.create_aos_v_sigmas(sigma_max, sigma_min, num_steps, device)
        elif "AOS-ε" in scheduler:
            return AdeptSchedulers.create_aos_e_sigmas(sigma_max, sigma_min, num_steps, device)
        elif scheduler == "Entropic":
            return AdeptSchedulers.create_entropic_sigmas(sigma_max, sigma_min, num_steps, 6.0, device)
        else:
            # Fallback to entropic with neutral power
            return AdeptSchedulers.create_entropic_sigmas(sigma_max, sigma_min, num_steps, 1.0, device)