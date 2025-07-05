import math
import numpy as np
import torch
import json

from modules import scripts
import gradio as gr
import k_diffusion.sampling

# Import shared for RNG state management
try:
    from modules import shared
    from modules.processing import StableDiffusionProcessing
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

# Global settings that control sampler behavior
current_sampler_settings = {
    'enabled': False,
    'eta': 1.0,
    's_noise': 1.0,
    'use_dynamic_threshold': True,
    'threshold_percentile': 0.995,
    'use_adaptive_thresholding': False,
    'adaptive_threshold_end': 0.95,
    'solver_order': 2,
    'debug_reproducibility': False,
    'use_entropic_scheduler': False,
    'entropic_scheduler_power': 6.0,
    'use_anime_schedule': False,
    'use_dynamic_ancestral_noise': False,
    'use_heun_corrector': False,
    'use_enhanced_detail_phase': True,
    'detail_enhancement_strength': 0.1,
    'detail_separation_radius': 0.5,
}



def patch_samplers_globally():
    """Patch sampling functions once at startup"""
    if 'euler_ancestral' not in original_samplers:
        if hasattr(k_diffusion.sampling, 'sample_euler_ancestral'):
            original_samplers['euler_ancestral'] = k_diffusion.sampling.sample_euler_ancestral
            
            def smart_euler_ancestral_wrapper(model, x, sigmas, extra_args=None, callback=None, disable=None, generator=None, **kwargs):
                """Smart wrapper that checks global settings"""
                if not current_sampler_settings['enabled']:
                    # Use original sampler if disabled
                    return original_samplers['euler_ancestral'](model, x, sigmas, extra_args, callback, disable, **kwargs)
                
                # Use Enhanced custom sampler with current settings
                script_instance = AdeptSamplerForge()
                
                # Potentially override sigmas with entropic schedule
                final_sigmas = sigmas
                if current_sampler_settings.get('use_entropic_scheduler', False) and not current_sampler_settings.get('debug_reproducibility', False):
                    print("🔄 Overriding sigma schedule with Entropic Time Scheduler.")
                    power = current_sampler_settings.get('entropic_scheduler_power', 3.0)
                    if len(sigmas) > 1:
                        final_sigmas = script_instance.create_entropic_sigmas(
                            sigmas[0], sigmas[-2], len(sigmas) - 1, power, sigmas.device
                        )
                elif current_sampler_settings.get('use_anime_schedule', False) and not current_sampler_settings.get('debug_reproducibility', False):
                    print("🎨 Overriding sigma schedule with Anime-Optimized Schedule (AOS).")
                    if len(sigmas) > 1:
                        final_sigmas = script_instance.create_anime_optimized_sigmas(
                            sigmas[0], sigmas[-2], len(sigmas) - 1, sigmas.device
                        )
                
                return script_instance.sample_enhanced_euler_ancestral(
                    model, x, final_sigmas, extra_args=extra_args, callback=callback, disable=disable,
                    eta=current_sampler_settings.get('eta', 1.0),
                    s_noise=current_sampler_settings.get('s_noise', 1.0),
                    solver_order=current_sampler_settings.get('solver_order', 1),
                    generator=generator
                )
            
            # Apply the patch
            k_diffusion.sampling.sample_euler_ancestral = smart_euler_ancestral_wrapper
            
            # Also patch other common names
            for attr_name in ['sample_euler_a', 'sample_euler_ancestral_discrete']:
                if hasattr(k_diffusion.sampling, attr_name):
                    if attr_name not in original_samplers:
                        original_samplers[attr_name] = getattr(k_diffusion.sampling, attr_name)
                        setattr(k_diffusion.sampling, attr_name, smart_euler_ancestral_wrapper)
            
            print("🔧 Custom Euler Ancestral samplers patched globally")


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
            gr.HTML('Adept Sampler: An advanced Euler Ancestral sampler with custom schedulers and detail enhancement.')
            
            self.enable_custom = gr.Checkbox(label='Enable Adept Sampler', value=False)
            
            # This group will be shown/hidden based on the checkbox
            with gr.Group(visible=False) as main_options:
                with gr.Tabs():
                    with gr.TabItem("Scheduler"):
                        gr.Markdown("### Scheduler Override\nReplace the default time steps with a custom schedule.")
                        
                        self.scheduler_override = gr.Radio(
                            ["None", "Anime-Optimized (AOS)", "Entropic"],
                            label="Scheduler",
                            value="None"
                        )

                        with gr.Group(visible=False) as entropic_options:
                            self.entropic_scheduler_power = gr.Slider(
                                label='Entropic Power', 
                                minimum=1.0, maximum=8.0, 
                                value=6.0, step=0.1,
                                info="Controls timestep clustering. >1 clusters steps at the start (high detail)."
                            )
                        
                        with gr.Group(visible=False) as aos_plus_options:
                            gr.Markdown("⚠️ **Compatibility Warning:** AOS is heavily optimized for **v-prediction** models. Using it with **epsilon-prediction** models may break the generation.")
                            self.use_content_aware_pacing = gr.Checkbox(label='Enable Content-Aware Pacing (AOS Only)', value=False, info="Dynamically adjusts pacing based on image coherence. May improve low-step-count results.")
                            self.pacing_coherence_sensitivity = gr.Slider(
                                label='Coherence Sensitivity',
                                minimum=0.1, maximum=1.0, value=0.75, step=0.05,
                                info="Controls when to switch from composition to detail. Lower values switch sooner."
                            )
                            self.debug_stop_after_coherence = gr.Checkbox(label='[Debug] Stop after coherence', value=False, info="Stops generation immediately after coherence is detected, skipping the detail phase.")

                        def on_scheduler_change(scheduler):
                            return {
                                aos_plus_options: gr.update(visible=scheduler == "Anime-Optimized (AOS)"),
                                entropic_options: gr.update(visible=scheduler == "Entropic")
                            }

                        self.scheduler_override.change(
                            on_scheduler_change,
                            inputs=[self.scheduler_override],
                            outputs=[aos_plus_options, entropic_options]
                        )

                        gr.Markdown("ℹ️ **Note:** When using a custom scheduler, you may need to **lower your CFG Scale** (e.g., by 1-2 points) to prevent oversaturated or 'burnt' images.")

                    with gr.TabItem("Detail Enhancement"):
                        gr.Markdown("### Detail Enhancement Settings")
                        self.use_enhanced_detail_phase = gr.Checkbox(label="Enable Detail Enhancement", value=True, info="Separates and enhances high-frequency details during sampling. Works with all schedulers.")

                        with gr.Group(visible=True) as enhancer_settings:
                            self.detail_enhancement_strength = gr.Slider(label="Detail Enhancement Strength", minimum=0.0, maximum=1.0, value=0.1, step=0.05)
                            self.detail_separation_radius = gr.Slider(label="Detail Separation Radius (Sigma)", minimum=0.1, maximum=2.0, value=0.5, step=0.05, info="Controls what is considered a 'detail'. Higher values sharpen larger features.")
                        
                        self.use_enhanced_detail_phase.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[self.use_enhanced_detail_phase],
                            outputs=[enhancer_settings]
                        )

                    with gr.TabItem("Advanced"):
                        gr.Markdown("### Advanced Noise & Solver Settings")
                        self.eta = gr.Slider(label='Eta (Ancestral)', minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        self.s_noise = gr.Slider(label='Noise Scale', minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        
                        gr.Markdown("---")
                        gr.Markdown("### Debugging")
                        self.debug_reproducibility = gr.Checkbox(label='Debug Reproducibility (disables advanced features)', value=False)
            
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
            (self.debug_stop_after_coherence, lambda p: str(p.get('debug_stop_after_coherence')).lower() == 'true' if 'debug_stop_after_coherence' in p else gr.update()),
            (self.use_enhanced_detail_phase, lambda p: str(p.get('enhanced_detail_phase')).lower() == 'true' if 'enhanced_detail_phase' in p else gr.update()),
            (self.detail_enhancement_strength, lambda p: gr.update() if p.get('detail_enhancement_strength') in (None, 'N/A') else float(p['detail_enhancement_strength'])),
            (self.detail_separation_radius, lambda p: gr.update() if p.get('detail_separation_radius') in (None, 'N/A') else float(p['detail_separation_radius'])),
        ]

        def scheduler_getter(params):
            if 'adept_sampler_enabled' not in params:
                return gr.update()
            
            if str(params.get('anime_optimized_schedule')).lower() == 'true':
                return "Anime-Optimized (AOS)"
            if str(params.get('entropic_scheduler')).lower() == 'true':
                return "Entropic"
            
            return "None"

        self.infotext_fields.append((self.scheduler_override, scheduler_getter))

        return [
            self.enable_custom,
            self.eta, self.s_noise, self.debug_reproducibility, 
            self.scheduler_override, self.entropic_scheduler_power,
            self.use_content_aware_pacing, self.pacing_coherence_sensitivity,
            self.debug_stop_after_coherence,
            self.use_enhanced_detail_phase,
            self.detail_enhancement_strength, self.detail_separation_radius,
        ]

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (
            enable_custom,
            eta, s_noise, debug_reproducibility,
            scheduler_override, entropic_scheduler_power,
            use_content_aware_pacing, pacing_coherence_sensitivity,
            debug_stop_after_coherence,
            use_enhanced_detail_phase,
            detail_enhancement_strength, detail_separation_radius,
        ) = script_args

        # Set scheduler flags based on the radio button choice
        use_anime_schedule = (scheduler_override == "Anime-Optimized (AOS)")
        use_entropic_scheduler = (scheduler_override == "Entropic")

        # Progressive enhancement is now tied to the detail enhancement checkbox
        use_progressive_enhancement = use_enhanced_detail_phase

        # Update global settings (this happens immediately)
        current_sampler_settings.update({
            'enabled': enable_custom,
            'eta': eta,
            's_noise': s_noise,
            'solver_order': 1, # Simplified to 1st order
            'debug_reproducibility': debug_reproducibility,
            'use_entropic_scheduler': use_entropic_scheduler,
            'entropic_scheduler_power': entropic_scheduler_power,
            'use_anime_schedule': use_anime_schedule,
            'use_content_aware_pacing': use_content_aware_pacing and use_anime_schedule, # Only works with AOS
            'pacing_coherence_sensitivity': pacing_coherence_sensitivity,
            'debug_stop_after_coherence': debug_stop_after_coherence and use_content_aware_pacing,
            'use_enhanced_detail_phase': use_enhanced_detail_phase,
            'detail_enhancement_strength': detail_enhancement_strength,
            'detail_separation_radius': detail_separation_radius,
            'use_progressive_enhancement': use_progressive_enhancement,
        })
        
        if enable_custom:
            if debug_reproducibility:
                print(f"🔬 Debug mode: Adept Sampler - simplified for reproducibility testing")
            else:
                print(f"✅ Adept Sampler is now active!")
            
            # Add parameters to generation info
            p.extra_generation_params.update({
                'adept_sampler_enabled': True,
                'adept_sampler_type': 'Enhanced (AOS Focused)',
                'custom_eta': eta,
                'custom_s_noise': s_noise,
                'solver_order': 1, # Simplified
                'custom_sampler_deterministic': True,
                'debug_reproducibility': debug_reproducibility,
                'entropic_scheduler': use_entropic_scheduler and not debug_reproducibility,
                'entropic_power': entropic_scheduler_power if use_entropic_scheduler and not use_anime_schedule else 'N/A',
                'anime_optimized_schedule': use_anime_schedule and not debug_reproducibility,
                'content_aware_pacing': use_content_aware_pacing and use_anime_schedule,
                'coherence_sensitivity': pacing_coherence_sensitivity if use_content_aware_pacing and use_anime_schedule else 'N/A',
                'debug_stop_after_coherence': debug_stop_after_coherence and use_content_aware_pacing and use_anime_schedule,
                'use_progressive_enhancement': use_progressive_enhancement,
                'enhanced_detail_phase': use_enhanced_detail_phase,
                'detail_enhancement_strength': detail_enhancement_strength if use_enhanced_detail_phase else 'N/A',
                'detail_separation_radius': detail_separation_radius if use_enhanced_detail_phase else 'N/A',
            })
        else:
            print("🔄 Using standard Euler Ancestral sampler")
    

    def sample_enhanced_euler_ancestral(self, model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., solver_order=1, generator=None, use_dynamic_ancestral_noise=False, use_heun_corrector=False):
        """Simplified custom Euler Ancestral with dynamic thresholding, focused on AOS."""
        # --- Read settings from global config to ensure they are always correct ---
        use_enhanced_detail_phase = current_sampler_settings.get('use_enhanced_detail_phase', True)

        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        
        # Get the proper noise sampler for reproducibility
        noise_sampler = self.get_noise_sampler(x)
        
        # --- Content-Aware Pacing Setup ---
        total_steps = len(sigmas) - 1
        original_sigmas = sigmas.clone() # Keep a copy for rescheduling
        use_pacing = current_sampler_settings.get('use_content_aware_pacing', False) and total_steps > 0

        if use_pacing:
            # --- Simplified PACING ---
            is_coherent = False
            initial_variance = None
            last_composition_derivative = None
            last_composition_dt = None
            
            # Safety fallback
            fallback_step_pct = 0.7

            # --- Composition Phase ---
            print("🧠 Pacing: Starting composition phase...")
            composition_steps_taken = 0
            i = 0
            last_composition_sigma_idx = 0
            
            # The composition loop runs at double speed
            while i < (total_steps - 1) and composition_steps_taken < int(total_steps * fallback_step_pct):
                composition_steps_taken += 1
                last_composition_sigma_idx = i
                
                current_sigma = original_sigmas[i]
                next_sigma_idx = min(i + 2, total_steps)
                next_sigma = original_sigmas[next_sigma_idx]

                if current_sigma < next_sigma: break

                denoised = model(x, current_sigma * s_in, **extra_args)
                
                derivative = (x - denoised) / current_sigma
                last_composition_derivative = derivative
                
                # --- Coherence Calculation ---
                variance = torch.var(derivative.flatten(1), dim=1).mean().item()

                if composition_steps_taken == 2:
                    # Establish a stable baseline variance after the initial large drop from pure noise.
                    initial_variance = variance
                elif composition_steps_taken > 2 and initial_variance is not None:
                    # Start checking for coherence against the post-drop baseline.
                    sensitivity = current_sampler_settings.get('pacing_coherence_sensitivity', 0.75)
                    threshold_percentage = sensitivity * 0.4 + 0.5
                    coherence_threshold = initial_variance * threshold_percentage
                
                    if variance < coherence_threshold:
                        print(f"🧠 Pacing: Coherence achieved at iteration {composition_steps_taken} (Sigma Step {i}). Rescheduling detail phase.")
                        is_coherent = True
                        break

                if callback is not None: callback({'x': x, 'i': i, 'sigma': original_sigmas[i], 'sigma_hat': original_sigmas[i], 'denoised': denoised})
                
                sigma_down, sigma_up = self.get_ancestral_step(current_sigma, next_sigma, eta)
                dt = sigma_down - current_sigma
                last_composition_dt = dt
                x = x + derivative * dt
                if next_sigma > 0: x = x + noise_sampler(current_sigma, next_sigma) * s_noise * sigma_up
                
                i = next_sigma_idx

            # --- Detail Phase ---
            if is_coherent and current_sampler_settings.get('debug_stop_after_coherence', False):
                print("🛑 [Debug] Coherence achieved. Stopping generation before detail phase as requested.")
                return x
            
            # The number of detail steps is the total steps minus how many composition steps we took.
            remaining_iterations = total_steps - composition_steps_taken
            
            if not is_coherent:
                print("🧠 Pacing: Composition phase finished. Switching to detail phase for remaining steps.")
                is_coherent = True # Enable detail-phase logic

            if remaining_iterations <= 0 and total_steps > 0:
                print(f"⚠️ Warning: No steps remaining for detail phase. Composition took all {composition_steps_taken} steps.")
                # Ensure we still return the final image from the composition phase
                return x

            if remaining_iterations > 0:
                print(f"🧠 Pacing: Starting detail phase with {remaining_iterations} steps.")
                
                sigma_at_switch = original_sigmas[last_composition_sigma_idx]
                sigma_min = original_sigmas[-2] # The one before zero

                detail_sigmas = self.create_detail_schedule(sigma_at_switch, sigma_min, remaining_iterations, x.device)
                
                # We don't need a derivative history for the simplified approach
                derivatives = []
                dts = []

                if last_composition_derivative is not None:
                    derivatives.append(last_composition_derivative)
                    if last_composition_dt is not None:
                        dts.append(last_composition_dt)

                for j in range(len(detail_sigmas) - 1):
                    current_sigma = detail_sigmas[j]
                    next_sigma = detail_sigmas[j+1]

                    if current_sigma < next_sigma: break
                    
                    denoised = model(x, current_sigma * s_in, **extra_args)

                    derivative = (x - denoised) / current_sigma

                    progress = (last_composition_sigma_idx + j) / total_steps
                    callback_step = last_composition_sigma_idx + j
                    if callback is not None: callback({'x': x, 'i': callback_step, 'sigma': current_sigma, 'sigma_hat': current_sigma, 'denoised': denoised})

                    sigma_down, sigma_up = self.get_ancestral_step(current_sigma, next_sigma, eta)
                    dt = sigma_down - current_sigma

                    x = x + derivative * dt
                    
                    # --- High-Frequency Detail Enhancement ---
                    if use_enhanced_detail_phase and TORCHVISION_AVAILABLE:
                        base_strength = current_sampler_settings.get('detail_enhancement_strength', 0.1)
                        progress = (last_composition_sigma_idx + j) / total_steps
                        strength = self.apply_progressive_enhancement(base_strength, 'detail', progress)
                        
                        radius = current_sampler_settings.get('detail_separation_radius', 0.5)
                        low_freq = gaussian_blur(denoised, kernel_size=3, sigma=radius)
                        high_freq = denoised - low_freq
                        
                        enhancement_amount = dt.abs() / current_sigma.clamp(min=1e-6)
                        x = x + high_freq * enhancement_amount * strength
                    
                    if next_sigma > 0: x = x + noise_sampler(current_sigma, next_sigma) * s_noise * sigma_up
        else:
            # --- Pacing Disabled: Standard Single-Phase Sampling ---
            print("Pacing disabled. Running in standard single-phase mode.")

            for i in range(total_steps):
                denoised = model(x, sigmas[i] * s_in, **extra_args)

                derivative = (x - denoised) / sigmas[i]

                if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

                sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i+1], eta)
                dt = sigma_down - sigmas[i]

                x = x + derivative * dt
                
                # --- High-Frequency Detail Enhancement ---
                if use_enhanced_detail_phase and TORCHVISION_AVAILABLE:
                    base_strength = current_sampler_settings.get('detail_enhancement_strength', 0.1)
                    strength = self.apply_progressive_enhancement(base_strength, 'single_phase', i/total_steps)
                    
                    radius = current_sampler_settings.get('detail_separation_radius', 0.5)
                    low_freq = gaussian_blur(denoised, kernel_size=3, sigma=radius)
                    high_freq = denoised - low_freq

                    enhancement_amount = dt.abs() / sigmas[i].clamp(min=1e-6)
                    x = x + high_freq * enhancement_amount * strength
                
                if sigmas[i+1] > 0: x = x + noise_sampler(sigmas[i], sigmas[i+1]) * s_noise * sigma_up
        
        return x

    def create_detail_schedule(self, sigma_max, sigma_min, num_steps, device):
        """Creates a schedule for the detail phase, respecting the original scheduler choice."""
        if current_sampler_settings.get('use_anime_schedule'):
            return self.create_anime_optimized_sigmas(sigma_max, sigma_min, num_steps, device)
        elif current_sampler_settings.get('use_entropic_scheduler'):
            power = current_sampler_settings.get('entropic_scheduler_power', 3.0)
            return self.create_entropic_sigmas(sigma_max, sigma_min, num_steps, power, device)
        else:
            # Fallback to entropic with neutral power, as it's self-contained.
            return self.create_entropic_sigmas(sigma_max, sigma_min, num_steps, 1.0, device)

    def apply_progressive_enhancement(self, base_strength, phase, progress):
        """Applies enhancement based on the current sampling phase."""
        if phase == 'composition':
            return base_strength * 0.5  # Lighter enhancement
        elif phase == 'detail':
            return base_strength * 1.5  # Stronger enhancement
        else:  # single_phase
            return base_strength * (0.5 + progress) # Gradual increase

    def get_ancestral_step(self, sigma, sigma_next, eta=1.):
        """Calculate ancestral step sizes"""
        sigma_up = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
        sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up
    
    def get_noise_sampler(self, x):
        """Get the proper noise sampler using k_diffusion's approach"""
        if hasattr(k_diffusion.sampling, 'default_noise_sampler'):
            return k_diffusion.sampling.default_noise_sampler(x)
        else:
            # Fallback: create a simple noise sampler
            def simple_noise_sampler(sigma_from, sigma_to):
                return torch.randn_like(x)
            return simple_noise_sampler

    def create_anime_optimized_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """Creates a three-phase noise schedule optimized for anime aesthetics."""
        rho = 7.0  # karras-ve rho
        
        # Phase boundaries (as fractions of total steps)
        p1_frac, p2_frac = 0.2, 0.6

        # Ramp values at phase boundaries, defining the sigma drop steepness
        ramp_p1_val, ramp_p2_val = 0.6, 0.9

        # Step indices for boundaries
        p1_steps = int(num_steps * p1_frac)
        p2_steps = int(num_steps * p2_frac)

        # Phase 1: Composition Lock-in (aggressive, starts fast)
        # Power < 1 starts fast then slows down.
        phase1_ramp = torch.linspace(0, 1, p1_steps, device=device) ** 0.5 * ramp_p1_val

        # Phase 2: Color Blocking (medium, linear steps)
        phase2_ramp = torch.linspace(ramp_p1_val, ramp_p2_val, p2_steps - p1_steps, device=device)

        # Phase 3: Detail Refinement (slow, extended tail)
        # Power > 1 starts slow then speeds up at the very end.
        phase3_base = torch.linspace(0, 1, num_steps - p2_steps, device=device) ** 3
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


# Initialize the extension when script loads
patch_samplers_globally()
print("Adept Sampler for reForge loaded successfully!") 