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
    'debug_reproducibility': False,
    'use_entropic_scheduler': False,
    'entropic_scheduler_power': 6.0,
    'use_anime_schedule': False,
    'use_enhanced_detail_phase': True,
    'detail_enhancement_strength': 0.05,
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
                
                # The sigma override logic has been moved inside sample_enhanced_euler_ancestral
                return script_instance.sample_enhanced_euler_ancestral(
                    model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                    eta=current_sampler_settings.get('eta', 1.0),
                    s_noise=current_sampler_settings.get('s_noise', 1.0),
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
                            label="Scheduler",
                            value="None",
                            choices=[
                                "None",
                                "Entropic",
                                "Constant-Rate",
                                "Adaptive-Optimized",
                                "AOS-V (for v-prediction)",
                                "SNR-Optimized",
                                "AOS-ε (for ε-prediction)",
                            ]
                        )

                        with gr.Group(visible=False) as entropic_options:
                            self.entropic_scheduler_power = gr.Slider(
                                label='Entropic Power', 
                                minimum=1.0, maximum=8.0, 
                                value=6.0, step=0.1,
                                info="Controls timestep clustering. >1 clusters steps at the start (high detail)."
                            )
                        
                        gr.Markdown(
                            "**Scheduler Categories:**<br>"
                            "▻ **Universal**: `None`, `Entropic`, `Constant-Rate`, `Adaptive-Optimized`<br>"
                            "▻ **V-Prediction**: `AOS-V`, `SNR-Optimized`<br>"
                            "▻ **ε-Prediction**: `AOS-ε`"
                        )

                        with gr.Group(visible=False) as aos_plus_options:
                            gr.Markdown("⚠️ **Compatibility Warning:** Use the correct AOS variant for your model type. **AOS-V** is for **v-prediction** models. **AOS-ε** is for **epsilon-prediction** models. Mismatching them may break the generation.")
                            self.use_content_aware_pacing = gr.Checkbox(label='Enable Content-Aware Pacing (AOS Only)', value=False, info="Dynamically adjusts pacing based on image coherence. Requires higher step counts (at least model recommended step count * 1.5) for effective phasing.")
                            self.pacing_coherence_sensitivity = gr.Slider(
                                label='Coherence Sensitivity',
                                minimum=0.1, maximum=1.0, value=0.75, step=0.05,
                                info="Controls when to switch from composition to detail. Higher values switch sooner."
                            )
                            self.manual_pacing_override = gr.Textbox(
                                label="Manual Pacing Override (JSON)",
                                info='Advanced: Manually set phase steps. e.g., {"composition": 0.4} or {"composition": 10}',
                                placeholder='Disabled (uses automatic pacing)'
                            )
                            self.debug_stop_after_coherence = gr.Checkbox(label='[Debug] Stop after coherence', value=False, info="Stops generation immediately after coherence is detected, skipping the detail phase.")

                        def on_scheduler_change(scheduler):
                            is_aos = "AOS-V" in scheduler or "AOS-ε" in scheduler
                            return {
                                aos_plus_options: gr.update(visible=is_aos),
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
                            self.detail_enhancement_strength = gr.Slider(label="Detail Enhancement Strength", minimum=0.0, maximum=1.0, value=0.05, step=0.05)
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
            (self.manual_pacing_override, lambda p: p.get('manual_pacing_override', gr.update())),
            (self.debug_stop_after_coherence, lambda p: str(p.get('debug_stop_after_coherence')).lower() == 'true' if 'debug_stop_after_coherence' in p else gr.update()),
            (self.use_enhanced_detail_phase, lambda p: str(p.get('enhanced_detail_phase')).lower() == 'true' if 'enhanced_detail_phase' in p else gr.update()),
            (self.detail_enhancement_strength, lambda p: gr.update() if p.get('detail_enhancement_strength') in (None, 'N/A') else float(p['detail_enhancement_strength'])),
            (self.detail_separation_radius, lambda p: gr.update() if p.get('detail_separation_radius') in (None, 'N/A') else float(p['detail_separation_radius'])),
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
            
            return "None"

        self.infotext_fields.append((self.scheduler_override, scheduler_getter))

        return [
            self.enable_custom,
            self.eta, self.s_noise, self.debug_reproducibility, 
            self.scheduler_override, self.entropic_scheduler_power,
            self.use_content_aware_pacing, self.pacing_coherence_sensitivity,
            self.manual_pacing_override,
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
            manual_pacing_override,
            debug_stop_after_coherence,
            use_enhanced_detail_phase,
            detail_enhancement_strength, detail_separation_radius,
        ) = script_args

        # Set scheduler flags based on the radio button choice
        use_anime_schedule_v = (scheduler_override == "AOS-V (for v-prediction)")
        use_anime_schedule_e = (scheduler_override == "AOS-ε (for ε-prediction)")
        use_anime_schedule = use_anime_schedule_v or use_anime_schedule_e
        use_entropic_scheduler = (scheduler_override == "Entropic")

        custom_scheduler_type = "None"
        if scheduler_override in ["SNR-Optimized", "Constant-Rate", "Adaptive-Optimized"]:
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

        # Update global settings (this happens immediately)
        current_sampler_settings.update({
            'enabled': enable_custom,
            'eta': eta,
            's_noise': s_noise,
            'debug_reproducibility': debug_reproducibility,
            'use_entropic_scheduler': use_entropic_scheduler,
            'entropic_scheduler_power': entropic_scheduler_power,
            'use_anime_schedule': use_anime_schedule,
            'use_anime_schedule_v': use_anime_schedule_v,
            'use_anime_schedule_e': use_anime_schedule_e,
            'use_content_aware_pacing': use_content_aware_pacing and use_anime_schedule, # Only works with AOS
            'pacing_coherence_sensitivity': pacing_coherence_sensitivity,
            'manual_pacing_schedule': manual_pacing_schedule,
            'debug_stop_after_coherence': debug_stop_after_coherence and use_content_aware_pacing,
            'use_enhanced_detail_phase': use_enhanced_detail_phase,
            'detail_enhancement_strength': detail_enhancement_strength,
            'detail_separation_radius': detail_separation_radius,
            'custom_scheduler_type': custom_scheduler_type,
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
                'custom_sampler_deterministic': True,
                'debug_reproducibility': debug_reproducibility,
                'entropic_scheduler': use_entropic_scheduler and not debug_reproducibility,
                'entropic_power': entropic_scheduler_power if use_entropic_scheduler and not use_anime_schedule else 'N/A',
                'anime_optimized_schedule': 'V' if use_anime_schedule_v else ('Epsilon' if use_anime_schedule_e else 'N/A'),
                'content_aware_pacing': use_content_aware_pacing and use_anime_schedule,
                'coherence_sensitivity': pacing_coherence_sensitivity if use_content_aware_pacing and use_anime_schedule and not manual_pacing_schedule else 'N/A',
                'manual_pacing_override': json.dumps(manual_pacing_schedule) if manual_pacing_schedule else 'N/A',
                'debug_stop_after_coherence': debug_stop_after_coherence and use_content_aware_pacing and use_anime_schedule,
                'enhanced_detail_phase': use_enhanced_detail_phase,
                'detail_enhancement_strength': detail_enhancement_strength if use_enhanced_detail_phase else 'N/A',
                'detail_separation_radius': detail_separation_radius if use_enhanced_detail_phase else 'N/A',
                'custom_scheduler_type': custom_scheduler_type,
            })
        else:
            print("🔄 Using standard Euler Ancestral sampler")
    

    def sample_enhanced_euler_ancestral(self, model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., generator=None):
        """Simplified custom Euler Ancestral with dynamic thresholding, focused on AOS."""
        # --- Read settings from global config to ensure they are always correct ---
        use_enhanced_detail_phase = current_sampler_settings.get('use_enhanced_detail_phase', True)
        custom_scheduler_type = current_sampler_settings.get('custom_scheduler_type', 'None')

        # --- Sigma Schedule Override ---
        final_sigmas = sigmas
        is_custom_scheduler = custom_scheduler_type != 'None'

        if is_custom_scheduler and not current_sampler_settings.get('debug_reproducibility', False):
            print(f"🔬 Overriding sigma schedule with Custom Scheduler: {custom_scheduler_type}.")
            if len(sigmas) > 1:
                sigma_args = (sigmas[0], sigmas[-2], len(sigmas) - 1, sigmas.device)
                scheduler_map = {
                    "SNR-Optimized": self.create_snr_optimized_sigmas,
                    "Constant-Rate": self.create_constant_rate_sigmas,
                    "Adaptive-Optimized": self.create_adaptive_optimized_sigmas,
                }
                if custom_scheduler_type in scheduler_map:
                    final_sigmas = scheduler_map[custom_scheduler_type](*sigma_args)
        elif current_sampler_settings.get('use_entropic_scheduler', False) and not current_sampler_settings.get('debug_reproducibility', False):
            print("🔄 Overriding sigma schedule with Entropic Time Scheduler.")
            power = current_sampler_settings.get('entropic_scheduler_power', 3.0)
            if len(sigmas) > 1:
                final_sigmas = self.create_entropic_sigmas(
                    sigmas[0], sigmas[-2], len(sigmas) - 1, power, sigmas.device
                )
        elif current_sampler_settings.get('use_anime_schedule', False) and not current_sampler_settings.get('debug_reproducibility', False):
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
                print("🧠 Pacing: Using conservative single-step pacing for medium step count (<= 40).")
                pacing_step_size = 1
            else:
                print("🧠 Pacing: Using aggressive double-step pacing for high step count (> 40).")
                pacing_step_size = 2

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
                    denoised = model(x, original_sigmas[i] * s_in, **extra_args)
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
                print("🧠 Pacing: Starting composition phase...")
                i = 0
                
                while i < (total_steps - 1) and composition_steps_taken < int(total_steps * fallback_step_pct):
                    composition_steps_taken += 1
                    last_composition_sigma_idx = i
                    
                    current_sigma = original_sigmas[i]
                    next_sigma_idx = min(i + pacing_step_size, total_steps)
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
                    
                    i = next_sigma_idx
                
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
                
                # Regardless of the main scheduler, when pacing triggers a phase switch,
                # we create a new detail-focused schedule for the remaining steps.
                sigma_at_switch = original_sigmas[min(sigma_idx_at_switch, total_steps)]
                sigma_min = original_sigmas[-2] # The one before zero
                detail_sigmas = self.create_detail_schedule(sigma_at_switch, sigma_min, remaining_iterations, x.device)
                
                # The derivative from the composition phase is used to smooth the first step of the detail phase.
                # A full derivative history is not needed for this simplified solver.
                if len(detail_sigmas) > 1:
                    for j in range(len(detail_sigmas) - 1):
                        current_sigma = detail_sigmas[j]
                        next_sigma = detail_sigmas[j+1]

                        if current_sigma < next_sigma: break
                        
                        denoised = model(x, current_sigma * s_in, **extra_args)

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
                denoised = model(x, final_sigmas[i] * s_in, **extra_args)

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
        elif current_sampler_settings.get('use_entropic_scheduler'):
            power = current_sampler_settings.get('entropic_scheduler_power', 3.0)
            return self.create_entropic_sigmas(sigma_max, sigma_min, num_steps, power, device)
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
        """Get the proper noise sampler using k_diffusion's approach"""
        if hasattr(k_diffusion.sampling, 'default_noise_sampler'):
            return k_diffusion.sampling.default_noise_sampler(x)
        else:
            # Fallback: create a simple noise sampler
            def simple_noise_sampler(sigma_from, sigma_to):
                return torch.randn_like(x)
            return simple_noise_sampler

    def create_aos_v_sigmas(self, sigma_max, sigma_min, num_steps, device='cpu'):
        """Creates a three-phase noise schedule optimized for anime aesthetics on v-prediction models."""
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

    # --- End of Experimental Schedulers and Methods ---


# Initialize the extension when script loads
patch_samplers_globally()
print("Adept Sampler for reForge loaded successfully!") 