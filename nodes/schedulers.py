"""
Scheduler implementations for Adept Sampler ComfyUI nodes.
Adapted from the reForge implementation.
"""

import math
import torch


class AdeptSchedulers:
    """Collection of advanced schedulers for diffusion sampling."""
    
    @staticmethod
    def create_aos_v_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
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

    @staticmethod
    def create_aos_e_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
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

    @staticmethod
    def create_entropic_sigmas(sigma_max, sigma_min, num_steps, power=3.0, device='cpu'):
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

    @staticmethod
    def create_snr_optimized_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
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

    @staticmethod
    def create_constant_rate_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
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

    @staticmethod
    def create_adaptive_optimized_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
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
        
        if (torch.max(combined_t) - torch.min(combined_t)) > 1e-6:
            combined_t = (combined_t - torch.min(combined_t)) / (torch.max(combined_t) - torch.min(combined_t))
        
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + combined_t * (min_inv_rho - max_inv_rho)) ** rho
        
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    @staticmethod
    def get_scheduler_fn(scheduler_type):
        """Get the scheduler function by name."""
        scheduler_map = {
            "AOS-V": AdeptSchedulers.create_aos_v_sigmas,
            "AOS-ε": AdeptSchedulers.create_aos_e_sigmas,
            "Entropic": AdeptSchedulers.create_entropic_sigmas,
            "SNR-Optimized": AdeptSchedulers.create_snr_optimized_sigmas,
            "Constant-Rate": AdeptSchedulers.create_constant_rate_sigmas,
            "Adaptive-Optimized": AdeptSchedulers.create_adaptive_optimized_sigmas,
        }
        return scheduler_map.get(scheduler_type)