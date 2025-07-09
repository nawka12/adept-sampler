"""
Adept Scheduler Node for ComfyUI.
Generates custom sigma schedules for advanced sampling.
"""

import torch
from .schedulers import AdeptSchedulers


class AdeptSchedulerNode:
    """ComfyUI node for generating custom sigma schedules."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (["karras", "AOS-V", "AOS-ε", "Entropic", "SNR-Optimized", "Constant-Rate", "Adaptive-Optimized"],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "entropic_power": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"
    
    def get_sigmas(self, scheduler, steps, denoise, entropic_power=6.0, model=None):
        """Generate custom sigma schedule."""
        
        # Get device (prefer model device if available)
        device = "cpu"
        if model is not None:
            try:
                # Try to get device from model
                device = next(model.parameters()).device
            except:
                device = "cpu"
        
        # Use default karras schedule as fallback or when explicitly selected
        if scheduler == "karras" or scheduler not in ["AOS-V", "AOS-ε", "Entropic", "SNR-Optimized", "Constant-Rate", "Adaptive-Optimized"]:
            # ComfyUI's default karras schedule
            return (self._get_karras_sigmas(steps, denoise, device),)
        
        # Calculate sigma range for custom schedulers
        # Using typical SD ranges
        sigma_max = 14.6146  # Typical max for SD models
        sigma_min = 0.0291   # Typical min for SD models
        
        # Apply denoise factor
        if denoise < 1.0:
            t = int(steps * denoise)
            steps = t
            
        scheduler_fn = AdeptSchedulers.get_scheduler_fn(scheduler)
        if scheduler_fn is None:
            # Fallback to karras
            return (self._get_karras_sigmas(steps, denoise, device),)
            
        if scheduler == "Entropic":
            sigmas = scheduler_fn(sigma_max, sigma_min, steps, entropic_power, device)
        else:
            sigmas = scheduler_fn(sigma_max, sigma_min, steps, device)
            
        return (sigmas,)
    
    def _get_karras_sigmas(self, steps, denoise, device):
        """Generate standard Karras schedule as fallback."""
        # Simple Karras schedule implementation
        sigma_max = 14.6146
        sigma_min = 0.0291
        rho = 7.0
        
        if denoise < 1.0:
            t = int(steps * denoise)
            steps = t
            
        step_indices = torch.arange(0, steps, dtype=torch.float32, device=device)
        
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        
        sigmas = (max_inv_rho + step_indices / (steps - 1) * (min_inv_rho - max_inv_rho)) ** rho
        
        return torch.cat([sigmas, torch.zeros(1, device=device)])