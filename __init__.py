from .nodes.adept_sampler_node import AdeptSamplerNode
from .nodes.adept_scheduler_node import AdeptSchedulerNode
from .nodes.comfy_adept_sampler import AdeptSamplerComfy

NODE_CLASS_MAPPINGS = {
    "AdeptSamplerNode": AdeptSamplerNode,
    "AdeptSchedulerNode": AdeptSchedulerNode,
    "AdeptSamplerComfy": AdeptSamplerComfy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdeptSamplerNode": "Adept Sampler (Simple)",
    "AdeptSchedulerNode": "Adept Scheduler",
    "AdeptSamplerComfy": "Adept Sampler (ComfyUI)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]