# Adept Sampler for ComfyUI

This is the ComfyUI implementation of the Adept Sampler, converted from the original Stable Diffusion WebUI reForge version. It provides advanced sampling techniques including custom schedulers, content-aware pacing, and detail enhancement for ComfyUI workflows.

## 🌟 Features

- **Advanced Schedulers**: Multiple custom scheduler types including AOS-V, AOS-ε, Entropic, SNR-Optimized, Constant-Rate, and Adaptive-Optimized
- **Content-Aware Pacing**: Dynamic two-phase sampling that switches from composition to detail based on image coherence (AOS schedulers only)
- **Detail Enhancement**: High-frequency detail separation and enhancement for improved image quality
- **ComfyUI Integration**: Native ComfyUI nodes that work seamlessly with existing workflows
- **Flexible Configuration**: Extensive parameters for fine-tuning the sampling process

## 📦 Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-repo/adept-sampler-comfy
   ```

2. Restart ComfyUI

3. The nodes will appear in the `sampling/custom_sampling` category

## 🛠️ Available Nodes

### 1. Adept Scheduler
- **Purpose**: Generates custom sigma schedules for advanced sampling
- **Location**: `sampling/custom_sampling/schedulers`
- **Inputs**:
  - `scheduler`: Choose from various scheduler types
  - `steps`: Number of sampling steps
  - `denoise`: Denoising strength (0.01-1.0)
  - `entropic_power`: Power setting for Entropic scheduler (1.0-8.0)
  - `model`: (Optional) Model for device detection
- **Output**: `SIGMAS` - Custom sigma schedule

### 2. Adept Sampler (ComfyUI)
- **Purpose**: Enhanced Euler Ancestral sampler with advanced features
- **Location**: `sampling/custom_sampling`
- **Inputs**:
  - Standard ComfyUI sampling inputs (`model`, `noise`, `guider`, `sampler`, `sigmas`, `latent_image`)
  - `scheduler_type`: Type of scheduler to use
  - `eta`: Ancestral noise amount (0.0-2.0)
  - `s_noise`: Noise scale (0.0-2.0)
  - `entropic_power`: Power for Entropic scheduler
  - Content-aware pacing settings
  - Detail enhancement settings
- **Output**: `LATENT` - Generated latent image

### 3. Adept Sampler (Simple)
- **Purpose**: Simplified all-in-one sampler node
- **Location**: `sampling/custom_sampling`
- **Inputs**: Similar to ComfyUI version but with simplified interface

## 📚 Scheduler Types

### Universal Schedulers (Recommended for all models)
- **karras**: Standard Karras schedule (baseline)
- **Entropic**: Power-based clustering with configurable power
- **Constant-Rate**: Ensures constant rate of distributional change
- **Adaptive-Optimized**: Hybrid approach blending multiple strategies

### Specialized Schedulers
- **AOS-V**: Anime-Optimized Schedule for v-prediction models
- **AOS-ε**: Anime-Optimized Schedule for epsilon-prediction models
- **SNR-Optimized**: Concentrates steps around critical logSNR = 0 point

## 🎛️ Key Parameters

### Scheduler Settings
- **scheduler_type**: Choose the scheduling algorithm
- **entropic_power**: Controls timestep clustering (higher = more detail focus early)

### Content-Aware Pacing (AOS only)
- **enable_content_aware_pacing**: Enable dynamic two-phase sampling
- **coherence_sensitivity**: Controls when to switch from composition to detail (0.1-1.0)
- **manual_pacing_override**: JSON string for manual phase control
- **debug_stop_after_coherence**: Stop after composition phase for debugging

### Detail Enhancement
- **enable_detail_enhancement**: Enable high-frequency detail enhancement
- **detail_enhancement_strength**: Strength of detail enhancement (0.0-1.0)
- **detail_separation_radius**: Sigma for detail separation (0.1-2.0)

### Noise Control
- **eta**: Ancestral noise amount (1.0 = standard Euler Ancestral)
- **s_noise**: Additional noise scaling

## 🔧 Usage Examples

### Basic Usage
1. Add an **Adept Scheduler** node
2. Set scheduler type (e.g., "Entropic")
3. Connect to **Adept Sampler (ComfyUI)** node
4. Configure your model, conditioning, and other standard inputs
5. Adjust parameters as needed

### Advanced Content-Aware Pacing
1. Use **AOS-V** or **AOS-ε** scheduler (match your model type)
2. Enable **Content-Aware Pacing**
3. Set **coherence_sensitivity** (0.75 is a good starting point)
4. Use higher step counts (at least 30+ for effective phasing)
5. Consider lowering CFG scale by 1-2 points

### Manual Pacing Control
Use the **manual_pacing_override** with JSON format:
```json
{"composition": 0.4}
```
This allocates 40% of steps to composition phase.

## ⚠️ Important Notes

### Model Compatibility
- **AOS-V**: Use with v-prediction models only
- **AOS-ε**: Use with epsilon-prediction models only
- **Universal schedulers**: Work with any model type

### CFG Scale Adjustment
When using custom schedulers, you may need to **lower your CFG Scale** by 1-2 points to prevent oversaturated images.

### Step Count Recommendations
- **Content-Aware Pacing**: Use at least 26 steps, ideally 30+
- **Standard sampling**: Works with any step count
- **Detail enhancement**: More effective with higher step counts

## 🔍 Comparison with Original

This ComfyUI version maintains full feature parity with the original reForge implementation:

| Feature | reForge | ComfyUI |
|---------|---------|---------|
| Custom Schedulers | ✅ | ✅ |
| Content-Aware Pacing | ✅ | ✅ |
| Detail Enhancement | ✅ | ✅ |
| Manual Pacing Override | ✅ | ✅ |
| All Scheduler Types | ✅ | ✅ |
| Debug Features | ✅ | ✅ |

## 🐛 Troubleshooting

### Common Issues
1. **No visible improvement**: Try lowering CFG scale and using higher step counts
2. **Oversaturated images**: Lower CFG scale by 1-2 points
3. **Pacing not working**: Ensure you're using AOS schedulers and sufficient steps
4. **Detail enhancement not working**: Check that torchvision is installed

### Performance Tips
- Start with **Entropic** scheduler for general use
- Use **AOS** schedulers for anime/illustration style content
- Experiment with **coherence_sensitivity** between 0.5-0.9
- **Detail enhancement strength** of 0.05-0.1 works well for most cases

## 📄 License

This project maintains the same **GNU General Public License v3.0 (GPL-3.0)** as the original implementation.

## 🙏 Credits

This ComfyUI version is adapted from the original [Adept Sampler for Stable Diffusion WebUI reForge](https://github.com/nawka12/adept-sampler) by nawka12/KayfaHaarukku.

---

For more detailed information about the sampling techniques and research behind this implementation, please refer to the original project's README.