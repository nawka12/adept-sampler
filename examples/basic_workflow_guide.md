# ComfyUI Workflow Guide for Adept Sampler

This guide provides step-by-step instructions for setting up Adept Sampler in your ComfyUI workflows.

## Basic Setup

### Method 1: Using Adept Scheduler + Adept Sampler (Recommended)

1. **Add Nodes:**
   - Add `Adept Scheduler` node (Category: `sampling/custom_sampling/schedulers`)
   - Add `Adept Sampler (ComfyUI)` node (Category: `sampling/custom_sampling`)

2. **Configure Scheduler:**
   - Set `scheduler` to desired type (e.g., "Entropic" for general use)
   - Set `steps` to your desired step count
   - Set `denoise` to 1.0 for full denoising
   - Adjust `entropic_power` if using Entropic scheduler

3. **Connect Scheduler:**
   - Connect Scheduler's `SIGMAS` output to Sampler's `sigmas` input

4. **Configure Sampler:**
   - Connect your `MODEL`, `CONDITIONING` (positive/negative), and `LATENT`
   - Set `scheduler_type` to match your scheduler choice
   - Adjust other parameters as needed

### Method 2: Using Adept Sampler (Simple)

1. **Add Node:**
   - Add `Adept Sampler (Simple)` node

2. **Configure:**
   - Connect all standard inputs (`model`, `positive`, `negative`, `latent_image`)
   - Set scheduler and other parameters directly in the node

## Recommended Settings by Use Case

### General Purpose (Most Models)
```
Scheduler: Entropic
Steps: 20-30
Entropic Power: 6.0
Detail Enhancement: Enabled
Detail Enhancement Strength: 0.05
CFG: Your normal CFG - 1
```

### Anime/Illustration (v-prediction models)
```
Scheduler: AOS-V
Steps: 30-40
Content-Aware Pacing: Enabled
Coherence Sensitivity: 0.75
Detail Enhancement: Enabled
Detail Enhancement Strength: 0.05-0.1
CFG: Your normal CFG - 2
```

### Anime/Illustration (epsilon-prediction models)
```
Scheduler: AOS-ε
Steps: 30-40
Content-Aware Pacing: Enabled
Coherence Sensitivity: 0.75
Detail Enhancement: Enabled
Detail Enhancement Strength: 0.05-0.1
CFG: Your normal CFG - 2
```

### High Detail Focus
```
Scheduler: Entropic
Steps: 25-35
Entropic Power: 7.0-8.0
Detail Enhancement: Enabled
Detail Enhancement Strength: 0.1-0.15
Detail Separation Radius: 0.3-0.5
CFG: Your normal CFG - 1
```

## Advanced Features

### Content-Aware Pacing Setup
1. Use AOS-V or AOS-ε scheduler
2. Enable `enable_content_aware_pacing` 
3. Set `coherence_sensitivity` (0.5-0.9 range)
4. Use at least 30 steps for effective phasing
5. Optional: Use `manual_pacing_override` for precise control

### Manual Pacing Examples
```json
{"composition": 0.4}          # 40% composition, 60% detail
{"composition": 12}           # Exactly 12 steps composition
```

### Debug Mode
- Enable `debug_stop_after_coherence` to see composition phase only
- Useful for testing coherence detection sensitivity

## Workflow Integration

### Standard Text-to-Image
```
[Text Encode] → [Adept Scheduler] → [Adept Sampler] → [VAE Decode] → [Save Image]
     ↓                ↓                    ↓
[CLIP] → [Model] → [Empty Latent] → [Conditioning]
```

### ControlNet Integration
```
[ControlNet] → [Apply ControlNet] → [Adept Sampler]
```

### LoRA Integration
```
[Load LoRA] → [Model] → [Adept Sampler]
```

### Hires Fix Workflow
```
[Text-to-Image with Adept] → [Upscale Latent] → [Adept Sampler] → [VAE Decode]
```

## Performance Tips

1. **Start Simple**: Begin with Entropic scheduler and default settings
2. **CFG Adjustment**: Always reduce CFG by 1-2 points with custom schedulers
3. **Step Count**: Higher step counts (25+) show more benefit from advanced features
4. **Model Matching**: Use correct AOS variant for your model's prediction type
5. **Experimentation**: Fine-tune coherence sensitivity and detail enhancement for your content

## Troubleshooting

### Issue: No visible difference from default sampler
**Solution**: 
- Lower CFG scale
- Increase step count
- Try different scheduler types
- Increase detail enhancement strength

### Issue: Oversaturated/burnt images
**Solution**:
- Reduce CFG scale by 2-3 points
- Lower detail enhancement strength
- Try different scheduler

### Issue: Content-aware pacing not working
**Solution**:
- Ensure using AOS scheduler
- Increase step count to 30+
- Adjust coherence sensitivity
- Check that pacing is enabled

### Issue: Very slow sampling
**Solution**:
- Disable detail enhancement if not needed
- Use lower step counts initially
- Ensure torchvision is properly installed

## Example Workflows

See the `workflows/` directory for example `.json` workflow files that you can load directly into ComfyUI.

---

**Note**: Always save your working configurations! The extensive parameter space means you might find settings that work particularly well for your specific use cases.