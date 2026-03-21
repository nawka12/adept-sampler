# Adept Sampler

This repository integrates multiple research implementations—including **SA-Solver**, **UniPC**, and **SMEA**—into a unified, highly customizable sampler pipeline for Stable Diffusion WebUI reForge. It optimizes these methods for reForge, providing a tunable sampler that combines the stability of predictor-correctors with the texture retention of stochastic sampling.

## 📝 Disclaimer

While this sampler implements advanced techniques to improve generation, the field of generative images is complex and results can be sensitive to settings. This tool is not guaranteed to work perfectly in all scenarios, and some experimentation with the parameters may be required to achieve your desired outcome.

## ⚠️ Compatibility

This extension is developed and tested on **Stable Diffusion WebUI reForge**. Compatibility with other versions, such as the original WebUI or WebUI Forge, is not guaranteed.

> Also available for ComfyUI: https://github.com/nawka12/ComfyUI-Adept-Sampler

### Sampler/Solver/Scheduler compatibility

- Adept can run with the WebUI sampler's native solver or one of the Adept solvers.
- When an Adept Solver is enabled, it replaces the WebUI solver but still respects the chosen sigma schedule.
- When the Adept Solver is disabled, Adept preserves the sampler's native solver and can optionally override only the sigma schedule.
- Not all schedulers are equally stable with every sampler choice. Euler a remains a strong baseline for stability.

## 🌟 Features

### Solvers

| Solver | Description |
|--------|-------------|
| **Adept Solver** | A hybrid **Predictor-Corrector** pipeline. It utilizes a multi-step Adams-Bashforth integrator (derived from **DEIS**) for prediction and a **UniPC**-style corrector step. It integrates **DC-Solver**'s compensation logic to align predictor-corrector steps and uses **DPM-Solver++** dynamic thresholding for high-CFG stability. |
| **Adept Ancestral Solver** | Enhanced ancestral sampling with adaptive step sizing, phase-aware noise injection, enhanced derivative computation, and dynamic eta scheduling. |
| **AkashicSolver** | An optimized implementation of the **Stochastic Adams Solver (SA-Solver)** tailored for SDXL/EQ-VAE. It augments the standard SA-Solver with **SMEA** (Sinusoidal Multipass) interpolation for high-res coherence, phase-aware stochasticity control (Tau) to interpolate between ODE and SDE behaviors, and **Adaptive Noise Scale** which auto-calibrates s_noise based on model behavior. |
| **Mirror Correction Euler** | Euler Ancestral with a semantic reflection probe. In the first `Correction Phase` fraction of steps, uses a 3-call Heun correction: `x_probe = 2·D(x) − x` (reflection of x through its own denoised prediction). Unlike a naive `-x` probe, this probe lies on the denoising trajectory, giving a meaningful curvature estimate. Remaining steps are standard single-call Euler Ancestral. Optional **Smooth Phase Decay** mode replaces the binary step cutoff with a continuous log-sigma weight modulated by gradient agreement for smoother transitions. |

### Schedulers

Organized by category for easy selection:

| Category | Schedulers |
|----------|------------|
| **Universal** | `None`, `Entropic`, `SNR-Optimized`, `Constant-Rate`, `Adaptive-Optimized`, `Cosine-Annealed`, `LogSNR-Uniform`, `Tanh Mid-Boost`, `Exponential Tail`, `Jittered-Karras`, `Hybrid JYS-Karras`, `AYS-SDXL`, `Stochastic`, `JYS (Dynamic)` |
| **V-Prediction** | `AOS-V (for v-prediction)`, `SNR-Optimized` |
| **ε-Prediction / EQ-VAE** | `AOS-ε (for ε-prediction)`, `AkashicAOS`, `AkashicAOS Alt`, `AkashicEQFlow` |

#### Akashic Scheduler Family (ε / EQ-VAE)

**AkashicAOS v2**

A continuous power-function schedule designed for the smooth latent space of EQ-VAE:
- **Single continuous curve** — unlike stepwise schedules, uses a single continuous curve with no discrete phase boundaries
- **Detail-progressive** — shifts ~18% more steps into the low-sigma (detail) region
- **Mid-range enhancement** — applies a sinusoidal density boost at logSNR ≈ 0 to enhance mid-range structure formation
- **AkashicSolver compatible** — designed for stable multi-step integration

**AkashicAOS Alt**
- **Stronger detail bias** — emphasizes low-sigma refinement more aggressively than AkashicAOS
- **Shifted crossover shaping** — concentrates around a slightly detail-shifted transition region
- **Step-count adaptive behavior** — tuned to stay useful from low to high step counts

**AkashicEQFlow**
- **Crossover-focused density** — concentrates steps around the structure-to-detail transition in logSNR space
- **Robust hybrid mapping** — blends EQFlow crossover placement with a Karras prior for stability
- **Multi-step safety controls** — uses ratio caps and ratio slew-rate limiting for high-step robustness

### Enhancement Features

| Feature | Description |
|---------|-------------|
| **Detail Enhancement** | High-frequency detail boosting using frequency separation. Configurable strength and separation radius. |
| **Native Detail Boost** | A frequency-separation tool for the AkashicSolver. It applies a Gaussian blur to the input noise to isolate high-frequency components, then selectively boosts them during the sampling process to maximize texture emergence at native resolutions. |
| **SMEA** | High-resolution coherency feature (for >1024px) that prevents duplicated subjects and warped anatomy. |
| **Adaptive Noise Scale** | Available on AkashicSolver, Adept Ancestral, and Mirror Correction Euler. Runs a short calibration pass to measure sigma-relative prediction divergence, then restarts the generation with an auto-computed s_noise correction. Replaces the old EQ-VAE Mode — works with any model, not just EQ-VAE. |
| **Content-Aware Pacing** | An adaptive scheduling algorithm for Euler Ancestral. It monitors the variance of the latent derivative at every step; once variance drops below a stability threshold (indicating structure coherence), it automatically switches the sampler from the composition phase to the detail refinement phase. |
| **Spectral Modulation** | (AkashicSolver) Frequency-domain CFG correction based on Clybius's approach. Applies a spectral boost to noise predictions, with a configurable percentile threshold to control how aggressively high-frequency components are emphasized. |
| **Combat CFG Drift** | (AkashicSolver) Recenters the latent mean to counter cumulative drift caused by high CFG values. Configurable correction intensity. Auto-disabled for inpaint/ADetailer passes to avoid composite artifacts. |
| **VAE Reflection Padding** | Patches all Conv2d layers in the VAE to use reflect padding. Fixes edge artifacts when using VAEs trained with reflect padding (e.g., Anzhc's EQ-VAE). |

### Other Features

- **Full XYZ Grid Support** — All settings available for parameter sweeps
- **Complete Metadata** — All settings saved to image metadata and restored on load
- **Global Sampler Support** — Works with all k-diffusion samplers

## 🛠️ Installation

**Method 1: Using the `Install from URL` Feature**

1. Navigate to the **Extensions** tab in your WebUI.
2. Click on the **Install from URL** sub-tab.
3. Paste the following URL:
   ```
   https://github.com/nawka12/adept-sampler
   ```
4. Click **Install**.
5. Navigate to **Installed** tab and click **Apply and restart UI**.

**Method 2: Manual Installation**

```bash
git clone https://github.com/nawka12/adept-sampler extensions/adept-sampler
```

## 📖 Usage

1. Select your base sampling method (e.g., **Euler a**, **DPM++ 2M SDE**, etc.).
2. Navigate to "Scripts" and select **"Adept Sampler"**.
3. Enable the **"Enable Adept Sampler"** checkbox.

### UI Tabs

| Tab | Options |
|-----|---------|
| **Solver** | Solver Type (None / Adept Solver / Adept Ancestral Solver / AkashicSolver / Mirror Correction Euler), per-solver options |
| **Scheduler** | Category selection, Scheduler dropdown, Scheduler-specific options |
| **Detail Enhancement** | Enable toggle, Strength, Separation Radius |
| **Advanced** | Eta, Noise Scale, Disable for Hires. fix, Debug Reproducibility, VAE Reflection Padding |
| **Experimental** | CFG-to-zero after 40% |

## 🔧 Recommended Settings

### AkashicSolver with AkashicAOS (for EQ-VAE models)

```
Solver: AkashicSolver
Scheduler: AkashicAOS
τ (tau): 0.5-0.6
Order: 2
η (eta): 1.0
Noise Scale: 1.0
Adaptive Eta: On
Phase Strength: 0.5-0.6
SMEA: 0.0 (off for native res) / 0.1-0.2 (for >1.5x native)
Native Detail Boost: 0.3-0.5 (for native res detail)
Adaptive Noise Scale: On (auto-calibrates s_noise for your model)
```

### AkashicSolver with AkashicEQFlow (for EQ-VAE models)

```
Solver: AkashicSolver
Scheduler: AkashicEQFlow
τ (tau): 0.55-0.65
Order: 2
η (eta): 1.0
Noise Scale: 1.0
Adaptive Eta: On
Phase Strength: 0.5-0.6
SMEA: 0.0 (off for native res) / 0.1-0.2 (for >1.5x native)
Native Detail Boost: 0.2-0.4
Adaptive Noise Scale: On
```

**Tip**: Enable Additional CFG Fixes (Spectral Modulation, Combat CFG Drift) for EQ-VAE models if needed.

### Mirror Correction Euler

```
Solver: Mirror Correction Euler
Eta: 1.0
Noise Scale: 1.0
Correction Phase: 0.5  (first half of steps; increase for more correction, 0 = plain Euler Ancestral)
```

**Tip**: Correction Phase 0.3–0.5 is a good starting point. Higher values apply the Heun probe to more steps (more GPU cost) with diminishing returns.

### Adept Solver

| Goal | Settings |
|------|----------|
| **Quality-first** | Order 2–3, Corrector On |
| **Speed-first** | Order 1, Corrector Off |
| **High CFG (≥7)** | Order 2+, Corrector On (dynamic thresholding auto-activates) |

### Adept Ancestral Solver

| Preset | η | s_noise | Adaptive Eta | Phase Noise | Enhanced Derivative |
|--------|---|---------|--------------|-------------|---------------------|
| **Classic** | 1.0 | 1.0 | Off | Off | Off |
| **Balanced** | 1.0 | 1.0 | On | On | On |
| **High Diversity** | 1.1 | 1.05 | On | On | On |
| **Stable** | 0.9 | 0.95 | On | On | On |

## 🔍 Mirror Correction Euler Settings Reference

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| **Eta (η)** | 0.0–2.0 | 1.0 | Ancestral noise coefficient. 0 = deterministic ODE, 1 = full ancestral |
| **Noise Scale** | 0.0–2.0 | 1.0 | Multiplier on the ancestral noise added each step |
| **Correction Phase** | 0.0–1.0 | 0.5 | Fraction of steps that receive the 3-call semantic probe. 0.5 = first half (default), 0.0 = plain Euler Ancestral |
| **Smooth Phase Decay** | On/Off | Off | Replaces binary cutoff with a continuous log-sigma weight (sqrt curve) scaled by gradient stability. Best for EQ-VAE smooth latents — produces a more natural phase transition visible in correction_phase XYZ sweeps |

### Smooth Phase Decay details

When enabled, correction strength is computed as:

```
correction_weight = sqrt(t)       # t ∈ [0,1] in log-sigma space, 1 = sigma_max
effective_weight  = correction_weight × gradient_agreement
d = d + effective_weight × (d_heun − d)   # soft blend instead of hard replace
```

- `gradient_agreement` = `max(0, 1 − ‖d − d_probe‖ / mean_norm)` — high when curvatures agree (smooth manifold)
- Calls 2 and 3 are skipped automatically when `effective_weight < 0.001`, saving NFE at the tail of the phase
- `smooth_phase=False` (default) preserves exact original binary behavior with zero algorithmic change

## 🔍 AkashicSolver Settings Reference

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| **Tau (τ)** | 0.0–1.0 | 0.5 | Stochasticity control. 0=ODE (deterministic), 1=full SDE (stochastic) |
| **Order** | 1–3 | 2 | Multi-step integration order. Higher = more accurate but less stable |
| **Eta (η)** | 0.5–1.5 | 1.0 | Noise magnitude within the stochastic component |
| **Noise Scale** | 0.5–1.5 | 1.0 | Overall noise scaling factor |
| **Adaptive Eta** | On/Off | On | Phase-aware eta adjustment |
| **Phase Strength** | 0.0–1.0 | 0.5 | Intensity of phase-aware adaptations |
| **SMEA Strength** | 0.0–1.0 | 0.0 | High-res coherency (use for >1024px only) |
| **Native Detail Boost** | 0.0–1.0 | 0.0 | High-frequency noise boost for native res detail |
| **Adaptive Noise Scale** | On/Off | Off | Auto-calibrates s_noise via prediction divergence analysis. Replaces EQ-VAE Mode |
| **Spectral Modulation** | On/Off | Off | Frequency-domain CFG correction |
| **Spectral Percentile** | 1.0–15.0 | 5.0 | Frequency threshold (lower = gentler boost) |
| **Combat CFG Drift** | On/Off | Off | Recenters latent mean to remove cumulative CFG drift |
| **Drift Correction Intensity** | 0.1–1.0 | 0.5 | How much drift to remove (lower = subtler) |

## 📊 Samples

### Using AkashicPulse v4.0
![xyz_grid-0010-185404508](https://github.com/user-attachments/assets/b5a4d8d3-2b30-4a2e-861e-c02a95f16c93)

## 📄 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

### License Summary

- ✅ Commercial use, Modification, Distribution, Patent use, Private use

**Requirements:**
- 📋 License and copyright notice, State changes, Disclose source, Same license

**Limitations:**
- ❌ No Liability or Warranty

### Why GPL-3.0?

This license ensures compatibility with **Stable Diffusion WebUI reForge** and its ecosystem, while protecting the open-source nature of the project.

Full license text: https://www.gnu.org/licenses/gpl-3.0.html

---

Copyright (C) 2025 nawka12/KayfaHaarukku. This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it under certain conditions as specified in the GPL-3.0 license.
