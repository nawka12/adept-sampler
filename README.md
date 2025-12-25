# Adept Sampler

This repository contains an advanced, highly customizable sampler for Stable Diffusion WebUI reForge. It integrates state-of-the-art techniques to provide enhanced detail, flexible scheduling, and improved image composition.

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
| **Adept Solver** | Training-free deterministic solver synthesizing DPM-Solver++, UniPC, DEIS, and DC-Solver. Features multistep predictor, optional corrector, dynamic thresholding, and phase-aware adaptive compensation. |
| **Adept Ancestral Solver** | Enhanced ancestral sampling with adaptive step sizing, phase-aware noise injection, enhanced derivative computation, and dynamic eta scheduling. |
| **AkashicSolver [EXPERIMENTAL]** | Advanced solver optimized for EQ-VAE SDXL models (e.g., AkashicPulse). Combines SA-Solver multi-step integration, phase-aware tau control, and Native Detail Boost. |

### Schedulers

Organized by category for easy selection:

| Category | Schedulers |
|----------|------------|
| **Universal** | `None`, `Entropic`, `SNR-Optimized`, `Constant-Rate`, `Adaptive-Optimized`, `Cosine-Annealed`, `LogSNR-Uniform`, `Tanh Mid-Boost`, `Exponential Tail`, `Jittered-Karras`, `Hybrid JYS-Karras`, `AYS-SDXL`, `Stochastic`, `JYS (Dynamic)` |
| **V-Prediction** | `AOS-V (for v-prediction)` |
| **ε-Prediction** | `AOS-ε (for ε-prediction)` |
| **Experimental** | `AkashicAOS [EXPERIMENTAL]` |

#### AkashicAOS v2 [EXPERIMENTAL]

Detail-progressive schedule designed specifically for EQ-VAE SDXL models:
- **Single continuous curve** — no discrete phase boundaries, smooth step ratios
- **Detail-progressive** — ~18% more steps in detail region (exploits EQ-VAE's fine detail capability)  
- **Mid-range enhancement** — subtle boost around logSNR ≈ 0 for structure formation
- **AkashicSolver compatible** — designed for stable multi-step integration

### Enhancement Features

| Feature | Description |
|---------|-------------|
| **Detail Enhancement** | High-frequency detail boosting using frequency separation. Configurable strength and separation radius. |
| **Native Detail Boost** | AkashicSolver-exclusive feature that boosts high-frequency noise components for enhanced detail emergence at native resolution (without the blur that SMEA causes). |
| **SMEA** | High-resolution coherency feature (for >1024px) that prevents duplicated subjects and warped anatomy. |
| **Content-Aware Pacing** | Dynamically switches from composition to detail focus based on image coherence (AOS only). |

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
| **Solver** | Solver Type, Order, Corrector, Ancestral settings, AkashicSolver settings |
| **Scheduler** | Category selection, Scheduler dropdown, Scheduler-specific options |
| **Detail Enhancement** | Enable toggle, Strength, Separation Radius |
| **Advanced** | Eta, Noise Scale, Disable for HR |
| **Experimental** | CFG-to-zero after 40% |

## 🔧 Recommended Settings

### AkashicSolver with AkashicAOS (for EQ-VAE models)

```
Solver: AkashicSolver [EXPERIMENTAL]
Scheduler: AkashicAOS [EXPERIMENTAL]
τ (tau): 0.5
Order: 2
η (eta): 1.0
Noise Scale: 1.0
Adaptive Eta: On
Phase Strength: 0.5
SMEA: 0.0 (off for native res) / 0.1-0.2 (for >1.5x native)
Native Detail Boost: 0.3-0.5 (for native res detail)
```

**Important**: Use external rescaleCFG at 0.7 for EQ-VAE models.

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
