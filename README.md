# Adept Sampler

This repository contains an advanced, highly customizable sampler for Stable Diffusion WebUI reForge. It integrates state-of-the-art techniques to provide enhanced detail, flexible scheduling, and improved image composition.

## 📝 Disclaimer

While this sampler implements advanced techniques to improve generation, the field of generative images is complex and results can be sensitive to settings. This tool is not guaranteed to work perfectly in all scenarios, and some experimentation with the parameters may be required to achieve your desired outcome.

## ⚠️ Compatibility

This extension is developed and tested on **Stable Diffusion WebUI reForge**. Compatibility with other versions, such as the original WebUI or WebUI Forge, is not guaranteed.

## 🌟 Features

- **Advanced Ancestral Sampler**: A custom implementation that patches the default Euler Ancestral sampler to provide advanced features.
- **Detail Enhancement**: A unique method to enhance high-frequency details, which can be used with any scheduler. The **Detail Separation Radius** controls what is considered a 'detail,' with higher values sharpening larger features.
- **Custom Schedulers**: A suite of schedulers to control the denoising process.
    > **Scheduler Categories:**
    > - **Universal**: Recommended for all model types. Includes `Entropic`, `Constant-Rate`, and `Adaptive-Optimized`.
    > - **V-Prediction**: Specialized for `v-prediction` models. Includes `AOS-V` and `SNR-Optimized`.
    > - **ε-Prediction**: Specialized for `epsilon-prediction` models. Includes `AOS-ε`.

    - **Anime-Optimized Schedule for V-Prediction (AOS-V)**: A three-phase scheduler designed to improve composition and detail for anime-style images on **v-prediction** models.
    - **Anime-Optimized Schedule for Epsilon-Prediction (AOS-ε)**: A three-phase scheduler optimized for **epsilon-prediction** models, with adjusted phase boundaries and power curves.
      > ⚠️ **Compatibility Warning**: Use the correct AOS variant for your model type. **AOS-V** is for **v-prediction** models, while **AOS-ε** is for **epsilon-prediction** models. Mismatching them may break the generation.
    - **Entropic Schedule**: A power-based schedule that clusters steps for fine-tuning detail generation. The **Entropic Power** setting controls this clustering, with values greater than 1.0 focusing more steps at the beginning of the sampling process.
    - **SNR-Optimized**: Concentrates steps around the critical `logSNR = 0` point, where the model transitions from noise-dominant to signal-dominant.
        - *Based on: [Hang, T., et al. (2024). Improved Noise Schedule for Diffusion Training.](https://arxiv.org/abs/2407.03297)*
    - **Constant-Rate**: Aims to ensure a constant rate of change in the data's probability distribution, preventing large, unstable jumps in the sampling process.
        - *Based on: [Okada, S., et al. (2024). Constant Rate Scheduling.](https://arxiv.org/abs/2411.12188)*
    - **Adaptive-Optimized**: A hybrid approach inspired by data-driven methods, blending multiple strategies for a robust, general-purpose curve.
        - *Inspired by: [Sabour, A., et al. (2024). Align Your Steps: Optimizing Sampling Schedules in Diffusion Models.](https://arxiv.org/abs/2404.14507)*
- **Content-Aware Pacing (AOS Only)**: Dynamically adjusts the sampling process, switching from composition to detail focus based on image coherence. The **Coherence Sensitivity** slider controls when this switch occurs. Works with both AOS-V and AOS-ε variants.
- **Full UI Integration**: All features are configurable through a custom accordion panel in the WebUI or reForge interface.
- **Experimental Features**: Optional settings in a dedicated tab, including experimental support for Content-Aware Pacing with the SNR-Optimized scheduler. Includes custom sensitivity and manual override controls.

## 🛠️ Installation

There are two ways to install the extension:

**Method 1: Using the `Install from URL` Feature**

1.  Navigate to the **Extensions** tab in your WebUI.
2.  Click on the **Install from URL** sub-tab.
3.  Paste the following URL into the **URL for extension's git repository** field:
    ```
    https://github.com/nawka12/adept-sampler
    ```
4.  Click **Install**.
5.  Once installation is complete, navigate to the **Installed** tab and click **Apply and restart UI**.

**Method 2: Manual Installation (git clone)**

1.  Clone this repository into your `extensions` directory in your Stable Diffusion WebUI reForge installation.
    ```bash
    git clone https://github.com/nawka12/adept-sampler extensions/adept-sampler
    ```
2.  Restart your Stable Diffusion WebUI reForge.

## 📖 Usage

1.  Navigate to the "Scripts" section at the bottom of the `txt2img` or `img2img` tabs.
2.  Select **"Adept Sampler"** from the script dropdown.
3.  Enable the **"Enable Adept Sampler"** checkbox to activate the custom features.
4.  The settings are organized into tabs for easy configuration:
    - **Scheduler**:
        - Choose a scheduler from the dropdown. See the "Features" section for a description of each.
        - **Entropic Power**: If using the `Entropic` scheduler, this slider controls timestep clustering. Higher values focus on detail earlier.
        - **Content-Aware Pacing (AOS Only)**: For `AOS` schedulers, this enables dynamic adjustment from composition to detail focus. You can control its sensitivity.
    - **Detail Enhancement**:
        - Toggle the detail enhancer and adjust its `Strength`.
        - Use the `Detail Separation Radius` to define what counts as a "detail." Higher values sharpen larger features.
    - **Advanced**:
        - Fine-tune `Eta` and `Noise Scale` for different ancestral noise effects.
        - Option to automatically disable the sampler for the Hires. fix pass.
      > ℹ️ **Note**: When using a custom scheduler, you may often need to **lower your CFG Scale** (e.g., by 1-2 points) to prevent oversaturated or 'burnt' images.
    - **Advanced Noise Settings**: Fine-tune `Eta` and `Noise Scale` for different effects. 
    - **Experimental**:
        - Toggle experimental features like enabling Content-Aware Pacing for the SNR-Optimized scheduler.
        - Adjust **Experimental Coherence Sensitivity** to control phase switching timing.
        - Use **Experimental Manual Pacing Override** for manual phase control via JSON (e.g., {"composition": 0.4}).
      > ⚠️ **Warning**: These features are experimental and may cause instability or unexpected results. Use with caution and test thoroughly.

## 🔍 Sampling Method Comparison

| **Method** | **Key Characteristics** | **Best For** | **Key Settings** | **Sample Image** | **Notes** |
|------------|------------------------|--------------|------------------|------------------|-----------|
| **Euler Ancestral** | Standard sampling with noise injection | General purpose, baseline comparison | CFG Scale: Standard values | ![image](https://github.com/user-attachments/assets/10f7087e-0b79-4b34-bd7b-73cd1263e24b) | Default sampler, good baseline performance |
| **Adept + AOS + Content-Aware Pacing** | Three-phase scheduler with dynamic composition-to-detail switching | Anime/illustration style, complex compositions | AOS-V/AOS-ε (match model type)<br/>Coherence Sensitivity<br/>CFG Scale: -1 to -2 from normal | ![image](https://github.com/user-attachments/assets/f0a036e1-1fa6-4941-a08e-1e364979f05e) | Automatically adapts focus from composition to details based on image coherence |
| **Adept + Entropic** | Power-based clustering with concentrated early steps | Fine detail work, texture enhancement | Entropic Power: >1.0 for early focus<br/>Detail Enhancement<br/>CFG Scale: -1 to -2 from normal | ![image](https://github.com/user-attachments/assets/0a62c159-43c9-4dd5-9579-78a231d1e5d6) | Clusters more steps at beginning for better detail control |
| **Adept + SNR-Optimized** | Concentrates steps around the critical `logSNR = 0` point for balanced sampling | Balanced compositions, preventing over/under-exposure | CFG Scale: -1 to -2 from normal | Image not available | Based on recent research to improve stability. Now supports experimental Content-Aware Pacing for dynamic phase switching. |
| **Adept + Constant-Rate** | Ensures a constant rate of change, preventing unstable jumps in sampling | Smooth, stable, and predictable generations | CFG Scale: -1 to -2 from normal | Image not available | Ideal for preventing artifacts from sudden changes in sampling speed |
| **Adept + Adaptive-Optimized** | Hybrid approach blending multiple strategies for a robust, general-purpose curve | General-purpose use across a wide variety of models | CFG Scale: -1 to -2 from normal | Image not available | A "best-of-all-worlds" approach inspired by data-driven methods |

## 📄 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

### License Summary

- ✅ **Commercial use** - You may use this software commercially
- ✅ **Modification** - You may modify the software  
- ✅ **Distribution** - You may distribute the software
- ✅ **Patent use** - You may use any patents that contributors grant you
- ✅ **Private use** - You may use the software privately

**Requirements:**
- 📋 **License and copyright notice** - Include the license and copyright notice with the software
- 📋 **State changes** - Indicate significant changes made to the software
- 📋 **Disclose source** - You must make the source code available when you distribute the software
- 📋 **Same license** - You must license derivative works under the same license

**Limitations:**
- ❌ **Liability** - The software is provided without warranty
- ❌ **Warranty** - No warranties are provided with the software

### Why GPL-3.0?

This license was chosen to ensure compatibility with **Stable Diffusion WebUI reForge** and its ecosystem, while protecting the open-source nature of the project. It ensures that improvements and modifications remain available to the community.

### Full License Text

The complete license text can be found at: https://www.gnu.org/licenses/gpl-3.0.html

---

Copyright (C) 2025 nawka12/KayfaHaarukku. This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it under certain conditions as specified in the GPL-3.0 license.
