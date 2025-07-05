# Adept Sampler

This repository contains an advanced, highly customizable sampler for Stable Diffusion WebUI and reForge. It integrates state-of-the-art techniques to provide enhanced detail, flexible scheduling, and improved image composition.

## ⚠️ Compatibility

This extension is developed and tested on **Stable Diffusion WebUI reForge**. Compatibility with other versions, such as the original WebUI or WebUI Forge, is not guaranteed.

## 🌟 Features

- **Advanced Ancestral Sampler**: A custom implementation that patches the default Euler Ancestral sampler to provide advanced features.
- **Detail Enhancement**: A unique method to enhance high-frequency details, which can be used with any scheduler.
- **Custom Schedulers**:
    - **Anime-Optimized Schedule (AOS)**: A three-phase scheduler designed to improve composition and detail for anime-style images.
    - **Entropic Schedule**: A power-based schedule that clusters steps for fine-tuning detail generation.
- **Content-Aware Pacing (AOS Only)**: Dynamically adjusts the sampling process, switching from composition to detail focus based on image coherence.
- **Full UI Integration**: All features are configurable through a custom accordion panel in the WebUI or reForge interface.

## 🛠️ Installation

1.  Clone this repository into your `extensions` directory in your Stable Diffusion WebUI installation.
    ```bash
    git clone https://github.com/nawka12/adept-sampler extensions/adept-sampler
    ```
2.  Restart your Stable Diffusion WebUI.

## 📖 Usage

1.  Navigate to the "Scripts" section at the bottom of the `txt2img` or `img2img` tabs.
2.  Select **"Adept Sampler"** from the script dropdown.
3.  Enable the **"Enable Adept Sampler"** checkbox to activate the custom features.
4.  Configure the settings as desired:
    - **Scheduler**: Choose between the default, Anime-Optimized (AOS), or Entropic schedulers.
    - **Detail Enhancement**: Toggle and adjust the strength of high-frequency detail enhancement.
    - **Advanced Noise Settings**: Fine-tune `Eta` and `Noise Scale` for different effects. 