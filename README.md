# Adept Sampler

This repository contains an advanced, highly customizable sampler for Stable Diffusion WebUI reForge. It integrates state-of-the-art techniques to provide enhanced detail, flexible scheduling, and improved image composition.

## ⚠️ Compatibility

This extension is developed and tested on **Stable Diffusion WebUI reForge**. Compatibility with other versions, such as the original WebUI or WebUI Forge, is not guaranteed.

## 🌟 Features

- **Advanced Ancestral Sampler**: A custom implementation that patches the default Euler Ancestral sampler to provide advanced features.
- **Detail Enhancement**: A unique method to enhance high-frequency details, which can be used with any scheduler. The **Detail Separation Radius** controls what is considered a 'detail,' with higher values sharpening larger features.
- **Custom Schedulers**:
    - **Anime-Optimized Schedule (AOS)**: A three-phase scheduler designed to improve composition and detail for anime-style images.
      > ⚠️ **Compatibility Warning**: AOS is heavily optimized for **v-prediction** models. Using it with **epsilon-prediction** models may break the generation.
    - **Entropic Schedule**: A power-based schedule that clusters steps for fine-tuning detail generation. The **Entropic Power** setting controls this clustering, with values greater than 1.0 focusing more steps at the beginning of the sampling process.
- **Content-Aware Pacing (AOS Only)**: Dynamically adjusts the sampling process, switching from composition to detail focus based on image coherence. The **Coherence Sensitivity** slider controls when this switch occurs.
- **Full UI Integration**: All features are configurable through a custom accordion panel in the WebUI or reForge interface.

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
4.  Configure the settings as desired:
    - **Scheduler**: Choose between the default, Anime-Optimized (AOS), or Entropic schedulers.
      > ℹ️ **Note**: When using a custom scheduler, you may need to **lower your CFG Scale** (e.g., by 1-2 points) to prevent oversaturated or 'burnt' images.
    - **Detail Enhancement**: Toggle and adjust the strength of high-frequency detail enhancement.
    - **Advanced Noise Settings**: Fine-tune `Eta` and `Noise Scale` for different effects. 