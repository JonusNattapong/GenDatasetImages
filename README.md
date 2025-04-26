# GenDatasetImages: Automatic Text-to-Image Dataset Generation Tool 🖼️

[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)](LICENSE)

**GenDatasetImages** is a Python tool designed to streamline the creation of image datasets using the **Automatic1111 Stable Diffusion Web UI API**. Define your desired image variations (prompts, styles, parameters) in a simple YAML configuration file, and let the tool automatically generate the images, organize them, and compile detailed metadata in a machine-learning-friendly format.

---

## ✨ Features

* **Configuration-Driven:** Define complex generation tasks using a clear YAML format.
* **Parameter Variation:** Automatically generate images for all combinations of specified parameters (prompts, seeds, CFG scales, steps, etc.).
* **A1111 API Integration:** Connects seamlessly to a running Automatic1111 instance (requires `--api` flag).
* **Structured Output:** Creates datasets with a clean folder structure (`dataset_name/images/`, `dataset_name/metadata.jsonl`).
* **JSON Lines Metadata:** Saves detailed metadata for each image incrementally to a `.jsonl` file, ideal for large datasets and ML pipelines.
* **Descriptive Filenames:** Generates informative image filenames based on key parameters (seed, CFG, prompt prefix) and image hash.
* **Standard Copyright:** Includes a standard copyright notice (All Rights Reserved).

---

## 📁 Project Structure

```
GenDatasetImages/
├── src/                     # Main Python source code
│   ├── __init__.py
│   ├── config_loader.py     # Loads and validates config files
│   ├── a1111_client.py      # Interacts with the A1111 API
│   ├── generator.py         # Orchestrates the generation process
│   ├── dataset_builder.py   # Saves images and metadata
│   └── main.py              # Main script entry point
├── configs/                 # Directory for configuration files
│   └── default_config.yaml  # Example configuration file
├── output_datasets/         # Default output directory for generated datasets
│   └── example_dataset/     # Example output structure
│       ├── images/
│       │   └── img_s11111_cfg7_a_high_quality_a1b2c3d4.png
│       │   └── ...
│       └── metadata.jsonl   # JSON Lines file with metadata for all images
├── README.md                # This file
├── LICENSE                  # Copyright and license information
└── requirements.txt         # Python dependencies
```

---

## ⚙️ Prerequisites

1. **Python:** Version 3.8 or higher recommended.
2. **Automatic1111 Web UI:** A running instance of the Stable Diffusion Web UI by AUTOMATIC1111.
    * **API Enabled:** You **MUST** launch A1111 with the `--api` command-line flag (e.g., modify `webui-user.bat` or `webui.sh`).
    * **Network Access:** The script needs network access to the A1111 API URL (default: `http://127.0.0.1:7860`).

---

## 🚀 Installation & Setup

1. **Clone or Download:**

    ```bash
    git clone <repository_url> # Or download and extract the ZIP
    cd GenDatasetImages
    ```

2. **Create Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    # Activate:
    # Windows (cmd): venv\Scripts\activate.bat
    # Windows (PS):  .\venv\Scripts\Activate.ps1
    # macOS/Linux:   source venv/bin/activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure:**
    * Copy `configs/default_config.yaml` to a new file (e.g., `configs/my_project.yaml`).
    * Edit your new config file:
        * Verify `api_url`.
        * Set `dataset_name`.
        * Define `base_parameters` (defaults for A1111).
        * Specify the desired `parameter_variations` (the core of your generation task).

---

## ▶️ Usage

This tool can be used in two ways:

**1. Command-Line Interface (Original)**

* **Configure:** Edit a YAML file in the `configs/` directory (e.g., `my_project.yaml`).
* **Run:**
    1. Ensure Automatic1111 is running with `--api`.
    2. Activate your virtual environment.
    3. Execute:

        ```bash
        python src/main.py configs/your_config_file.yaml
        ```

* **Output:** Monitor the terminal. Results appear in `output_datasets/your_dataset_name/`.

**2. Web Application Interface**

* **Run:**
    1. Ensure Automatic1111 is running with `--api`.
    2. Activate your virtual environment.
    3. Execute:

        ```bash
        streamlit run app.py
        ```

* **Access:** Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.
* **Configure:** Use the sidebar in the web interface to set API URL, dataset name, base parameters, and basic variations (prompt, seed).
* **Generate:** Click the "Start Generation" button.
* **Output:** Monitor progress (status, logs, image previews) in the main area of the web app. The UI should remain responsive during generation. Results are saved to `output_datasets/your_dataset_name/`.

---

## 📄 Metadata (`metadata.jsonl`)

The output `metadata.jsonl` uses the **JSON Lines** format:

* Each line is a complete JSON object for one image.
* Efficient for processing large datasets line by line.

Each JSON object includes:

* `filename`: The image filename (e.g., `img_s123_cfg7_cat_on_a_f1a2b3c4.png`).
* `filepath`: Relative path within the dataset (e.g., `images/img_s123...png`).
* `request_parameters`: The exact parameters sent to the API for this image.
* All other fields returned by the A1111 API's `info` object (actual seed, model hash, etc.).

---

## 📜 License

Copyright (c) 2025 Your Name or Organization. **All Rights Reserved.**

This software is proprietary and confidential. Unauthorized copying, modification, distribution, or use is strictly prohibited. See the [LICENSE](LICENSE) file for details.

---

## 💡 TODO / Potential Improvements

* More robust error handling and optional retries for API calls.
* Support for other A1111 API endpoints (e.g., `img2img`, `interrogate`).
* Add a progress bar visualization (e.g., using `tqdm`).
* Implement optional handling for custom parameter combinations (like `prompt_suffix`).
* Explore alternative metadata formats (e.g., CSV output option).
* Consider integration with ComfyUI API as an alternative backend.
* **Web UI Improvements:**
  * Improve UI/UX for defining parameter variations (beyond simple text areas).
  * Add dataset download functionality (e.g., as a zip file).
  * Implement a "Stop" button to cancel ongoing generation.
  * Add more robust error handling and display within the UI.
