# 🧬 Synthetic Data & Phantom Generation Pipeline (`data-gen`)

The `data-gen` directory contains standalone python simulation tools for generating synthetic multi-coil MRI k-space datasets, physical brain phantoms (PhantomNet), and simulated pathology slice sequences for testing and offline model calibration in MedMatrix.

---

## 🚀 Overview & Modules

* **`generate_demo_patients.py`**: Synthesizes multi-coil complex K-space data for synthetic patients, supporting various artifact corruptions (motion, noise, phase errors) across 11 pathology types.
* **`generate_phantomnet_demo.py`**: Generates numerical brain phantom slices (T1-weighted, T2-weighted, FLAIR contrast characteristics) with complex multi-coil sensitivity maps.
* **`generate_step3_images.py`**: Utility script to extract and visualize intermediate frequency and spatial domain feature representations.
* **`output_preview/`**: Contains generated preview images showcasing reconstructed magnitude slices, K-space frequency maps, and multi-class anomaly overlays.

---

## ⚙️ Usage

Activate the Python virtual environment (with PyTorch and NumPy installed) and run:

```bash
# Generate synthetic patient dataset
python data-gen/generate_demo_patients.py

# Generate PhantomNet numerical brain phantom previews
python data-gen/generate_phantomnet_demo.py

# Generate intermediate feature slice visualizations
python data-gen/generate_step3_images.py
```
