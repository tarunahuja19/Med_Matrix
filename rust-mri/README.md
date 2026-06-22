# 🦀 Rust MRI Processing Prototype (`rust-mri`)

This directory houses the **MedMatrix high-performance Rust processing prototype**. It implements clinical-grade, low-level mathematical operations in native code, offering sub-millisecond execution times for 2D Inverse Fast Fourier Transforms (IFFT), phase correction calibrations, Root-Sum-of-Squares (RSS) coil combinations, and compilation of clinical PDF reports.

---

## 🛠️ Stack & Dependencies

* **Language:** Rust (2024 Edition)
* **Libraries:**
  * `rustfft`: High-performance, cache-friendly Fast Fourier Transforms in native Rust.
  * `num-complex`: Complex number data structures and polar coordinate transformations.
  * `printpdf`: Pure Rust library for structured PDF document generation.
  * `image`: Image decoding and format handling.
  * `serde` & `serde_json`: JSON schema serialization and deserialization.

---

## 🗺️ File Map & Details

### 1. Main Reconstruction CLI (`src/main.rs`)
* **Purpose:** Processes raw multi-coil, multi-slice complex frequency (K-space) signals into clinical magnitude images.
* **Process Steps:**
  1. **Binary Parsing:** Reads raw double-precision float values (`f64`) representing stacked complex points ($2 \times \text{slices} \times \text{coils} \times \text{height} \times \text{width}$).
  2. **Zero-Order Phase Correction:** Identifies the frequency peak phase angle within each coil channel and rotates all coordinates to align their real components.
  3. **2D Centered IFFT:** Applies row-wise and column-wise Inverse Fourier Transforms, utilizing `fftshift` operations to center frequencies in the spatial domain.
  4. **Low-Resolution Calibration (First-Order):** Extracts a central $24 \times 24$ calibration window from the frequency grid, reconstructs it, and applies the local phase alignment back to the high-resolution image to resolve phase wrap.
  5. **Root-Sum-of-Squares (RSS) Fusion:** Combines multi-coil slices into single intensity maps:
     $$I(x, y) = \sqrt{\sum_{c=1}^{C} |S_c(x, y)|^2}$$
  6. **Export:** Dumps the resulting flat float matrix back to disk.

### 2. PDF Reporting Compiler (`src/bin/report_pdf.rs`)
* **Purpose:** Compiles diagnostic PDF reports based on patient metadata and RAG-generated text.
* **Features:**
  * Strict clinical page layouts with tables, headers, and signature fields.
  * Embeds patient summaries, symptom logs, and diagnostic impressions dynamically.

---

## ⚙️ Compilation & CLI Usage

### Prerequisites
Make sure you have Rust installed via `rustup` or your package manager:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 1. Build the Binaries
Compile the reconstruction tool and PDF compiler in Release mode:
```bash
cargo build --release
```
Compiles target executable files into `target/release/rust-mri` and `target/release/report_pdf`.

### 2. Run Reconstruction
To execute reconstruction via command line:
```bash
cargo run --release -- <input_kspace_bin> <output_magnitude_bin> <slices> <coils> <height> <width> <phase_correction>
```
* **Parameters:**
  * `input_kspace_bin`: Path to raw f64 complex K-space data.
  * `output_magnitude_bin`: Path to save double-precision magnitude values.
  * `slices` / `coils` / `height` / `width`: Shape dimensions (e.g. `8`, `16`, `128`, `128`).
  * `phase_correction`: `true` or `false` (to toggle Zero/First-order calibrations).
