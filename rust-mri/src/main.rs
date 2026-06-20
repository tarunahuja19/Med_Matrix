use std::env;
use std::fs::File;
use std::io::{Read, Write};
use num_complex::Complex;
use rustfft::FftPlanner;

fn fftshift_2d<T: Clone>(data: &mut [T], height: usize, width: usize) {
    let half_h = height / 2;
    let half_w = width / 2;
    for r in 0..half_h {
        for c in 0..half_w {
            let i0 = r * width + c;
            let i3 = (r + half_h) * width + (c + half_w);
            data.swap(i0, i3);

            let i1 = r * width + (c + half_w);
            let i2 = (r + half_h) * width + c;
            data.swap(i1, i2);
        }
    }
}

fn fft2d(data: &mut [Complex<f64>], height: usize, width: usize, inverse: bool) {
    let mut planner = FftPlanner::new();
    
    // 1. Row FFTs
    let fft_row = if inverse {
        planner.plan_fft_inverse(width)
    } else {
        planner.plan_fft_forward(width)
    };
    let mut scratch = vec![Complex::new(0.0, 0.0); fft_row.get_inplace_scratch_len()];
    for r in 0..height {
        let row_start = r * width;
        let row_end = row_start + width;
        fft_row.process_with_scratch(&mut data[row_start..row_end], &mut scratch);
    }
    
    // 2. Col FFTs (using transpose)
    let mut transposed = vec![Complex::new(0.0, 0.0); height * width];
    for r in 0..height {
        for c in 0..width {
            transposed[c * height + r] = data[r * width + c];
        }
    }
    
    let fft_col = if inverse {
        planner.plan_fft_inverse(height)
    } else {
        planner.plan_fft_forward(height)
    };
    let mut scratch_col = vec![Complex::new(0.0, 0.0); fft_col.get_inplace_scratch_len()];
    for c in 0..width {
        let col_start = c * height;
        let col_end = col_start + height;
        fft_col.process_with_scratch(&mut transposed[col_start..col_end], &mut scratch_col);
    }
    
    for c in 0..width {
        for r in 0..height {
            data[r * width + c] = transposed[c * height + r];
        }
    }
    
    if inverse {
        let scale = (height * width) as f64;
        for val in data.iter_mut() {
            *val /= scale;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 8 {
        eprintln!("Usage: {} <input_kspace_bin> <output_magnitude_bin> <slices> <coils> <height> <width> <phase_correction>", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let slices: usize = args[3].parse()?;
    let coils: usize = args[4].parse()?;
    let height: usize = args[5].parse()?;
    let width: usize = args[6].parse()?;
    let phase_correction: bool = args[7].parse().unwrap_or(true);

    // 1. Read complex raw k-space (slices * coils * height * width * 2 f64 values)
    let mut file = File::open(input_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let num_elements = buffer.len() / 8;
    let expected_elements = slices * coils * height * width * 2;
    if num_elements != expected_elements {
        return Err(format!("Input file size mismatch. Expected {} elements ({} bytes), got {} elements ({} bytes)",
            expected_elements, expected_elements * 8, num_elements, buffer.len()).into());
    }

    // Parse bytes into Complex<f64> slice-by-slice, coil-by-coil
    let slice_size = height * width;
    let coil_data_len = coils * slice_size;
    let mut magnitude_img = vec![0.0; slices * slice_size];

    for s in 0..slices {
        let mut coil_images = vec![vec![Complex::new(0.0, 0.0); slice_size]; coils];
        
        for c in 0..coils {
            let offset_base = (s * coil_data_len + c * slice_size) * 2;
            for i in 0..slice_size {
                let idx = offset_base + i * 2;
                let real_bytes = &buffer[idx * 8..(idx * 8 + 8)];
                let imag_bytes = &buffer[(idx + 1) * 8..((idx + 1) * 8 + 8)];
                
                let re = f64::from_le_bytes(real_bytes.try_into()?);
                let im = f64::from_le_bytes(imag_bytes.try_into()?);
                coil_images[c][i] = Complex::new(re, im);
            }

            if phase_correction {
                // Zero-order phase correction (subtract phase at the peak of 2D k-space)
                let mut peak_val = 0.0;
                let mut peak_phase = 0.0;
                for val in coil_images[c].iter() {
                    let mag = val.norm();
                    if mag > peak_val {
                        peak_val = mag;
                        peak_phase = val.arg();
                    }
                }
                let correction = Complex::from_polar(1.0, -peak_phase);
                for val in coil_images[c].iter_mut() {
                    *val *= correction;
                }
            }

            // Save the raw k-space for this coil for low-res phase alignment
            let kspace_coil = coil_images[c].clone();

            // Centering & 2D IFFT
            fftshift_2d(&mut coil_images[c], height, width);
            fft2d(&mut coil_images[c], height, width, true);
            fftshift_2d(&mut coil_images[c], height, width);

            if phase_correction {
                // Low-res coil phase alignment (using 24x24 center calibration)
                let cal_h = std::cmp::min(height, 24);
                let cal_w = std::cmp::min(width, 24);
                let h_start = (height - cal_h) / 2;
                let w_start = (width - cal_w) / 2;

                let mut cal_kspace = vec![Complex::new(0.0, 0.0); slice_size];
                for r in h_start..(h_start + cal_h) {
                    for col in w_start..(w_start + cal_w) {
                        let idx = r * width + col;
                        cal_kspace[idx] = kspace_coil[idx];
                    }
                }

                fftshift_2d(&mut cal_kspace, height, width);
                fft2d(&mut cal_kspace, height, width, true);
                fftshift_2d(&mut cal_kspace, height, width);

                for i in 0..slice_size {
                    let phase = cal_kspace[i].arg();
                    coil_images[c][i] *= Complex::from_polar(1.0, -phase);
                }
            }
        }

        // RSS Combination for this slice
        let slice_out_offset = s * slice_size;
        for i in 0..slice_size {
            let mut sum_sq = 0.0;
            for c in 0..coils {
                sum_sq += coil_images[c][i].norm_sqr();
            }
            magnitude_img[slice_out_offset + i] = sum_sq.sqrt();
        }
    }

    // 6. Save reconstructed magnitude to binary file
    let mut magnitude_bytes = Vec::with_capacity(magnitude_img.len() * 8);
    for val in magnitude_img.iter() {
        magnitude_bytes.extend_from_slice(&val.to_le_bytes());
    }
    let mut magnitude_file = File::create(output_path)?;
    magnitude_file.write_all(&magnitude_bytes)?;

    println!("Success! Rust reconstruction complete for {} slices, {} coils, {}x{} resolution.", slices, coils, width, height);
    Ok(())
}
