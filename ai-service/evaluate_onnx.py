import os
import sys
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add current directory to path to import train_anomaly_detector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_anomaly_detector import KSpaceAnomalyDataset, get_status, print_patient_report

def evaluate_onnx(onnx_path: str, data_dir: str, coils: int, batch_size: int):
    import onnxruntime
    
    print("=" * 60)
    print("EVALUATING ONNX ANOMALY DETECTOR MODEL")
    print("=" * 60)
    print(f"ONNX Model Path: {onnx_path}")
    print(f"Data Directory:  {data_dir}")
    print(f"Coils:           {coils}")
    
    # 1. Load manifest and setup samples
    manifest_csv = os.path.join(data_dir, "dataset_manifest.csv")
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest not found at {manifest_csv}. Please generate data first.")
        
    samples = []
    with open(manifest_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'patient_index': int(row['patient_index']),
                'filename': row['filename']
            })
            
    num_patients = len(samples)
    print(f"Loaded: {num_patients} patients ({num_patients * 8} slices)")
    
    # Use all generated samples for evaluation
    val_dataset = KSpaceAnomalyDataset(samples, data_dir, coils=coils, is_train=False, preload=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 2. Load ONNX Model Session
    print("\nInitializing ONNX Runtime inference session...")
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # 3. Run Inference and collect outputs
    val_targets_all = []
    val_preds_all = []
    
    print("Running inference on dataset...")
    for batch_idx, (x, contrast, targets) in enumerate(val_loader):
        B_patient, S, C2, H, W = x.shape
        # Flatten patients and slices dimension to fit model input
        x = x.view(B_patient * S, C2, H, W)
        contrast = contrast.view(B_patient * S)
        targets = targets.view(B_patient * S, 3)
        
        # Convert to numpy for ONNX runtime
        x_np = x.numpy()
        contrast_np = contrast.numpy()
        
        ort_inputs = {
            'kspace': x_np,
            'contrast': contrast_np
        }
        
        # Run ONNX inference
        outputs = ort_session.run(None, ort_inputs)[0]
        
        val_targets_all.append(targets.numpy())
        val_preds_all.append(outputs)
        
        print(f"  Processed batch {batch_idx + 1}/{len(val_loader)}...")
        
    val_targets_all = np.concatenate(val_targets_all, axis=0)
    val_preds_all = np.concatenate(val_preds_all, axis=0)
    
    print("\nInference completed successfully.")
    
    # 4. Compute Regression Metrics
    diff = val_preds_all - val_targets_all
    mses = np.mean(diff ** 2, axis=0)
    maes = np.mean(np.abs(diff), axis=0)
    overall_mse = np.mean(diff ** 2)
    overall_mae = np.mean(np.abs(diff))
    
    # 5. Compute Classification Metrics (Accuracies)
    val_targets_class = np.zeros_like(val_targets_all, dtype=np.int32)
    val_preds_class = np.zeros_like(val_preds_all, dtype=np.int32)
    
    # Severity thresholds: Severe (>0.5) is 2, Mild (>0.15) is 1, None <= 0.15 is 0
    val_targets_class[val_targets_all > 0.15] = 1
    val_targets_class[val_targets_all > 0.5] = 2
    
    val_preds_class[val_preds_all > 0.15] = 1
    val_preds_class[val_preds_all > 0.5] = 2
    
    noise_acc = np.mean(val_targets_class[:, 0] == val_preds_class[:, 0])
    motion_acc = np.mean(val_targets_class[:, 1] == val_preds_class[:, 1])
    phase_acc = np.mean(val_targets_class[:, 2] == val_preds_class[:, 2])
    overall_acc = np.mean(val_targets_class == val_preds_class)
    
    print("\n" + "=" * 60)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 60)
    print(f"Overall Regression MSE:          {overall_mse:.6f}")
    print(f"Overall Regression MAE:          {overall_mae:.6f}")
    print(f"Overall Classification Accuracy: {overall_acc*100:.2f}%\n")
    
    print("Per-Parameter Metrics:")
    print(f"  - Noise:  MSE = {mses[0]:.6f}, MAE = {maes[0]:.6f}, Severity Acc = {noise_acc*100:.2f}%")
    print(f"  - Motion: MSE = {mses[1]:.6f}, MAE = {maes[1]:.6f}, Severity Acc = {motion_acc*100:.2f}%")
    print(f"  - Phase:  MSE = {mses[2]:.6f}, MAE = {maes[2]:.6f}, Severity Acc = {phase_acc*100:.2f}%")
    print("=" * 60 + "\n")
    
    # 6. Generate Clinical Diagnostic Reports for the first 2 patients
    print("Clinical Diagnostic Reports for the first two patients in validation dataset:")
    for p in range(min(2, num_patients)):
        p_slice_preds = val_preds_all[p*8 : (p+1)*8]
        p_slice_gts = val_targets_all[p*8 : (p+1)*8]
        print_patient_report(samples[p]['patient_index'], p_slice_preds, p_slice_gts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate exported ONNX MRI Anomaly Estimator model.")
    parser.add_argument("--onnx_path", type=str, default="ai-service/anomaly_detector.onnx", help="Path to ONNX model")
    parser.add_argument("--data_dir", type=str, default="eval_data", help="Directory containing synthetic evaluation dataset")
    parser.add_argument("--coils", type=int, default=16, help="Number of coils in K-space")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (number of patients per batch)")
    args = parser.parse_args()
    
    evaluate_onnx(args.onnx_path, args.data_dir, args.coils, args.batch_size)
