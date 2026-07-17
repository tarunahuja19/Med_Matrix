"""
KVision 4.0 — Evaluation Framework

Comprehensive metrics suite for all model components.
This is the FIRST thing that must work — nothing can be validated without it.

Metrics per task:
    Pathology Classification: Macro-F1, per-class AUC-ROC, confusion matrix, Cohen's κ, ECE
    Anomaly Estimation: MAE, MSE, Spearman ρ per corruption type
    Reconstruction: SSIM, PSNR, NMSE (when learned reconstruction is implemented)
    Progression: MAE on volume at future timepoints, calibration (ECE)
    Uncertainty: ECE, Brier score, AUPRC on selective prediction
"""

import numpy as np
from collections import defaultdict
from typing import Optional


# ── Classification Metrics ───────────────────────────────────────────────────

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute NxN confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """Compute precision, recall, F1 for each class."""
    cm = compute_confusion_matrix(y_true, y_pred, n_classes)
    metrics = {}

    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[c, :].sum()),
        }

    return metrics


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 score across all classes."""
    per_class = compute_per_class_metrics(y_true, y_pred, n_classes)
    f1_scores = [per_class[c]["f1"] for c in range(n_classes)]
    return float(np.mean(f1_scores))


def compute_cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Cohen's Kappa: agreement corrected for chance."""
    cm = compute_confusion_matrix(y_true, y_pred, n_classes)
    n = cm.sum()
    if n == 0:
        return 0.0

    po = np.diag(cm).sum() / n  # Observed agreement
    pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (n * n)  # Expected agreement

    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))


def compute_auc_roc(y_true: np.ndarray, y_probs: np.ndarray, n_classes: int) -> dict[int, float]:
    """
    Per-class AUC-ROC using the trapezoidal rule.
    
    Args:
        y_true: True labels [N]
        y_probs: Predicted probabilities [N, n_classes]
        
    Returns:
        Dict mapping class_idx → AUC-ROC
    """
    aucs = {}
    for c in range(n_classes):
        binary_true = (y_true == c).astype(float)
        scores = y_probs[:, c]

        # Sort by predicted probability (descending)
        sorted_idx = np.argsort(-scores)
        sorted_true = binary_true[sorted_idx]

        # Compute TPR and FPR at each threshold
        tp = np.cumsum(sorted_true)
        fp = np.cumsum(1 - sorted_true)

        tpr = tp / max(1, binary_true.sum())
        fpr = fp / max(1, (1 - binary_true).sum())

        # Trapezoidal AUC
        auc = float(np.trapz(tpr, fpr))
        aucs[c] = max(0.0, min(1.0, auc))

    return aucs


# ── Calibration Metrics ──────────────────────────────────────────────────────

def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE).
    
    Measures how well predicted confidence aligns with actual accuracy.
    Perfect calibration: ECE = 0.
    
    Args:
        y_true: True labels [N]
        y_probs: Predicted probabilities [N, n_classes]
        n_bins: Number of confidence bins
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() == 0:
            continue

        bin_accuracy = accuracies[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        bin_weight = in_bin.sum() / n_total

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_probs: np.ndarray, n_classes: int) -> float:
    """
    Brier score (multi-class): mean squared error of predicted probabilities.
    Lower is better. Perfect = 0.
    """
    one_hot = np.eye(n_classes)[y_true]  # [N, n_classes]
    return float(np.mean(np.sum((y_probs - one_hot) ** 2, axis=1)))


def compute_reliability_diagram(
    y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10
) -> dict:
    """
    Compute data for reliability diagram visualization.
    
    Returns:
        Dict with bin_centers, bin_accuracies, bin_confidences, bin_counts
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = int(in_bin.sum())
        bin_counts.append(count)

        if count > 0:
            bin_centers.append(float((bin_boundaries[i] + bin_boundaries[i + 1]) / 2))
            bin_accuracies.append(float(accuracies[in_bin].mean()))
            bin_confidences.append(float(confidences[in_bin].mean()))
        else:
            bin_centers.append(float((bin_boundaries[i] + bin_boundaries[i + 1]) / 2))
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


# ── Regression Metrics (Anomaly Estimation) ──────────────────────────────────

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def compute_spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    n = len(y_true)
    if n < 2:
        return 0.0

    # Rank
    rank_true = np.argsort(np.argsort(y_true)).astype(float)
    rank_pred = np.argsort(np.argsort(y_pred)).astype(float)

    d = rank_true - rank_pred
    return float(1 - (6 * np.sum(d ** 2)) / (n * (n ** 2 - 1)))


# ── Reconstruction Metrics ───────────────────────────────────────────────────

def compute_ssim_2d(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """
    Structural Similarity Index (SSIM) for 2D images.
    
    Implements Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity"
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(numerator / denominator)


def compute_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0  # Perfect match
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_nmse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Normalized Mean Squared Error."""
    return float(np.sum((img1 - img2) ** 2) / np.sum(img2 ** 2))


# ── Full Evaluation Report ───────────────────────────────────────────────────

def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    n_classes: int,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Run full evaluation suite on a classifier.
    
    Returns a comprehensive report dict with all metrics.
    """
    per_class = compute_per_class_metrics(y_true, y_pred, n_classes)
    aucs = compute_auc_roc(y_true, y_probs, n_classes)

    report = {
        "macro_f1": compute_macro_f1(y_true, y_pred, n_classes),
        "cohens_kappa": compute_cohens_kappa(y_true, y_pred, n_classes),
        "ece": compute_ece(y_true, y_probs),
        "brier_score": compute_brier_score(y_true, y_probs, n_classes),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred, n_classes).tolist(),
        "per_class": {},
        "reliability_diagram": compute_reliability_diagram(y_true, y_probs),
        "n_samples": len(y_true),
    }

    for c in range(n_classes):
        name = class_names[c] if class_names else f"class_{c}"
        report["per_class"][name] = {
            **per_class[c],
            "auc_roc": aucs.get(c, 0.0),
        }

    return report


def evaluate_anomaly_estimator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    corruption_names: list[str],
) -> dict:
    """
    Evaluate the anomaly estimator on each corruption type.
    
    Args:
        y_true: Ground truth severities [N, n_corruptions]
        y_pred: Predicted severities [N, n_corruptions]
        corruption_names: List of corruption type names
    """
    report = {"per_corruption": {}, "overall": {}}

    for i, name in enumerate(corruption_names):
        report["per_corruption"][name] = {
            "mae": compute_mae(y_true[:, i], y_pred[:, i]),
            "mse": compute_mse(y_true[:, i], y_pred[:, i]),
            "spearman_rho": compute_spearman_rho(y_true[:, i], y_pred[:, i]),
        }

    report["overall"]["mean_mae"] = float(np.mean([
        report["per_corruption"][n]["mae"] for n in corruption_names
    ]))
    report["overall"]["mean_spearman"] = float(np.mean([
        report["per_corruption"][n]["spearman_rho"] for n in corruption_names
    ]))

    return report
