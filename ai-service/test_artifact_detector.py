import os
import numpy as np
import torch
import pytest
from artifact_detector import (
    preprocess_image,
    get_model,
    detect_artifacts,
    MRIArtifactDataset,
    train_model,
    generate_phantom_image,
    generate_synthetic_dataset,
    add_ghosting_artifact,
    add_wrap_around_artifact,
    add_zipper_noise_artifact
)
from torch.utils.data import DataLoader

def test_preprocess_image():
    # Test 2D image
    img_2d = np.random.rand(128, 128)
    tensor = preprocess_image(img_2d, target_size=(64, 64))
    assert tensor.shape == (1, 64, 64)
    
    # Test 3D image (C, H, W)
    img_3d_c = np.random.rand(3, 128, 128)
    tensor_3d = preprocess_image(img_3d_c, target_size=(64, 64))
    assert tensor_3d.shape == (1, 64, 64)
    
    # Test 3D image (H, W, C)
    img_3d_w = np.random.rand(128, 128, 3)
    tensor_3dw = preprocess_image(img_3d_w, target_size=(64, 64))
    assert tensor_3dw.shape == (1, 64, 64)


def test_model_instantiation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Custom CNN model
    model_custom = get_model(model_type='custom', device=device)
    assert isinstance(model_custom, torch.nn.Module)
    
    # ResNet-18 model
    model_resnet = get_model(model_type='resnet18', device=device)
    assert isinstance(model_resnet, torch.nn.Module)
    
    # Check input shape compatibility
    dummy_input = torch.randn(2, 1, 256, 256).to(device)
    with torch.no_grad():
        out_custom = model_custom(dummy_input)
        out_resnet = model_resnet(dummy_input)
        
    assert out_custom.shape == (2, 3)
    assert out_resnet.shape == (2, 3)


def test_detect_artifacts():
    img = generate_phantom_image(size=128)
    
    # Test with custom model
    probs = detect_artifacts(img, model_type='custom')
    assert isinstance(probs, dict)
    assert 'ghosting' in probs
    assert 'wrap_around' in probs
    assert 'zipper_noise' in probs
    for val in probs.values():
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0


def test_training_loop():
    # Generate tiny dataset
    images, labels = generate_synthetic_dataset(num_samples=4, size=64)
    dataset = MRIArtifactDataset(images, labels)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = get_model(model_type='custom')
    
    # Train for 1 epoch
    trained_model = train_model(
        train_loader=loader,
        val_loader=None,
        model=model,
        epochs=1,
        lr=1e-3,
        save_path=None
    )
    assert isinstance(trained_model, torch.nn.Module)


def test_artifact_simulators():
    base_img = np.ones((64, 64), dtype=np.float32) * 0.5
    
    # Ghosting
    ghosted = add_ghosting_artifact(base_img, num_ghosts=2, intensity=0.2)
    assert ghosted.shape == (64, 64)
    assert not np.allclose(ghosted, base_img)
    
    # Wrap-around
    wrapped = add_wrap_around_artifact(base_img, wrap_fraction=0.2)
    assert wrapped.shape == (64, 64)
    assert not np.allclose(wrapped, base_img)
    
    # Zipper noise
    zippered = add_zipper_noise_artifact(base_img, spike_intensity=2.0)
    assert zippered.shape == (64, 64)
    assert not np.allclose(zippered, base_img)