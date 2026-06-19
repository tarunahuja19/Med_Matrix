"""
Motion Correction Module for KVISION AI Service.
Provides retrospective rigid motion correction using SimpleITK.
"""

import logging
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def _apply_transform_2d_3d(
    moving_np: np.ndarray,
    fixed_image_sitk: sitk.Image,
    transform: sitk.Transform,
    is_complex: bool,
) -> np.ndarray:
    """
    Helper to apply a SimpleITK transform to a real or complex numpy array.
    """
    if is_complex:
        moving_real = np.real(moving_np).astype(np.float32)
        moving_imag = np.imag(moving_np).astype(np.float32)

        real_sitk = sitk.GetImageFromArray(moving_real)
        imag_sitk = sitk.GetImageFromArray(moving_imag)

        resampled_real = sitk.Resample(
            real_sitk,
            fixed_image_sitk,
            transform,
            sitk.sitkLinear,
            0.0,
            real_sitk.GetPixelID(),
        )
        resampled_imag = sitk.Resample(
            imag_sitk,
            fixed_image_sitk,
            transform,
            sitk.sitkLinear,
            0.0,
            imag_sitk.GetPixelID(),
        )

        return sitk.GetArrayFromImage(resampled_real) + 1j * sitk.GetArrayFromImage(resampled_imag)
    else:
        moving_sitk = sitk.GetImageFromArray(moving_np.astype(np.float32))
        resampled = sitk.Resample(
            moving_sitk,
            fixed_image_sitk,
            transform,
            sitk.sitkLinear,
            0.0,
            moving_sitk.GetPixelID(),
        )
        return sitk.GetArrayFromImage(resampled)


def correct_motion(
    image: np.ndarray,
    ref_idx: int = None,
    learning_rate: float = 1.0,
    min_step: float = 1e-4,
    number_of_iterations: int = 100,
) -> np.ndarray:
    """
    Perform retrospective rigid motion correction using SimpleITK.
    Aligns slices or volumes in a sequence to a reference slice/volume.

    Parameters:
    -----------
    image : np.ndarray
        Reconstructed image of shape:
        - (N, H, W) for a series of N 2D slices/frames
        - (T, Z, H, W) for a series of T 3D volumes
        - (H, W) or (Z, H, W) if it's a single 2D or 3D frame (returned as is)
        Can be real or complex-valued.
    ref_idx : int, optional
        The index of the reference slice/volume to align others to.
        If None, defaults to the middle slice/volume (N // 2 or T // 2).
    learning_rate : float, optional
        Learning rate for the Regular Step Gradient Descent optimizer. Default is 1.0.
    min_step : float, optional
        Minimum step size for the optimizer. Default is 1e-4.
    number_of_iterations : int, optional
        Maximum number of iterations for the optimizer. Default is 100.

    Returns:
    --------
    aligned_image : np.ndarray
        Aligned image of the same shape and data type as the input.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy.ndarray")

    ndim = image.ndim
    if ndim < 2 or ndim > 4:
        raise ValueError(f"Input image must be 2D, 3D, or 4D. Got dimension: {ndim}")

    if ndim == 2:
        logger.info("Input image is 2D. No motion correction is performed.")
        return image.copy()

    is_complex = np.iscomplexobj(image)

    if ndim == 3:
        # N 2D frames/slices: shape (N, H, W)
        n_frames = image.shape[0]
        if n_frames <= 1:
            logger.info("Sequence contains only 1 frame. Returning a copy.")
            return image.copy()

        if ref_idx is None:
            ref_idx = n_frames // 2
        elif ref_idx < 0 or ref_idx >= n_frames:
            raise ValueError(f"ref_idx {ref_idx} is out of bounds for sequence of size {n_frames}")

        logger.info(f"Correcting motion for {n_frames} 2D frames using frame {ref_idx} as reference.")

        ref_frame = image[ref_idx]
        ref_frame_mag = np.abs(ref_frame) if is_complex else ref_frame
        fixed_image = sitk.GetImageFromArray(ref_frame_mag.astype(np.float32))

        aligned_slices = []
        for i in range(n_frames):
            if i == ref_idx:
                aligned_slices.append(image[i].copy())
                continue

            moving_frame = image[i]
            moving_frame_mag = np.abs(moving_frame) if is_complex else moving_frame
            moving_image = sitk.GetImageFromArray(moving_frame_mag.astype(np.float32))

            initial_transform = sitk.Euler2DTransform()
            try:
                initial_transform = sitk.CenteredTransformInitializer(
                    fixed_image,
                    moving_image,
                    initial_transform,
                    sitk.CenteredTransformInitializerFilter.GEOMETRY,
                )
            except Exception:
                center = [fixed_image.GetSpacing()[d] * (fixed_image.GetSize()[d] - 1) / 2.0 for d in range(2)]
                initial_transform.SetCenter(center)

            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMeanSquares()
            registration_method.SetOptimizerAsRegularStepGradientDescent(
                learningRate=learning_rate,
                minStep=min_step,
                numberOfIterations=number_of_iterations,
                gradientMagnitudeTolerance=1e-8,
            )
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.SetInitialTransform(initial_transform, inPlace=False)

            try:
                final_transform = registration_method.Execute(fixed_image, moving_image)
            except Exception as e:
                logger.warning(f"Registration failed for frame {i}: {e}. Keeping original frame.")
                aligned_slices.append(image[i].copy())
                continue

            aligned_slice = _apply_transform_2d_3d(moving_frame, fixed_image, final_transform, is_complex)
            aligned_slices.append(aligned_slice)

        return np.stack(aligned_slices, axis=0)

    elif ndim == 4:
        # T 3D volumes: shape (T, Z, H, W)
        t_frames = image.shape[0]
        if t_frames <= 1:
            logger.info("Sequence contains only 1 volume. Returning a copy.")
            return image.copy()

        if ref_idx is None:
            ref_idx = t_frames // 2
        elif ref_idx < 0 or ref_idx >= t_frames:
            raise ValueError(f"ref_idx {ref_idx} is out of bounds for sequence of size {t_frames}")

        logger.info(f"Correcting motion for {t_frames} 3D volumes using volume {ref_idx} as reference.")

        ref_volume = image[ref_idx]
        ref_volume_mag = np.abs(ref_volume) if is_complex else ref_volume
        fixed_image = sitk.GetImageFromArray(ref_volume_mag.astype(np.float32))

        aligned_volumes = []
        for t in range(t_frames):
            if t == ref_idx:
                aligned_volumes.append(image[t].copy())
                continue

            moving_volume = image[t]
            moving_volume_mag = np.abs(moving_volume) if is_complex else moving_volume
            moving_image = sitk.GetImageFromArray(moving_volume_mag.astype(np.float32))

            initial_transform = sitk.Euler3DTransform()
            try:
                initial_transform = sitk.CenteredTransformInitializer(
                    fixed_image,
                    moving_image,
                    initial_transform,
                    sitk.CenteredTransformInitializerFilter.GEOMETRY,
                )
            except Exception:
                center = [fixed_image.GetSpacing()[d] * (fixed_image.GetSize()[d] - 1) / 2.0 for d in range(3)]
                initial_transform.SetCenter(center)

            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMeanSquares()
            registration_method.SetOptimizerAsRegularStepGradientDescent(
                learningRate=learning_rate,
                minStep=min_step,
                numberOfIterations=number_of_iterations,
                gradientMagnitudeTolerance=1e-8,
            )
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.SetInitialTransform(initial_transform, inPlace=False)

            try:
                final_transform = registration_method.Execute(fixed_image, moving_image)
            except Exception as e:
                logger.warning(f"Registration failed for volume {t}: {e}. Keeping original volume.")
                aligned_volumes.append(image[t].copy())
                continue

            aligned_volume = _apply_transform_2d_3d(moving_volume, fixed_image, final_transform, is_complex)
            aligned_volumes.append(aligned_volume)

        return np.stack(aligned_volumes, axis=0)