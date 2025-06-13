import numpy as np
import nibabel as nib
import tensorflow as tf
from patchify import patchify, unpatchify
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import os
from tensorflow.keras.applications.vgg16 import preprocess_input
from skimage.morphology import remove_small_objects, binary_closing

from classification_models_3D.tfkeras import Classifiers
from skimage.morphology import remove_small_objects # For post-processing

def post_process_mask(mask: np.ndarray, min_size: int = 100):
    """
    Cleans up a binary mask by removing small noisy objects.
    """
    processed_mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    return processed_mask.astype(np.uint8)


def run_3d_inference(image_path: str, model_path: str):
    """
    Runs patch-based 3D inference, precisely matching the training preprocessing.
    """
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    patch_size = 64

    print(f"Loading image: {image_path}")
    img_nii = nib.load(image_path)
    original_image_data = img_nii.get_fdata()

    image_float = original_image_data.astype(np.float32)

    original_shape = image_float.shape
    pad_width = []
    for dim_size in original_shape:
        remainder = dim_size % patch_size
        if remainder == 0:
            pad_width.append((0, 0))
        else:
            pixels_to_add = patch_size - remainder
            pad_before = pixels_to_add // 2
            pad_after = pixels_to_add - pad_before
            pad_width.append((pad_before, pad_after))

    padded_image = np.pad(image_float, pad_width, mode='constant', constant_values=0)
    padded_shape = padded_image.shape
    
    patches = patchify(padded_image, (patch_size, patch_size, patch_size), step=patch_size)
    patches_reshaped = patches.reshape(-1, patch_size, patch_size, patch_size)
    patches_3_channel = np.repeat(patches_reshaped[..., np.newaxis], 3, axis=-1)

    print("Applying preprocessing to patches...")
    _, specific_preprocess_input_wrapped = Classifiers.get('vgg16')
    if hasattr(specific_preprocess_input_wrapped, '__wrapped__'):
        actual_preprocess_function = specific_preprocess_input_wrapped.__wrapped__
    else:
        actual_preprocess_function = specific_preprocess_input_wrapped
    preprocessed_patches = actual_preprocess_function(patches_3_channel)

    print(f"Running prediction on {len(preprocessed_patches)} patches...")
    prediction_patches = model.predict(preprocessed_patches, batch_size=4)
    foreground_probs = prediction_patches[..., 1]
    threshold = 0.5 # You can now tune this threshold meaningfully
    predicted_mask_patches = (foreground_probs >= threshold).astype(np.uint8)

    predicted_mask_grid = predicted_mask_patches.reshape(patches.shape)
    reconstructed_mask = unpatchify(predicted_mask_grid, padded_shape)
    crop_slices = tuple(slice(pad[0], dim_size + pad[0]) for dim_size, pad in zip(original_shape, pad_width))
    final_prediction_mask = reconstructed_mask[crop_slices]

    final_prediction_mask = post_process_mask(final_prediction_mask, min_size=500)
    print("Inference complete.")

    ground_truth_data = None
    mask_path = image_path.replace("ImagesVl", "labelsVl")
    if os.path.exists(mask_path):
        ground_truth_data = nib.load(mask_path).get_fdata().astype(np.uint8)
    else:
        print(f"Warning: Ground truth mask not found at {mask_path}")

    return original_image_data, final_prediction_mask, ground_truth_data

def visualize_with_slider(image_volume: np.ndarray, predicted_mask: np.ndarray, ground_truth_mask: np.ndarray = None):
    """
    Visualizes the image, predicted mask, and ground truth mask with a slider.
    """
    D = image_volume.shape[2]
    initial_slice = D // 2

    has_ground_truth = ground_truth_mask is not None
    num_plots = 3 if has_ground_truth else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 7))
    fig.subplots_adjust(bottom=0.2)
    
    ax_img = axes[0]
    ax_img.set_title("Original Image")
    im_display = ax_img.imshow(np.rot90(image_volume[:, :, initial_slice]), cmap='gray')
    ax_img.axis('off')

    ax_pred = axes[1]
    ax_pred.set_title("Predicted Mask Overlay")
    pred_img_display = ax_pred.imshow(np.rot90(image_volume[:, :, initial_slice]), cmap='gray')
    pred_masked_overlay = np.ma.masked_where(np.rot90(predicted_mask[:, :, initial_slice]) == 0, np.rot90(predicted_mask[:, :, initial_slice]))
    pred_mask_display = ax_pred.imshow(pred_masked_overlay, cmap='jet', alpha=0.5)
    ax_pred.axis('off')

    if has_ground_truth:
        ax_gt = axes[2]
        ax_gt.set_title("Ground Truth Mask Overlay")
        gt_img_display = ax_gt.imshow(np.rot90(image_volume[:, :, initial_slice]), cmap='gray')
        
        gt_masked_overlay = np.ma.masked_where(np.rot90(ground_truth_mask[:, :, initial_slice]) == 0, np.rot90(ground_truth_mask[:, :, initial_slice]))
        
        gt_mask_display = ax_gt.imshow(gt_masked_overlay, cmap='spring', alpha=0.6)
        ax_gt.axis('off')
        
    fig.suptitle(f"Slice: {initial_slice + 1}/{D}", fontsize=16)
    ax_slider = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    slice_slider = Slider(ax=ax_slider, label='Slice', valmin=0, valmax=D-1, valinit=initial_slice, valstep=1)

    def update(val):
        slice_idx = int(slice_slider.val)
        im_display.set_data(np.rot90(image_volume[:, :, slice_idx]))
        pred_img_display.set_data(np.rot90(image_volume[:, :, slice_idx]))
        pred_masked_overlay = np.ma.masked_where(np.rot90(predicted_mask[:, :, slice_idx]) == 0, np.rot90(predicted_mask[:, :, slice_idx]))
        pred_mask_display.set_data(pred_masked_overlay)
        
        if has_ground_truth:
            gt_img_display.set_data(np.rot90(image_volume[:, :, slice_idx]))
            
            gt_masked_overlay = np.ma.masked_where(np.rot90(ground_truth_mask[:, :, slice_idx]) == 0, np.rot90(ground_truth_mask[:, :, slice_idx]))

            gt_mask_display.set_data(gt_masked_overlay)
            
        fig.suptitle(f"Slice: {slice_idx + 1}/{D}", fontsize=16)
        fig.canvas.draw_idle()
        
    slice_slider.on_changed(update)
    plt.show()

if __name__ == "__main__":    
    model_path = "best_model3d_64x64x64.keras"
    img_path = "Dataset001_BREAST/ImagesVl/ISPY1_1238.nii.gz"

    original_image, predicted_mask, ground_truth_mask = run_3d_inference(img_path, model_path)
    
    if predicted_mask is not None:
        visualize_with_slider(original_image, predicted_mask, ground_truth_mask)
