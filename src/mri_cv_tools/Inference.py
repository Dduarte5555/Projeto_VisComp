import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import resize
from pathlib import Path
import pandas as pd
from datetime import datetime
import os

def _load_nii(path: Path):
    arr = nib.load(str(path)).get_fdata(dtype=np.float32)
    # Move depth axis to the front if it's at the end (H, W, D) -> (D, H, W)
    if arr.shape[2] != arr.shape[0] and arr.shape[2] != arr.shape[1]:
         return np.moveaxis(arr, 2, 0)
    return arr


def load_and_run_inference_2_5D(image_path: str, model_path: str, window_size: int = 3):
    """
    Loads an image and its corresponding ground truth mask, performs inference on all
    slices using a probability threshold, and returns the image, the full 3D prediction,
    and the ground truth mask.
    
    The window_size parameter must match the number of input channels of the model.
    """
    assert window_size % 2 == 1, "Window size must be an odd number (e.g., 1, 3, 5...)."
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading PyTorch model: {model_path}")
    print(f"--> Assuming model has {window_size} input channels.")
    
    try:
        model = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except Exception:
        print("weights_only=True failed. Attempting to load the full model object with weights_only=False.")
        model = torch.load(model_path, map_location=DEVICE, weights_only=False)

    model.eval()

    print(f"Loading image: {image_path}")
    image_data = _load_nii(Path(image_path))
    D, H, W = image_data.shape
    
    full_prediction_mask = np.zeros_like(image_data, dtype=np.uint8)
    
    half = window_size // 2
    print(f"Processing {D - 2 * half} windows with 2.5D PyTorch model...")
    for z in range(half, D - half):
        window_np = image_data[z - half : z + half + 1, :, :]

        mean = window_np.mean()
        std = window_np.std() + 1e-6
        normalized_window = (window_np - mean) / std
        
        input_tensor = torch.from_numpy(normalized_window).float().unsqueeze(0).to(DEVICE)
        input_resized = F.interpolate(input_tensor, size=(64, 64), mode='bilinear', align_corners=False)

        with torch.no_grad():
            logits = model(input_resized)
            
            logits_resized = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            probs = logits_resized.sigmoid()
        
            threshold = 0.5 
            final_mask_slice = (probs.squeeze() > threshold).cpu().numpy().astype(np.uint8)
        
        full_prediction_mask[z, :, :] = final_mask_slice

    print("Inference complete.")
    
    ground_truth_data = None
    mask_path = image_path.replace("ImagesVl", "labelsVl")
    if os.path.exists(mask_path):
        ground_truth_data = _load_nii(Path(mask_path)).astype(np.uint8)

    return image_data, full_prediction_mask, ground_truth_data

def visualize_with_slider(image_volume: np.ndarray, predicted_mask: np.ndarray, ground_truth_mask: np.ndarray = None):
    """
    Visualizes (Depth, Height, Width) data correctly, with an interactive slider.
    """
    D = image_volume.shape[0]
    initial_slice = D // 2

    has_ground_truth = ground_truth_mask is not None
    num_plots = 3 if has_ground_truth else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 7))
    fig.subplots_adjust(bottom=0.2)
    
    ax_img = axes[0]
    ax_img.set_title("Original Image")
    im_display = ax_img.imshow(np.rot90(image_volume[initial_slice, :, :]), cmap='gray')
    ax_img.axis('off')

    ax_pred = axes[1]
    ax_pred.set_title("Predicted Mask Overlay")
    pred_img_display = ax_pred.imshow(np.rot90(image_volume[initial_slice, :, :]), cmap='gray')
    pred_masked_overlay = np.ma.masked_where(np.rot90(predicted_mask[initial_slice, :, :]) == 0, np.rot90(predicted_mask[initial_slice, :, :]))
    pred_mask_display = ax_pred.imshow(pred_masked_overlay, cmap='jet', alpha=0.5)
    ax_pred.axis('off')

    if has_ground_truth:
        ax_gt = axes[2]
        ax_gt.set_title("Ground Truth Mask Overlay")
        gt_img_display = ax_gt.imshow(np.rot90(image_volume[initial_slice, :, :]), cmap='gray')
        gt_masked_overlay = np.ma.masked_where(np.rot90(ground_truth_mask[initial_slice, :, :]) == 0, np.rot90(ground_truth_mask[initial_slice, :, :]))
        gt_mask_display = ax_gt.imshow(gt_masked_overlay, cmap='spring', alpha=0.6)
        ax_gt.axis('off')
        
    fig.suptitle(f"Slice: {initial_slice + 1}/{D}", fontsize=16)
    
    ax_slider = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    slice_slider = Slider(ax=ax_slider, label='Slice', valmin=0, valmax=D-1, valinit=initial_slice, valstep=1)

    def update(val):
        slice_idx = int(slice_slider.val)
        
        # Update all plots by slicing the first dimension
        im_display.set_data(np.rot90(image_volume[slice_idx, :, :]))
        pred_img_display.set_data(np.rot90(image_volume[slice_idx, :, :]))
        
        pred_masked_overlay = np.ma.masked_where(np.rot90(predicted_mask[slice_idx, :, :]) == 0, np.rot90(predicted_mask[slice_idx, :, :]))
        pred_mask_display.set_data(pred_masked_overlay)
        
        if has_ground_truth:
            gt_img_display.set_data(np.rot90(image_volume[slice_idx, :, :]))
            gt_masked_overlay = np.ma.masked_where(np.rot90(ground_truth_mask[slice_idx, :, :]) == 0, np.rot90(ground_truth_mask[slice_idx, :, :]))
            gt_mask_display.set_data(gt_masked_overlay)
            
        fig.suptitle(f"Slice: {slice_idx + 1}/{D}", fontsize=16)
        fig.canvas.draw_idle()
        
    slice_slider.on_changed(update)
    plt.show()

from sklearn.metrics import accuracy_score

def calculate_segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculates standard segmentation metrics for a 3D volume.
    
    Parameters:
    - y_true: The ground truth mask (3D numpy array).
    - y_pred: The predicted mask (3D numpy array).
    
    Returns:
    A dictionary containing the calculated scores.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # --- Voxel-wise Accuracy (for demonstration) ---
    # NOTE: This metric is often misleadingly high due to class imbalance.
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    
    # --- Dice Coefficient and IoU (the better metrics) ---
    # Add a small epsilon to the denominator to avoid division by zero
    epsilon = 1e-6
    
    intersection = np.sum(y_true_flat * y_pred_flat)
    
    dice_score = (2. * intersection + epsilon) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + epsilon)
    
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection
    iou_score = (intersection + epsilon) / (union + epsilon)
    
    return {
        "accuracy": accuracy,
        "dice_coefficient": dice_score,
        "iou_score": iou_score
    }

def evaluate_model_performance(image_path: str, model_path: str, inference_function):
    """Runs inference, compares the prediction to the ground truth, and returns scores."""
    original_image, predicted_mask, ground_truth_mask = inference_function(image_path, model_path)
    if ground_truth_mask is None:
        print(f"\n[Warning] Ground truth mask not found for {os.path.basename(image_path)}. Skipping.")
        return None
    metrics = calculate_segmentation_metrics(ground_truth_mask, predicted_mask)
    return metrics

def evaluate_folder(images_dir: str, model_path: str, inference_function):
    """
    Evaluates a model on all NIfTI files in a folder and reports aggregate metrics.

    Parameters:
    - images_dir: Path to the directory containing validation images.
    - model_path: Path to the trained model file.
    - inference_function: The inference function to use.
    """
    image_folder = Path(images_dir)
    image_paths = sorted(list(image_folder.glob("*.nii.gz")))

    if not image_paths:
        print(f"[Error] No .nii.gz files found in the directory: {images_dir}")
        return

    all_metrics = []
    num_files = len(image_paths)

    print("=" * 70)
    print(f"Starting Batch Evaluation of {num_files} files...")
    print(f"Model: {os.path.basename(model_path)}")
    print("=" * 70)

    for i, img_path in enumerate(image_paths):
        print(f"\n--- Processing file {i+1}/{num_files}: {img_path.name} ---")
        try:
            metrics = evaluate_model_performance(str(img_path), model_path, inference_function)
            if metrics:
                # Add the filename to the dictionary for context
                metrics['filename'] = img_path.name
                all_metrics.append(metrics)
        except Exception as e:
            print(f"!!! An error occurred while processing {img_path.name}: {e}")
            continue # Move to the next file

    if not all_metrics:
        print("\n[Error] No files could be successfully evaluated.")
        return

    results_df = pd.DataFrame(all_metrics)
    average_metrics = results_df.mean(numeric_only=True)

    print("COMPLETE BATCH EVALUATION RESULTS")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total files evaluated successfully: {len(results_df)}/{num_files}")
    
    print("\n--- Average Scores ---")
    print(f"Average Accuracy:       {average_metrics['accuracy']:.4f}")
    print(f"Average Dice Coeff:     {average_metrics['dice_coefficient']:.4f}")
    print(f"Average IoU Score:      {average_metrics['iou_score']:.4f}")
    print("-" * 25)

    print("\n--- Detailed Results per File ---")
    # Set display options to show all columns and rows
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df[['filename', 'dice_coefficient', 'iou_score', 'accuracy']].to_string(index=False))
    print("=" * 70)

    return results_df, average_metrics.to_dict()

def load_and_run_inference_2D(image_path: str, model_path: str):
    return load_and_run_inference_2_5D(image_path, model_path, window_size = 1)

if __name__ == "__main__":
    image_data, full_prediction_mask, ground_truth_data = load_and_run_inference_2_5D("Dataset001_BREAST/ImagesVl/ISPY1_1238.nii.gz", "models/best_model2_5D_64x64x3.pth")
    image_data, full_prediction_mask, ground_truth_data = load_and_run_inference_2D("Dataset001_BREAST/ImagesVl/ISPY1_1238.nii.gz", "models/best_model2D_64x64_side_view.pth")
    visualize_with_slider(image_data, full_prediction_mask, ground_truth_data)