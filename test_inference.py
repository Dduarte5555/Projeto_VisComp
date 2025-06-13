from mri_cv_tools import Inference3d, Inference

model_path = "models/best_model3d_64x64x64.keras"
img_path = "Dataset001_BREAST/ImagesVl/ISPY1_1238.nii.gz"

original_image, predicted_mask, ground_truth_mask = Inference3d.run_3d_inference(img_path, model_path)
Inference3d.visualize_with_slider(original_image, predicted_mask, ground_truth_mask)

print("-"*70)
print("Visualizando inferencia do modelo 3D")

image_data, full_prediction_mask, ground_truth_data = Inference.load_and_run_inference_2_5D("Dataset001_BREAST/ImagesVl/ISPY1_1238.nii.gz", "models/best_model2_5D_64x64x3.pth")
Inference.visualize_with_slider(image_data, full_prediction_mask, ground_truth_data)
print("-"*70)
print("Visualizando inferencia do modelo 2.5D")

image_data, full_prediction_mask, ground_truth_data = Inference.load_and_run_inference_2D("Dataset001_BREAST/ImagesVl/ISPY1_1238.nii.gz", "models/best_model2D_64x64_side_view.pth")
Inference.visualize_with_slider(image_data, full_prediction_mask, ground_truth_data)
print("-"*70)
print("Visualizando inferencia do modelo 2D")
