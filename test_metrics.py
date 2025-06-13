from mri_cv_tools.Inference import load_and_run_inference_2_5D, evaluate_folder, load_and_run_inference_2D
from mri_cv_tools.Inference3d import run_3d_inference

model2d   = "best_model2D_64x64_side_view.pth"
model2_5d = "best_model2_5D_64x64x3.pth"
model3d   = "best_model3d_64x64x64.keras"

# Path to the folder containing the validation images
validation_images_dir = "Dataset001_BREAST/ImagesVl"

print("Evaluating 2D model")
evaluate_folder(
    images_dir=validation_images_dir, 
    model_path=model2d, 
    inference_function=load_and_run_inference_2D
)
print("Evaluating 2.5D model")
evaluate_folder(
    images_dir=validation_images_dir, 
    model_path=model2_5d, 
    inference_function=load_and_run_inference_2_5D
)
print("Evaluating 3D model")
evaluate_folder(
    images_dir=validation_images_dir, 
    model_path=model3d, 
    inference_function=run_3d_inference
)