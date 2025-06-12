from Inference import load_and_run_inference_2_5D, evaluate_folder
from Inference3d import run_3d_inference

if __name__ == "__main__":
    model2_5d = "best_model2_5D_64x64x3.pth"
    model3d   = "best_model3d_64x64x64.h5"
    
    # Path to the folder containing the validation images
    validation_images_dir = "Dataset001_BREAST/ImagesVl"
    
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
