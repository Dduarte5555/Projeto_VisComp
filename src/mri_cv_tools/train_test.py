from mri_cv_tools.train2_5d import train_model2_5d, train_model2d
from mri_cv_tools.train3d import train_model3d

def test_training(dataset_path):
    print("Beggining training on 2D model")
    train_model2d(dataset_path, "test_model2_5D", N_EPOCHS=5)
    print("Beggining training on 2.5D model")
    train_model2_5d(dataset_path,"test_model_2D", N_EPOCHS=5)
    print("Beggining training on 3D model")
    train_model3d(dataset_path,"3d_64x64x64", epochs=5)

if __name__ == "__main__":
    test_training('test_dataset')