from train2_5d import train_model2_5d, train_model2d
from train3d import train_model3d

def train_all(dataset_path,
                    batch_size=8,
                    epochs=100,
                    patch_size = 64,
                    n_classes = 2,
                    THRESH_IoU_2D=0.2,
                    THRESH_IoU_2_5D=0.2,
                    LR_2D = 0.001,
                    LR_2_5D = 0.001,
                    LR_3D = 0.0001):


    print("Beggining training on 2D model")
    train_model2d(dataset_path, "2d_64x64", 
                    BATCH_SIZE = batch_size,
                    N_EPOCHS=epochs,
                    classes=n_classes-1,
                    LR=LR_2D,
                    THRESH_IoU=THRESH_IoU_2D,
                    )
    print("Beggining training on 2.5D model")
    train_model2_5d(dataset_path,"2_5d_64x64x3",                     
                    BATCH_SIZE = batch_size,
                    N_EPOCHS=epochs,
                    classes=n_classes-1,
                    LR=LR_2_5D,
                    THRESH_IoU=THRESH_IoU_2D,
                    )

    print("Beggining training on 3D model")
    train_model3d(dataset_path,"3d_64x64x64",
                    batch_size=batch_size,
                    epochs=epochs,
                    patch_size = patch_size,
                    n_classes = n_classes,
                    LR = LR_3D)

if __name__ == "__main__":
    train_all("Dataset001_BREAST")