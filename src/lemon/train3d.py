import segmentation_models_3D as sm
from classification_models_3D.tfkeras import Classifiers
from skimage import io
import tensorflow as tf
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import nibabel as nib
import os
from pathlib import Path

def load_and_patch_images_masks(
    path: str,                          # <- dataset root
    subset: str = "train",              # "train" or "val"
    image_marker_segment: str = "_DCE_0002_N3_zscored.nii.gz",
    mask_filename_transform: tuple = ("_DCE_0002_N3_zscored.nii.gz", ".nii.gz"),
    patch_size: tuple = (64, 64, 64),
    step_size: int = 64
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Loads images & masks from the decathlon-style folder structure
        ├── imagesTr/     (training images)
        ├── labelsTr/     (training masks)
        ├── ImagesVl/     (validation images)
        └── labelsVl/     (validation masks)

    Parameters
    ----------
    path : str
        Root directory that contains the four folders above.
    subset : str, optional
        `"train"` / `"training"`  → imagesTr & labelsTr  
        `"val"`   / `"validation"`→ ImagesVl & labelsVl
    image_marker_segment, mask_filename_transform, patch_size, step_size
        Same meaning as in the original function.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        Lists of image-patch arrays and mask-patch arrays (aligned 1-to-1).
    """

    # ------------------------------------------------------------------
    # 1. Resolve which sub-folders to use
    # ------------------------------------------------------------------
    subset_lc = subset.lower()
    if subset_lc.startswith("train"):
        images_dir = Path(path) / "imagesTr"
        masks_dir  = Path(path) / "labelsTr"
    elif subset_lc.startswith(("val", "test")):       # allow “validation” or “val”
        images_dir = Path(path) / "ImagesVl"
        masks_dir  = Path(path) / "labelsVl"
    else:
        raise ValueError("subset must be 'train' or 'val' (got {!r})".format(subset))

    # ------------------------------------------------------------------
    # 2. Safety checks
    # ------------------------------------------------------------------
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found:  {masks_dir}")

    collected_image_patches: list[np.ndarray] = []
    collected_mask_patches:  list[np.ndarray] = []

    # ------------------------------------------------------------------
    # 3. Iterate through every image file in the chosen folder
    # ------------------------------------------------------------------
    for img_path in images_dir.iterdir():
        if (
            img_path.is_file() and
            img_path.name.endswith(".nii.gz") and
            image_marker_segment in img_path.name
        ):
            # Derive mask filename by replacing the marker segment
            mask_name = img_path.name.replace(*mask_filename_transform)
            mask_path = masks_dir / mask_name
            if not mask_path.exists():
                print(f"[skip] mask not found for {img_path.name}")
                continue

            # ------------------------------------------------------------------
            # 4. Load, sanity-check, and patchify
            # ------------------------------------------------------------------
            try:
                img_data  = nib.load(str(img_path)).get_fdata()
                mask_data = nib.load(str(mask_path)).get_fdata()

                if img_data.shape != mask_data.shape:
                    print(f"[skip] shape mismatch {img_path.name} vs {mask_path.name}")
                    continue

                if any(d < p for d, p in zip(img_data.shape, patch_size)):
                    print(f"[skip] volume smaller than patch for {img_path.name}")
                    continue

                img_patches  = patchify(img_data,  patch_size, step=step_size)
                mask_patches = patchify(mask_data, patch_size, step=step_size)

                collected_image_patches.append(img_patches)
                collected_mask_patches.append(mask_patches)

            except Exception as e:
                print(f"[skip] error on {img_path.name}: {e}")

    if not collected_image_patches:
        print("No valid image-mask pairs found.")

    return collected_image_patches, collected_mask_patches

def load_dataset(path):
    train_imgs, train_msks = load_and_patch_images_masks(path, subset="train")
    val_imgs,   val_msks   = load_and_patch_images_masks(path, subset="val")


    for i in range(len(train_imgs)):
        train_imgs[i] = np.reshape(train_imgs[i], (-1, train_imgs[i].shape[3], train_imgs[i].shape[4], train_imgs[i].shape[5]))
        train_msks[i] = np.reshape(train_msks[i], (-1, train_msks[i].shape[3], train_msks[i].shape[4], train_msks[i].shape[5]))

    for i in range(len(val_imgs)):
        val_imgs[i] = np.reshape(val_imgs[i], (-1, val_imgs[i].shape[3], val_imgs[i].shape[4], val_imgs[i].shape[5]))
        val_msks[i] = np.reshape(val_msks[i], (-1, val_msks[i].shape[3], val_msks[i].shape[4], val_msks[i].shape[5]))
    
    return train_imgs, train_msks, val_imgs, val_msks

#Convert grey image to 3 channels by copying channel 3 times.
#We do this as our unet model expects 3 channel input.
def format_loaded_to_categorical(train_imgs, train_msks, val_imgs, val_msks, n_classes=2):
    train_imgs = np.stack((train_imgs,)*3, axis=-1)
    val_imgs = np.stack((val_imgs,)*3, axis=-1)


    train_msks = np.expand_dims(train_msks, axis=4)
    train_msks = to_categorical(train_msks, num_classes=n_classes)

    val_msks = np.expand_dims(val_msks, axis=4)
    val_msks = to_categorical(val_msks, num_classes=n_classes)

    #masks_list = np.concatenate(masks_list, axis=0)

    print("Finished split to categorical")
    X_train, X_test = train_imgs, val_imgs
    Y_train, Y_test = train_msks, val_msks
    return X_train, X_test, Y_train, Y_test


# Loss Function and coefficients to be used during training:
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

#Define parameters for our model.
def train_model(path, filename):

    train_imgs, train_msks, val_imgs, val_msks = load_dataset(path)
    X_train, X_test, Y_train, Y_test = format_loaded_to_categorical(train_imgs, train_msks, val_imgs, val_msks, n_classes=2)

    encoder_weights = 'imagenet'
    BACKBONE = 'vgg16'  #Try vgg16, efficientnetb7, inceptionv3, resnet50
    activation = 'softmax'
    patch_size = 64
    n_classes = 2
    channels=3

    LR = 0.0001

    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.992, 0.008]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]



    model_builder, specific_preprocess_input_wrapped = Classifiers.get('vgg16')

    # Attempt to get the actual underlying function
    if hasattr(specific_preprocess_input_wrapped, '__wrapped__'):
        actual_preprocess_function = specific_preprocess_input_wrapped.__wrapped__
        print("Using __wrapped__ to get the original preprocess_input function.")
    else:
        # Fallback if __wrapped__ is not present for some reason,
        # though the traceback suggests it should be.
        actual_preprocess_function = specific_preprocess_input_wrapped
        print("Warning: __wrapped__ not found, using the function as is.")


    X_train_prep = actual_preprocess_function(X_train)
    X_test_prep = actual_preprocess_function(X_test)


    preprocess_input = sm.get_preprocessing("vgg16")


    model = sm.Unet(BACKBONE, classes=n_classes,
                    input_shape=(patch_size, patch_size, patch_size, channels),
                    encoder_weights=encoder_weights,
                    activation=activation)

    model.compile(optimizer = optim, loss=total_loss, metrics=metrics)

    history=model.fit(X_train_prep,
              Y_train,
              batch_size=8,
              epochs=100,
              verbose=1,
              validation_data=(X_test_prep, Y_test))

    model.save(f'{filename}.h5')
