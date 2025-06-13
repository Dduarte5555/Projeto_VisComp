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
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

def load_and_patch_images_masks(
    path: str,                          # <- dataset root
    subset: str = "train",              # "train" or "val"
    image_marker_segment: str = ".nii.gz",
    mask_filename_transform: tuple = (".nii.gz", ".nii.gz"),
    patch_size: tuple = (64, 64, 64),
    step_size: int = 64
) -> tuple[list[np.ndarray], list[np.ndarray]]:

    subset_lc = subset.lower()
    if subset_lc.startswith("train"):
        images_dir = Path(path) / "imagesTr"
        masks_dir  = Path(path) / "labelsTr"
    elif subset_lc.startswith(("val", "test")):
        images_dir = Path(path) / "ImagesVl"
        masks_dir  = Path(path) / "labelsVl"
    else:
        raise ValueError("subset must be 'train' or 'val' (got {!r})".format(subset))

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found:  {masks_dir}")

    collected_image_patches: list[np.ndarray] = []
    collected_mask_patches:  list[np.ndarray] = []

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

def format_loaded_to_categorical(train_imgs, train_msks,
                                 val_imgs,   val_msks,
                                 n_classes=2):
    train_imgs = np.concatenate(train_imgs, axis=0).astype(np.float32)
    val_imgs   = np.concatenate(val_imgs,   axis=0).astype(np.float32)

    train_msks = np.concatenate(train_msks, axis=0).astype(np.int32)
    val_msks   = np.concatenate(val_msks,   axis=0).astype(np.int32)

    train_imgs = np.repeat(train_imgs[..., np.newaxis], 3, axis=-1)
    val_imgs   = np.repeat(val_imgs[..., np.newaxis],   3, axis=-1)

    train_msks = to_categorical(train_msks[..., np.newaxis],
                                num_classes=n_classes)
    val_msks   = to_categorical(val_msks[..., np.newaxis],
                                num_classes=n_classes)

    print("Finished split to categorical")
    return train_imgs, val_imgs, train_msks, val_msks


def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def train_model3d(path, filename,
                    batch_size=8,
                    epochs=100,
                    encoder_weights = 'imagenet',
                    BACKBONE = 'vgg16',  #Try vgg16, efficientnetb7, inceptionv3, resnet50
                    activation = 'softmax',
                    patch_size = 64,
                    n_classes = 2,
                    channels=3,
                    LR = 0.0001):

    train_imgs, train_msks, val_imgs, val_msks = load_dataset(path)
    X_train, X_test, Y_train, Y_test = format_loaded_to_categorical(
        train_imgs, train_msks, val_imgs, val_msks, n_classes=2)
        
    optim = keras.optimizers.Adam(LR)

    # set class weights for dice_loss
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.992, 0.008]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

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

    checkpoint = ModelCheckpoint(f'{filename}.keras',
                             monitor='val_iou_score',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

    history=model.fit(X_train_prep,
              Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test_prep, Y_test),
              callbacks=[checkpoint])


if __name__ == "__main__":
    dataset_path = "test_dataset"
    train_model3d(dataset_path,"3d_64x64x64", epochs=5)