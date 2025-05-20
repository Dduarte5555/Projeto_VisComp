import os
import pydicom
import nibabel as nib
import numpy as np

def dicom_to_nifti(dicom_dir: str, output_path: str) -> None:
    """
    Converte uma série DICOM em um único arquivo NIfTI.
    :param dicom_dir: pasta contendo .dcm
    :param output_path: caminho de saída (.nii ou .nii.gz)
    """
    # Exemplo: ler todos os .dcm, empilhar em array 3D e salvar com nibabel
    slices = []
    for fname in sorted(os.listdir(dicom_dir)):
        if fname.lower().endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(dicom_dir, fname))
            slices.append(ds.pixel_array)
    volume = np.stack(slices, axis=-1)
    img = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(img, output_path)