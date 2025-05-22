# LeMON

Biblioteca Python para pré-processamento de imagens médicas:
1. **DICOM → NIfTI**  
2. **Geração de máscaras**  
3. **Cálculo de distância (Hausdorff, etc.)**  

## Instalação

```bash
git clone https://…/lemon.git
cd lemon
pip install .
```

## Links relevantes
[Dataset usado para treinamento](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101942541)

[Modelo de segmentação 2D usado](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101942541)


### Exemplo de uso

from lemon.file_conversion import dicom_to_nifti
from lemon.mask import create_mask
from lemon.distance import hausdorff_distance

# 1. Converter
dicom_to_nifti("dados/dicom/", "output/volume.nii")

# 2. Carregar NIfTI e criar máscara
import nibabel as nib
img = nib.load("output/volume.nii").get_fdata()
mask = create_mask(img, threshold=100)

# 3. Calcular distância
dist = hausdorff_distance(mask, img > 0)
print("Hausdorff:", dist)
