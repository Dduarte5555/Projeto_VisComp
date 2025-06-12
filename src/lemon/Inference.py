import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def Inference(image_path: str, model_path: str):
    """
    Realiza a inferência de uma imagem .nii.gz com modelo treinado em 2.5D (3 canais).
    """

    # Carrega a imagem NIfTI
    img_nii = nib.load(image_path)
    image_data = img_nii.get_fdata()

    # Seleciona 3 slices adjacentes para formar os 3 canais
    z = image_data.shape[2] // 2
    if z - 1 < 0 or z + 1 >= image_data.shape[2]:
        raise ValueError("A imagem não possui fatias suficientes para gerar 3 canais.")
    
    slices = image_data[:, :, z-1:z+2]  # shape: [H, W, 3]
    slices = slices.astype(np.float32)

    # Normaliza os valores para [0, 1]
    min_val = np.min(slices)
    max_val = np.max(slices)
    if max_val > min_val:
        slices_norm = (slices - min_val) / (max_val - min_val)
    else:
        slices_norm = slices

    # Transpõe para [C, H, W] e converte para tensor
    slices_norm = slices_norm.transpose(2, 0, 1)  # [3, H, W]
    tensor_img = torch.tensor(slices_norm, dtype=torch.float32)

    # Aplica resize e normalização
    preprocess = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_tensor = preprocess(tensor_img).unsqueeze(0)  # [1, 3, 192, 192]

    # Carrega o modelo
    model = torch.load(model_path, weights_only=False, map_location='cpu')
    model.eval()

    # Inferência
    with torch.no_grad():
        output = model(input_tensor)  # [1, num_classes, H, W]
        pred_mask = torch.argmax(output, dim=1)  # [1, H, W]

    pred_mask_np = pred_mask.squeeze().cpu().numpy()

    # Visualização
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_data[:, :, z], cmap='gray')
    plt.title("Slice central da imagem")

    plt.subplot(1, 2, 2)
    plt.imshow(image_data[:, :, z], cmap='gray')
    plt.imshow(pred_mask_np, cmap='jet', alpha=0.5)
    plt.title("Máscara prevista (overlay)")

    plt.tight_layout()
    plt.show()

    print("Valores únicos previstos:", torch.unique(pred_mask))
    return pred_mask_np

# Exemplo de uso
model_path = "best_model2_5D_64x64x3.pth"
img_path = "Dataset001_BREAST/ImagesVl/ISPY1_1011.nii.gz"
Inference(img_path, model_path)
