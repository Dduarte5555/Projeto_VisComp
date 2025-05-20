import numpy as np

def create_mask(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    Gera máscara binária a partir de um threshold.
    :param image: array 3D ou 2D de intensidades
    :param threshold: valor mínimo para pertencer à máscara
    :return: máscara binária (0 ou 1)
    """
    return (image >= threshold).astype(np.uint8)