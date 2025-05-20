import numpy as np
from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(mask: np.ndarray, image: np.ndarray) -> float:
    """
    Calcula a distância de Hausdorff entre dois conjuntos de pontos (ex.: contornos).
    :param mask: máscara binária
    :param image: máscara real ou estrutura para comparar
    """
    pts_a = np.argwhere(mask > 0)
    pts_b = np.argwhere(image > 0)
    d_ab = directed_hausdorff(pts_a, pts_b)[0]
    d_ba = directed_hausdorff(pts_b, pts_a)[0]
    return max(d_ab, d_ba)