import numpy as np
from matplotlib.colors import hsv_to_rgb

from nfractals.runner import CONVERGIDO

# Brillo mínimo de un punto convergido (el negro puro se reserva a los que no convergen).
BRILLO_MIN = 0.35


def plot_basins(raiz, iters, estado, n_raices, ax=None, iter_ref=None):
    """Devuelve una imagen RGB (alto, ancho, 3) de las cuencas.

    Matiz: según la raíz (equiespaciado en el círculo cromático).
    Brillo: baja con el log de las iteraciones, de 1 (1 iteración) a
    BRILLO_MIN (iter_ref iteraciones o más, recortando por encima); más
    oscuro = más iteraciones. Si no se pasa `iter_ref`, se usa el percentil 99
    de las iteraciones de los puntos convergidos de este mapa.
    Negro: puntos con estado distinto de CONVERGIDO.
    Si se pasa `ax`, la imagen también se dibuja en él.
    """
    conv = estado == CONVERGIDO

    matiz = np.where(conv, raiz, 0) / n_raices
    brillo = np.ones(raiz.shape)
    if conv.any():
        it = iters[conv].astype(float)
        if iter_ref is None:
            iter_ref = np.percentile(it, 99)
        # Escala logarítmica: 1 iteración -> rel 0 (brillo 1); iter_ref o más -> rel 1.
        if iter_ref > 1:
            rel = np.clip(np.log(it) / np.log(iter_ref), 0.0, 1.0)
        else:
            rel = np.zeros_like(it)
        brillo[conv] = 1.0 - (1.0 - BRILLO_MIN) * rel

    hsv = np.stack([matiz, np.ones(raiz.shape), brillo], axis=-1)
    img = hsv_to_rgb(hsv)
    img[~conv] = 0.0

    if ax is not None:
        ax.imshow(img, origin="lower")
        ax.set_axis_off()
    return img
