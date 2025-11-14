import numpy as np
from .morph_core import indices_delaunay  # os alunos podem reutilizar

# ------------------------- Funções a implementar pelos estudantes -------------------------

def pontos_medios(pA, pB):
    return (pA + pB) / 2.0

def indices_pontos_medios(pA, pB):
    return indices_delaunay(pontos_medios(pA, pB))

# Interpoladoras
def linear(t, a=1.0, b=0.0):
    return a * t + b

def sigmoide(t, k):
    val = 1.0 / (1.0 + np.exp(-k * (t - 0.5)))
    
    inicio = 1.0 / (1.0 + np.exp(k / 2.0))
    fim = 1.0 / (1.0 + np.exp(-k / 2.0))

    return (val - inicio) / (fim - inicio)

def dummy(t):
    return t

# Geometria / warping por triângulos
def _det3(a, b, c):
    return a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])

def _transf_baricentrica(pt, tri):
    v1, v2, v3 = tri[0], tri[1], tri[2]
    
    area_total_x2 = _det3(v1, v2, v3)

    if abs(area_total_x2) < 1e-9:
        return None

    w1 = _det3(pt, v2, v3) / area_total_x2
    w2 = _det3(v1, pt, v3) / area_total_x2
    w3 = _det3(v1, v2, pt) / area_total_x2
    
    return np.array([w1, w2, w3])

def _check_bari(w1, w2, w3, eps=1e-6):
    return w1 >= -eps and w2 >= -eps and w3 >= -eps

def _tri_bbox(tri, W, H):
    x_coords = tri[:, 0]
    y_coords = tri[:, 1]

    xmin = np.min(x_coords)
    xmax = np.max(x_coords)
    ymin = np.min(y_coords)
    ymax = np.max(y_coords)

    xmin_int = int(np.maximum(0, np.floor(xmin)))
    xmax_int = int(np.minimum(W - 1, np.ceil(xmax)))
    ymin_int = int(np.maximum(0, np.floor(ymin)))
    ymax_int = int(np.minimum(H - 1, np.ceil(ymax)))

    return xmin_int, xmax_int, ymin_int, ymax_int

def _amostra_bilinear(img_float, x, y):
    h, w, _ = img_float.shape

    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return np.array([0, 0, 0], dtype=img_float.dtype)

    l, c = int(y), int(x)
    dl, dc = y - l, x - c

    a1 = img_float[l, c]      # Top-left
    a2 = img_float[l, c + 1]    # Top-right
    a3 = img_float[l + 1, c]    # Bottom-left
    a4 = img_float[l + 1, c + 1]  # Bottom-right

    # Interpolação bilinear
    top = (1 - dc) * a1 + dc * a2
    bottom = (1 - dc) * a3 + dc * a4

    pixel = (1 - dl) * top + dl * bottom

    return pixel

def gera_frame(A, B, pA, pB, triangles, alfa, beta):
    H, W, _ = A.shape
    frame_out = np.zeros_like(A)

    pT = (1 - alfa) * pA + alfa * pB

    for tri_indices in triangles:
        tri_A = pA[tri_indices]
        tri_B = pB[tri_indices]
        tri_T = pT[tri_indices]

        xmin, xmax, ymin, ymax = _tri_bbox(tri_T, W, H)

        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                pt = np.array([x, y])
                
                pesos = _transf_baricentrica(pt, tri_T)

                if pesos is not None and _check_bari(pesos[0], pesos[1], pesos[2]):
                    pt_A = pesos @ tri_A
                    pt_B = pesos @ tri_B

                    # Amostra as cores em A e B usando interpolação bilinear
                    cor_A = _amostra_bilinear(A, pt_A[0], pt_A[1])
                    cor_B = _amostra_bilinear(B, pt_B[0], pt_B[1])

                    cor_final = (1 - beta) * cor_A + beta * cor_B
                    
                    frame_out[y, x] = cor_final
                    
    return frame_out
