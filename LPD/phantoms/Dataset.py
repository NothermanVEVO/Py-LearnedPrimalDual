import numpy as np
import random

import os

from tensorflow import keras
from keras.utils import load_img, img_to_array

import odl

def generate_custom_data_set(size : int, quant_of_phantoms : int, x_path : str, y_path, projections : list[int]) -> None:

    ## Generating y
    
    for i in range(quant_of_phantoms):
        print(i)

        phantom = generate_random_phantom(size, num_ellipses=random.randrange(2, 14), seed=None)

        np.save(os.path.join(y_path, f"{i}.npy"), phantom.astype(np.float32))

    ## Generating x

    # Espaço do ODL (mesmo do modelo)
    space = odl.uniform_discr(
        [-size/2, -size/2],
        [size/2, size/2],
        [size, size],
        dtype='float32'
    )

    for n_proj in projections:
        os.makedirs(os.path.join(x_path, str(n_proj)), exist_ok=True)

        # Definindo geometria com número de projeções
        angle_partition = odl.uniform_partition(0.0, np.pi, n_proj)
        detector_partition = odl.uniform_partition(-size/2, size/2, size)

        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
        operator = odl.tomo.RayTransform(space, geometry)

        image_files = [f for f in os.listdir(y_path) if f.lower().endswith(('.npy'))]

        for img_name in image_files:
            print(f"Processando {img_name}...")

            # Converter para ODL
            image_odl = space.element(phantom)

            # Forward projection (ODL)
            sinogram = operator(image_odl)

            # (Opcional) adicionar ruído
            sinogram = sinogram + 0.01 * odl.phantom.white_noise(operator.range)

            # Converter para numpy
            sinogram_np = sinogram.asarray()

            sinogram_np = sinogram.asarray().astype(np.float32)

            out_path = os.path.join(x_path, str(n_proj), img_name)
            np.save(out_path.replace(".png", ".npy"), sinogram_np.astype(np.float32))

def load_dataset_X_n_Y(x_path : str, y_path : str) -> tuple[np.ndarray, np.ndarray]:
    X = []
    Y = []

    x_files = sorted(os.listdir(x_path))
    y_files = sorted(os.listdir(y_path))

    for x_file, y_file in zip(x_files, y_files):
        # Carregar sinograma
        x = np.load(os.path.join(x_path, x_file))

        # Carregar phantom (imagem)
        y_img = load_img(os.path.join(y_path, y_file), color_mode="grayscale")
        y = img_to_array(y_img) / 255.0

        # Adicionar canal
        x = np.expand_dims(x, axis=-1)

        X.append(x.astype(np.float32))
        Y.append(y.astype(np.float32))

    return np.array(X), np.array(Y)

def load_full_dataset_X_n_Y(x_path : str, y_path : str, projections : list[int]) -> tuple[np.ndarray, np.ndarray]:
    X_total = []
    Y_total = []

    for p in projections:
        x, y = load_dataset_X_n_Y(x_path + f"/{p}", y_path)
        X_total.append(x)
        Y_total.append(y)

    return np.concatenate(X_total, axis=0), np.concatenate(Y_total, axis=0)

def generate_random_phantom(size : int = 256, num_ellipses : int = 10, seed : int | None = None) -> np.ndarray:
    """
    Gera um phantom aleatório baseado em elipses.
    
    Parâmetros:
        size (int): tamanho da imagem (size x size)
        num_ellipses (int): quantidade de elipses
        seed (int): define a semente aleatória (para reprodutibilidade)
    
    Retorna:
        np.ndarray: imagem phantom normalizada entre [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Cria grade normalizada (-1 a 1)
    y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    img = np.zeros((size, size), dtype=float)
    
    for _ in range(num_ellipses):
        # Parâmetros aleatórios da elipse
        cx, cy = np.random.uniform(-0.5, 0.5, 2)         # centro
        a, b = np.random.uniform(0.05, 0.4, 2)           # eixos
        angle = np.random.uniform(0, np.pi)              # rotação
        intensity = np.random.uniform(-1.0, 1.0)         # brilho
        
        # Equação da elipse rotacionada
        x_rot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
        y_rot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
        
        mask = (x_rot / a)**2 + (y_rot / b)**2 <= 1
        img[mask] += intensity

    # Normaliza entre 0 e 1
    img -= img.min()
    img /= img.max() if img.max() != 0 else 1
    
    return img