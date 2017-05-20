import random

import numpy as np

from xraylib_np import CS_Energy
from helper_functions import spectrum

Nx, Ny = 256, 256
Ne = 100

materials = [35, 40, 45, 50]
Es, Is = spectrum(100, 1e7)
material_profile = CS_Energy(np.array(materials), np.array(Es))
rect_profile = CS_Energy(np.array([14]), np.array(Es))


def poly_phantom(template_array):

    air_array = template_array.reshape(Nx, Ny, 4)[:, :, 0]
    material_array = template_array.reshape(Nx, Ny, 4)[:, :, 1]

    airidx = np.where(air_array == 0)
    rectidx = np.where(material_array == 14)
    midx = [np.where(material_array == m) for m in materials]

    img = np.zeros((Ne, Nx, Ny))

    for ind, m in enumerate(midx):
        img[:, m[0], m[1]
            ] += np.tile(material_profile[ind, :], (len(m[0]), 1)).T

    img[:, rectidx[0], rectidx[1]] = np.tile(
        rect_profile, (len(rectidx[0]), 1)).T

    return img
