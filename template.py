import random

import numpy as np

from PIL import Image, ImageDraw

materials = [35, 40, 45, 50]


def rand_circs():
    '''
    creates random number (between 2 and 11) of circles. Each circle
    has a random position and a random radius as well as random choice of 
    material... 
    '''
    N = np.random.randint(2, 11)
    rs = np.random.uniform(3, 19, size=N)
    ps = np.random.uniform(58.4, 197.6, size=(2, N))
    ms = np.array([random.choice(materials) for i in range(N)])
    return rs, ps, ms


def make_circ(p, r, d, m):
    '''
    adds circle to base image d
    '''
    x, y = p
    bbox = [(x - r, y - r), (x + r, y + r)]
    d.ellipse(bbox, fill=(1, m, 0))


def template_array():

    Nx, Ny = 256, 256

    base = Image.new('RGBA', (Nx, Ny), (0, 0, 0))
    d = ImageDraw.Draw(base)

    rectx1, recty1, rectx2, recty2 = 0.15, 0.15, 0.85, 0.85
    rectx1 = rectx1 * Nx
    recty1 = (1 - recty1) * Ny
    rectx2 = rectx2 * Nx
    recty2 = (1 - recty2) * Ny

    rectbbox = [(rectx1, recty1), (rectx2, recty2)]
    d.rectangle(rectbbox, fill=(1, 14, 0))

    rs, ps, ms = rand_circs()
    [make_circ(p, r, d, m) for p, r, m in zip([x for x in ps.T], rs, ms)]

    base_array = np.array(base.getdata())

    return base_array


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = template_array()

    plt.figure()
    plt.imshow(x[:, 0].reshape(256, 256))

    plt.figure()
    plt.imshow(x[:, 1].reshape(256, 256))

    plt.show()
