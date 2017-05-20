import odl
import numpy as np
from helper_functions import spectrum
from projections import poly_projection

Nx, Ny = 256, 256
Np_full, Nd_full = 1024, 512

reco_space = odl.uniform_discr([-1, -1], [1, 1], [Nx, Ny], dtype='float32')
detector_partition = odl.uniform_partition(-1.2, 1.2, Nd_full)
angle_partition_full = odl.uniform_partition(0, 2 * np.pi, Np_full)
geometry_full = odl.tomo.Parallel2dGeometry(
    angle_partition_full, detector_partition)
ray_trafo_full = odl.tomo.RayTransform(
    reco_space, geometry_full, impl="astra_cuda", use_cache=True)
fbp_full = odl.tomo.fbp_op(ray_trafo_full)


def fwd(X): return ray_trafo_full(X).asarray()


def bwd(X): return ray_trafo_full.adjoint(X).asarray()


Es, Is = spectrum(100, 1e7)


def sparse_sample_recon(Np):

    angle_partition_lim = odl.uniform_partition(0, 2 * np.pi, Np)
    geometry_lim = odl.tomo.Parallel2dGeometry(
        angle_partition_lim, detector_partition)
    ray_trafo_lim = odl.tomo.RayTransform(
        reco_space, geometry_lim, impl="astra_cuda", use_cache=True)
    fbp_lim = odl.tomo.fbp_op(ray_trafo_lim)

    def gen(img):
        array = img[50, ...]
        lim_sino = ray_trafo_lim(array)
        full_sino = ray_trafo_full(array)

        lim_recon = fbp_lim(lim_sino)
        full_recon = fbp_full(full_sino)

        input = lim_recon.asarray()
        target = full_recon.asarray()

        return input, target

    return gen


def sparse_sample_sino(sf):

    coord_lim = angle_partition_full.coord_vectors[0][::sf]
    i_prod = odl.IntervalProd(0, 2 * np.pi)
    g_lim = odl.RectGrid(coord_lim)
    angle_partition_lim = odl.RectPartition(i_prod, g_lim)
    geometry_lim = odl.tomo.Parallel2dGeometry(
        angle_partition_lim, detector_partition)
    ray_trafo_lim = odl.tomo.RayTransform(
        reco_space, geometry_lim, impl="astra_cuda", use_cache=True)

    def gen(img):
        array = img[50, ...]
        sino_full = ray_trafo_full(array).asarray()
        sino_lim = ray_trafo_lim(array).asarray()
        sino_padded = np.zeros_like(sino_full)
        sino_padded[::sf] = sino_lim

        input = sino_padded
        target = sino_full

        return input, target

    return gen


def poly(img):

    sino = poly_projection(fwd, img, Is)
    recon = fbp_full(sino)

    min_diff = np.argmin(
        np.array([np.linalg.norm(img[i, :] - recon.asarray()) for i in range(100)]))
    min_diff = img[min_diff, :]
    mid_energy = img[50, :]

    return recon.asarray(), min_diff, mid_energy


# sino_foos = {'sino_{}'.format(i):sparse_sample_sino(i) for i in [8,16,32,64]}
# sino_foos
# sino_foos = {'recon_{}'.format(i):sparse_sample_recon(i) for i in [5,10,20,50]}
# from inputs import sparse_sample_recon
# recon_foos = {'recon_{}'.format(i):sparse_sample_recon(i) for i in [5,10,20,50]}
# sino_foos
# recon_foos
# all_foos = {}
# all_foos.update(sino_foos)
# all_foos.update(recon_foos)
# all_foos
# out = {}
# x = poly_phantom(template_array())
# for name,foo in all_foos.items():
#     inp,targ = foo(x)
#     out[name+'inp'] = inp
#     out[name+'targ'] = targ
# out
# for name,val in out.items():
#     imsave(name+'.png',val)
