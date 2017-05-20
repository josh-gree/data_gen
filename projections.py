from numpy import pi, exp ,log, vstack, newaxis, array
import odl


def prj_factory(Nx,Ny,Np,Nd,ret_A=False):
    '''
    returns cuda fwd and bwd projectors for given 2d geometry.

    Inputs
    Nx,Ny -> voxels in x and y dims
    Np -> number of angles
    Nd -> number of det elements

    Outputs
    fwd -> forward projector, calculates Ax
        Input Nx*Ny matrix
        Output Np*Nd matrix
    bwd -> backward projector, calculates A^Tx
        Input Nx*Ny matrix
        Output Np*Nd matrix
    '''
    reco_space = odl.uniform_discr([0, 0],[1, 1], [Nx, Ny],dtype='float32')
    angle_partition = odl.uniform_partition(0, 2 * pi, Np)
    detector_partition = odl.uniform_partition(-0.1, 1.1, Nd)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
    fwd = lambda X : ray_trafo(X).asarray()
    bwd = lambda X : ray_trafo.adjoint(X).asarray()
    return fwd, bwd


def multi_mono_prj(fwd_bwd,X):
    '''
    monoprojection for each energy, calculates AX or A^TX

    Inputs
    fwd_bwd -> fwd or bwd projection operator should be viable for shape of x
               This will come from prj_factory
    x -> poly energy image (Ne x Nx x Ny)

    Outputs
    out -> basicly this is AX or A^TX

    '''

    if len(X.shape) == 2:
        out = fwd_bwd(X)
        return out
    Ne = X.shape[0]
    out = vstack([fwd_bwd(X[i,...])[newaxis,...] for i in range(Ne)])
    return out


def poly_projection(fwd,X,I):
    '''
    calculates -log(ZI), where Z = exp(-AX)
    '''
    if not I.shape:
        I = array([I])
    Z = exp(-multi_mono_prj(fwd,X))/I.sum()
    out = -log((Z*I[:,newaxis,newaxis]).sum(axis=0))
    return out
