from numpy import (
    vectorize,
    zeros,
    vstack,
    linspace,
    interp,
    array,
    log,
    argsort,
    newaxis,
)

import pickle
from sklearn.cluster import KMeans


@vectorize
def photo_electric(E):
    '''
    Basis function for the photo-electric effect
    Eq 11 in doi:  10.1088/0031-9155/55/4/016
    '''
    return 10**2 * (1 / E**3.05)


@vectorize
def compton(E):
    '''
    Klein-nishina basis function for compton scatter
    Eq 11 in doi:  10.1088/0031-9155/55/4/016
    '''
    a = E / 511.0
    term_1 = (1 / 100) * ((1 + a) / a**2) * \
        ((2 + 2 * a) / (1 + 2 * a) - (1 / a) * log(1 + 2 * a))
    term_2 = (1 / 2 * a) * log(1 + 2 * a)
    term_3 = ((1 + 3 * a) / (1 + 2 * a)**2)
    return term_1 + term_2 + term_3

# vectorize the energy function from xraylib
# could this be done in the library?
#energy_profile = vectorize(xraylib.CS_Energy)


def PE_CO_combi(Es, a, b):
    '''
    forms linear combination of basis functions
    for photo-electric effect and compton scatter
    '''
    return a * compton(Es) + b * photo_electric(Es)


def form_small_im():
    '''
    create small 10x10 image with 30 energy levels
    materials: carbon + calcium
    '''
    Nx, Ny = 10, 10
    im_mask = zeros((Nx, Ny))
    im_mask[2:8, 2:8] = 1
    im_mask[4:6, 4:6] = 2
    im_mask = im_mask.flatten()

    S = zeros((Nx * Ny, 2))
    S[:, 0][im_mask == 1] = 1
    S[:, 1][im_mask == 2] = 1

    Es = linspace(12, 100, 30)
    m1 = energy_profile(12, Es)  # carbon
    m2 = energy_profile(20, Es)  # calcium
    M = vstack([m1, m2])
    im = S.dot(M)

    return im


def spectrum(N, flux):
    '''
    create simulated spectrum at N points
    flux param can be thought of as number of emitted photons
    '''
    spectra = pickle.load(open("E_spectra.p", "rb"), encoding='latin1')
    es = linspace(12, 100, N + 1)
    fs = interp(es, spectra[:, 0], spectra[:, 1])
    Es = [(x[0] + x[1]) / 2 for x in zip(es[:-1], es[1:])]
    Fs = [(x[0] + x[1]) / 2 for x in zip(fs[:-1], fs[1:])]
    ESminus = [x[1] - x[0] for x in zip(es[:-1], es[1:])]
    Is = [x[0] * x[1] for x in zip(Fs, ESminus)]
    Is = array(Is)
    Is = Is / Is.sum()
    Is = Is * flux
    return Es, Is


def form_SME(S, M, E):
    return S.dot(M.dot(E))


def formS_SME(S):
    S = S.reshape(Nx * Ny, Nm)
    return form_SME(S, M)


def formM_SME(M):
    M = M.reshape(Nm, 2)
    return form_SME(S, M)


def tensor_to_matrix(X, Ne):
    return vstack([X[z, ...].flatten() for z in range(Ne)]).T


def matrix_to_tensor(X, Ne, Nx, Ny):
    return array([X[:, z].reshape(Nx, Ny) for z in range(Ne)])


def vec_to_tensor(X, Ne, Nx, Ny):
    return array([X[i * Nx * Ny:(i + 1) * Nx * Ny].reshape(Nx, Ny) for i in range(Ne)])


def formS(data, Nx, Ny, Nm):
    kmeans = KMeans(n_clusters=Nm, random_state=0).fit(
        data.flatten().reshape(-1, 1))
    S = zeros((Nx * Ny, Nm))
    for ind, lab in enumerate(kmeans.labels_):
        S[ind, lab] = 1
    return S
