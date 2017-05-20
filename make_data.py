from template import template_array
from poly_phantom import poly_phantom
from inputs import sparse_sample_recon, sparse_sample_sino, poly

import deepdish as dd

recon_foos = {'recon_{}'.format(i): sparse_sample_recon(i)
              for i in [5, 10, 20, 50]}
sino_foos = {'sino_{}'.format(i): sparse_sample_sino(i)
             for i in [8, 16, 32, 64]}


def one_sample():

    x = poly_phantom(template_array())

    out = {}
    for ind, (name, foo) in enumerate(recon_foos.items()):
        if ind == 0:
            inp, targ = foo(x)
            out['recon_targ'] = targ
            out[name + 'inp'] = inp
            del inp
            del targ
        else:
            inp, _ = foo(x)
            out[name + 'inp'] = inp
            del inp

    for ind, (name, foo) in enumerate(sino_foos.items()):
        if ind == 0:
            inp, targ = foo(x)
            out['sino_targ'] = targ
            out[name + 'inp'] = inp
            del inp
            del targ
        else:
            inp, _ = foo(x)
            out[name + 'inp'] = inp
            del inp

    poly_recon, min_diff, mid_energy = poly(x)
    out['poly_recon'] = poly_recon
    out['min_diff'] = min_diff
    out['mid_energy'] = mid_energy
    del poly_recon
    del min_diff
    del mid_energy

    return out


if __name__ == "__main__":
    N_train = 10
    N_test = 10
    N_val = 5

    for i in range(N_train):
        out = one_sample()
        dd.io.save('{}.h5'.format(i), out)
