from template import template_array
from poly_phantom import poly_phantom
from inputs import sparse_sample_recon, sparse_sample_sino, poly
from pathlib import Path

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
    N_train = 500
    N_test = 500
    N_val = 100

    train_path = Path('data/train')
    test_path = Path('data/test')
    val_path = Path('data/val')

    train_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    for i in range(N_train):
        out = one_sample()
        dd.io.save(bytes(train_path / '{}.h5'.format(i)), out)

    for i in range(N_test):
        out = one_sample()
        dd.io.save(bytes(test_path / '{}.h5'.format(i)), out)

    for i in range(N_val):
        out = one_sample()
        dd.io.save(bytes(val_path / '{}.h5'.format(i)), out)
