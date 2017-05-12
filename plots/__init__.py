import numpy as np
import matplotlib.pyplot as plt

import h5py
# Set Latex font for figures
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


sixk = False
latent_space = 200  # size of the vector z
if sixk:
    training_file = 'data/prepared_6k.hdf'
    generated_file = 'data/generated_01_6k.hdf'

    n_jets = 60000

    gen_weights = 'models/weights_1_6k/params_generator_epoch_049.hdf5'
    disc_weights = 'models/weights_1_6k/params_discriminator_epoch_049.hdf5'

else:
    training_file = 'data/prepared_24k.hdf'
    generated_file = 'data/generated_02_24k.hdf'

    n_jets = 25000
    latent_space = 200  # size of the vector z

    gen_weights = 'models/weights_2_24k/params_generator_epoch_049.hdf5'
    disc_weights = 'models/weights_2_24k/params_discriminator_epoch_049.hdf5'

grid = 0.5 * (np.linspace(-1.25, 1.25, 26)[:-1] + np.linspace(-1.25, 1.25, 26)[1:])
eta = np.tile(grid, (25, 1))
phi = np.tile(grid[::-1].reshape(-1, 1), (1, 25))