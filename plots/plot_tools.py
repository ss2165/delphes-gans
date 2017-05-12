"""
Based on plots.ipynb
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
from tqdm import tqdm

import h5py
# Set Latex font for figures
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

#TODO get rid of plot_jet and average_image


def plot_jet(image_array, etaran=(-1.25,1.25), phiran=(-1.25,1.25), output_name=None, vmin=1e-6, vmax=300, title=''):
    """
        Adapted from Luke de Oliveira, & Michela Paganini. (2017). lukedeo/adversarial-jets: Initial Release [Data set]. Zenodo. http://doi.org/10.5281/zenodo.400708
        Function to help you visualize a jet image on a log scale
        Args:
        -----
           content : numpy array of dimensions 25x25, first arg to imshow, content of the image
                     e.g.: generated_images.mean(axis=0) --> the average generated image
                           real_images.mean(axis=0) --> the average Pythia image
                           generated_images[aux_out == 1].mean(axis=0) --> the average generated image labeled as real by the discriminator
                           etc...
           output_name : string, name of the output file where the plot will be saved. Note: it will be located in ../plots/
           vmin : (default = 1e-6) float, lower bound of the pixel intensity scale before saturation
           vmax : (default = 300) float, upper bound of the pixel intensity scale before saturation
           title : (default = '') string, title of the plot, to be displayed on top of the image
        Outputs:
        --------
           no function returns
           saves file in ../plots/output_name
        """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = plt.gca()
    extent = phiran + etaran
    im = ax.imshow(image_array, interpolation='nearest', norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent, cmap='jet')
    cbar = plt.colorbar(im, fraction=0.05, pad=0.05)
    cbar.set_label(r'Pixel $p_T$ (GeV)', y=0.85)
    plt.xlabel(r'[Transformed] Pseudorapidity $(\eta)$')
    plt.ylabel(r'[Transformed] Azimuthal Angle $(\phi)$')
    plt.title(title)
    if output_name is None:
        plt.show()
    else:
        plt.savefig(output_name)


def average_image(images_array):
    #avereage image from array of images
    return np.mean(images_array, axis=0)



def discrete_mass(jet_image):
    '''
    Calculates the jet mass from a pixelated jet image
    Args:
    -----
        jet_image: numpy ndarray of dim (1, 25, 25)
    Returns:
    --------
        M: float, jet mass
    '''
    Px = np.sum(jet_image * np.cos(phi), axis=(1, 2))
    Py = np.sum(jet_image * np.sin(phi), axis=(1, 2))

    Pz = np.sum(jet_image * np.sinh(eta), axis=(1, 2))
    E = np.sum(jet_image * np.cosh(eta), axis=(1, 2))

    PT2 = np.square(Px) + np.square(Py)
    M2 = np.square(E) - (PT2 + np.square(Pz))
    M = np.sqrt(M2)
    return M

def discrete_pt(jet_image):
    '''
    Calculates the jet transverse momentum from a pixelated jet image
    Args:
    -----
        jet_image: numpy ndarray of dim (1, 25, 25)
    Returns:
    --------
        float, jet transverse momentum
    '''
    Px = np.sum(jet_image * np.cos(phi), axis=(1, 2))
    Py = np.sum(jet_image * np.sin(phi), axis=(1, 2))
    return np.sqrt(np.square(Px) + np.square(Py))


def dphi(phi1, phi2):
    '''
    Calculates the difference between two angles avoiding |phi1 - phi2| > 180 degrees
    '''
    import math
    return math.acos(math.cos(abs(phi1 - phi2)))


def _tau1(jet_image):
    """
    Calculates the normalized tau1 from a pixelated jet image
    Args:
    -----
        jet_image: numpy ndarray of dim (1, 25, 25)
    Returns:
    --------
        float, normalized jet tau1
    """
    # find coordinate of most energetic pixel, then use formula to compute tau1
    tau1_axis_eta = eta.ravel()[np.argmax(jet_image)]
    tau1_axis_phi = phi.ravel()[np.argmax(jet_image)]
    tau1 = np.sum(jet_image *
            np.sqrt(np.square(tau1_axis_eta - eta) + np.square([dphi(tau1_axis_phi, p) for p in phi.ravel()]).reshape(25, 25))
                 )
    return tau1 / np.sum(jet_image) # normalize by the total intensity


def _tau2(jet_image):
    '''
    Calculates the normalized tau2 from a pixelated jet image
    Args:
    -----
        jet_image: numpy ndarray of dim (1, 25, 25)
    Returns:
    --------
        float, normalized jet tau2
    Notes:
    ------
        slow implementation
    '''
    # print(jet_image[jet_image != 0])
    proto = np.array(list(zip(jet_image[jet_image != 0],
                         eta[jet_image != 0],
                         phi[jet_image != 0])))
    while len(proto) > 2:
        candidates = [
            (
                (i, j),
                (min(pt1, pt2) ** 2) * ((eta1 - eta2) ** 2 + (phi1 - phi2) ** 2)
            )
            for i, (pt1, eta1, phi1) in enumerate(proto)
            for j, (pt2, eta2, phi2) in enumerate(proto)
            if j > i
        ]

        index, value = zip(*candidates)
        pix1, pix2 = index[np.argmin(value)]
        if pix1 > pix2:
            # swap
            pix1, pix2 = pix2, pix1

        (pt1, eta1, phi1) = proto[pix1]
        (pt2, eta2, phi2) = proto[pix2]

        e1 = pt1 / np.cosh(eta1)
        e2 = pt2 / np.cosh(eta2)
        choice = e1 > e2

        eta_add = (eta1 if choice else eta2)
        phi_add = (phi1 if choice else phi2)
        pt_add = (e1 + e2) * np.cosh(eta_add)

        proto[pix1] = (pt_add, eta_add, phi_add)

        proto = np.delete(proto, pix2, axis=0).tolist()

    if len(proto)>0:
        (_, eta1, phi1), (_, eta2, phi2) = proto
        np.sqrt(np.square(eta - eta1) + np.square(phi - phi1))

        grid = np.array([
            np.sqrt(np.square(eta - eta1) + np.square(phi - phi1)),
            np.sqrt(np.square(eta - eta2) + np.square(phi - phi2))
        ]).min(axis=0)

        return np.sum(jet_image * grid) / np.sum(jet_image) # normalize by the total intensity
    else:
        return 0.0

def tau21(jet_image):
    '''
    Calculates the tau21 from a pixelated jet image using the functions above
    Args:
    -----
        jet_image: numpy ndarray of dim (1, 25, 25)
    Returns:
    --------
        float, jet tau21
    Notes:
    ------
        slow implementation
    '''
    # sh = jet_image.shape
    # jet_image = jet_image.reshape(())
    ar = []
    for image in tqdm(jet_image):
        # image = image.reshape((1, 25, 25))
        tau1 = _tau1(image)
        if tau1 <= 0:
            ar.append(0)
        else:
            tau2 = _tau2(image)
            ar.append(tau2 / tau1)
    return np.array(ar)


##PIXEL INTENSITY
def pixel_intensity(real_images, generated_images, outdir):
    fig, ax = plt.subplots(figsize=(6, 6))

    _, bins, _ = plt.hist(real_images.ravel(),
                          bins=np.linspace(0, 300, 50), histtype='step', label='Pythia', color='purple')
    _ = plt.hist(generated_images.ravel(),
                 bins=bins, histtype='step', label='GAN', color='green')

    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.yscale('log')
    plt.legend(loc='upper right')

    # plt.savefig(os.path.join(outdir, 'pixel_intensity.pdf'))

##MASS
def mass_dist(real_images, generated_imagesm, title='Title'):
    # fig, ax = plt.subplots(figsize=(6, 6))
    bins = np.linspace(50, 200, 50)
    _ = plt.hist(discrete_mass(generated_images[sampled_labels == 1]),
                 bins=bins, histtype='step', label=r"generated ($W' \rightarrow WZ$)", normed=True, color='red')
    _ = plt.hist(discrete_mass(real_images[real_labels == 1]),
                 bins=bins, histtype='step', label=r"Pythia ($W' \rightarrow WZ$)", normed=True, color='red', linestyle='dashed')

    _ = plt.hist(discrete_mass(generated_images[sampled_labels == 0]),
                 bins=bins, histtype='step', label=r'generated (QCD dijets)', normed=True, color='blue')
    _ = plt.hist(discrete_mass(real_images[real_labels == 0]),
                 bins=bins, histtype='step', label=r'Pythia (QCD dijets)', normed=True, color='blue', linestyle='dashed')

    plt.xlabel(r'Discretized $m$ of Jet Image')
    plt.ylabel(r'Units normalized to unit area')
    plt.legend()

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.ylim(0, 0.03)
    plt.title(title)
    # plt.savefig(os.path.join(outdir, 'mass.pdf'))


def pt_dist(real_images, generated_images, title='Title'):
    ##PT
    # fig, ax = plt.subplots(figsize=(6, 6))
    bins = np.linspace(100, 600, 50)
    _ = plt.hist(discrete_pt(generated_images[sampled_labels == 1]),
                 bins=bins, histtype='step', label=r"generated ($W' \rightarrow WZ$)", normed=True, color='C2')
    _ = plt.hist(discrete_pt(real_images[real_labels == 1]),
                 bins=bins, histtype='step', label=r"PD ($W' \rightarrow WZ$)", normed=True, color='C2',
                 linestyle='dashed')

    _ = plt.hist(discrete_pt(generated_images[sampled_labels == 0]),
                 bins=bins, histtype='step', label=r'generated (QCD dijets)', normed=True, color='C3')
    _ = plt.hist(discrete_pt(real_images[real_labels == 0]),
                 bins=bins, histtype='step', label=r'PD (QCD dijets)', normed=True, color='C3',
                 linestyle='dashed')
    plt.xlabel(r'Discretized $p_T$ of Jet Image')
    plt.ylabel(r'Units normalized to unit area')
    plt.legend()
    plt.ylim(0, 0.015)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title(title)

    # plt.savefig(os.path.join(outdir, 'pt.pdf'))

# def tau21_dist(real_images, generated_images, outdir):
#     ##PT
#     fig, ax = plt.subplots(figsize=(6, 6))
#     bins = np.linspace(0, 1, 50)
#     _ = plt.hist(tau21(generated_images[sampled_labels == 1]),
#                  bins=bins, histtype='step', label=r"generated ($W' \rightarrow WZ$)", normed=True, color='red')
#     _ = plt.hist(tau21(real_images[real_labels == 1]),
#                  bins=bins, histtype='step', label=r"Pythia ($W' \rightarrow WZ$)", normed=True, color='red',
#                  linestyle='dashed')
#
#     _ = plt.hist(tau21(generated_images[sampled_labels == 0]),
#                  bins=bins, histtype='step', label=r'generated (QCD dijets)', normed=True, color='blue')
#     _ = plt.hist(tau21(real_images[real_labels == 0]),
#                  bins=bins, histtype='step', label=r'Pythia (QCD dijets)', normed=True, color='blue',
#                  linestyle='dashed')
#     plt.xlabel(r'Discretized $p_T$ of Jet Image')
#     plt.ylabel(r'Units normalized to unit area')
#     plt.legend()
#     plt.ylim(0, 6.0)
#     # plt.savefig(os.path.join(outdir, 'tau21.pdf'))


def load_images(filename):
    with h5py.File(os.path.abspath(filename), 'r') as f:
        images = f['image'][:]
        labels = f['signal'][:]
        images[images < 1e-3] = 0.0  # everything below 10^-3 is unphysical and due to instabilities in the rotation

    return images, labels


def plot_diff_jet_image(
                    content,
                    output_name=None,
                    extr=None,
                    title='',
                    cmap='PiYG'):
    '''
    Function to help you visualize the difference between two sets of jet images on a linear scale
    Args:
    -----
       content : numpy array of dimensions 25x25, first arg to imshow, content of the image
                 e.g.: generated_images.mean(axis=0) - real_images.mean(axis=0) --> difference between avg generated and avg Pythia image
                       etc...
       output_name : string, name of the output file where the plot will be saved. Note: it will be located in ../plots/
       extr : (default = None) float, magnitude of the upper and lower bounds of the pixel intensity scale before saturation (symmetric around 0)
       title : (default = '') string, title of the plot, to be displayed on top of the image
       cmap : (default = matplotlib.cm.PRGn_r) matplotlib colormap, ideally white in the middle
    Outputs:
    --------
       no function returns
       saves file in ../plots/output_name
    '''
    ax = plt.gca()  # fig, ax = plt.subplots(figsize=(6, 6))
    extent=[-1.25, 1.25, -1.25, 1.25]
    if extr == None:
        extr = max( abs(content.min()), abs(content.max()))
    im = ax.imshow(content,
                   interpolation='nearest', norm=Normalize(vmin=-extr, vmax=+extr), extent=extent,
                   cmap=cmap)
    cbar = plt.colorbar(im, fraction=0.05, pad=0.05)
    cbar.set_label(r'(PD - G) $p_T$ (GeV)', y=0.7)
    plt.xlabel(r'[Transformed] Pseudorapidity $(\eta)$')
    plt.ylabel(r'[Transformed] Azimuthal Angle $(\phi)$')
    plt.title(title)
    # plt.savefig(os.path.join('..', outdir, output_name))


def plot_diff_jet_image_log(
                    content,
                    output_name=None,
                    extr=None,
                    title='',
                    cmap='PiYG'):
    '''
    Function to help you visualize the difference between two sets of jet images on a linear scale
    Args:
    -----
       content : numpy array of dimensions 25x25, first arg to imshow, content of the image
                 e.g.: generated_images.mean(axis=0) - real_images.mean(axis=0) --> difference between avg generated and avg Pythia image
                       etc...
       output_name : string, name of the output file where the plot will be saved. Note: it will be located in ../plots/
       extr : (default = None) float, magnitude of the upper and lower bounds of the pixel intensity scale before saturation (symmetric around 0)
       title : (default = '') string, title of the plot, to be displayed on top of the image
       cmap : (default = matplotlib.cm.PRGn_r) matplotlib colormap, ideally white in the middle
    Outputs:
    --------
       no function returns
       saves file in ../plots/output_name
    '''
    fig, ax = plt.subplots(figsize=(6, 6))
    extent = [-1.25, 1.25, -1.25, 1.25]
    content = content + np.min(content)
    if extr == None:
        extr = max( abs(content.min()), abs(content.max()))
    im = ax.imshow(content,
                   interpolation='nearest', norm=LogNorm(), extent=extent)
    plt.colorbar(im, fraction=0.05, pad=0.05)
    plt.xlabel(r'[Transformed] Pseudorapidity $(\eta)$')
    plt.ylabel(r'[Transformed] Azimuthal Angle $(\phi)$')
    plt.title(title)
    # plt.savefig(os.path.join('..', outdir, output_name))


def plot_just_jet(image_array, etaran=(-1.25,1.25), phiran=(-1.25,1.25), output_name=None, vmin=1e-6, vmax=300, title=''):
    # fig, ax = plt.subplots(figsize=(6, 6))
    axi = plt.gca()
    extent = phiran + etaran
    im = axi.imshow(image_array, interpolation='nearest', norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent, cmap='jet')
    axi.axis('off')
    # if output_name is None:
    #     plt.show()
    # else:
    #     plt.savefig(output_name)

grid = 0.5 * (np.linspace(-1.25, 1.25, 26)[:-1] + np.linspace(-1.25, 1.25, 26)[1:])
eta = np.tile(grid, (25, 1))
phi = np.tile(grid[::-1].reshape(-1, 1), (1, 25))

# sixk = False
# latent_space = 200  # size of the vector z
# if sixk:
#     training_file = 'data/prepared_6k.hdf'
#     generated_file = 'data/generated_01_6k.hdf'
#
#     n_jets = 60000
#
#     gen_weights = 'models/weights_1_6k/params_generator_epoch_049.hdf5'
#     disc_weights = 'models/weights_1_6k/params_discriminator_epoch_049.hdf5'
#
# else:
#     training_file = 'data/prepared_24k.hdf'
#     generated_file = 'data/generated_02_24k.hdf'
#
#     n_jets = 25000
#     latent_space = 200  # size of the vector z
#
#     gen_weights = 'models/weights_2_24k/params_generator_epoch_049.hdf5'
#     disc_weights = 'models/weights_2_24k/params_discriminator_epoch_049.hdf5'

#EPOCH TRACKING
epoch = '{0:02d}'.format(1)
generated_file = 'data/generated_e{}.hdf'.format(epoch)

# real_images, real_labels = load_images(training_file)
generated_images, sampled_labels = load_images(generated_file)

# from models.networks.lagan import generator as build_generator
# g = build_generator(latent_space, return_intermediate=False)
# g.load_weights(os.path.abspath(gen_weights))
#
# noise = np.random.normal(0, 1, (n_jets, latent_space))
# sampled_labels = np.random.randint(0, 2, n_jets)
# generated_images = g.predict(
#     [noise, sampled_labels.reshape(-1, 1)], verbose=False, batch_size=64)
# generated_images *= 100
# generated_images = np.squeeze(generated_images)

#PLOT IMAGES

signal_gen = generated_images[sampled_labels == 1]
noise_gen = generated_images[sampled_labels == 0]
# signal_real = real_images[real_labels == 1]
# noise_real = real_images[real_labels == 0]

av_sig_gen = average_image(signal_gen)
av_noise_gen = average_image(noise_gen)
# av_sig_real = average_image(signal_real)
# av_noise_real = average_image(noise_real)

# print(av_sig_real[12][12], av_sig_gen[12][12])
# print(av_noise_real[12][12], av_noise_gen[12][12])

##SAMPLE
# fig, ax = plt.subplots(figsize=(10, 2))
# for i in range(6):
#     ax1 = plt.subplot(1, 6, i+1)
#     plot_just_jet(noise_gen[600+i])
# fig.patch.set_visible(False)
# plt.tight_layout()
# plt.savefig('samples_noi_gen.pdf')

# , output_name='samples_sig_gen_{}.svg'.format(i+1)

# for i in range(3):
#     plot_jet(signal_gen[34+i])
#     plt.show()

# PLOT AVERAGE
plot_jet(av_sig_gen, title='W\' signal', output_name='av_sig_e{}.svg'.format(epoch))
# plt.show()
plot_jet(av_noise_gen, title='QCD noise', output_name='av_noi_e{}.svg'.format(epoch))

# ##PLOTTING AVERAGES + DIF
# av_p = av_sig_real
# av_d = av_sig_gen
#
# f, ax  = plt.subplots(1, 3, figsize=(10, 4))
#
# plt.subplot(131)
# plot_jet(av_p, title='Pythia+Delphes')
# plt.subplot(132)
# plot_diff_jet_image(av_p- av_d, title='Difference')
#
# plt.subplot(133)
# plot_jet(av_d, title='Generator')



# # tau21_dist(real_images, generated_images, outdir)
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
#
# #load 6k
# training_file = 'data/prepared_6k.hdf'
# generated_file = 'data/generated_01_6k.hdf'
#
# real_images, real_labels = load_images(training_file)
# generated_images, sampled_labels = load_images(generated_file)
#
# #6kmass
# plt.subplot(121)
# # mass_dist(real_images, generated_images,title='6k training set')
# #
# # #6kpt
# pt_dist(real_images, generated_images,title='6k training set')
#
# # # load 24k
# training_file = 'data/prepared_24k.hdf'
# generated_file = 'data/generated_02_24k.hdf'
# real_images, real_labels = load_images(training_file)
# generated_images, sampled_labels = load_images(generated_file)
# #
# # 24kmass
# plt.subplot(122)
# # mass_dist(real_images, generated_images,title='24k training set')
# #
# # # 24kpt
# pt_dist(real_images, generated_images,title='24k training set')

##N-subjettiness
# bins = np.linspace(0, 1, 50)
# plt.subplot(121)
# _ = plt.hist(np.load('data/tau21_sig_gen_6k.npy'),
#              bins=bins, histtype='step', label=r"generated ($W' \rightarrow WZ$)", normed=True, color='C2')
# _ = plt.hist(np.load('data/tau21_sig_real_6k.npy'),
#              bins=bins, histtype='step', label=r"PD ($W' \rightarrow WZ$)", normed=True, color='C2',
#              linestyle='dashed')
#
# _ = plt.hist(np.load('data/tau21_noise_gen_6k.npy'),
#              bins=bins, histtype='step', label=r'generated (QCD dijets)', normed=True, color='C3')
# _ = plt.hist(np.load('data/tau21_noise_real_6k.npy'),
#              bins=bins, histtype='step', label=r'PD (QCD dijets)', normed=True, color='C3',
#              linestyle='dashed')
# plt.title(r'6k training set')
# plt.xlabel(r'Discretized $\tau_{21}$ of Jet Image')
# plt.ylabel(r'Units normalized to unit area')
# plt.legend()
# plt.ylim(0, 6.0)
#
# plt.subplot(122)
# _ = plt.hist(np.load('data/tau21_sig_gen_24k.npy'),
#              bins=bins, histtype='step', label=r"generated ($W' \rightarrow WZ$)", normed=True, color='C2')
# x = np.load('data/tau21_sig_real_24k.npy')
# _ = plt.hist(x[~np.isnan(x)],
#              bins=bins, histtype='step', label=r"PD ($W' \rightarrow WZ$)", normed=True, color='C2',
#              linestyle='dashed')
#
# _ = plt.hist(np.load('data/tau21_noise_gen_24k.npy'),
#              bins=bins, histtype='step', label=r'generated (QCD dijets)', normed=True, color='C3')
# _ = plt.hist(np.load('data/tau21_noise_real_24k.npy'),
#              bins=bins, histtype='step', label=r'PD (QCD dijets)', normed=True, color='C3',
#              linestyle='dashed')
# plt.title(r'24k training set')
#
# plt.xlabel(r'Discretized $\tau_{21}$ of Jet Image')
# plt.ylabel(r'Units normalized to unit area')
# plt.legend()
# plt.ylim(0, 8.0)

# plt.show()

##SAVE TAU21 CALCULATIONS
# tau21_sig_gen = tau21(generated_images[sampled_labels == 1])
# np.save('tau21_sig_gen_24k.npy', tau21_sig_gen)
# tau21_sig_real = tau21(real_images[real_labels == 1])
# np.save('tau21_sig_real_24k.npy', tau21_sig_real)
# tau21_noise_gen = tau21(generated_images[sampled_labels == 0])
# np.save('tau21_noise_gen_24k.npy', tau21_noise_gen)
# tau21_noise_real = tau21(real_images[real_labels == 0])
# np.save('tau21_noise_real_24k.npy', tau21_noise_real)
##END TAU21 SAVE


