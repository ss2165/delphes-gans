from plots.plot_tools import *
import h5py
import os


def total_dist(pythia, delphes, title='Title'):
    # fig, ax = plt.subplots(figsize=(6, 6))
    bins = np.linspace(200, 1000, 50)
    _ = plt.hist(delphes,
                 bins=bins, histtype='step', label=r"P+D", normed=True)
    _ = plt.hist(pythia,
                 bins=bins, histtype='step', label=r"P", normed=True)

    plt.xlabel(r'Discretized $p_T$ of Jet Image')
    plt.ylabel(r'Units normalized to unit area')
    plt.legend()
    plt.ylim(0, 0.05)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title(title)

d_images, d_labels = load_images(training_file)
# generated_images, sampled_labels = load_images(generated_file)

with h5py.File(os.path.abspath('data/pythia_24k.hdf5'), 'r') as f:
    p_images = f['image'][:]
    p_labels = f['signal'][:]
    p_images[p_images < 1e-3] = 0.0


av_d_sig = average_image(d_images[d_labels==1])
av_p_sig = average_image(p_images[p_labels==1])
av_d_noi = average_image(d_images[d_labels==0])
av_p_noi = average_image(p_images[p_labels==0])

# total_p = np.sum(p_images.reshape((p_images.shape[0], 625)), axis=1)
# total_d = np.sum(d_images.reshape((d_images.shape[0], 625)), axis=1)
# total_dist(total_p, total_d)

##PLOTTING AVERAGES
# av_p = av_p_noi
# av_d = av_d_noi
#
# f, ax  = plt.subplots(1, 3, figsize=(10, 4))
#
# plt.subplot(131)
# plot_jet(av_p, title='Pythia only')
# plt.subplot(132)
# plot_diff_jet_image(np.clip(av_p- av_d, -20, 20), title='Difference')
#
# plt.subplot(133)
# plot_jet(av_d, title='Pythia+Delphes')

##PLOTTING CENTRAL AXIS
# f, ax  = plt.subplots(1, 2, figsize=(8, 4))
#
#
# plt.xlabel(r'[Transformed] Azimuthal Angle $(\phi)$')
# plt.ylabel(r'$p_T$ (GeV)')
# plt.title('W\' signal')
# plt.grid()
#
# plt.subplot(121)
# plt.plot(phi[:, 0], av_p_sig[:, 12], '+-', label='P')
# plt.plot(phi[:, 0], av_d_sig[:, 12], '+-', label='PD')
# plt.xlabel(r'[Transformed] Azimuthal Angle $(\phi)$')
# plt.ylabel(r'$p_T$ /GeV')
# plt.title('W\' signal')
# plt.grid()
# plt.ylim([-10, 120])
# plt.legend()
#
# plt.subplot(122)
# plt.plot(phi[:, 0], av_p_noi[:, 12], '+-', label='P')
# plt.plot(phi[:, 0], av_d_noi[:, 12], '+-', label='PD')
# plt.ylim([-10, 120])
# plt.grid()
# plt.legend()
#
# plt.xlabel(r'[Transformed] Azimuthal Angle $(\phi)$')
# plt.ylabel(r'$p_T$ /GeV')
# plt.title('QCD noise')

plt.show()


# with h5py.File(os.path.abspath('data/pythia_24k.hdf5'), 'w') as f:
#     dset = f.create_dataset('image', data=p_images)
#     sigs = f.create_dataset('signal', data=p_labels)