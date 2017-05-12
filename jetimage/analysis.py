"""analysis.py allows some image visualisation, averaging and batching"""
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm


def plot_jet(image_array, etaran=(-1.25, 1.25), phiran=(-1.25, 1.25), output_name=None, vmin=1e-6, vmax=300, title=''):
    """
        Adapted from Luke de Oliveira, & Michela Paganini. (2017). lukedeo/adversarial-jets: 
        Initial Release [Data set]. Zenodo. http://doi.org/10.5281/zenodo.400708
        Function to help you visualize a jet image on a log scale
        Args:
        -----
           content : numpy array of dimensions 25x25, first arg to imshow, content of the image
                     e.g.: generated_images.mean(axis=0) --> the average generated image
                           real_images.mean(axis=0) --> the average Pythia image
                           generated_images[aux_out == 1].mean(axis=0) --> the average generated image labeled as real
                           by the discriminator
                           etc...
           output_name : string, name of the output file where the plot will be saved. 
           vmin : (default = 1e-6) float, lower bound of the pixel intensity scale before saturation
           vmax : (default = 300) float, upper bound of the pixel intensity scale before saturation
           title : (default = '') string, title of the plot, to be displayed on top of the image
        Outputs:
        --------
           no function returns
           saves file in ../plots/output_name
        """
    fig, ax = plt.subplots(figsize=(6, 6))
    extent = phiran + etaran
    im = ax.imshow(image_array, interpolation='nearest', norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent,
                   cmap='nipy_spectral')
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
    """"avereage image from array of images"""
    return np.mean(images_array, axis=0)


def image_set(images):
    """"convert list of JetImages to array of image arrays shape (N_images, Nphi, Neta)"""
    return np.array([image.image_array for image in images])
