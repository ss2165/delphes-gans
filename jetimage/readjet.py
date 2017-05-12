"""
readjet.py Contains classes ans helper functions for generating jet-images from Delphes ROOT files.
Ensure your Delphes installation directory is in the ROOT dynamic path (set in .rootrc)
"""
import ROOT
import ROOT.Tower
import ROOT.Jet
from ROOT import TLorentzVector

from tqdm import tqdm
import time
import subprocess
import os
import numpy as np
import skimage.transform as sk

ROOT.gSystem.Load("libDelphes")  # should be in Delphes directory


class Cell:
    """Cell object. Captures energy, azimuthal angle and energy from tower cell constituent of a jet.
        Can be used as array of said variables."""
    def __init__(self, constituent_tower):
        """
        
        :param constituent_tower: ExRootTreeReader Tower object
        """
        self.E = constituent_tower.E
        self.eta = constituent_tower.Eta
        self.phi = constituent_tower.Phi
        self.pT = self.E/np.cosh(self.eta)

    def __array__(self):
        return np.array([self.eta, self.phi, self.E], dtype='float_')

    def __len__(self):
        return len(self.__array__())

    def notempty(self):
        return len(self) > 1


class Jet:
    """
    Jet object for storing and manipulating jets from Delphes ROOT files.
    """
    def __init__(self, jet_obj, subjet_array):
        """
        
        :param jet_obj: ExRootTreeReader jet object
        :param subjet_array: 2D numpy array of subjet 4-momenta. 
        """
        self.pT = jet_obj.PT
        self.eta = jet_obj.Eta
        self.phi = jet_obj.Phi
        self.mass = jet_obj.Mass
        self.momentum = TLorentzVector()
        self.momentum.SetPtEtaPhiM(self.pT, self.eta, self.phi, self.mass)
        self.full_obj = jet_obj
        momenta = [array2lorentz(a) for a in subjet_array]  # convert to lorentz vectors
        self.trimmed_momentum = momenta[0]  # momentum of overall trimmed jet
        self.sj_momentum = momenta[1:]  # subjet momenta
        self.cells = self.read_constituents(self.full_obj.Constituents)

    def read_constituents(self, constituents):
        """
        :param constituents: jet object constituents attribute
        :return: array of Cell objects
        """
        cells = []
        for const in constituents:
            if isinstance(const, ROOT.Tower):
                c = Cell(const)
                if c.notempty():
                    cells.append(c)
        return cells

    def __array__(self):
        ar = np.zeros((len(self.cells), 3))
        for i, cell in enumerate(self.cells):
            ar[i, :] = np.array(cell)
        return ar

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, item):
        return self.cells[item]


class RootEvents:
    """
    Class for reading jet calorimeter cell values from Delphes ROOT output.
    """
    def __init__(self, fname, ptmin, ptmax, mass_min=60, mass_max=100):
        """
        :param fname: .root file to extract from 
        :param ptmin: minimum pT value allowed for jets in GeV
        :param ptmax: maximum pT value allowed for jets in GeV
        :param mass_min: minimum mass value allowed for jets in GeV
        :param mass_max: maximum mass value allowed for jets in GeV
        """
        # run external root macro to extract subjet momenta as PyROOT does not presently support this
        print("Running ROOT macro.")
        start = time.clock()
        trimp4 = self.extract_subjets(fname)
        print(" Time taken: {}".format(time.clock() - start))
        # Load ExRootTreeReader
        chain = ROOT.TChain("Delphes")
        chain.Add(fname)
        self.tree_read = ROOT.ExRootTreeReader(chain)

        nev = self.tree_read.GetEntries()
        bjet = self.tree_read.UseBranch("Jet")
        btow = self.tree_read.UseBranch("Tower")

        self.events = []  # list to store events in

        print("Reading Jets.")
        for entry, subjets in tqdm(trimp4):
            self.tree_read.ReadEntry(entry)
            # take leading jet and its subjets
            j = bjet.At(0)
            if ptmin <= j.PT <= ptmax and mass_min <= j.Mass <= mass_max:
                self.events.append(Jet(j, subjets))

    def extract_subjets(self, fname):
        """
        Run root macro to extract 4 momenta of subjets
        :param fname: 
        :return: 
        """
        file_dir = os.path.dirname(os.path.realpath(__file__))
        # trim_macro.cxx must exist in same directory as this file
        script_name = os.path.join(file_dir, 'trim_macro.cxx')
        # compile root macro call
        root_command = ['root', '-q', '-b', "{}(\"{}\")".format(script_name, fname)]
        proc = subprocess.Popen(root_command, stdout=subprocess.PIPE)
        tmp = proc.stdout.read()
        seek_text = 'Processing {}("{}")...\n'.format(script_name, fname)  # skip preamble of output
        idx = tmp.find(seek_text)
        ar = np.fromstring(tmp[idx + len(seek_text):-1], dtype=float, sep=",")
        # expects 5 4-momenta vectors
        trimp4 = [(int(ar[x]), ar[x + 1: x + 21].reshape((5, 4))) for x in range(0, ar.size, 21)]
        return trimp4

    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        # enough to make iterable
        return self.events[item]


class JetImage:
    """
    Pixelises jet in to jet object and processes. 
    """
    def __init__(self, jet, dim=(25, 25), etaran=(-1.25, 1.25), phiran=(-1.25, 1.25)):
        """
        Read in jet image
        :param jet: Jet object (defined in this file)
        :param dim: Dimensions of image in pixels
        :param etaran: tuple range of pseudorapidity values of image (min, max)
        :param phiran: tuple range of azimuthal angle values of image (min, max)
        """
        self.momentum = jet.momentum
        self.subjets = jet.sj_momentum
        jet = np.array(jet)  # convert to array of cell values

        # number of pixels per axis
        Ne, Np = dim
        im = np.zeros((Np, Ne))  # array containing image

        self.ec, self.pc = (self.subjets[1].Eta(), self.subjets[1].Phi())  # leading subjet location
        for e, p, E in jet:
            # find pixel closest to coordinate values
            translated = self.centre(e, p)  # subtract leading subjet location
            # find corresponding discretised image array index
            j, i = self.find_index(translated, (etaran, phiran), dim)
            if (0 <= i < Np) and (0 <= j < Ne):
                im[i][j] += E/np.cosh(e)  # pT, invariant under translations
        self.image_array = im

        self.dim = dim
        self.phiran = phiran
        self.etaran = etaran

        # store discrete sets of coordinates
        self.etavals = self.generate_coords(etaran, Ne, Np)
        self.phivals = np.flipud(self.generate_coords(phiran, Np, Ne).T)

    @staticmethod
    def generate_coords(rang, N, N_other):
        """
        generate array of coordinate values
        :param rang: range of coordinate axis
        :param N: number of pixels in coordinate dimension
        :param N_other: number of pixels in other dimension
        :return: return N x N_other array of coordinate values
        """
        # generate array of coordinate values in shape of image, over range=rang,
        # N steps, N_other is size of other axis
        step = (rang[1]-rang[0])/N
        vals = np.linspace(rang[0]+step/2, rang[1]-step/2, N)

        return np.tile(vals, (N_other, 1))

    def centre(self, e, p):
        # translate coordinates to central value
        return e-self.ec, delta_phi(p, self.pc)

    def sj2_rotate(self):
        # rotate second subjet to -pi/2
        e, p = self.centre(self.subjets[2].Eta(), self.subjets[2].Phi())  # second subjet location
        if e < -10 or p < -10:
            e, p = self.pca_dir()  # subjet is too far away, do pca

        try:
            if e == 0:
                angle = np.pi
            else:
                angle = np.arctan(p / e) + np.pi/2
        except RuntimeWarning:
            return np.pi

        if (-np.sin(angle) * e + np.cos(angle) * p) > 0:
            angle -= np.pi
        self.image_array = sk.rotate(self.image_array, np.rad2deg(-angle), order=3)
        return self.image_array

    def pca_dir(self):
        """
        perform principle component analysis
        :return: (x, y) unit vector of principle axis
        """
        I = self.image_array
        norm = np.sum(I)
        mux = np.sum(self.etavals*I)/norm
        muy = np.sum(self.phivals*I)/norm

        x = self.etavals-mux
        y = self.phivals-muy

        sigmax2 = np.sum(x*x*I)/norm - mux**2
        sigmay2 = np.sum(y*y*I)/norm - muy**2
        sigmaxy = np.sum(x*y*I)/norm - mux*muy

        lamb_min = 0.5*(sigmax2 + sigmay2 - np.sqrt((sigmax2-sigmay2)*(sigmax2-sigmay2) + 4*sigmaxy*sigmaxy))

        dir_x = sigmax2 + sigmaxy - lamb_min
        dir_y = sigmay2 + sigmaxy - lamb_min

        # The first PC is only defined up to a sign. Let's have it point toward the side of the jet with the most energy
        dotprod = dir_x*x + dir_y*y
        if np.sum(I[dotprod > 0]) > np.sum(I[dotprod < 0]):
            dir_x *= -1
            dir_y *= -1

        return dir_x, dir_y

    def flip(self, side='r'):
        """
        Flips image so that most energetic half is on 'side'
        :param side: 'r' or 'l', the side which the most energetic side should be on
        :return: transformed image array
        """
        jet = self.image_array
        weight = jet.sum(axis=0)

        halfway = jet.shape[0] / 2.
        l, r = np.int(np.floor(halfway)), np.int(np.ceil(halfway))
        l_weight, r_weight = np.sum(weight[:l]), np.sum(weight[r:])

        if 'r' in side:
            if r_weight > l_weight:
                return jet
            return np.fliplr(jet)
        else:
            if l_weight > r_weight:
                return jet
            return np.fliplr(jet)

    def __array__(self):
        return self.image_array

    @staticmethod
    def find_index(var, ran, dim):
        """
        find index of point var in 2D image array of dimensions dim, with co-ordinate ranges ran 
        :param var: point to find index of
        :param ran: range of co-ordinates ((horizontal_min, horizontal_max),(vertical_min, vertical_max))
        :param dim: image dimensions horizontal x vertical
        :return: array indices (horizontal, vertical)
        """
        j = int(np.floor(dim[0] * (var[0] - ran[0][0]) / (ran[0][1] - ran[0][0])))
        i = dim[1] - 1 - int(np.floor(dim[1] * (var[1] - ran[1][0]) / (ran[1][1] - ran[1][0])))
        return j, i

    def normalise(self, normaliser=None):
        """
        Divide image by specified value of total intensity
        :param normaliser: value to divide by
        :return: 
        """
        if normaliser is None:
            self.image_array /= np.linalg.norm(self.image_array)
        else:
            self.image_array /= normaliser
        return self.image_array


def array2lorentz(arr):
    """convert (1,4) array lorentz vector to TLorentzVector object"""
    px, py, pz, E = arr
    return TLorentzVector(px, py, pz, E)


def theta2eta(theta):
    """convert polar angle to pseudorapidity"""
    return -np.log(np.tan(theta/2))


def eta2theta(eta):
    """convert pseudorapidity to polar angle"""
    return 2*np.arctan(np.exp(-eta))


def delta_phi(p1, p2):
    """calculate phi separation accounting for discontinuity"""
    return np.arccos(np.cos(abs(p1 - p2)))


