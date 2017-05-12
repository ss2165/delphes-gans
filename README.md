Generate jet-images from Delphes output and train a LAGAN to mimic them
==========================================================================
Code from my Part III research project, originally housed at [pt3proj](https://github.com/ss2165/pt3proj). 
All processes tested on `Python 2.7`.
## Batch run Delphes

To batch produce events from Pythia and Delphes use `run_delphes.py` in the directory jetimage.
Requires Delphes installation with internal Pythia8 
[https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook/Pythia8](https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook/Pythia8)
Make sure the directory containing DelphesPythia8 command is in your PATH.

Requirements, tested on: 
* `ROOT v5.34` 
* `Delphes v3.3.3` (Delphes installation directory must be in ROOT dynamic path, set in .rootrc).

`run_delphes.py` generates Pythia+Delphes events for wprime and qcd processes
```
Usage:
    run_delphes.py <process> <file_name> <n_events> [-s=SEED] [--boson-mass=BOSON_MASS] [--pt-hat-min=PTHMIN] [--pt-hat-max=PTHMAX]

Arguments:
    <process>    Process to generate events for
    <file_name>  Root file output name
    <n_events>   Number of events

Options:
    -s=SEED                   Random seed
    --boson-mass=BOSON_MASS   Specify boson mass GeV [default: 800]
    --pt-hat-min=PTHMIN       pthatmin GeV [default: 100]
    --pt-hat-max=PTHMAX       pthatmax GeV [default: 500]
```

## Image processing
`data_gen.py` converts Delphes ROOT files to jet images and outputs hdf files
```
Usage:
    data_gen.py <in_file> [-w] [-o <out_file>] [--ptmin=<ptmin>] [--ptmax=<ptmax>]
    data_gen.py -h | --help

Arguments:
    <in_file>  Root file to extract from

Options:
    -h --help        Show this screen
    -o <out_file>    file to save to
    --ptmin=<ptmin>  Minimum pT of jets in GeV [default: 250]
    --ptmax=<ptmax>  Maximum pT of jets in GeV [default: 300]

```
Requirements, tested on:
* `numpy v1.12.1`
* `scikit-image v0.12.3`
* `h5py v2.7.0`
* `tqdm`

## Training
See helper files in `/proc` for converting image hdf files in to a format suitable for training.
Contents of `/models` contains training code, from [adversarial-jets](https://github.com/hep-lbdl/adversarial-jets/), 
see that repository for usage, or just run `python models/train.py -h`.

Addditional requirements, tested on:
* `keras v1.2`
* `tensorflow v0.11`

