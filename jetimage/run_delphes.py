"""run_delphes.py generates Pythia+Delphes events for wprime and qcd processes
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
"""
import time
from os import getpid, path, remove, pardir
from docopt import docopt
import subprocess

arguments = docopt(__doc__, help=True)
print(arguments)
process = arguments['<process>']
fname = arguments['<file_name>']
seed = arguments['-s']

if seed is None:
    tseed = time.clock()
    seed = abs(((tseed * 181) * ((getpid() - 83) * 359)) % 104729)

file_dir = path.dirname(path.realpath(__file__))
# template.cmnd must exist in same directory as this file
template_file = path.abspath(path.join(file_dir, 'template.cmnd'))


with open(template_file, "r") as f:
    lines = f.readlines()

# find locations of various chunks of pythia config in template file
r_index = lines.index('Random:setSeed = on\n')
lines[r_index+1] = 'Random:seed = {}\n'.format(seed)
lines[0] = 'Main:numberOfEvents = {}\n'.format(int(arguments['<n_events>']))
z_index = lines.index('!1MASS\n')
wzl_index = lines.index('!2MASS\n')
wzh_index = lines.index('!3MASS\n')
qcd_index = lines.index('HardQCD:all = on\n')

if process == "wprime":
    lines[wzl_index] = '34:m0={}\n'.format(arguments['--boson-mass'])
    lines = lines[:z_index] + lines[wzl_index:wzh_index]

elif process == "qcd":
    ind = lines.index('ptHatMin.str()\n')
    lines[ind] = 'PhaseSpace:pTHatMin  ={}\n'.format(arguments['--pt-hat-min'])
    lines[ind+1] = 'PhaseSpace:pTHatMin  ={}\n'.format(arguments['--pt-hat-max'])
    lines = lines[:z_index] + lines[qcd_index:]
else:
    raise ValueError("process must be wprime/qcd")

home = path.expanduser('~')

pythia_card = path.abspath(path.join(home, '.tmppythcard.cmnd'))  # temporary pythia config file
with open(pythia_card, "w") as f:
    f.writelines(lines)


updir = path.abspath(path.join(file_dir, pardir))
delphes_card = path.abspath(path.join(updir, 'configs', 'delphes_card.tcl'))
shell_command = ['DelphesPythia8', delphes_card, pythia_card, fname]
t0 = time.time()
subprocess.call(shell_command)
print("Runtime for {}".format(arguments['<n_events>']))
print(time.time()-t0)
remove(pythia_card)
