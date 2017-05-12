"""
bigdatarun.py personal batch condor submit script
generate lots of data of specified kind 'wprime' or 'qcd' - remember to change command
For batch of jobs, edits condor_submit script, writes job script, submits job scipt  
"""
import subprocess
import os

jobran = range(100) # number of jobs
runset = 4 # label of run
for jobno in jobran:
    seed = runset*(jobno+1)
    padjob = '{0:03d}'.format(jobno)
    jobfile = 'q_job{}.sh'.format(padjob)
    with open(os.path.abspath("/usera/ss2165/pt3proj/configs/base_condor.sh"), 'r') as f:
        lines = f.readlines()
    lines[1] = 'export UUIIDD=\'{}\'\n'.format(padjob)
    with open(os.path.abspath("/usera/ss2165/scripts/condor_submit_me"), 'w') as f:
        f.writelines(lines)
    with open(jobfile, 'w') as f:
        f.write("#!/bin/sh\n")
        f.write("python ~/pt3proj/jetimage/run_delphes.py qcd /r02/atlas/ss2165/qcd{}_{}.root 100000 -s {}\n".format(runset,padjob, seed))

    shell_command = ['condor_submit_me', jobfile]
    subprocess.call(shell_command)
