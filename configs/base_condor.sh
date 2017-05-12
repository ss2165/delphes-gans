#!/bin/bash
export UUIIDD='filler'
if ( [ $# -ne 1 ] ) ; then
export TMPF=/tmp/ss2165.fakestdin.'filler'.tmp
#export TMPF=/tmp/ss2165.fakestdin.`increment`.tmp
cat > $TMPF
echo Executable = $TMPF >  /tmp/ss2165.condor.$UUIIDD.tmp
else
echo Executable = `/usera/ss2165/scripts/fqfilename $1` > /tmp/ss2165.condor.$UUIIDD.tmp
fi
cat << EOF >> /tmp/ss2165.condor.$UUIIDD.tmp
#Requirements = (Arch == "INTEL" || Arch == "X86_64") && (POOL == "GENERAL" || POOL == "GEN_FARM") && Memory > 700 && LoadAvg < 0.8 && (OSTYPE == "SLC3")
#Requirements = (Arch == "INTEL" || Arch == "X86_64") && (POOL == "GENERAL" ) && (OSTYPE == "SLC6" && Machine != "pced.hep.phy.cam.ac.uk" && Machine != "pcey.hep.phy.cam.ac.uk" && Machine != "pcfp.hep.phy.cam.ac.uk" && Machine != "pcfo.hep.phy.cam.ac.uk" )
Requirements = (Arch == "INTEL" || Arch == "X86_64") && (POOL == "GENERAL" || POOL == "GEN_FARM" ) && (OSTYPE == "SLC6" && Machine != "pced.hep.phy.cam.ac.uk" && Machine != "pcey.hep.phy.cam.ac.uk" && Machine != "pcfp.hep.phy.cam.ac.uk" && Machine != "pcfo.hep.phy.cam.ac.uk"  && Machine != "pcla.hep.phy.cam.ac.uk")
#Requirements = (Arch == "INTEL" || Arch == "X86_64") && (POOL == "GEN_FARM" ) && (OSTYPE == "SLC6" && Machine != "pced.hep.phy.cam.ac.uk" && Machine != "pcey.hep.phy.cam.ac.uk" && Machine != "pcfp.hep.phy.cam.ac.uk" && Machine != "pcfo.hep.phy.cam.ac.uk"  && Machine != "pcla.hep.phy.cam.ac.uk")
Universe   = vanilla
output = $HOME/condor.$UUIIDD.output
error = $HOME/condor.$UUIIDD.error
Log = $HOME/condor.$UUIIDD.log
getenv = true
copy_to_spool           = true
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
# Rank = Memory
Rank = KFlops
# Rank = Mips
Queue
EOF
condor_submit /tmp/ss2165.condor.$UUIIDD.tmp
#rm /tmp/ss2165.condor.$UUIIDD.tmp
echo Using /tmp/ss2165.condor.$UUIIDD.tmp
echo Submitted job condor . $UUIIDD