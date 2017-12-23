#!/bin/bash
#SBATCH -J test          # Job Name
#SBATCH -A A-ccsc
#SBATCH -o test.o%j      # Output and error file name (%j expands to jobID)
#SBATCH -N 1 
#SBATCH -n 1          # Total number of mpi tasks requested
#SBATCH -p normal      # Queue (partition) name -- normal, development, etc.
#SBATCH -t 5:00:00     # Run time (hh:mm:ss) - 24 hours

#./label_knl /tmp/video04.mov 64 &> log_knl_64_video4
ibrun ./label_skx very_short.mov 47 &> log

