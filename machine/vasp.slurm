#!/bin/bash -l 
#SBATCH --partition=k2-hipri        
#SBATCH --job-name=default_slurm         
#SBATCH --time=24:00:00             
#SBATCH --nodes=1-1          
#SBATCH --ntasks=32           
#SBATCH --cpus-per-task=1     
#SBATCH --mem=10G             
#SBATCH --output=slurm.o%j    
#SBATCH --error=slurm.e%j     

module purge
module load services/s3cmd
module load libs/intel/2016u1
module load libs/intel-mkl/2016u1/bin
module load mpi/intel-mpi/2016u1/bin

mpirun -n 32 /mnt/scratch/chemistry-apps/dkb01416/vasp/installed/intel-2016/5.4.1-TS/vasp_std 2>&1 > vasp.out