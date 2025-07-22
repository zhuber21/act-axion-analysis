#!/bin/bash -l
# For getting best fit amp for full timestream for ALP timestream analysis on NERSC Perlmutter
# Submit this script as: "./run_assess_detection.sh" instead of "sbatch run_assess_detection.sh"

# Prepare user env needed for Slurm batch job
# such as module load, setup runtime environment variables, or copy input files, etc.
# Basically, these are the commands you usually run ahead of the srun command

conda activate /global/common/software/act/zhuber/alp_mpi_env
module load cray-mpich-abi

# I believe this should be the same as srun's number of logical cpus
# when OMP_PLACES=threads - this ensures the FFTs can use all threads
export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export RUN_TAG=run_20250721   # CHANGE THIS FOR EACH RUN!!

# Generate the Slurm batch script below with the here document,
# then when sbatch the script later, the user env set up above will run on the login node
# instead of on a head compute node (if included in the Slurm batch script),
# and inherited into the batch job.

cat << EOF > prepare-detection-env.sl
#!/bin/bash
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --account=mp107b
#SBATCH --mail-user=zbh5@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -J $RUN_TAG
#SBATCH --time 00:20:00
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=128
#SBATCH --output=/pscratch/sd/z/zbh5/results/$RUN_TAG.out

srun -n 5120 -c 2 --cpu_bind=cores python3 get_amp_assess_detection.py assess_detection_config.yaml $RUN_TAG

# Call collect_assess_detection_npy_files.py to group all output npy files for the splits of the sims into single npy files
python3 collect_assess_detection_npy_files.py /pscratch/sd/z/zbh5/results/assess_detection_results_$RUN_TAG/
# Amend path for output result copying location to match job name
cp -r /pscratch/sd/z/zbh5/results/assess_detection_results_$RUN_TAG/ /global/homes/z/zbh5/alp_outputs/assess_detection_results_$RUN_TAG/
cp /pscratch/sd/z/zbh5/results/$RUN_TAG.out /global/homes/z/zbh5/alp_outputs/assess_detection_results_$RUN_TAG/
EOF

# Now submit the batch job
sbatch prepare-detection-env.sl
